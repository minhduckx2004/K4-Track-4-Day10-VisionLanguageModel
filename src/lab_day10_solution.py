"""Student-side implementation for Lab Day 10.

This module deliberately lives outside ``scripts/`` because the assignment
forbids editing grader and data-preparation utilities.  The functions are
written to run in Colab with a T4 GPU after ``pip install -r requirements-colab.txt``.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl


PROMPT_NORM1000 = (
    "Detect road damage in the image. Return only a valid JSON list. "
    "Each item must have label and bbox_2d, for example "
    '[{"label":"pothole","bbox_2d":[x1,y1,x2,y2]}]. '
    "Allowed labels are pothole, crack, and manhole. "
    "Coordinates must be integers on the 0-1000 normalized image scale. "
    "If there is no road damage, return []."
)

PROMPT_PIXEL = (
    "Detect road damage in the image. Return only a valid JSON list. "
    "Each item must have label and bbox_2d. "
    "Allowed labels are pothole, crack, and manhole. "
    "Coordinates must be integer pixel coordinates in the original image. "
    "If there is no road damage, return []."
)

PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


@dataclass
class RunPaths:
    root: Path = Path(".")
    work: Path = Path("work")
    evidence: Path = Path("report/evidence")
    adapters: Path = Path("work/adapters")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def yolo_to_box(label: str, yolo: Iterable[float], coord_scale: str = "norm1000",
                width: int = 640, height: int = 360) -> dict[str, Any]:
    """Convert one YOLO normalized box to the requested JSON answer format."""
    xc, yc, w, h = [float(v) for v in yolo]
    scale_x = 1000 if coord_scale == "norm1000" else width
    scale_y = 1000 if coord_scale == "norm1000" else height
    x1 = round((xc - w / 2) * scale_x)
    y1 = round((yc - h / 2) * scale_y)
    x2 = round((xc + w / 2) * scale_x)
    y2 = round((yc + h / 2) * scale_y)
    x1 = max(0, min(scale_x, x1))
    y1 = max(0, min(scale_y, y1))
    x2 = max(0, min(scale_x, x2))
    y2 = max(0, min(scale_y, y2))
    return {"label": label, "bbox_2d": [int(x1), int(y1), int(x2), int(y2)]}


def objects_to_answer(objects: list[dict[str, Any]], coord_scale: str = "norm1000",
                      width: int = 640, height: int = 360) -> str:
    boxes = [
        yolo_to_box(o["label"], o["yolo"], coord_scale=coord_scale, width=width, height=height)
        for o in objects
    ]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"]))
    return json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))


def answer_to_pixel_boxes(answer: str, width: int, height: int,
                          coord_scale: str = "norm1000") -> list[dict[str, Any]]:
    boxes = json.loads(answer)
    if coord_scale == "pixel":
        return boxes
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b["bbox_2d"]
        out.append({
            "label": b["label"],
            "bbox_2d": [
                round(x1 / 1000 * width), round(y1 / 1000 * height),
                round(x2 / 1000 * width), round(y2 / 1000 * height),
            ],
        })
    return out


def inspect_splits(splits_dir: str | Path = "data/splits") -> dict[str, Any]:
    summary: dict[str, Any] = {}
    sessions: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        rows = read_jsonl(Path(splits_dir) / f"{split}.jsonl")
        counts = {c: 0 for c in CLASSES}
        for r in rows:
            for o in r["objects"]:
                counts[o["label"]] += 1
        summary[split] = {"images": len(rows), **counts}
        sessions[split] = {r["session"] for r in rows}
    leaks = []
    for a in sessions:
        for b in sessions:
            if a < b and sessions[a] & sessions[b]:
                leaks.append((a, b, sorted(sessions[a] & sessions[b])[:5]))
    summary["session_leaks"] = leaks
    return summary


def draw_boxes(image_path: str | Path, boxes: list[dict[str, Any]], out_path: str | Path,
               color: str = "red") -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox_2d"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 12)), b["label"], fill=color)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def visualize_split_examples(split: str = "train", n: int = 6,
                             data_dir: str | Path = "data",
                             out_dir: str | Path = "report/figures") -> list[Path]:
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")[:n]
    paths = []
    for r in rows:
        answer = objects_to_answer(r["objects"], width=r["width"], height=r["height"])
        boxes = answer_to_pixel_boxes(answer, r["width"], r["height"])
        out = Path(out_dir) / f"{split}_{r['image_id']}.jpg"
        draw_boxes(Path(data_dir) / r["image_path"], boxes, out)
        paths.append(out)
    return paths


def load_qwen(model_id: str = "Qwen/Qwen3-VL-4B-Instruct"):
    """Load Qwen3-VL in 4-bit QLoRA mode, keeping visual and lm_head out of int8 skip list."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def _message(image: Image.Image, prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


@torch.inference_mode()
def predict_split(model, processor, split: str = "test", out_path: str | Path = "work/pred.jsonl",
                  data_dir: str | Path = "data", prompt: str = PROMPT_NORM1000,
                  batch_size: int = 4, max_new_tokens: int = 256,
                  limit: int | None = None) -> None:
    rows = read_jsonl(Path(data_dir) / "splits" / f"{split}.jsonl")
    if limit:
        rows = rows[:limit]
    out_rows = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        messages = []
        for r in batch:
            img = Image.open(Path(data_dir) / r["image_path"]).convert("RGB")
            messages.append(_message(img, prompt))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        out_rows.extend({"image_id": r["image_id"], "raw": raw.strip()} for r, raw in zip(batch, decoded))
    write_jsonl(out_path, out_rows)


class RoadDamageDataset(Dataset):
    def __init__(self, split: str = "train", data_dir: str | Path = "data",
                 prompt: str = PROMPT_NORM1000, coord_scale: str = "norm1000",
                 max_items: int | None = None):
        self.data_dir = Path(data_dir)
        self.prompt = prompt
        self.coord_scale = coord_scale
        self.rows = read_jsonl(self.data_dir / "splits" / f"{split}.jsonl")
        if max_items:
            self.rows = self.rows[:max_items]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        image = Image.open(self.data_dir / r["image_path"]).convert("RGB")
        answer = objects_to_answer(
            r["objects"], coord_scale=self.coord_scale, width=r["width"], height=r["height"]
        )
        return {"image": image, "prompt": self.prompt, "answer": answer, "image_id": r["image_id"]}


def encode_example(processor, image: Image.Image, prompt: str, answer: str) -> dict[str, torch.Tensor]:
    prompt_enc = processor.apply_chat_template(
        _message(image, prompt),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        _message(image, prompt, answer),
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("Prompt tokens are not a prefix of the full conversation.")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100
    full_enc["labels"] = labels
    # Sequence-like tensors are padded later, so remove only their batch axis.
    # Vision tensors such as image_grid_thw must stay 2D ([n_images, 3]).
    out = {}
    for k, v in full_enc.items():
        if isinstance(v, torch.Tensor) and k in {"input_ids", "attention_mask", "labels", "mm_token_type_ids"}:
            out[k] = v.squeeze(0)
        else:
            out[k] = v
    return out


def make_collate(processor):
    pad_id = processor.tokenizer.pad_token_id

    def collate(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [encode_example(processor, e["image"], e["prompt"], e["answer"]) for e in examples]
        batch = {
            "input_ids": pad_sequence([e["input_ids"] for e in encoded], batch_first=True, padding_value=pad_id),
            "attention_mask": pad_sequence(
                [e["attention_mask"] for e in encoded], batch_first=True, padding_value=0
            ),
            "labels": pad_sequence([e["labels"] for e in encoded], batch_first=True, padding_value=-100),
        }
        for key in ("pixel_values", "image_grid_thw", "second_per_grid_ts"):
            vals = [e[key] for e in encoded if key in e]
            if vals:
                batch[key] = torch.cat(vals, dim=0)
        if "mm_token_type_ids" in encoded[0]:
            batch["mm_token_type_ids"] = pad_sequence(
                [e["mm_token_type_ids"] for e in encoded], batch_first=True, padding_value=0
            )
        return batch

    return collate


def _module_target_regex(targets: str) -> str:
    llm = (
        r".*language_model\.layers\.\d+\."
        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    )
    if targets != "llm+vit":
        return llm
    vit = (
        r".*visual\.blocks\.\d+\."
        r"(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))$"
    )
    return rf"({llm})|({vit})"


def apply_setup(model, setup: dict[str, Any]):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = setup["lora"]
    target_regex = _module_target_regex(str(lora["targets"]))
    matched_modules = [n for n, _ in model.named_modules() if re.match(target_regex, n)]
    cfg = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)

    if setup.get("train_projector"):
        for name, p in model.named_parameters():
            if any(k in name for k in PROJECTOR_KEYS):
                p.data = p.data.float()
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"matched_lora_modules={len(matched_modules)} "
        f"trainable={trainable:,} total={total:,} ({100 * trainable / total:.3f}%)"
    )
    return model


def save_extra_trainable(model, path: str | Path) -> None:
    state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if any(k in n for k in PROJECTOR_KEYS) and p.requires_grad
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_extra_trainable(model, path: str | Path) -> None:
    state = torch.load(path, map_location="cpu")
    params = dict(model.named_parameters())
    missing = []
    for n, v in state.items():
        if n in params:
            params[n].data.copy_(v.to(params[n].device, dtype=params[n].dtype))
        else:
            missing.append(n)
    if missing:
        raise KeyError(f"Missing projector parameters: {missing[:3]}")


def _move_to_device(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def train_experiment(config_path: str | Path, data_dir: str | Path = "data",
                     work_dir: str | Path = "work", max_items: int | None = None) -> dict[str, Any]:
    setup = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed_everything(int(setup.get("seed", 42)))
    model, processor = load_qwen(setup["model_id"])
    model = apply_setup(model, setup)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    prompt = PROMPT_PIXEL if setup.get("coord_scale") == "pixel" else PROMPT_NORM1000
    dataset = RoadDamageDataset(
        "train", data_dir=data_dir, prompt=prompt,
        coord_scale=setup.get("coord_scale", "norm1000"), max_items=max_items
    )
    loader = DataLoader(
        dataset,
        batch_size=int(setup["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(processor),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(setup["lr"]), weight_decay=0.0)
    max_steps = int(setup["max_steps"])
    grad_accum = int(setup["grad_accum"])
    warmup = max(1, int(max_steps * 0.05))

    def lr_at(step: int) -> float:
        if step < warmup:
            return float(setup["lr"]) * (step + 1) / warmup
        t = (step - warmup) / max(1, max_steps - warmup)
        return float(setup["lr"]) * 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.amp.GradScaler("cuda")
    out_dir = Path(work_dir) / "adapters" / setup["name"].lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    step = 0
    model.train()
    iterator = iter(loader)
    opt.zero_grad(set_to_none=True)
    while step < max_steps:
        total_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _move_to_device(batch, model.device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            total_loss += loss.item()
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0:
            print(f"{setup['name']} step={step}/{max_steps} loss={total_loss:.4f} lr={lr_at(step):.2e}")
        if step % 100 == 0 or step == max_steps:
            model.save_pretrained(out_dir)
            save_extra_trainable(model, out_dir / "extra_trainable.pt")

    minutes = (time.time() - start) / 60
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    run = {
        "name": setup["name"],
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "peak_vram_gb": float(round(peak, 3)),
        "train_minutes": float(round(minutes, 3)),
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def infer_adapter(config_path: str | Path, split: str = "test", data_dir: str | Path = "data",
                  work_dir: str | Path = "work") -> Path:
    from peft import PeftModel

    setup = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model, processor = load_qwen(setup["model_id"])
    adapter_dir = Path(work_dir) / "adapters" / setup["name"].lower()
    model = PeftModel.from_pretrained(model, adapter_dir)
    extra = adapter_dir / "extra_trainable.pt"
    if extra.exists():
        load_extra_trainable(model, extra)
    prompt = PROMPT_PIXEL if setup.get("coord_scale") == "pixel" else PROMPT_NORM1000
    out = Path(work_dir) / f"{setup['name'].lower()}.jsonl"
    predict_split(model, processor, split=split, out_path=out, data_dir=data_dir, prompt=prompt)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def score_prediction(pred_path: str | Path, name: str, out_path: str | Path,
                     split: str = "test", coord: str = "norm1000",
                     meta: dict[str, Any] | None = None) -> None:
    cmd = [
        "python", "scripts/score.py",
        "--pred", str(pred_path),
        "--name", name,
        "--out", str(out_path),
        "--split", split,
        "--coord", coord,
    ]
    for k, v in (meta or {}).items():
        cmd += ["--meta", f"{k}={v}"]
    subprocess.run(cmd, check=True)


def score_experiment(config_path: str | Path, split: str = "test",
                     work_dir: str | Path = "work") -> Path:
    setup = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    name = setup["name"]
    pred = Path(work_dir) / f"{name.lower()}.jsonl"
    out = Path("report/evidence") / f"{name.lower()}.json"
    run_path = Path(work_dir) / "adapters" / name.lower() / "run.json"
    meta = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    coord = setup.get("coord_scale", "norm1000")
    score_prediction(pred, name, out, split=split, coord=coord, meta=meta)
    return out


def reward_iou(completion: str, gt_boxes: list[dict[str, Any]]) -> float:
    from scripts.score import iou_xyxy, parse_output

    pred = parse_output(completion)
    if pred is None:
        return 0.0
    gt = [{"label": g["label"], "bbox": tuple(g.get("bbox", g.get("bbox_2d")))} for g in gt_boxes]
    if not gt and not pred:
        return 1.0
    pairs = []
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if g["label"] == p["label"]:
                pairs.append((iou_xyxy(g["bbox"], p["bbox"]), i, j))
    pairs.sort(reverse=True)
    used_g, used_p, total_iou = set(), set(), 0.0
    for value, i, j in pairs:
        if value < 0.5:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total_iou += value
    return 0.2 + 0.8 * total_iou / max(len(gt), len(pred), 1)


def run_reward_tests() -> None:
    gt = [{"label": "pothole", "bbox_2d": [100, 100, 200, 200]}]
    assert reward_iou('[{"label":"pothole","bbox_2d":[100,100,200,200]}]', gt) == 1.0
    assert reward_iou("not json", gt) == 0.0
    assert reward_iou('[{"label":"crack","bbox_2d":[100,100,200,200]}]', gt) == 0.2
    assert reward_iou("[]", []) == 1.0
    extra = '[{"label":"pothole","bbox_2d":[100,100,200,200]},{"label":"pothole","bbox_2d":[300,300,400,400]}]'
    assert 0.55 < reward_iou(extra, gt) < 0.65
