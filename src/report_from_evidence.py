"""Fill the numeric summary table in REPORT.md from report/evidence/*.json.

Run after scoring E0-E4:

    python -m src.report_from_evidence --name "Your Name"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SETUPS = {
    "e0": "zero-shot",
    "e1": "LoRA LLM",
    "e2": "+ projector",
    "e3": "+ LoRA ViT",
    "e4": "rank ablation r=4",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "CHUA_CHAY"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _row(code: str, ev: dict | None) -> str:
    if ev is None:
        return f"| {code.upper()} | {SETUPS[code]} | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |"
    meta = ev.get("meta", {})
    micro = ev.get("micro", {})
    per = ev.get("per_class", {})
    ci = ev.get("f1_ci95", [None, None])
    ci_text = "CHUA_CHAY" if ci[0] is None else f"{ci[0]:.3f}-{ci[1]:.3f}"
    return (
        f"| {code.upper()} | {SETUPS[code]} | {_fmt(meta.get('trainable_params'), 0)} | "
        f"{_fmt(meta.get('peak_vram_gb'))} | {_fmt(meta.get('train_minutes'), 1)} | "
        f"{_fmt(micro.get('f1'))} | {ci_text} | {_fmt(ev.get('fp_per_image'))} | "
        f"{_fmt(per.get('pothole', {}).get('f1'))} | {_fmt(per.get('crack', {}).get('f1'))} | "
        f"{_fmt(per.get('manhole', {}).get('f1'))} |"
    )


def build_report(name: str) -> str:
    ev_dir = Path("report/evidence")
    evidence = {k: _load(ev_dir / f"{k}.json") for k in ("e0", "e1", "e2", "e3")}
    e4_files = sorted(ev_dir.glob("e4*.json"))
    evidence["e4"] = _load(e4_files[0]) if e4_files else None
    dataset_version = next(
        (e["dataset_version"] for e in evidence.values() if e and e.get("dataset_version")),
        "CHUA_CHAY",
    )
    rows = "\n".join(_row(k, evidence[k]) for k in ("e0", "e1", "e2", "e3", "e4"))
    return f"""Ho ten: {name}
`dataset_version`: {dataset_version}

## Bang tong hop

| Ma | Setup | Tham so huan luyen | VRAM dinh (GB) | Phut | F1 | CI95 | FP/anh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
{rows}

## 10 cau hoi

### Câu 1 - Kien truc
Qwen3-VL-4B gom LLM khoang 4.0B tham so, ViT khoang 415M tham so, projector gom visual.merger khoang 27M va visual.deepstack_merger_list khoang 82M. E1 chi cap nhat LoRA tren LLM; E2 them projector day du; E3 them LoRA tren cac khoi attention/MLP cua ViT.

### Câu 2 - E0
Lay parse_valid_rate va vi du output tho tu report/evidence/e0.json va work/e0.jsonl sau khi chay. Hai loi can kiem tra la JSON khong hop le/bi boc markdown va sai thang toa do hoac bia box.

### Câu 3 - Thang 0-1000
Qwen grounding dung toa do tuong doi 0-1000 de doc lap voi kich thuoc anh. Neu train/cham lech pixel voi norm1000, IoU bi sai ty le nen F1 thuong gan 0 du JSON hop le. Kiem chung bang E4 coord_scale hoac chay score.py voi --coord nguoc lai.

### Câu 4 - Mask loss
Chi tinh loss tren cau tra loi vi prompt va token anh la dieu kien dau vao, khong phai noi dung can bat model sinh lai. Neu tinh loss tren prompt/token anh, gradient bi tieu ton de hoc copy template va token dac biet, lam yeu tin hieu hoc JSON box.

### Câu 5 - Projector va 4-bit
Projector can cap nhat day du nen khong duoc de no nam trong trong so luong tu 4-bit; tham so 4-bit khong phu hop cho full fine-tune truc tiep. Cach xu ly la skip visual/lm_head khi quantize va dua visual.merger, visual.deepstack_merger_list ve float32 khi E2/E3.

### Câu 6 - Split theo phien quay
Chia ngau nhien lam ro ri vi cac frame lien ke rat giong nhau, khien train va test co canh gan trung lap va F1 cao gia. Dien so anh/box tung split tu inspect_splits() sau khi prepare_data.py sinh data/splits.

### Câu 7 - So sanh E1/E2/E3
Dien ket luan tu bang tren sau khi chay. Chi nen noi setup tot hon ro rang neu chen lech F1 vuot dang ke so voi CI95 va chi phi VRAM/thoi gian chap nhan duoc.

### Câu 8 - Tap test nho
Tap test 311 anh lam CI95 co the rong, nen cac chen lech F1 nho hoac CI chong lap khong du manh de ket luan setup A hon setup B.

### Câu 9 - Loi theo lop
Dua vao F1 tung lop va anh minh hoa trong report/figures. Lop kem thuong do box nho, it mau trong split, hoac bien dang hinh hoc de nham voi nen duong.

### Câu 10 - E4
E4 hien cau hinh rank=4 so voi baseline E1 rank=16, chi doi mot bien la LoRA rank. Dien F1/CI95 cua E4 va E1 sau khi chay de ket luan rank thap co lam giam ket qua dang tin hay khong.

## Nhin lai
Can bo sung cac so CHUA_CHAY bang ket qua that tu Colab truoc khi nop. Khong commit data/, work/, adapter hoac trong so model.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="CHUA_DIEN")
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()
    Path(args.out).write_text(build_report(args.name), encoding="utf-8")


if __name__ == "__main__":
    main()
