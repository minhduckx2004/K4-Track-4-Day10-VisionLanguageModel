# Bao cao loi va luu y ky thuat

File nay tom tat cac bay ky thuat can nho khi chay lab tren Colab T4 va khi nop bai.

## 1. Unicode tren Windows

Trieu chung: PowerShell hien tieng Viet bi loi hoac script bao `UnicodeEncodeError`.

Cach xu ly:

```powershell
$env:PYTHONIOENCODING="utf-8"
python scripts/check_submission.py
```

Co the dung them `chcp 65001` neu console van hien sai.

## 2. Du lieu Kaggle

Trieu chung: `prepare_data.py --kaggle` bao loi xac thuc.

Nguyen nhan thuong gap:

- Chua tao Kaggle API token.
- Chua them `KAGGLE_USERNAME` va `KAGGLE_KEY` vao Colab Secrets.
- Secret chua duoc bat quyen truy cap notebook.

Cach xu ly: tao lai token trong Kaggle Account Settings, them vao Colab Secrets, roi nap bang `google.colab.userdata`.

## 3. Ro ri du lieu do random split

Khong duoc tu chia anh ngau nhien. Anh la frame cat tu video, nen frame gan nhau rat giong nhau. Random split se dua cac frame gan trung lap vao ca train va test, lam F1 cao gia.

Dung duy nhat split sinh boi:

```bash
python scripts/prepare_data.py --kaggle
```

## 4. Sai thang toa do

Qwen3-VL grounding dung toa do normalized `0-1000`, khong phai pixel mac dinh. Neu output JSON hop le nhung F1 gan 0, hay kiem tra:

- Ham YOLO -> JSON co nhan voi `1000`.
- Prompt co noi ro `0-1000`.
- Neu chay E4 pixel thi phai cham bang `--coord pixel`.

## 5. Mask loss sai

Chi tinh loss tren token cau tra loi assistant. Prompt va token anh phai co label `-100`.

Kiem tra nhanh trong notebook: decode cac token co `labels != -100`; ket qua phai la JSON dap an va token ket thuc assistant.

## 6. Projector cua E2/E3

PEFT `model.save_pretrained()` chi luu LoRA adapter. Projector duoc mo full train trong E2/E3 nen phai luu rieng bang `torch.save`.

Trong repo nay ham `save_extra_trainable()` luu `visual.merger` va `visual.deepstack_merger_list` vao:

```text
work/adapters/e2/extra_trainable.pt
work/adapters/e3/extra_trainable.pt
```

Khong commit cac file `.pt` nay.

## 7. CUDA out of memory

Neu Colab T4 bao OOM:

- Giam `batch_size` tu `2` xuong `1`.
- Tang `grad_accum` tu `4` len `8`.
- Dam bao da bat gradient checkpointing.
- Chay tung E1, E2, E3 rieng, giai phong runtime giua cac lan neu can.

## 8. Loss `nan` hoac `inf`

Nguyen nhan thuong gap la tran so fp16, dac biet khi train projector.

Cach xu ly:

- Giam learning rate xuong `1e-4`.
- Dam bao projector da duoc dua ve `float32`.
- Kiem tra loss mask, khong tinh loss tren prompt/visual tokens.

## 9. `parse_valid_rate` thap

Nguyen nhan co the la model sinh van ban thua, markdown fence, hoac bi cat output.

Cach xu ly:

- Giu prompt ngan va bat model "Return only JSON".
- Tang `max_new_tokens=256`.
- Kiem tra output tho trong `work/e*.jsonl`.

## 10. Quy tac nop bai

Truoc khi push:

```bash
git status --short
git rev-list --count HEAD
python scripts/check_submission.py
```

Chi nop khi `check_submission.py` bao `0 FAIL` va repo GitHub ca nhan dang Public.
