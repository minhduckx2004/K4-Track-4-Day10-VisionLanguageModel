# Checklist nop bai Lab Day 10

Dung file nay de tu ra soat truoc khi nop link GitHub public len VinUni LMS.

## 1. Repository

- [ ] Da fork repo goc ve tai khoan GitHub ca nhan.
- [ ] Repo GitHub ca nhan dang o che do Public.
- [ ] Khong clone/lap bai truc tiep tren repo `VinUni-AI20k`.
- [ ] `git rev-list --count HEAD` tra ve it nhat `5`.
- [ ] `git status --short` khong co file cam dang track.

## 2. Moi truong Colab

- [ ] Runtime da chon `T4 GPU`.
- [ ] Da chay `!pip -q install -r requirements-colab.txt`.
- [ ] Da tao Colab Secrets `KAGGLE_USERNAME` va `KAGGLE_KEY`.
- [ ] Da mount Google Drive de sao luu adapter/checkpoint.
- [ ] Da chay `python -m pytest -q` va thay `68 passed, 1 skipped`.

## 3. Du lieu

- [ ] Da chay `python scripts/prepare_data.py --kaggle`.
- [ ] Co `data/images/`, `data/splits/train.jsonl`, `val.jsonl`, `test.jsonl`, `meta.json`.
- [ ] `dataset_version` trong `data/splits/meta.json` bat dau bang `d10-`.
- [ ] Split khop: train 1422 anh, val 276 anh, test 311 anh.
- [ ] Khong co session nao xuat hien o hai split.
- [ ] Da tao it nhat 6 anh minh hoa box trong `report/figures/`.

## 4. Thi nghiem

- [ ] E0 da sinh `work/e0.jsonl` va `report/evidence/e0.json`.
- [ ] E1 da train, infer va sinh `report/evidence/e1.json`.
- [ ] E2 da train, infer va sinh `report/evidence/e2.json`.
- [ ] E3 da train, infer va sinh `report/evidence/e3.json`.
- [ ] E4 da chay ablation don bien va sinh it nhat mot file `report/evidence/e4*.json`.
- [ ] E1-E3 co du `meta.trainable_params`, `meta.peak_vram_gb`, `meta.train_minutes`.
- [ ] E1 co `parse_valid_rate >= 0.95`.
- [ ] F1(E1) cao hon F1(E0) it nhat `0.10`.

## 5. Bao cao

- [ ] Da chay `python -m src.report_from_evidence --name "Ho Ten"` sau khi co evidence.
- [ ] `REPORT.md` da dien ho ten va `dataset_version` that.
- [ ] Bang tong hop lay dung so tu `report/evidence/*.json`.
- [ ] 10 cau hoi da duoc tra loi bang so lieu that cua minh.
- [ ] Khong con chuoi `CHUA_CHAY`, `CHUA_DIEN`, hoac placeholder nao trong `REPORT.md`.
- [ ] Khong con chuoi `<dien>` hoac `<điền>` trong `REPORT.md`.

## 6. Kiem tra truoc khi nop

- [ ] Khong commit `data/`, `work/`, `.env`, adapter, checkpoint, hoac file trong so.
- [ ] Da chay `$env:PYTHONIOENCODING="utf-8"; python scripts/check_submission.py` tren Windows.
- [ ] Ket qua `check_submission.py` la `0 FAIL`.
- [ ] Da push toan bo commit len GitHub ca nhan.
- [ ] Link nop LMS co dang `https://github.com/<your-username>/K4-Track-4-Day10-VisionLanguageModel`.
