Ho ten: CHUA_DIEN
`dataset_version`: CHUA_CHAY

## Bang tong hop

| Ma | Setup | Tham so huan luyen | VRAM dinh (GB) | Phut | F1 | CI95 | FP/anh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | zero-shot | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |
| E1 | LoRA LLM | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |
| E2 | + projector | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |
| E3 | + LoRA ViT | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |
| E4 | rank ablation r=4 | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY | CHUA_CHAY |

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
