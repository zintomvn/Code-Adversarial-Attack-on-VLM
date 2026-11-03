# Chạy My-Attack-V3 bằng Python trên Molab

Mở Terminal của Molab trong runtime GPU được tài khoản cho phép. Nếu tài khoản vẫn báo `Your account is restricted`, cần giải quyết qua hỗ trợ Molab trước khi chạy.

## 1. Lấy code và cài môi trường

```bash
git clone --single-branch --branch my-attack-v3-mpc-layout https://github.com/zintomvn/Code-Adversarial-Attack-on-VLM.git /marimo/vlm_attack_v3
cd /marimo/vlm_attack_v3/src/My-Attack-V3
git rev-parse HEAD
uv venv --python 3.11.15 /marimo/v3_env
source /marimo/v3_env/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt kagglehub==0.3.13
python generate_adversarial_examples_MPCAttack.py --help
python -X utf8 -m unittest discover -s tests -p 'test_*.py' -v
```

V3 chứa dependencies MPC cần thiết; không cần khởi tạo submodule. Lưu commit SHA, `uv pip freeze` và thông tin GPU vào bundle của run.

## 2. Duyệt phạm vi trước khi inference

Plan: [MY-ATTACK-V3-MPC-LAYOUT-N2-plan.md](../../experiments/MY-ATTACK-V3-MPC-LAYOUT-N2-plan.md).
Plan hiện **PENDING**, chưa được duyệt. Lệnh attack dưới đây chỉ dùng sau khi người dùng duyệt rõ ràng plan này và approval được ghi nhận trong plan. Không tự đổi trạng thái để vượt gate. Approval của bản chạy cũ không tự áp dụng cho executable mới.

Trước inference, kiểm tra đường backup bền vững bằng cách upload một file thử lên Drive rồi tải/đọc lại. Core hiện ghi file local, **không tự đồng bộ Drive hoặc Google Sheet**. `/marimo/outputs` có thể mất khi runtime bị thu hồi.

## 3. Chuẩn bị dữ liệu đúng phiên bản

```bash
python - <<'PY'
from pathlib import Path
import kagglehub

root = Path(kagglehub.dataset_download('zintom/blackbox-adversarial-attack-dataset/versions/1'))
for sample_id in (0, 1):
    assert (root / f'images/bigscale/nips17/{sample_id}.png').is_file()
    assert (root / f'images/target_images/1/{sample_id}.jpg').is_file()
Path('/marimo/v3_dataset_path.txt').write_text(str(root), encoding='utf-8')
print(root)
PY
```

## 4. Chạy hai mẫu bằng Python sau khi plan được duyệt

Chạy foreground trong Terminal để thấy lỗi và các điểm dừng review:

```bash
source /marimo/v3_env/bin/activate
cd /marimo/vlm_attack_v3/src/My-Attack-V3
DATA_ROOT="$(cat /marimo/v3_dataset_path.txt)"
OUTPUT=/marimo/outputs/v3_mpc_n2
PLAN=/marimo/vlm_attack_v3/experiments/MY-ATTACK-V3-MPC-LAYOUT-N2-plan.md
mkdir -p "$OUTPUT"
python -u generate_adversarial_examples_MPCAttack.py \
  --cle_data_path "$DATA_ROOT/images/bigscale" \
  --tgt_data_path "$DATA_ROOT/images/target_images" \
  --output "$OUTPUT" \
  --batch_size 1 --num_samples 2 --start_index 0 \
  --input_res 224 --clip_backbone B16 B32 Laion --ensemble True \
  --steps 300 --epsilon 16 --alpha 1 \
  --num_negatives 10 --lam 0.6 --tau 0.2 --omega 2 \
  --seed 2026 --save_every 30 --max_seconds 4500 \
  --model_name Qwen2.5-VL-3B-Instruct --approved_plan "$PLAN"
```

Muốn gọi từ Python cell, dùng `subprocess.run([...], cwd='/marimo/vlm_attack_v3/src/My-Attack-V3', check=True)` với interpreter `/marimo/v3_env/bin/python` và các arguments như trên. Không chạy detached vì cần theo dõi review và ngân sách.

## 5. Review, backup và kết quả

Runner dừng ở bank, update1 và complete của từng mẫu. Xem file trong `$OUTPUT/reviews/`: `0_data_bank.json`, `0_update1.json`, `0_complete.json`, tương tự mẫu1. Review phải chỉ rõ inspector, notes, evidence hash và trạng thái dựa trên ảnh/crop/bank/loss thực tế. Complete cần đủ bảy mục checklist, PASS hoặc N/A có lý do và artifact. Không điền PASS hàng loạt để tiếp tục.

Tại mỗi gate, đóng gói snapshot, upload vào thư mục con riêng trong [Drive thực nghiệm](https://drive.google.com/drive/folders/1daramuahKwdAVGQtsDZTzy1v0OSG-kiI), kiểm tra đọc lại rồi mới cho qua gate. Ví dụ tạo ZIP local (chưa phải backup bền vững):

```bash
python - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import shutil

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
archive = Path(shutil.make_archive(f'/marimo/v3_snapshot_{stamp}', 'zip', '/marimo/outputs/v3_mpc_n2'))
print(archive)
print(hashlib.sha256(archive.read_bytes()).hexdigest())
PY
```

Tải ZIP ra khỏi runtime/upload Drive và đối chiếu SHA256. Nếu backup thất bại, dừng và giữ partials. Saved adversarial PNG nằm dưới `img/<class>/<id>.png`; caption, provenance, bank, checkpoint và review nằm trong output. Caption source/target mặc định ở `descriptions_src_tgt` trong output.

Runner nối attack → verify PNG → caption local → diagnostic/review. Không cấu hình GPT/API judge trong pilot; official ASR vẫn PENDING. Sau mỗi lần chạy, kể cả partial, ghi kết quả/checklist/artifacts vào docs và dòng mới của [Google Sheet](https://docs.google.com/spreadsheets/d/1VbuyjnPBHmd_J09FSouRQU41NCcWQovyyGlijlqthW8/edit#gid=312174925), đọc lại để xác minh; giữ nguyên baseline. Nếu sync lỗi, lưu bản cập nhật pending vào bundle.

Khi lỗi, giữ nguyên output và traceback. Post-processing có cache riêng để tiếp tục trên PNG đã lưu; không xóa output hoặc chạy lại attack để cứu caption. Interrupted attack không tự retry. Không tăng lên toàn bộ dữ liệu trước khi pilot vượt checks và plan tiếp theo được duyệt.
