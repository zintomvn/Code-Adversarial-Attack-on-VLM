# Thực nghiệm giả thuyết KLTN v4.3.2-h1 — 200 cặp

## Phạm vi đã chuẩn bị

- 200 cặp đầu theo sorted ImageFolder pairing của dataset gốc, ảnh không resize/re-encode lúc đóng gói.
- 200 source duy nhất, 199 target duy nhất; không loại mẫu theo kết quả attack.
- 300 updates, epsilon 16/255, alpha 1 pixel, seed 2023, ensemble B16/B32/Laion, hybrid FP32.
- Model revisions attack được khóa theo run 1 ảnh đã thành công trên Molab ngày 16/10/2026;
  notebook gốc trên Molab có SHA-256 khớp bản local.
- 8 variants chính dùng chung captions/negative bank trong một run: `foa_reference`, `mpc_clip`,
  `hybrid_plain`, `hybrid_negative`, `hybrid_fixed`, `hybrid_dynamic`,
  `hybrid_random_region`, `hybrid_random_neg`.
- 2 controls H4: `hybrid_plain` với MPC weight = 0, và `hybrid_plain` với fusion lambda = 1.
  Chúng dùng lại captions và model revisions từ run chính.
- Tổng cộng 2.000 attack samples / 600.000 updates. Không phải tổng 200 ảnh chia giữa variants.
- Full MPC là baseline tùy chọn, chưa nằm trong campaign này; không gọi `mpc_clip` là full MPC.

## Ma trận đối chiếu

| Giả thuyết | Control | Treatment |
|---|---|---|
| H1: negative loss | hybrid_plain | hybrid_negative |
| H1: hard mining | hybrid_random_neg | hybrid_dynamic |
| H2: vị trí attention | hybrid_random_region | hybrid_dynamic |
| H3: refresh attention | hybrid_fixed | hybrid_dynamic |
| H4: thêm MPC term | h4_mpc_zero / hybrid_plain | primary / hybrid_plain |
| H4: image-text fusion | h4_image_only / hybrid_plain | primary / hybrid_plain |

Giữ toàn bộ 200 cặp trong báo cáo. Với H1 hard mining, báo thêm phân tích bank size > 2;
khi bank chỉ có 2 negatives và k = 2, hard và random lấy cùng một tập nên không kiểm chứng
được hard selection. Không tự sinh lại bank để chọn các mẫu có kết quả đẹp.

## Files và vị trí

Hai ZIP riêng:

- `attack_data_n200.zip`: 400 file ảnh gốc, `pairs_200.jsonl`, metadata/checksums.
- `molab_hypotheses_n200_code.zip`: notebook attack/evaluation gốc, campaign driver,
  kế hoạch JSON và hướng dẫn này. Không chứa credentials/API keys.

Đã tạo thư mục Drive riêng tư:
<https://drive.google.com/drive/folders/1daramuahKwdAVGQtsDZTzy1v0OSG-kiI>

Upload hai ZIP qua Molab Files hoặc tải từ Drive bằng một kết nối đã được người dùng chấp thuận.
Giải nén code vào `/marimo/hypotheses_n200` và data vào `/marimo/hypotheses_n200/data`.
Đường dẫn tuyệt đối này phải giữ nguyên khi resume vì notebook ghi paths trong run identity.

## Cài và chạy

Trong cell Molab:

```python
from pathlib import Path
from zipfile import ZipFile
import subprocess
import sys

_project = Path('/marimo/hypotheses_n200')
_project.mkdir(exist_ok=True)
with ZipFile('molab_hypotheses_n200_code.zip') as _archive:
    _archive.extractall(_project)
with ZipFile('attack_data_n200.zip') as _archive:
    _archive.extractall(_project / 'data')
subprocess.run([sys.executable, str(_project / 'molab_hypothesis_campaign.py'),
                'check', '--root', str(_project)], check=True)
```

Sau đó chạy `setup`, rồi `attack`, dùng hàm sau để stream log:

```python
def run_campaign(mode):
    import subprocess
    import sys
    command = [sys.executable, '-u', '/marimo/hypotheses_n200/molab_hypothesis_campaign.py',
               mode, '--root', '/marimo/hypotheses_n200', '--max-hours', '10']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        code = process.wait()
        if code not in (0, 75):
            raise RuntimeError(f'Campaign failed: {code}')
        if code == 75:
            print('Paused at a sample/text checkpoint. Resume in the next session.')
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)

run_campaign('setup')
# Chạy riêng sau khi setup hoàn tất và kết nối backup được cho phép:
# run_campaign('attack')
```

`setup` tạo hai venv dùng Torch/CUDA có sẵn, không thay Torch hệ thống.
Attack dùng transformers 4.57.1; evaluation dùng transformers 4.57.6. Driver thực thi
nguyên code cells của từng notebook và chỉ override cấu hình/vận hành.

## Backup và quyền truy cập

Driver dùng Google Drive fsspec với `token='cache'`; nó không tự mở OAuth trong worker.
Kết nối Drive cần được tạo trước trong Molab và được người dùng chấp thuận. Mặc định backup
vào `KLTN_Molab_Hypotheses_n200_v432_h1/results`. Trước khi chạy model, driver thử upload
và đọc lại một probe để kiểm tra checksum. Thiếu kết nối sẽ dừng trước attack.

Từng mẫu hoàn tất được upload thành ZIP riêng, kèm metadata run và checksum đọc lại.
Trong text preparation, metadata được checkpoint mỗi 5 lần ghi. Đến giới hạn thời gian,
driver dừng ở ranh giới mẫu hoặc text checkpoint với exit code 75; không reset history FOA
bằng cách chia lại selection thành những run nhỏ không tương đương.

Phiên Molab có giới hạn theo nền tảng; 2.000 attack samples sẽ cần nhiều phiên. Test 1 ảnh
đã ghi nhận khoảng 83 giây cho vòng hybrid_dynamic, chưa gồm text preparation/model load/backup.
Không suy ra ETA chính xác cho mọi variant từ một mẫu.

**Trạng thái quyền tại lúc chuẩn bị:** automatic approval review đã chặn việc mở OAuth
cho client pydata-google-auth vì yêu cầu quyền đọc/ghi toàn bộ Drive và offline access.
Chưa có token được cấp từ luồng đó. Người dùng có thể upload hai ZIP thủ công; để dùng
backup Drive tự động cần chấp thuận cụ thể quyền này hoặc chọn một backend hẹp hơn.

Nếu runtime mới mất workspace, upload/giải nén lại hai ZIP vào đúng project root,
kết nối lại storage được chấp thuận, rồi chạy `restore`, `setup`, `attack` theo thứ tự.
`restore` lấy metadata và sample ZIPs, kiểm tra CRC/path an toàn. Notebook kiểm tra thêm
identity, image/state hashes và phiên bản package trước khi resume. Thay code, data,
device count hoặc package versions sẽ bị từ chối resume.

## Evaluation và diễn giải

Chạy `evaluate` sau khi attack hoàn tất. Cần `OPENAI_API_KEY` trong Molab Secrets hoặc
environment; không đặt key trong notebook/package. Driver chạy nguyên
`text_gen_eval_v3.1.ipynb` cho từng thư mục variant, với:

- Victims: LLaVA-1.5-7B và InternVL3-8B; checkpoint SHA được khóa và lưu khi bắt đầu evaluation.
- Judge GPT-4o, threshold **score >= 0.3**, prompt/protocol theo notebook v3.1.
- Qwen3-VL-8B chỉ là diagnostic của surrogate; không dùng làm bằng chứng held-out chính.
- InternVL3-8B khác checkpoint captioner InternVL3-1B nhưng cùng họ model; báo rõ điều này.

Tổng 20 evaluation tasks, tối đa 4.000 cặp cần judge. Đây là lời gọi API có phí theo tài khoản
người dùng; thiếu key thì evaluation dừng và không tạo kết luận giả thuyết.

Báo N dự kiến, N attack hoàn tất, N caption/judge hợp lệ, coverage, ASR/AvgSim valid,
paired differences trên common valid IDs và operational success trên toàn bộ 200 cặp.
Khi bootstrap, nhóm các cặp dùng chung ảnh theo checksum; tập này có một target lặp.
Không thay threshold sau khi xem kết quả. Một seed là vòng nghiên cứu ban đầu,
chưa chứng minh robustness qua nhiều seeds hay mọi victim.

## Phạm vi kiểm tra

Đã kiểm tra package local: 200 cặp/400 file, ảnh decode được, checksum nguyên gốc,
ZIP integrity, cú pháp cả hai notebook và cấu hình 10 attack/20 evaluation tasks.
Các kiểm tra local không thay cho chạy GPU của campaign hoặc kiểm chứng storage thật.
