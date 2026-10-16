# Chạy KLTN attack v4.3.2-h1 trên Molab

Hướng dẫn đối chiếu ngày 15/10/2026. Bộ launcher này đã được kiểm tra cú pháp
và cấu hình ở máy local; chưa được chạy với model thật trên GPU Molab.

## 1. Chuẩn bị file

Upload qua **Files** của Molab:

- `kltn-attack-v4.3.2-h1.ipynb`: notebook gốc.
- `molab_run_v432_h1.py`: launcher đi kèm.
- `attack_data.zip`: ảnh source và target của bạn, giữ nguyên cấu trúc dataset.

Hai file đầu có trong `outputs/molab_v432_h1_starter.zip`; ZIP này **không chứa dataset**.
Không cần upload `src/FOA-Attack`, `src/MPCAttack`, `src/SAGA`, toàn bộ repository,
weights, notebook caption/evaluation, hay API key OpenAI. Notebook đã nhúng source
FOA/MPC cần thiết; model tải từ Hugging Face trong phiên chạy.

Cấu trúc dữ liệu mặc định phải theo `torchvision.datasets.ImageFolder`:

```text
data/
  bigscale/
    class_a/
      001.jpg
      002.jpg
  target_images/
    class_a/
      001.jpg
      002.jpg
```

`class_a` chỉ là ví dụ: giữ tên thư mục class hiện có của dataset. Không để ảnh
trực tiếp dưới `bigscale/` hoặc `target_images/` khi dùng ImageFolder.
Hai phía phải có cùng số ảnh. Notebook ghép theo thứ tự đường dẫn mà ImageFolder
sắp xếp, **không tìm ảnh target bằng tên source**. Xem cặp đầu ở preflight.
Để so sánh Kaggle/Molab, dùng cùng dataset, cùng selection và kiểm tra `input_sha256`.

Nếu muốn chỉ upload vài cặp, hoặc ảnh của bạn nằm trong thư mục phẳng, dùng manifest:

```text
data/
  pairs.jsonl
  source/001.jpg
  target/001.jpg
```

Mỗi dòng trong `pairs.jsonl` là một JSON độc lập, đường dẫn tương đối với chính manifest:

```json
{"index": 0, "source": "source/001.jpg", "target": "target/001.jpg"}
```

Khi chạy thêm `--pair-manifest data/pairs.jsonl` vào danh sách đối số. Giữ index
và đúng cặp ảnh của thực nghiệm gốc; không tự đánh lại index nếu cần đối chiếu.

## 2. Tạo phiên và bật GPU

Mở <https://molab.marimo.io>, tạo notebook, bấm **notebook specs** trên header và bật GPU.
Theo [tài liệu Molab](https://docs.marimo.io/guides/molab/), GPU là NVIDIA RTX Pro 6000
Blackwell 96 GB; RAM mặc định 32 GB. Kiểm tra phần cứng thực tế trong log preflight.

Molab dùng marimo. Launcher đọc các code cell của IPYNB và chạy tuần tự trong một
tiến trình Python riêng, tránh phải sửa toàn bộ notebook sang mô hình reactive.
Không cần mở IPYNB như một notebook marimo để thực thi attack.

Upload file, rồi giải nén trong một cell Python. Ví dụ cả hai ZIP nằm trong thư mục
hiện tại, và `attack_data.zip` chứa trực tiếp `bigscale/` và `target_images/`:

```python
from pathlib import Path
from zipfile import ZipFile

print(Path.cwd())
with ZipFile("molab_v432_h1_starter.zip") as _zip:
    _zip.extractall("attack_project")
with ZipFile("attack_data.zip") as _zip:
    _zip.extractall("attack_project/data")
```

Nếu ZIP dataset có thêm thư mục bọc ngoài, sửa `--dataset-root` ở bước sau đến thư
mục thực sự chứa `bigscale/` và `target_images/`. Tránh giải nén lại khi attack đang chạy.

## 3. Cài thư viện, kiểm tra, rồi attack

Dán đoạn sau vào **một cell** trong Molab. Chạy lần lượt bằng cách sửa `MODE`:
`setup` → `preflight` → `smoke` → `attack`. Chỉ chạy một tiến trình tại một thời điểm;
đặt cell ở chế độ chạy thủ công/lazy để tránh vô tình khởi động lại attack khi sửa cell.

```python
def run_molab_attack(mode):
    from pathlib import Path
    import subprocess
    import sys

    project = Path("attack_project").resolve()
    command = [
        sys.executable, "-u", "molab_run_v432_h1.py",
        "--mode", mode,
        "--dataset-root", "data",
        "--work-root", "work",
        "--num-samples", "1",
        "--steps", "300",
        "--variants", "hybrid_dynamic",
        # "--pair-manifest", "data/pairs.jsonl",
    ]
    process = subprocess.Popen(
        command, cwd=project,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise RuntimeError(f"Launcher failed: exit={code}; xem log phía trên.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

MODE = "setup"  # Sau đó: preflight, smoke, attack
run_molab_attack(MODE)
```

- `setup`: cài đúng pins đọc từ notebook; không tải weights. Giữ Torch/Torchvision
  do Molab cung cấp. Mỗi lần gọi launcher là một Python process mới.
- `preflight`: chạy self-checks, kiểm tra CUDA forward/backward đơn giản, đọc cặp
  ảnh và kiểm tra device. Cần thấy `PREFLIGHT PASSED`. Không tải weights.
- `smoke`: thêm kiểm tra GPU với encoder tổng hợp. Pass bước này **chưa chứng minh**
  Qwen NF4 và ensemble CLIP thật chạy được.
- `attack`: tải model cần thiết, sinh caption/negative bank, chạy attack và xuất ZIP.

Lần đầu giữ **1 cặp, 300 steps, hybrid_dynamic**. Muốn chạy chẩn đoán ngắn có thể
đặt 30 steps (tối thiểu cấu hình vùng mặc định cho phép); notebook gắn nhãn diagnostic.
Khi lấy số liệu chính thức, dùng 300 steps và output run mới.

Khi 1 cặp chạy thành công, tăng `--num-samples`. Để so sánh ba variant, sửa đoạn cuối
danh sách lệnh thành:

```python
"--variants", "foa_reference", "mpc_clip", "hybrid_dynamic",
```

Launcher tự đặt cả attack, Qwen attention và caption trên `cuda:0`; giữ
`HYBRID_AMP=False`, `GRADIENT_SCALE=1.0`, ensemble B16/B32/Laion và objective gốc.
Mặc định launcher tắt hai offload CPU để tận dụng VRAM 96 GB và giảm áp lực RAM 32 GB.
Đây là cấu hình khởi đầu, không phải cam kết đủ bộ nhớ ở mọi runtime.
Nếu CUDA OOM, thử thêm `--cpu-saved-tensors`; nếu cần thêm `--offload-clip`.
Hai tùy chọn sẽ dùng thêm RAM máy chủ. Theo dõi cả RAM và VRAM, không tự bỏ Laion
hay đổi dtype/objective để làm kết quả có vẻ chạy thành công.

## 4. Model và CUDA

`hybrid_dynamic` cần tải:

- `openai/clip-vit-base-patch16`
- `openai/clip-vit-base-patch32`
- `laion/CLIP-ViT-G-14-laion2B-s12B-b42K`
- `OpenGVLab/InternVL3-1B-hf`
- `Qwen/Qwen3-VL-8B-Instruct` (NF4 theo notebook)

Cache mặc định nằm trong `attack_project/work/hf_cache` trừ khi runtime đã đặt `HF_HOME`.
Cần Internet và đủ dung lượng đĩa; không cần upload weights từ máy cá nhân.
Không upload cache weights vào vùng lưu trữ notebook giới hạn.

Các pin chính là transformers 4.57.1, tokenizers 0.22.1, huggingface-hub 0.34.4,
accelerate 1.11.0, bitsandbytes 0.48.1; launcher lấy toàn bộ danh sách từ notebook.
Nếu preflight báo lỗi kernel CUDA/`sm_120`, kiểm tra Torch/CUDA trước khi tải model.
[PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/) bắt đầu cung cấp hỗ trợ Blackwell
với wheel CUDA 12.8. [bitsandbytes 0.48.1](https://huggingface.co/docs/bitsandbytes/v0.48.1/installation)
có binary Linux `sm120` cho CUDA 12.8–12.9. Không hạ về Torch/CUDA cũ của T4;
nếu phải đổi Torch, dùng Torch/Torchvision tương thích cùng nhau và ghi lại version.

## 5. Kết quả, lưu trữ và resume

Kết quả nằm ở:

```text
attack_project/work/kltn_attack_v4_3_2_h1_molab_outputs/
  v432_<timestamp>_<fingerprint>/
    run_config.json
    model_metadata.json
    self_checks.json
    pairs.json
    events.jsonl
    status.json
    summary.json
    hybrid_dynamic/
      manifest.jsonl
      status.json
      samples/00000/
        source.png
        target.png
        adversarial.png
        complete.json
        state.json
        steps.jsonl
        maps/
        checkpoints/
  v432_<timestamp>_<fingerprint>.zip
```

Chỉ coi run thành công khi `status.json` cấp run là `complete`, từng variant đủ số
mẫu và log có `ATTACK COMPLETE`. Có ZIP không có nghĩa mọi variant đã thành công.
Nếu lỗi, xem `summary.json` và `<variant>/errors.jsonl`.

**Tải ZIP kết quả xuống hoặc sao lưu sang remote storage trước khi dừng phiên.**
Theo [chính sách lưu trữ Molab](https://molab.marimo.io/blog/seamless-storage-in-molab),
file sinh bằng code không mặc nhiên tồn tại sau shutdown. Input ZIP upload qua Files
được lưu theo chính sách nền tảng; thư mục giải nén và output sinh ra cần chuẩn bị lại/
sao lưu. Không coi `work/` là Google Drive được mount và không dùng code
`google.colab.drive.mount`.

Molab hỗ trợ **Files → Add remote storage** cho Google Drive/S3/Hugging Face;
xem [hướng dẫn chính thức](https://molab.marimo.io/blog/drive-storage) để kết nối và
sao lưu kết quả. Phiên có giới hạn tối đa 12 giờ, idle quá 90 phút bị shutdown theo
tài liệu Molab; chia số mẫu mỗi run sau khi đo thời gian thực tế của 1 cặp.

Muốn tạo ZIP giữa chừng sau khi đã dừng process, dùng cell:

```python
import shutil

# Thay bằng RUN_DIR đúng được in trong log; không zip khi process đang ghi file.
_run_dir = "attack_project/work/kltn_attack_v4_3_2_h1_molab_outputs/v432_..."
print(shutil.make_archive(_run_dir + "_partial", "zip", root_dir=_run_dir))
```

Resume chỉ ở ranh giới mẫu đã hoàn tất. Thêm `--resume-run-dir` trỏ đúng run directory
và giữ nguyên cấu hình, code, dữ liệu, selection, package versions, đường dẫn ảnh
tuyệt đối và số GPU nhìn thấy. Mẫu bị ngắt giữa chừng chạy lại từ đầu. Không resume
run T4×2 bằng cấu hình một GPU: identity/device/RNG khác nhau. Không thay num-samples
của run đang resume để mở rộng run; tạo run mới cho batch tiếp theo.

Đánh giá bằng `notebooks/text_gen_eval_v2.ipynb` sau khi tải kết quả về, trỏ vào thư
mục **một variant** chứa `manifest.jsonl`. API key cho bước judge chỉ cần ở môi trường
evaluation, không cần trong phiên attack này.
