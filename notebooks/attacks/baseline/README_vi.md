# Baseline chính thức: FOA-Attack, MPC-Attack, SAGA

Pipeline mới cho **O-Attack và M-Attack-V2** nằm trong [om_attack/](om_attack/README_vi.md), có notebook Molab/Kaggle riêng và giữ nguyên bundle của ba method cũ.

Notebook Molab: [official_baselines_molab.py](official_baselines_molab.py). Notebook Jupyter/Kaggle: [official_baselines_kaggle.ipynb](official_baselines_kaggle.ipynb). Bản ipynb dùng giao diện Marimo: [official_baselines_n200.ipynb](official_baselines_n200.ipynb).

Phiên phục hồi ngày 21/10/2026: [Molab đang chạy SAGA](https://molab.marimo.io/notebooks/nb_B6DAdejhFQbHtZmtME5F8q). FOA và MPC đã phục hồi đủ 200/200 từ Drive. Xem [ghi chú phục hồi](RECOVERY_2026-10-21.md) và [trạng thái xác minh](recovery_status_2026-10-21.json). Bootstrap đã thêm `PYTHONPATH` để các môi trường Python con import được module của dự án; phiên cũ có thể dùng `molab_resume_cell.py`.

Mặc định **N=200, epsilon=16 pixel trong [0,255], steps=300, alpha=1**. Không truyền epsilon=16/255 vào CLI tác giả. Các tham số chỉnh ở cell `CONFIG`: phương pháp, dữ liệu, model revisions, MPC lambda/tau/omega, SAGA phases/top-k/layer, batch caption, giới hạn API, ngưỡng ASR và nơi lưu Drive. Đổi `RUN_NAME` khi thay đổi cấu hình attack. Giữ nguyên tên và cấu hình để chạy tiếp.

## Nguồn thuật toán

| Phương pháp | Repository / commit | Entry point |
| --- | --- | --- |
| FOA | [jiaxiaojunQAQ/FOA-Attack](https://github.com/jiaxiaojunQAQ/FOA-Attack/tree/6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b) | `generate_adversarial_samples_foa_attack.py` |
| MPC | [LiYuanBoJNU/MPCAttack](https://github.com/LiYuanBoJNU/MPCAttack/tree/7b195a5bcbaeaf1e8534d101199810a2782caad3) | `generate_adversarial_examples_MPCAttack.py` |
| SAGA | [jaehyun-kwak/SAGA](https://github.com/jaehyun-kwak/SAGA/tree/042274599b958d4d63760b9bf44bbcb88493811c) | `saga.pipeline.run` với CLI main của tác giả |

65 file nguồn được kiểm tra byte với Git object tại các commit này. `SOURCE_LOCK.json` khóa SHA-256 từng file. `source_audit.json` ghi khác biệt với bản trong `src`; phần FOA tại `src` có các chỉnh sửa trước đó nên gói chạy dùng bản upstream nguyên vẹn của cùng tác giả. SAGA tại `src` chỉ khác một chú thích trong optimizer. Các file trong `src` không bị sửa.

FOA dùng đúng quick-start generator: FGSM, K=10, ensemble B16/B32/Laion, source/target crop [0.5,0.9], 224 pixel, seed=2023. Không dùng `FOAttack.py`, vì entry point đó thêm lựa chọn K bằng phản hồi các model đích, là protocol khác.

MPC dùng đúng loss InfoNCE, MI-FGSM và các surrogate CLIP/InternVL3-1B/DINOv2 của tác giả. Bản phát hành **ghi cứng `range(300)`**, mặc dù có cờ `--steps`; notebook báo lỗi nếu chọn steps khác 300 cùng MPC. Batch attack giữ bằng 1 vì mã InternVL của tác giả chỉ xử lý phần tử đầu của batch.

SAGA giữ caption đích gốc từ `input/coco.jsonl`, ghép theo stem ID, seed chọn mẫu=43 và seed optimization=2023; giữ Qwen3-VL attention eager/layer29, 10 phases và top-k3. FOA/MPC có objective theo ảnh đích; SAGA có objective theo caption đích. Đây là khác biệt của phương pháp gốc, cần nêu khi báo cáo. Caption reference trong đánh giá là mô tả ảnh target bởi từng victim VLM, đúng notebook v3.2; metadata SAGA cũng lưu caption đích gốc để kiểm tra.

Không viết lại attack/loss/gradient/crop/attention. `author_worker.py` chỉ gắn hook dữ liệu, serialization, chọn revision model và checkpoint vào entry point gốc. Tất cả các file tác giả được kiểm tra trước khi chạy. Không bật AMP/TF32, không quantize surrogate attack và không thay precision của tác giả.

Dependency k-means dùng [upstream chính thức tại commit f7f36bd](https://github.com/subhadarship/kmeans_pytorch/tree/f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805). Bản PyPI cùng số phiên bản 0.3 thiếu `iter_limit` mà FOA gọi. Ba file package upstream này khớp nội dung với package k-means được tác giả MPC đóng gói. Không bỏ tham số, không sửa k-means hay thay bằng implementation khác; hash module thực tế được ghi vào từng checkpoint. Audit cũng phân biệt khác biệt CRLF/LF của checkout Windows với thay đổi nội dung.

## Chạy trên Molab

1. Import `official_baselines_molab.py` bằng **Import notebook**. Không cần upload ZIP hay dataset vào notebook. Gói mã khoảng 300 KB được nhúng sẵn, có checksum, không chứa trọng số hoặc thông tin xác thực.
2. Chọn **Configure compute → RTX Pro 6000 (Blackwell)**.
3. Chỉnh cell `CONFIG`. Mặc định dùng cùng 200 cặp trong `outputs/molab_hypotheses_n200/pairs_200.jsonl` của thí nghiệm hiện có, không chọn mẫu khác cho từng baseline. Dataset ZIP được khóa SHA-256 và tải từ Drive. Nếu dùng dữ liệu khác, cập nhật archive, SHA, manifest cặp ảnh và caption SAGA phù hợp; không tự tạo caption thay mã tác giả.
4. Run các cell. Mở link kết nối Drive, chọn account có quyền đọc dataset và ghi thư mục kết quả, cấp quyền rồi dán **URL callback** vào ô bảo mật. Notebook dùng OAuth PKCE và kiểm tra state.
5. Nhập `OPENAI_API_KEY` và `GEMINI_API_KEY` vào ô password hoặc Molab Secrets/environment trước khi khởi chạy controller. Không đưa khóa vào cell code. Khóa nhập sau khi worker đã khởi động chỉ áp dụng khi khởi chạy lại controller; các checkpoint vẫn được giữ.
6. Bấm **Chạy / tiếp tục baseline + evaluation**. Monitor cập nhật trạng thái, số ảnh và log; chạy lại monitor không khởi động worker mới.

Notebook đã được triển khai qua Chrome DevTools tại [Molab baseline](https://molab.marimo.io/notebooks/nb_oKkjtn1RrUc7gDm3bdGRsR). Xem `deployment_status.json` để biết bằng chứng thực nghiệm gần nhất; mã nguồn hoặc smoke test không có nghĩa là đã hoàn thành toàn bộ 600 ảnh và 15 tổ hợp đánh giá.

## Drive và chạy tiếp

Đích mặc định: `<drive_parent_id>/official_baselines/<RUN_NAME>/`. Thư mục được tạo riêng, không ghi đè thí nghiệm H1 hiện có.

- `identity.json`: cấu hình, source hashes, dữ liệu, mã điều phối và fingerprint.
- `checkpoints/<method>/<index>.zip`: ảnh source/target/adversarial PNG, trạng thái RNG Python/NumPy/Torch/CUDA và marker/checksum. SAGA có thêm ba artifact gốc để native resume kiểm tra run ID.
- `results/<method>/manifest.jsonl`: đầu vào tương thích protocol đánh giá v3.2; có thể dựng lại hoàn toàn từ các ZIP.
- `evaluation/<model>/captions.json`, `scores.json`, `scores.jsonl`, `summary.json`, `summary.csv`: checkpoint caption/judge, raw response metadata, ASR và AvgSim.

Mỗi ZIP attack được kiểm tra checksum upload trên Drive trước khi worker tiến sang ảnh tiếp theo. Resume chỉ chấp nhận prefix mẫu đã hoàn tất, đúng fingerprint, artifact checksum và môi trường attack. Hook khôi phục RNG **sau khi khởi tạo DataLoader iterator**, trước mẫu tiếp theo; network retry không thay RNG của thuật toán. Nếu ngắt giữa một ảnh, ảnh đó chạy lại từ checkpoint sau ảnh trước. Không lưu optimizer giữa 300 bước và không hứa khôi phục giữa một bước.

Caption/judge lưu sau từng nhóm hữu hạn, lỗi API không được gán score=0 hay tính là thất bại. ASR/AvgSim của kết quả thiếu mẫu được gắn `complete=false` cùng số `judged/pending`; chỉ báo kết quả N=200 khi đủ 200 score hợp lệ.

Khi hết phiên: mở lại cùng notebook, chọn GPU, kết nối Drive, nhập lại secrets nếu cần, giữ `RUN_NAME`/tham số và bấm chạy tiếp. Không cần khôi phục bằng tay từng file. Khi đổi account Molab: dùng cùng thủ tục, chỉnh `drive_parent_id` và dataset ID/path; account mới cần được cấp quyền thư mục/checkpoint đã có nếu muốn nối tiếp run cũ. Notebook không thay đổi chia sẻ Drive tự động.

Sau một lỗi timeout Drive thực tế, có thêm `storage_watchdog.py` và cell tùy chọn `molab_storage_watchdog_cell.py`. Chạy cell này trong notebook đang hoạt động để giám sát controller. Supervisor giữ nguyên runner, source tác giả, fingerprint và checkpoint; chỉ khởi chạy lại sau traceback lỗi mạng từ Google Drive/httplib2, tối đa 5 lần với backoff. Nó dừng khi gặp lỗi thuật toán/CUDA/API, khi hết ngân sách phiên, hoặc khi controller kết thúc bình thường. Đây không phải cơ chế duy trì compute sau khi Molab thu hồi phiên. Trạng thái nằm trong `storage_watchdog_status.json`, log trong `logs/storage_watchdog.log`.

## Kaggle và môi trường GPU

Import `official_baselines_kaggle.ipynb`, bật Internet và GPU, chỉnh `data_archive` tới dataset ZIP trong `/kaggle/input/...`, hoặc dùng cùng OAuth Drive. Output nằm dưới `/kaggle/working/baselines`. Notebook hỏi secrets qua `getpass`; các adapter evaluation cũng hỗ trợ Kaggle Secrets. Các subprocess kế thừa API key trong environment của kernel.

Attack FOA/MPC cần Python3.10 vì dataclass nguyên bản; SAGA cần Python3.11. `uv` tạo các môi trường riêng. PyTorch2.8/CUDA12.8 được dùng để hỗ trợ compute capability12.0 của Blackwell; đây là **thay đổi môi trường tương thích**, không phải tái tạo nguyên xi GPU/CUDA cũ trong paper. Transformers4.57.1, model commit và package versions được ghi lại. Resume attack từ môi trường/GPU khác sẽ bị chặn; dùng run mới để có provenance rõ ràng. Notebook có thể chạy mới trên Kaggle, không cam kết bitwise giữa Blackwell và T4/P100.

RTX Pro 6000 hiện có 97,887 MiB VRAM, host Molab có 32 GiB RAM và 4 CPU. Mỗi method dùng đầy đủ GPU khả dụng theo mã gốc; khởi chạy tuần tự tránh quá RAM khi nạp Laion/Qwen. Không cam kết utilization=100% vì code tác giả có k-means, crop và batch1. Tăng `evaluation.local_batch_size` để tăng throughput caption; wrapper v3.2 tự chia batch khi OOM. Mặc định quantization caption là NF4 như v3.2; có thể chọn `none` nhưng cần coi đó là cấu hình evaluation khác.

## Protocol đánh giá

| Vai trò | Model |
| --- | --- |
| Victim VLM open | LLaVA-1.5-7B, InternVL3-8B, Qwen3-VL-8B |
| Victim VLM API | Gemini 2.5 Flash, GPT-5 mini |
| Judge | GPT-4o |

`eval_protocol_v32.py` lấy **nguyên văn cells 5,6,7** của `notebooks/text_gen_eval_v3.2.ipynb`: `CaptionGenerator`, `GPTScorer`, rubric, bộ giới hạn RPM/TPM và retry. `evaluation_source.json` khóa hash notebook gốc và mã trích xuất. Prompt: một câu mô tả tối đa 20 từ; generation64 tokens; GPT judge temperature0/max_tokens100. Mặc định ASR dùng **score >=0.3**, đúng v3.2, không tự thay bằng ngưỡng ở README cũ. Các model API là victim sinh caption, không thay judge.

Caption target được dùng lại giữa các baseline chỉ khi model/settings và checksum ảnh giống hoàn toàn. Judge được dùng lại chỉ khi hai caption và protocol giống hoàn toàn. Thiếu key/quota, refusal hoặc lỗi chưa hết sau retry được ghi `pending/error` và có thể chạy lại; không bịa caption hoặc score. Chi phí API và thời gian thực tế phụ thuộc quota của account.

## Kiểm tra và tái dựng

`local_validation.json` ghi kết quả kiểm tra source, đủ 200 cặp, và phép thử resume với **hàm FGSM nguyên bản** trên CPU. Phép thử này không thay thế xác minh các run GPU thực tế.

Các notebook đã tạo có thể chạy độc lập với thư mục repository. Để cập nhật lớp điều phối, sửa các file `.py` rồi chạy `python build_notebooks.py` trong thư mục này. Builder lấy source từ các Git object đã khóa nếu có checkout, hoặc từ gói `baseline_code.zip` hiện có và kiểm tra lại hashes; không lấy source từ nhánh main thay đổi theo thời gian. Giữ notebook evaluation gốc ở đường dẫn repository khi tái dựng.
