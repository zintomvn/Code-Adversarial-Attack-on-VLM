# O-Attack / M-Attack-V2 trên Molab và Kaggle

Notebook Molab: [official_om_molab.py](official_om_molab.py). Notebook Jupyter/Kaggle: [official_om_kaggle.ipynb](official_om_kaggle.ipynb). Cấu hình mẫu: [default_config.json](default_config.json).

Phiên mới: [Official O-Attack + M-Attack-V2 v4 trên Molab](https://molab.marimo.io/notebooks/nb_NRE5b1ieMbBcgmP6pPV2HZ). Controller v4 đã khởi động; [folder kết quả v4 trên Drive](https://drive.google.com/drive/folders/13bB-pBlCpevcn3rn2qdWXiLEDRHUrQQ2) chứa retrieval đủ 200 target và bản release notebook. Kết quả lịch sử của run `official_om_n200_e16_s300_v3`: [Drive v3](https://drive.google.com/drive/folders/1S6VZfKXFycU92-S58IVziNrn9yFeuQSF), gồm checkpoint O-Attack một mẫu. Tiến độ xác nhận được ghi riêng trong [deployment_status.json](deployment_status.json); chưa hoàn tất 200 mẫu hay đánh giá cho hai attack mới.

Folder `release/` của run trên Drive chứa source notebook Molab đã triển khai, bundle code và `SOURCE_LOCK.json`. Source được kiểm tra không chứa giá trị khóa API trước khi xuất; `.env` và OAuth credentials không nằm trong release. License MIT gốc của M-Attack-V2 đi kèm bundle và [M_ATTACK_V2_LICENSE.txt](M_ATTACK_V2_LICENSE.txt).

## Chạy trên tài khoản Molab mới

1. Import `official_om_molab.py` vào Molab, chọn RTX Pro 6000 và Internet.
2. Chỉnh duy nhất cell `CAMPAIGN_CONFIG`: `drive_parent_id` là folder mà tài khoản hiện tại có quyền đọc/ghi; `data_drive_file_id` là dataset ZIP được share hoặc copy cho tài khoản đó. Giữ SHA của dataset nếu bytes không thay đổi.
3. Chạy các cell, kết nối Drive bằng form OAuth. Khóa API nhập qua ô password hoặc nạp `.env` riêng tư; không ghi khóa vào source notebook.
4. Bấm **Chạy / tiếp tục O-Attack + M-Attack-V2**. Bảng tiến độ và log là output hiển thị trực tiếp, kể cả ở kiosk mode, cập nhật 30 giây.
5. Hết phiên: mở lại cùng notebook, giữ `run_name` và thông số, kết nối Drive rồi bấm tiếp tục. Ảnh dở dang chạy lại từ checkpoint batch trước.

Kaggle/Jupyter dùng `official_om_kaggle.ipynb`; bật GPU và Internet. Cell OAuth dùng `getpass`, không lưu callback/khóa trong source. Các thuật toán và dữ liệu giống bản Molab.

Các tài khoản chạy đồng thời dùng `run_name` riêng. Để chuyển tiếp cùng một run sang account khác, dừng phiên cũ rồi dùng cùng cấu hình và cả folder checkpoint đã share/copy. Hai account không cùng ghi vào một run. Kaggle có GPU/VRAM khác Molab; batch mặc định của M-Attack-V2 là 10 theo tác giả. Nếu đổi batch để vừa VRAM, đặt run mới và ghi nhận thay đổi đó trong thí nghiệm.

## Thông số và nguồn thuật toán

Mặc định N=200, epsilon=16 pixel, 300 iterations. Cohort đúng 200 cặp của thí nghiệm trước; kiểm tra local xác nhận source/target bytes trùng dataset phát hành của cả hai tác giả. Mỗi phương pháp có learning rate và batch riêng, không áp một alpha chung lên hai optimizer.

| Method | Commit | Cấu hình giữ lại |
|---|---|---|
| O-Attack | `5585398e89a4d3415d0f29ff924368e8ba3c9cef` | G3, Adam alpha=1.0, batch=1, 10 source crops; caption banks 1000 cặp gốc |
| M-Attack-V2 | `8faca6d95e1caa190d676575489ab81bb31e4d03` | `pgd_multi_pass`, Adam alpha=0.005, batch=10, K=10, beta=0.3, bốn CLIP surrogate, target_num=3 |

File nguồn được lấy theo commit, đóng gói nguyên bản và khóa SHA-256 trong `SOURCE_LOCK.json`. Builder đối chiếu `src` sau chuẩn hóa CRLF/LF và ghi khác biệt vào `source_comparison.json`; các thay đổi local được giữ nguyên, baseline dùng nguồn chính thức đã khóa. Không thay loss, optimizer, crop hoặc gradient của tác giả. Adapter chỉ điều phối dữ liệu, model revisions, entry point một GPU và checkpoint. M-Attack-V2 gọi worker `process_batch` từ script parallel để giữ đúng K=10 và crop schedule của config; script single có các mặc định khác.

Môi trường độc lập theo method: O-Attack Python 3.10 / torch 2.7.0 / Transformers 4.57.0; M-Attack-V2 Python 3.10 / torch 2.10.0 / Transformers 4.51.3. Python 3.11 từ `pyproject.toml` của M-Attack-V2 không import được `config_schema.py` do dataclass mutable defaults; dùng Python 3.10 để giữ nguyên file tác giả, với SciPy 1.15.3 tương thích. M-Attack-V2 gọi CLIPVisionTransformer với `return_dict`; bản 4.57.0 gây TypeError ở forward, nên môi trường mới dùng 4.51.3 có wrapper tương thích. Đây là lựa chọn dependency; cần xác nhận lại bằng forward/backward GPU và batch attack thật trước khi coi lỗi đã được sửa. Dùng các wheel CUDA 12.8 để chạy Blackwell. O-Attack giữ precision gốc; M-Attack-V2 giữ AMP mặc định của tác giả. Không tăng batch tự động vì batch có thể ảnh hưởng gradient và chuỗi RNG.

Trên Molab, toàn bộ môi trường Python, cache và dữ liệu làm việc nằm trong `/tmp/vlm_baselines/<run_name>`, tránh đồng bộ hàng nghìn file nguồn của venv vào workspace khi mở lại sandbox. `/tmp` có thể mất khi đổi sandbox; Drive giữ checkpoint và retrieval để khôi phục. Kaggle dùng `/kaggle/working/baselines/<run_name>`.

Kiểm tra GPU của v4 đã tái hiện TypeError trên 4.57.0 và xác nhận forward/backward AMP có gradient đầu vào hữu hạn trên 4.51.3; xem [m_clip_api_regression.json](m_clip_api_regression.json). Cả hai method đã hoàn tất batch attack thật đủ 300 bước: O-Attack 1 ảnh, M-Attack-V2 10 ảnh. Cả hai checkpoint vượt kiểm tra tải từ Drive vào thư mục mới, checksum ảnh, giới hạn nhiễu 16 và RNG; xem [o_attack_drive_restore.json](o_attack_drive_restore.json) và [m_attack_v2_drive_restore.json](m_attack_v2_drive_restore.json). Controller tiếp tục campaign N=200 sau các batch kiểm tra; số ảnh hiện tại phải xem trên Molab, không suy ra từ số batch đã xác minh.

## Dữ liệu retrieval M-Attack-V2

ATA cần hai ảnh auxiliary khác bên cạnh primary target. Bản phát hành chỉ có retrieval cho 100 ID; trong cohort cũ có 179 ID chưa được phủ. Pipeline tải các asset phát hành có sẵn theo SHA, rồi gọi **retrieval.py gốc** để bổ sung những ID thiếu từ COCO train2014. Không chấp nhận fallback lặp primary target khi thiếu retrieval.

COCO train2014 có 82.783 ảnh, ZIP khoảng 13,5 GB. Pipeline dùng bản gốc từ bucket COCO và SHA-256 công bố trong checksum registry TensorFlow Datasets: `ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775`. Có thể đặt `retrieval.reference_root` tới thư mục sẵn có chứa `train2014/`, hoặc tắt `allow_download` để báo thiếu dữ liệu rõ ràng.

Giải nén kiểm tra đủ số ảnh và kích thước từng file, ghi qua file tạm rồi mới đổi tên. Nếu phiên ngắt giữa chừng, bước này bổ sung ảnh thiếu hoặc dở dang trước khi chạy retrieval. Molab coi model cache, môi trường Python và dataset tải bằng code là dữ liệu tạm; checkpoint và retrieval hoàn tất được lấy lại từ Drive khi đổi sandbox. Xem [chính sách lưu trữ Molab](https://marimo.io/pages/molab/storage).

Retrieval hoàn tất được lưu riêng thành `assets/retrieval.zip` trong folder run trên Drive. Khi chạy tiếp ở account khác, share/copy **cả folder run**, gồm `identity.json`, `checkpoints/` và `assets/`, rồi đặt `drive_parent_id` tới folder cha tương ứng. Không cần tải lại COCO nếu archive đã có. Rebuild retrieval dùng cùng code và reference pool; nó không chứng minh byte-identical với các ảnh auxiliary tác giả chưa phát hành.

## Checkpoint và danh tính thí nghiệm

- O-Attack commit một ảnh; M-Attack-V2 commit toàn bộ batch 10 ảnh vào một ZIP. Batch không được xem là hoàn tất từng phần.
- ZIP chứa ảnh source/target/adversarial đã chuẩn hóa theo tác giả, complete markers và Python/NumPy/Torch/CUDA RNG. Tải Drive kiểm tra MD5; marker kiểm tra SHA-256 ảnh và RNG.
- Khi resume, bộ đọc kiểm tra contiguous prefix, environment và run fingerprint. Khôi phục RNG đúng trước điểm iterator/attack tương ứng. M-Attack-V2 giữ cuDNN benchmark của tác giả, nên không cam kết bitwise giống giữa các GPU/kernel.
- `run_name` phải đổi nếu thay epsilon, steps, batch, learning rate, overrides, models hoặc dataset. Checkpoint cũ không được trộn vào cấu hình mới.
- Drive listing/upload/download retry tối đa bốn lần với backoff khi có lỗi transport; validation/checksum lỗi thì dừng. Retry không thay thuật toán.

## Đánh giá và thêm paper sau này

Sau hai attack, pipeline dùng lại evaluator v3.2 đã khóa của thí nghiệm cũ: LLaVA-1.5-7B, InternVL3-8B, Qwen3-VL-8B, Gemini 2.5 Flash, GPT-5 mini; judge GPT-4o, mặc định ASR >=0.3. Đây là protocol chung của dự án, không phải protocol paper của từng tác giả. Gemini thiếu khóa được ghi pending; các model còn lại vẫn chạy.

Muốn thêm method: bổ sung source/commit và hash trong `build_release.py`, profile environment/adapter trong `campaign_runtime.REGISTRY`, adapter I/O mới trong `campaign_worker.py`, rồi build và kiểm tra. Giữ các hàm attack của tác giả nguyên bản. Tạo run mới khi cập nhật bundle.

`verify_first_batch=True` chạy một batch thật của mỗi method trước, lưu checkpoint, rồi mới chạy phần còn lại. Trạng thái source/smoke test không được coi là đã hoàn thành N=200. Bằng chứng thực nghiệm thực tế nằm trong `deployment_status.json` khi triển khai.

## Kiểm tra local

```powershell
python notebooks/attacks/baseline/om_attack/build_release.py
python notebooks/attacks/baseline/om_attack/verify_release.py
```

Các kiểm tra local xác minh nguồn, caption coverage, atomic batch ZIP, khôi phục RNG, chuyển checkpoint sang thư mục mới và từ chối file hỏng. GPU và API phải xác minh riêng trên Molab.

`audit_checkpoint.py` tải checkpoint thật từ Drive, kiểm tra MD5/SHA, khôi phục trong thư mục trống bằng đúng Python của method và so sánh trạng thái RNG sau khôi phục. Bản triển khai có cell audit riêng; các báo cáo nằm ở `validation/` trên Drive. Script không chạy thêm attack và không ghi vào kết quả đã tạo.
