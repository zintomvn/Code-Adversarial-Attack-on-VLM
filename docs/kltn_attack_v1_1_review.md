# Review `kltn-attack-v1.1.ipynb`

Đã sửa trực tiếp notebook. Kết luận: chiều gradient ascent, projection vào `[-epsilon, epsilon]`, hệ số local OT `0.2` và đầu ra chia `255` khớp implementation FOA được đối chiếu. Lỗi xác định được nằm ở dependency, cách truyền cấu hình và xử lý run; không cần đổi objective để sửa các lỗi đó.

## Các lỗi đã sửa

| Vấn đề trước khi sửa                                                                         | Ảnh hưởng                                                                                         | Sửa trong notebook                                                                                                            |
| -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Cài K-means từ Git không pin commit; bản PyPI `0.3` thiếu `iter_limit`                       | Loss gọi `iter_limit=100` có thể lỗi `unexpected keyword argument`; không tái lập được dependency | Pin commit `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805`, force reinstall riêng K-means, giữ nguyên `iter_limit=100`             |
| Pip thay file nhưng session có thể còn module đã import                                      | Package trên disk và hàm đang chạy có thể khác nhau                                               | Yêu cầu restart nếu K-means đã import; kiểm tra signature và Git commit trong `direct_url.json` ngay khi import               |
| Upstream K-means import SoftDTW/Numba, nhưng không khai báo dependency trong `setup.py`      | Có thể lỗi import trên môi trường chưa có Numba                                                   | Thêm `numba` vào dependencies; pip chọn bản tương thích Python của session                                                    |
| `steps=STEPS` được chốt khi định nghĩa hàm, caller không truyền `steps`                      | Đổi `STEPS` sau đó có thể không đổi số bước thực tế, trong khi config ghi giá trị mới             | Caller truyền tường minh `steps=STEPS`, `epsilon=PAPER_EPSILON`, `alpha=PAPER_ALPHA`                                          |
| Parity guard chỉ dùng loss giả lập, không crop                                               | Chưa kiểm chứng nhánh crop mặc định hoặc loss OT/trọng số động                                    | So sánh cả crop on/off, thứ tự gọi loss, giới hạn pixel, giá trị loss thật và gradient qua ba lần forward                     |
| Chỉ kiểm tra `NaN` trong transport; không kiểm tra weights/objective/gradient                | Overflow hoặc gradient Inf có thể tạo ảnh không hợp lệ                                            | Kiểm tra số hữu hạn và dừng trước khi lưu ảnh; giữ nguyên công thức trọng số gốc                                              |
| `zip` dừng tại dataset ngắn hơn, config vẫn dùng số mẫu yêu cầu                              | Thiếu mẫu mà không báo lỗi                                                                        | Kiểm tra số cặp khả dụng trước run; hỗ trợ `NUM_SAMPLES=None`; ghi số mẫu thực tế vào status                                  |
| Manifest chỉ ghi cuối run                                                                    | Mất liên kết source/target của các ảnh đã hoàn tất nếu bị ngắt                                    | Ghi và flush từng record; ghi trạng thái complete/error/interrupted                                                           |
| Xóa repo và output cũ mỗi lần chạy                                                           | Mất checkout sửa tay hoặc kết quả thực nghiệm trước                                               | Reuse checkout cùng origin; yêu cầu chọn `OUTPUT_DIR` khác nếu thư mục output đã có dữ liệu; tên ZIP theo thư mục run thực tế |
| Preview lấy mọi entry từ `rglob('*')`, có thể có thư mục; preview ngẫu nhiên dùng global RNG | Lỗi mở ảnh hoặc làm đổi trạng thái random                                                         | Lấy đường dẫn từ `ImageFolder.samples`; dùng `random.Random(seed)` riêng cho preview                                          |

Tâm cụm target cũng được kiểm tra đủ số embedding và số hữu hạn. Log K-means ở loss được suppress theo cách của tác giả, tránh tạo lượng output lớn trong vòng lặp attack. Kernel OT được tính trong `torch.no_grad()` như implementation gốc; similarity vẫn truyền gradient.

## Logic được giữ theo tác giả

1. **Maximize similarity:** dùng `+ alpha * sign(gradient)` vì objective là similarity với target. Đổi thành dấu trừ sẽ đổi hướng tối ưu.
2. **Đơn vị pixel:** source/target đi vào attack ở `[0, 255]`; `epsilon=16`, `alpha=1` cùng đơn vị. Chỉ đầu ra cuối cùng chia `255` và clamp `[0, 1]`. Không tự đổi epsilon thành `16/255` ở bước cập nhật hiện tại.
3. **Crop target mỗi bước:** cập nhật ground truth sau random crop target ngay trong vòng lặp là hành vi gốc.
4. **Hai lần gọi loss khi source crop bật:** forward toàn ảnh trước, rồi forward crop và dùng score crop làm objective. Forward đầu còn cập nhật `previous_loss_list`, nên bỏ nó sẽ đổi dynamic weights dù score toàn ảnh không trực tiếp cộng vào objective.
5. **State qua các cặp ảnh:** tác giả tạo một loss bên ngoài loop dataset và không reset `previous_loss_list` ở mỗi ảnh. Đây là giới hạn của baseline: kết quả một cặp có thể phụ thuộc lịch sử các cặp trước. Notebook giữ hành vi này; reset sẽ là một thay đổi phương pháp và cần báo cáo riêng.
6. **K-means source không có hard iteration limit:** `EnsembleFeatureExtractor_ot.get_cluster_center` của tác giả không truyền `iter_limit`; dependency đã pin mặc định `iter_limit=0`. Target loss có `iter_limit=100`. Source clustering vẫn có rủi ro chạy lâu khi không hội tụ. Notebook không tự thêm limit source vì có thể đổi tâm cụm và baseline.
7. **Ghép theo thứ tự:** `ImageFolder` sắp đường dẫn và source/target được ghép bằng `zip`, như tác giả. Không có phép join theo filename. Manifest ghi chính xác `original_path` và `target_path` để đối chiếu từng cặp.
8. **Batch size 1:** extractor gốc dùng squeeze và chỉ lấy local embedding của phần tử đầu. Notebook kiểm tra batch một ảnh; không nên tăng batch size mà giữ nguyên extractor này.

## Đã kiểm chứng

- Validate notebook và compile tất cả code cell.
- Đọc wheel PyPI `kmeans-pytorch==0.3`: signature chỉ có `X`, `num_clusters`, `distance`, `tol`, `device`.
- Chạy đúng các hàm Euclidean của commit K-means đã pin trên CPU và CUDA: `K=10`, `iter_limit=100`, tâm cụm hữu hạn; gradient truyền qua trung bình embedding của cụm.
- So sánh attack với hàm `fgsm_attack` từ upstream FOA trên tensor không đồng nhất, crop on/off và 20 bước để chạm epsilon/clipping.
- So sánh loss OT/trọng số động và gradient với class `EnsembleFeatureLoss_OT_foa_attack` upstream cho ba surrogate giả lập, ba lần forward liên tiếp.
- Kiểm tra parity guard không làm đổi RNG Torch CPU/CUDA hoặc NumPy.
- Thử xuất PNG/JSONL/config/status trên ImageFolder nhỏ: thay `STEPS` sau khi định nghĩa vẫn được áp dụng, không ghi đè output cũ, phát hiện thiếu cặp, và giữ manifest khi mẫu sau bị lỗi.
- Thử objective hữu hạn nhưng gradient Inf: notebook dừng trước xuất ảnh.

Môi trường kiểm tra tensor: Python 3.14, Torch `2.12.0+cu126`, GPU NVIDIA GeForce RTX 4060 Laptop. Bài kiểm tra K-means thực thi code Euclidean upstream bằng AST, không kiểm tra import SoftDTW/Numba trên host. Chưa chạy cài đặt toàn bộ dependency hoặc tải/chạy ba backbone CLIP trên Kaggle; notebook có guard import và parity để kiểm tra thêm ở session Kaggle thật.

## Chạy lại trên Kaggle

Restart session rồi Run All để dùng dependency mới. Chạy ít mẫu trước. Nếu đã có output, đổi `OUTPUT_DIR` sang một thư mục run mới. Với các thực nghiệm độc lập, restart session và chạy từ đầu để khởi tạo lại loss và RNG; chỉ chạy lại cell attack trong cùng session sẽ tiếp tục lịch sử loss/RNG cũ. Không bỏ `iter_limit` để né lỗi import/version. Thông báo xung đột `huggingface-hub` với Gradio/Diffusers trong output cũ liên quan các package ngoài pipeline attack; chưa đủ để kết luận attack lỗi, và notebook không chạy hai package này.

## Nguồn đối chiếu

- [K-means upstream tại commit đã pin](https://github.com/subhadarship/kmeans_pytorch/blob/f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805/kmeans_pytorch/__init__.py): signature, iteration limit và Euclidean clustering.
- [SoftDTW upstream](https://github.com/subhadarship/kmeans_pytorch/blob/f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805/kmeans_pytorch/soft_dtw_cuda.py) và [setup.py](https://github.com/subhadarship/kmeans_pytorch/blob/f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805/setup.py): import Numba và dependencies khai báo rỗng.
- [FOA attack loop](https://github.com/jiaxiaojunQAQ/FOA-Attack/blob/main/generate_adversarial_samples_foa_attack.py) và [FOA surrogate/loss](https://github.com/jiaxiaojunQAQ/FOA-Attack/blob/main/surrogates/FeatureExtractors/Base.py): thuật toán tham chiếu. Notebook ghi SHA checkout thực tế vào `run_config.json`.
