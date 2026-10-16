# Đánh giá trước các cấu hình đã hoàn tất trên Molab

## Notebook cần dùng

- `notebooks/text_gen_eval_molab_completed_v1.ipynb`: notebook độc lập, nhúng sẵn runner, cấu hình và nguồn evaluation v3.1. Import vào Molab và chạy các cell theo thứ tự.
- `notebooks/attacks/molab_attack_remaining_parallel_v1.ipynb`: launcher riêng cho phần attack còn lại; mặc định không khởi chạy attack khi Run all.

Không cần chờ đủ 10 cấu hình mới đánh giá. Tại checkpoint 2026-10-16 11:58 UTC: FOA reference 200/200, MPC-CLIP 200/200, hybrid_plain 13/200. Hai baseline đủ điều kiện tạo 4 task evaluation (2 cấu hình × 2 victim). Trạng thái thực tế được kiểm tra lại bằng `prepare`.

## Molab: evaluation trước

1. Bật RTX PRO 6000 và import notebook evaluation mới. Dùng phiên mới nếu phiên cũ gần giới hạn 12 giờ.
2. Giữ `EVAL_ROOT=/marimo/eval_completed_n200`. Đặt `MAX_HOURS` phù hợp thời gian còn lại của phiên. Runner dừng ở checkpoint đã flush, nhưng cần chừa thời gian cho download model, lô đang chạy, request đang bay và backup.
3. Chạy bootstrap/setup. Các package evaluation được cài vào venv riêng; không đổi package của attack.
4. Kết nối Drive bằng cell hoặc Molab Files → Remote Storage → Google Drive. Không public thư mục và không đưa token vào notebook.
5. Chạy `prepare`: dùng output local còn tồn tại hoặc tải checkpoint Drive. Chỉ trích PNG và completion markers; không tải optimizer/model weights từ ZIP sample.
6. `prepare` yêu cầu đủ 200 index theo đúng thứ tự, cùng run fingerprint, checksum ảnh và complete marker khớp nhau. Các cấu hình thiếu ảnh được ghi trong `inventory.json`, không tự chọn subset để báo ASR toàn cohort.
7. Chạy `restore` để nhận evaluation checkpoint cũ, rồi `run`. Scheduler của notebook bàn giao tạo caption cho mọi cấu hình/victim sẵn sàng trước; sau đó mới chấm GPT.
8. Xem `summary_completed.json`, caption/judge status và `evaluation/scores.csv` của từng task.

Molab live lúc khởi chạy thử dùng engine trực tiếp và xử lý caption → judge cho từng task; không hot-edit tiến trình này. Notebook bàn giao dùng scheduler tách hai giai đoạn và có thể resume cùng fingerprint của engine.

## Batch, cache và giới hạn tính toán

- Batch caption của notebook mặc định 64, hỗ trợ 1–64. Batch lớn không tự bảo đảm throughput cao hơn; mức tối ưu cần benchmark cùng input. Engine v3.1 chia batch đệ quy khi OOM/batch lỗi và giữ một outcome cho từng ảnh.
- Smoke thực tế trên LLaVA/FOA, batch 32: 399 ảnh duy nhất / 200 cặp, khoảng 28,7 giây inference, peak allocated VRAM khoảng 15,2 GiB, 0 lỗi caption. Batch 64 cũng đã hoàn tất 200 cặp LLaVA/MPC (peak khoảng 26,6 GiB) và InternVL/FOA (peak khoảng 8,9 GiB). Không so speedup giữa các tập ảnh khác nhau; chưa chứng minh mức batch tối ưu.
- Các ảnh trùng checksum được sinh caption một lần trong task. Task sau được dùng lại caption khi model/revision, prompt, generation/preprocessing config, roles và seed giống nhau. Captions giữa hai victim không dùng chung.
- Model revision được khóa bằng commit SHA ở lần chạy đầu và lưu vào state. Prompt/rubric GPT-4o, temperature 0, max_tokens 100, threshold >= 0,3 giữ nguyên v3.1.
- GPT judge là API; tăng batch GPU không làm GPT judge nhanh hơn. Mặc định 4 worker, cùng rate governor 12 RPM / 8000 TPM và HTTP-aware retries. Chỉ tăng quota khi account thực sự hỗ trợ; không chạy nhiều judge process có rate governor riêng cho cùng quota.
- Đã quan sát HTTP 429 yêu cầu chờ khoảng 6912 giây trên key/project ban đầu. Guard mới không gửi tiếp request khi cooldown dài hơn 120 giây hoặc vượt ngân sách phiên; lưu score đã có và trả task partial để resume sau hoặc sau khi thay key/project. Không rút ngắn thời gian chờ của provider.
- Thiếu `OPENAI_API_KEY` thì chạy caption-only. Thêm key qua Secrets/environment hoặc `/marimo/.env`, sau đó chạy lại `run`; không in key ra output. Key không hợp lệ/quota lỗi sẽ hiện trạng thái lỗi và không được coi là score thành công.
- ASR toàn cohort chỉ hiện khi đủ 200 score hợp lệ. Khi coverage thiếu, giữ số mẫu lỗi/pending và không diễn giải như kết quả đầy đủ.

## Backup và resume evaluation

Checkpoint evaluation tách khỏi attack, tại Drive:

`KLTN_Molab_Hypotheses_n200_v432_h1/evaluation_completed_v1/<configuration>__<victim>/`

Mỗi snapshot có tên theo checksum, upload/readback trước khi cập nhật pointer. Do lưu snapshot bất biến, thư mục có thể tăng theo số lần backup; giữ ít nhất snapshot mà pointer hiện tại trỏ đến. Không xóa checkpoint trong khi job còn chạy.

Trong mỗi task có caption items JSONL, per-image hashes, caption metadata, judge scores JSONL/CSV, trace và summaries. Backup khoảng 120 giây và ở cuối worker. Nếu runtime bị tắt cưỡng bức, tối đa phần chưa được backup có thể cần sinh/chấm lại. Dừng theo time budget sẽ backup trước khi worker trả exit 75.

Mở lại cùng notebook/cùng package/cùng EVAL_ROOT → kết nối Drive → prepare → restore → run. Engine từ chối fingerprint khác. Bản guard chỉ cho phép migration từ đúng SHA engine đầu tiên `cc9de38eb2debb89c991bf4d1cd20d5db8ee9adf9899952cfe54a5efda30cc5a`: giữ fingerprint protocol cũ, kiểm tra mọi trường còn lại và ghi SHA engine mới vào execution_engine.json. Không tự ý bỏ qua identity checks cho các thay đổi khác.

Khi các cấu hình còn lại có đủ 200 ảnh: để `CONFIG_IDS=None`, chạy lại prepare → restore → run. Các task evaluation đã hoàn tất được bỏ qua.

## Attack còn lại: tăng concurrency, giữ semantics

Đã cung cấp launcher `molab_remaining_tasks.py` và notebook riêng:

```bash
python /marimo/hypotheses_n200/molab_remaining_tasks.py list --root /marimo/hypotheses_n200
python /marimo/hypotheses_n200/molab_remaining_tasks.py run --root /marimo/hypotheses_n200 --workers 2 --max-hours 8
```

Chạy sau khi evaluation không còn dùng GPU. Không khởi động cùng launcher attack cũ. Hai worker là hai job độc lập: primary_ablations tiếp tục phần hybrid đang dở; controls H4 dùng state riêng và captions có sẵn. Hàng đợi backup giới hạn 2 ZIP bất biến, có backpressure và checksum readback; lỗi upload sẽ dừng task. State được hợp nhất sau khi các worker kết thúc và vẫn restore được bằng package gốc.

Đây là tăng batch công việc ở cấp job, **không tăng image batch của FOA**. FOA giữ weights/history/RNG xuyên ảnh nên image batching tùy ý có thể thay đổi thuật toán. Hai worker chỉ là phương án benchmark trên VRAM >=80 GiB: chưa đo speedup GPU/OOM thực tế; nếu tranh CPU/VRAM hoặc chậm hơn thì dùng `--workers 1`. Caption/negative bank đã có được dùng lại, không sinh lại chỉ để tăng batch.

Trong log, LOCAL SAMPLE chỉ xác nhận local checkpoint; DRIVE BACKUP DRAINED AND VERIFIED mới xác nhận flush hoàn tất. Đừng tắt cưỡng bức runtime khi muốn giữ các mẫu cuối đang ở queue. Launcher ghi provenance riêng và không sửa notebook/driver/plan gốc đang được dùng cho resume.

## Kaggle sau này

Notebook evaluation có thể đổi `EVAL_ROOT` thành `/kaggle/working/eval_completed_n200`, `ATTACK_ROOT` thành campaign local đã tải từ Drive/dataset, giảm batch 2–4 trên T4 và dùng Kaggle Secret `OPENAI_API_KEY`. Bootstrap giữ Torch/CUDA có sẵn.

Đây là nhánh chuẩn bị cho Kaggle, chưa được GPU smoke trên Kaggle. Restore snapshot evaluation hiện yêu cầu đúng absolute root cũ; không chuyển nguyên pointer `/marimo/...` sang `/kaggle/...` rồi bỏ qua identity check. Để chuyển platform, tạo run evaluation mới từ cùng frozen attack inputs hoặc triển khai migration có kiểm tra riêng.

## Validation đã thực hiện

- Unit tests CPU: cohort thiếu, hash/complete marker sai, ZIP traversal, selective PNG restore, cache đúng protocol, upload readback corruption, immutable backup pointer; queued backup flush, immutable bytes và lỗi network.
- Compile mọi code cell của hai notebook mới; kiểm tra original attack notebook, campaign driver/plan giữ nguyên byte so với package đang chạy.
- GPU smoke Molab: cả 4 task FOA/MPC × LLaVA/InternVL đã có 200/200 cặp caption sẵn sàng; LLaVA/FOA dùng batch 32, ba task tiếp theo dùng batch 64. GPT-4o đã lưu 3 score hợp lệ trước HTTP 429; phần chấm đầy đủ đang chờ key/project có quota. Attack 2 worker còn cần benchmark GPU thực tế.

Nguồn chính thức: [OpenAI authentication](https://developers.openai.com/api/reference/overview#authentication), [rate limits](https://developers.openai.com/api/docs/guides/rate-limits), [LLaVA batch inference](https://huggingface.co/docs/transformers/v4.57.0/model_doc/llava).
