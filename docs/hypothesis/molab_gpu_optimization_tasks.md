# Tasks tối ưu GPU cho campaign Molab n200

Ngày lập: 2026-10-16. Trạng thái: backlog, chưa triển khai tối ưu.

Mục tiêu: tăng số mẫu attack hoàn thành mỗi giờ trên RTX PRO 6000, giữ nguyên ý nghĩa đối chiếu giả thuyết và khả năng khôi phục checkpoint. Không dùng mức chiếm RAM/VRAM làm mục tiêu tối ưu.

## Phạm vi và các ràng buộc

- Campaign gồm 200 cặp × 10 cấu hình × 300 bước = 2.000 lượt attack / 600.000 bước. Evaluation tính riêng.
- Giữ nguyên dữ liệu và thứ tự cặp, seed 2023, epsilon 16/255, alpha 1 pixel, model revisions, captions, negative bank và các đối chiếu H1–H4.
- `mpc_clip` không phải full MPC. Hai controls H4 là bỏ MPC term và image-only fusion.
- Không sửa driver/plan của lượt đang chạy: hash của hai file tham gia định danh resume.
- FOA giữ trạng thái loss/weights và RNG giữa các ảnh. Không mặc định coi các ảnh là các tác vụ độc lập để chia worker hoặc batch.
- Mỗi worker/job mới phải có thư mục kết quả, trạng thái và đích backup riêng. Không ghi chung checkpoint từ hai tiến trình hoặc hai tài khoản.
- Credentials chỉ ở Secrets/environment; không commit API key, OAuth token, `.env`, session URL hay dữ liệu thực nghiệm.

## Bằng chứng ban đầu

Log quan sát được cho `foa_reference`: khoảng 42.000/60.000 bước, thời gian vòng attack khoảng 2 giờ 58 phút; tốc độ trong vòng tối ưu có lúc khoảng 4,9 bước/giây. Thông lượng trung bình gồm thời gian giữa các ảnh khoảng 3,9 bước/giây. Đây là một đoạn quan sát, chưa phải benchmark được kiểm soát.

Code hiện chạy từng ảnh và từng cấu hình tuần tự. Caption/negative generation cũng tuần tự. Campaign chờ backup Drive tại các ranh giới checkpoint. Launcher tắt activation CPU offload và CLIP offload; hybrid mặc định FP32.

Chưa thu được GPU utilization, công suất, VRAM và CPU utilization trực tiếp. Chênh lệch thông lượng chưa thể quy hoàn toàn cho Drive. Ước lượng 45–65 giờ trước đó chỉ là ngoại suy từ smoke test một ảnh, không dùng làm baseline đã xác nhận.

## Thứ tự triển khai

T01 → T02 → T03. Sau đó thử riêng T04, T05 và T06; T07 dành cho evaluation. T08 tổng hợp kết quả trước khi áp dụng vào campaign mới.

### T01 — Đo tài nguyên và thời gian theo giai đoạn

- [ ] P0 · Chưa bắt đầu · Phụ thuộc: quyền truy cập runtime Molab.
- Thu telemetry chỉ đọc, tối thiểu 60 giây trong vòng tối ưu và qua ranh giới hai ảnh: GPU utilization, memory used/total, power draw/limit, CPU, RSS, timestamp và PID.
- Dùng `nvidia-smi` nếu runtime hỗ trợ; ghi rõ các trường không hỗ trợ. Không ghi toàn bộ environment/process command line có thể chứa secret.
- Ở bản benchmark riêng, đo riêng caption, negative generation, attention refresh, forward/backward/update, ghi checkpoint, upload và xác minh Drive.
- Đo CUDA bằng events hoặc synchronize tại ranh giới đo; không synchronize mỗi bước trong production.
- Đầu ra: telemetry CSV, bảng thời gian theo giai đoạn, thông lượng hoàn thành mẫu/giờ.
- Hoàn thành khi: có bằng chứng phân biệt được điểm nghẽn GPU compute, CPU, đọc/ghi hoặc mạng; tách warm-up khỏi phép đo chính.

### T02 — Tạo benchmark cố định và kiểm tra tính tương đương

- [ ] P0 · Chưa bắt đầu · Phụ thuộc: T01.
- Tạo runner riêng, không thay package đang chạy. Khóa tập cặp nhỏ, thứ tự, seed, revisions và text records trước khi đo.
- Bao phủ `foa_reference`, `mpc_clip`, `hybrid_plain`, `hybrid_dynamic`; chạy đủ 300 bước/ảnh, giữ trạng thái xuyên ảnh của FOA.
- Lưu baseline: thời gian từng ảnh, peak VRAM, loss history, trạng thái RNG/loss, checksum ảnh và artifact completeness.
- Kiểm tra resume bằng cách dừng ở ranh giới checkpoint rồi chạy tiếp, đối chiếu với chạy liên tục.
- Xác định và ghi trước dung sai số học phù hợp; với thay đổi chỉ về I/O, yêu cầu payload/checksum kết quả không đổi.
- Hoàn thành khi: benchmark tái chạy được và có tiêu chí pass/fail cho mọi phương án tối ưu.

### T03 — Tách backup khỏi vòng tính toán một cách an toàn

- [ ] P1 · Chưa bắt đầu · Phụ thuộc: T02 và T01 xác nhận thời gian I/O đáng kể.
- Tạo snapshot local bất biến sau khi ghi xong checkpoint; background worker chỉ upload snapshot đó, không đọc file đang bị thay đổi.
- Giữ upload + checksum readback. Chỉ cập nhật durable pointer sau khi xác minh thành công; local-complete và remote-durable phải được phân biệt.
- Dùng queue giới hạn dung lượng; có backpressure nếu mạng chậm. Không tạo queue tăng vô hạn hoặc nhiều bản trùng một mẫu.
- Khi lỗi backup: giữ snapshot để retry, báo lỗi rõ và dừng ở ranh giới an toàn; không âm thầm chạy tiếp hết campaign.
- Trước pause/exit: drain queue trong ngân sách thời gian còn lại; không tuyên bố checkpoint an toàn khi chưa durable.
- Hoàn thành khi: benchmark tăng thông lượng; test lỗi upload/checksum, mạng chậm, pause và restore đều không mất mẫu hoặc lệch trạng thái FOA.

### T04 — Benchmark batch cho caption và negative generation

- [ ] P1 · Chưa bắt đầu · Phụ thuộc: T02.
- Thử batch 1/2/4/8 theo peak VRAM; giữ nguyên preprocessing, prompt, token limit, decoding và revisions.
- Xử lý padding/slicing đầu ra đúng cho từng mẫu; giữ metadata, retries và lỗi ở cấp sample ID.
- Đối chiếu text và số negative hợp lệ với baseline. Batch có thể làm đổi kết quả sinh dù greedy: nếu khác, không thay bank của campaign hiện tại.
- Negative bank phục vụ H1 hard mining cần báo riêng tập bank size > 2; không loại mẫu hoặc sinh lại để chọn kết quả thuận lợi.
- Hoàn thành khi: có bảng throughput/VRAM/độ tương đương; phương án được chấp nhận giữ bank cũ hoặc tạo campaign mới với bank được khóa dùng chung cho mọi cấu hình.

### T05 — Benchmark chạy đồng thời các cấu hình độc lập

- [ ] P2 · Chưa bắt đầu · Phụ thuộc: T02, T03.
- Thử một worker so với hai worker trên cùng GPU với cùng tổng khối lượng công việc; đo aggregate samples/hour, peak VRAM, CPU và độ trễ backup.
- Mỗi worker sở hữu toàn bộ chuỗi 200 ảnh của một cấu hình, có model/RNG/loss state và output riêng. Không chia tùy ý ảnh của FOA giữa các worker.
- Tính VRAM cho cả model lẫn activations và dự phòng; không suy ra đủ bộ nhớ chỉ từ VRAM lúc idle.
- Caption/negative bank dùng bản khóa chung. Hai controls H4 chỉ chạy khi captions/revisions của primary đã sẵn sàng.
- Nếu triển khai nhiều tài khoản Molab: namespace backup riêng, ghi account/job provenance và hợp nhất theo configuration ID + pair ID sau khi xác minh hashes.
- Hoàn thành khi: không OOM hoặc checkpoint collision; tổng thông lượng tăng và trạng thái/kết quả đạt tiêu chí T02. Nếu chậm hơn, giữ một worker.

### T06 — Đánh giá batching/precision bên trong attack

- [ ] P2 · Chưa bắt đầu · Phụ thuộc: T02; chỉ thực hiện nếu T01 chỉ ra GPU còn dư tài nguyên tính toán.
- Phân tích semantics trước: FOA state, lịch sử loss, per-image KMeans, gradient, crop order, momentum và RNG.
- Chỉ vector hóa các phép toán độc lập đã chứng minh tương đương. Không đặt batch size > 1 cho toàn bộ attack chỉ vì VRAM còn trống.
- Thử mixed precision riêng nếu cần; giữ ghi nhận nonfinite và FP32 fallback. Xem đây là thay đổi số học cần kiểm chứng, không mặc định tương đương FP32.
- Hoàn thành khi: có bảng speed/memory/numerical difference và kiểm tra resume. Phương án thay đổi semantics phải trở thành thí nghiệm riêng, không thay reproduction hiện tại.

### T07 — Tối ưu evaluation sau khi có đủ attack artifacts

- [ ] P2 · Chưa bắt đầu · Phụ thuộc: T02 và attack outputs được xác minh.
- Benchmark caption batch 8/16/32 theo khả năng từng victim, bắt đầu từ batch 8 hiện tại; giữ model revisions và prompt.
- Kiểm tra padding, mapping sample ID, deterministic decoding và checkpoint trước khi tăng batch.
- GPT judge dùng concurrency có giới hạn theo quota thực tế, retry/backoff và cache; không suy ra tăng GPU sẽ tăng tốc judge API.
- Giữ nguyên protocol evaluation v3.1, threshold >= 0,3; báo coverage và lỗi. Không tự đổi sang protocol khác trong quá trình tối ưu.
- Hoàn thành khi: captions/scores map đúng mẫu, không duplicate judge calls khi resume, có thông lượng và chi phí đo được.

### T08 — Báo cáo benchmark, ước lượng thời gian và triển khai

- [ ] P1 · Chưa bắt đầu · Phụ thuộc: T01–T03; thêm kết quả các task tối ưu đã thử.
- Báo bảng baseline/treatment: completed samples/hour, runtime, peak VRAM, GPU utilization, backup overhead và kiểm tra tương đương.
- Đo từng cấu hình, không lấy tốc độ một ảnh hybrid làm tốc độ chung cho cả 10 cấu hình.
- Ước lượng thời gian còn lại = tổng số mẫu chưa hoàn thành × thời gian/mẫu của từng cấu hình, cộng setup/restore/text preparation chưa xong; evaluation tách riêng.
- Đóng gói version mới và branch riêng; kiểm tra metadata/hash trước khi chạy. Duy trì package cũ để resume campaign hiện tại.
- Chỉ chuyển ở ranh giới an toàn sau khi checkpoint đã durable; không hot-edit driver/plan đang chạy hoặc bỏ qua identity check.
- Hoàn thành khi: có quyết định giữ/loại từng tối ưu, hướng dẫn resume/rollback và smoke test đủ 300 bước kèm restore thành công.

## Nguồn code trong workspace

- `notebooks/attacks/kltn-attack-v4.3.2-h1.ipynb`: thuật toán, text preparation và vòng lặp experiment.
- `notebooks/attacks/molab_hypothesis_campaign.py`: orchestration, backup và resume.
- `notebooks/attacks/molab_hypotheses_n200.json`: ma trận 10 cấu hình và evaluation.
- `notebooks/text_gen_eval_v3.1.ipynb`: protocol đánh giá.

Tài liệu này là kế hoạch công việc. Các checkbox chưa đánh dấu không có nghĩa là đã triển khai hoặc benchmark thành công. Việc commit tài liệu task không tự xuất bản các notebook/code đang chưa được Git theo dõi trong workspace.
