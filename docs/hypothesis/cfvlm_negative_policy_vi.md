# Sửa negative caption theo CF-VLM

Notebook mới: `notebooks/attacks/kltn-attack-v4.3.3-h1-cfvlm.ipynb`.
Notebook v4.3.2 và driver cũ được giữ nguyên để bảo toàn kết quả/resume cũ.

Nguồn đối chiếu: [CF-VLM NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/541b6d155146d142a5e10d787d1d430f-Paper-Conference.pdf), mục 2.2 và Appendix A trang 17; [code chính thức](https://github.com/HCP-AI-Research-Lab/CF-VLM/blob/main/process.py).

Prompt cũ giới hạn thay đúng một cụm danh từ. Prompt mới cho phép thay một
thuộc tính của một đối tượng (màu, chất liệu, hình dạng, kích thước, vị trí/hướng,
trạng thái/tư thế, loại đối tượng, bộ phận/cấu trúc, chuyển động, số lượng,
biểu cảm), hoặc một quan hệ nhân quả đã có trong caption. Với nhân quả, đổi
nguyên nhân cùng hệ quả phụ thuộc vẫn là một can thiệp; không đổi thêm yếu tố
độc lập. Caption phải tự nhiên, khác biệt có ý nghĩa, giữ các chi tiết khác,
không trùng và mâu thuẫn với ảnh target.

Ví dụ: `A red car waits beside a tree.` → `A blue car waits beside a tree.`
Quan hệ: `A boy kicks the ball, causing it to roll forward.` →
`A boy misses the ball, causing it to stay still.`

`cfvlm_negatives.py` là nguồn prompt/parser, được nhúng trong notebook để chạy
độc lập. `build_cfvlm_attack.py` tái tạo notebook. JSON có `edit_type`,
`attribute`, `edits`, `caption`, và trace nguyên nhân/hệ quả khi cần. Parser
kiểm tra vị trí chỉnh sửa, giữ nguyên phần ngoài các span, span không nhập
nhằng/chồng lấn, loại hợp lệ, không lặp/no-op. Đây là kiểm tra cấu trúc;
không chứng minh được caption sai với ảnh hay chỉ đổi một yếu tố về ngữ nghĩa.
Raw output và trace được giữ để kiểm tra chất lượng negative bank.

Chạy thử phát hiện Qwen có thể gắn nhãn `color` cho thay đổi cả thời tiết và
bối cảnh. Bank thử v1 bị loại khỏi campaign chính. Policy
`cfvlm_text_edits_v2_audited` thêm lượt Qwen có ảnh để kiểm tra từng ứng viên:
một can thiệp, giữ chi tiết khác, mâu thuẫn với ảnh, mạch lạc, nhãn chính xác.
Thiếu quyết định, trùng ID, caption không khớp, boolean không đúng kiểu,
hoặc một tiêu chí sai/không chắc đều bị loại. Retry nhận lại lý do bị loại.
Audit dùng cùng Qwen checkpoint với lượt sinh, nên vẫn là kiểm tra tự động
có thể sai, không thay thế đánh giá chất lượng của con người.

Sinh 4 ứng viên mỗi ảnh (K mặc định của code chính thức), chọn/mine 2 như
thí nghiệm cũ. Negative variants dùng chung bank đã cache. Không đủ 4 sau
5 lượt retry thì ghi lỗi, không tạo fallback hay loại mẫu ngầm. Qwen sinh và audit batch 16,
OOM chia đôi; caption evaluation batch 64 có OOM splitting. Attack FOA giữ
chuỗi trạng thái và thứ tự ảnh, song song tối đa 2 job độc lập.

Đây là áp dụng chính sách chỉnh sửa văn bản vào attack, không phải tái hiện
toàn bộ CF-VLM: paper còn sinh ảnh counterfactual bằng SDXL và fine-tune VLM.
Mô hình sinh, JSON audit, validator và batching là lựa chọn triển khai của ta.

Campaign sửa dùng root, plan, bank, fingerprint và thư mục Drive mới
`cfvlm_v2`; không trộn bank thử v1 hoặc 13 kết quả hybrid cũ. Giữ lại caption source/target
và hai baseline FOA/MPC 200 mẫu làm provenance; baseline không dùng negative
bank. Khi mỗi cấu hình đủ 200 mẫu, controller tạm nhường GPU ở ranh giới mẫu
đã lưu, chạy caption 2 victim rồi GPT judge ngay; GPT có thể chạy cùng attack
tiếp theo. Quota dài thì lưu checkpoint và trì hoãn judge.

Notebook Molab chạy/resume: `notebooks/attacks/molab_cfvlm_overnight_v2.ipynb`.
Builder `build_molab_cfvlm_bundle.py` tạo ZIP và notebook với SHA cố định;
`molab_cfvlm_setup.py` khóa plan/provenance, khôi phục checkpoint và cài hai
môi trường attack/evaluation riêng. Control H4 chỉ khởi động sau khi cache
source/target đã đủ 200 mẫu để tránh tự sinh lại caption ngoài ý muốn.
