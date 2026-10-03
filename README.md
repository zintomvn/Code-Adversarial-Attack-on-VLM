# Code-Adversarial-Attack-on-VLM

Kaggle workflow: [hướng dẫn attack → caption → GPT-4o judge](src/FOA-Attack/config/README.md).
Mỗi notebook có 4 code cell; logic OOP nằm trong các file Python hiện có, không có module `pipeline_*.py` mới.

- [Sinh ảnh adversarial](notebooks/attacks/generate_adv_samlples_foa_v1.ipynb).
- [Sinh caption](notebooks/captioning/text_generation_v1.ipynb): chọn `caption_<model>_v2` cho GPT-5 mini, Gemini 2.5 Flash, LLaVA-1.5-7B, InternVL3-8B hoặc Qwen3-VL-8B.
- [LLM-as-a-judge](notebooks/evaluation/gpt_evaluate_v1.ipynb): dùng duy nhất `evaluate_gpt4o_judge_v2`, judge cố định `gpt-4o`, chỉ đổi `INPUT_DIR` để chọn caption đã sinh. Không tải/chạy model captioning ở bước evaluation.

Theo protocol paper: rubric Appendix C, thành công khi **score > 0.5**, báo cáo **ASR (%)** và **AvgSim**. Evaluation cần `OPENAI_API_KEY`, không cần GPU. Các config evaluation theo tên model captioning đã bị loại bỏ; baseline gốc vẫn được giữ nguyên.
