# FOA pipeline trên Kaggle — attack v1, caption/evaluation paper v2

## Chạy ba notebook

1. Upload repository này thành Kaggle Dataset (không upload `api_keys.*`). Trong notebook đặt `REPO_DIR` tới thư mục chứa `src/FOA-Attack`. Bật Internet. Bật GPU khi attack hoặc caption bằng model local.
2. Chạy `notebooks/attacks/generate_adv_samlples_foa_v1.ipynb`. Sửa `SOURCE_DIR`, `TARGET_DIR` tới các thư mục ImageFolder: mỗi thư mục phải có thư mục con chứa ảnh. Hai ImageFolder được ghép theo thứ tự đường dẫn đã sort, giống baseline; kiểm tra `original_paths` trong manifest để xác nhận pairing. `attack_foa_smoke_v1` chỉ kiểm tra 1 cặp/2 bước, không dùng báo cáo kết quả baseline.
3. Tải ZIP, giải nén và upload thư mục output thành Kaggle Dataset. Chạy `notebooks/captioning/text_generation_v1.ipynb`, đặt `INPUT_DIR` tới **một run attack**, rồi chọn `CONFIG_NAME` bên dưới. Không upload ZIP chưa giải nén. Output caption không chứa lại ảnh; giữ ZIP attack riêng.
4. Tải ZIP caption, giải nén/upload; chạy `notebooks/evaluation/gpt_evaluate_v1.ipynb`. Đặt `INPUT_DIR` tới **một run caption**. Giữ `CONFIG_NAME = "evaluate_gpt4o_judge_v2"` cho mọi model captioning: judge luôn là GPT-4o, không phải model sinh caption. Tên notebook giữ `_v1` theo yêu cầu đường dẫn ban đầu, nhưng config mặc định đã dùng protocol v2.

Mỗi notebook có 4 code cell: cấu hình đường dẫn, cài môi trường, gọi Python, xuất ZIP. Khi đổi model chỉ đổi `CONFIG_NAME`; đường dẫn input giữ nguyên nếu dùng cùng dataset. Mọi xử lý nằm trong các file Python cũ, không có Python module mới.

| Model | CONFIG_NAME caption | CONFIG_NAME evaluation |
|---|---|---|
| GPT-5 mini | `caption_gpt5mini_v2` | `evaluate_gpt4o_judge_v2` |
| Gemini 2.5 Flash | `caption_gemini25flash_v2` | `evaluate_gpt4o_judge_v2` |
| LLaVA-1.5-7B | `caption_llava15_7b_v2` | `evaluate_gpt4o_judge_v2` |
| InternVL3-8B | `caption_internvl3_8b_v2` | `evaluate_gpt4o_judge_v2` |
| Qwen3-VL-8B | `caption_qwen3vl_8b_v2` | `evaluate_gpt4o_judge_v2` |

Secrets trong Kaggle: cấp quyền `OPENAI_API_KEY` cho caption GPT và tất cả evaluation; `GEMINI_API_KEY` cho Gemini. API có phí; GPU Kaggle không chạy được trọng số GPT/Gemini. Code không tự thay model khi API ngừng hỗ trợ. GPT-5 mini dùng snapshot `gpt-5-mini-2025-08-07`; kiểm tra quyền truy cập trước khi chạy toàn bộ dataset.

## Baseline, OOP và version

- `FOABaselineRunner(ExperimentRunner)` gọi **nguyên hàm `fgsm_attack`, `get_models_ot`, `get_ensemble_loss_ot`** trong script baseline; K=10, seed=2023, state loss được giữ qua các cặp ảnh. Không sửa hàm/class baseline hay YAML `ensemble_3models.yaml`. Entry point cũ vẫn chạy như trước.
- `FOAClusterAblationRunner(FOABaselineRunner)` chỉ thay K/backbones. `attack_foa_k5_ablation_v1` dùng K=5; `attack_foa_smoke_v1` dùng B16/B32 và 2 bước. Đây là ablation/smoke, **không phải proposed method mới**. Muốn thêm thuật toán đề xuất, kế thừa runner trong cùng file hiện có và đăng ký vào `KaggleRuntime.execute`; giữ cùng artifact contract/evaluator.
- `VersionedImageDescriptionGenerator(ImageDescriptionGenerator)` bổ sung adapter mới; `VersionedGPTScorer(GPTScorer)` kế thừa đúng rubric/API/temperature baseline và chỉ bổ sung validation. `PaperGPT4oScorer(VersionedGPTScorer)` cố định judge `gpt-4o`. `FOAPaperEvaluationRunner(GPTEvaluationRunner)` dùng ngưỡng/rule main-paper; không rẽ nhánh metric theo phương pháp attack.
- Không import `FOAttack.py`/`MyAttack.py`: các file đó cố định `CUDA_VISIBLE_DEVICES=2` và gộp cả ba bước. Không sửa các file này.
- Khi đổi thông số, **copy config sang tên version mới**, đổi `experiment.name` và `experiment.version` tương ứng; không sửa config của run đã báo cáo. Ví dụ `caption_qwen3vl_8b_v2` với `experiment.version: v2`. Với biến thể v1 dùng tên có tag như `caption_qwen3vl_8b_fp16_v1`. Nếu đổi package, đặt `environment.name` mới để giữ môi trường cũ.
- Mỗi lần chạy tạo thư mục timestamp mới, không ghi đè. Output chứa `config_resolved.yaml`, SHA-256 config/input/ảnh, code/config snapshot và `requirements-resolved.txt`. Giữ cả snapshot, dữ liệu gốc và weights/revision để tái lập.
- HF mặc định `revision: main`; output ghi commit đã resolve. Để tái lập weights nghiêm ngặt, tạo config có `revision` là commit cụ thể. Hai môi trường attack/local VLM tách biệt (Transformers 4.49.0 vs 4.57.6); giữ Torch/torchvision GPU sẵn có của Kaggle và ghi lại phiên bản. Điều này bảo toàn thuật toán nhưng **không đảm bảo bitwise reproduction** giữa GPU/package khác nhau; dùng environment gốc khi yêu cầu đối chiếu bitwise.
- `kmeans-pytorch==0.3` trên PyPI không hỗ trợ `iter_limit=100` mà loss baseline gọi. Config attack pin commit upstream `f7f36bd1cb4e3a761d73d584866d0a9c6b4d2805` có tham số này; không bỏ tham số/patch hàm loss để che lỗi. Đây là dependency tương thích, không khẳng định là commit dependency trong môi trường paper ban đầu. Snapshot `pip freeze` ghi commit thực tế.
- Runner attack không resume giữa chừng vì baseline có state loss và RNG qua các ảnh. Khởi chạy lại sẽ tạo run mới. Caption/score ghi JSONL từng mẫu, lỗi được lưu riêng để không mất output đã có; hiện chưa resume/cache xuyên phiên. Có thể chạy cell export sau khi lỗi để tải output dở dang, nhưng không dùng run attack dở dang làm báo cáo đầy đủ.

## Model local và giới hạn GPU

Local configs dùng NF4 4-bit, một GPU `cuda:0`, greedy decoding, FP16 trên T4/P100 hoặc BF16 khi GPU hỗ trợ. Đây là evaluation **quantized**, không coi tương đương FP16/BF16 đầy đủ. Nếu so sánh baseline/proposed, dùng cùng config caption. Muốn full precision, tạo config riêng với `quantization: none`; 8B có thể vượt VRAM Kaggle. Không tự giảm backbone hay đổi quantization khi OOM.

InternVL dùng bản chuyển đổi chính thức `OpenGVLab/InternVL3-8B-hf`, không dùng remote code của checkpoint gốc. LLaVA dùng `llava-hf/llava-1.5-7b-hf`; Qwen dùng bản `Qwen/Qwen3-VL-8B-Instruct`. Ghi rõ các checkpoint khi viết luận văn.

Ensemble baseline có CLIP ViT-G/14 Laion lớn, có thể không vừa GPU Kaggle 16 GB. Smoke chỉ kiểm tra pipeline nhẹ; không âm thầm chuyển baseline sang smoke. Nếu OOM với baseline, cần GPU đủ VRAM hoặc nghiên cứu riêng phương án offload (không nằm trong phiên bản này).

## Metric và outputs

Attack xuất PNG source/target sau đúng Resize+CenterCrop baseline, PNG adversarial lossless, `manifest.jsonl` và kiểm tra L-infinity của PNG không vượt epsilon (thang 0–255). Caption cả ba ảnh bằng cùng model; các config v2 dùng prompt paper **“Describe this image.”**, không thêm giới hạn 20 từ. Không gửi ảnh target trong cùng request ảnh adversarial. Max token vẫn là giới hạn kỹ thuật; local v2 tăng lên 512 và báo lỗi khi output bị truncation.

Evaluation mặc định mới dùng **LLM-as-a-judge `gpt-4o`**, độc lập với cả năm model captioning. Theo §4.1 (trang 7) của PDF trong `docs/papers/(2025-NeurIPS) FOA-Attack.pdf`, cùng target MLLM sinh caption cho target/adversarial; main-paper định nghĩa thành công khi **target_adv > 0.5**. Rubric `GPTScorer.compute_similarity` khớp Appendix C/Figure 5 (trang 17), nên được kế thừa nguyên trạng; temperature=0. Config chung tự đọc một caption model từ input, chặn input trộn nhiều model/protocol/attack. Judge chỉ nhận **hai chuỗi caption**, không nhận ảnh và không gọi model captioning. GPT-4o là judge theo yêu cầu thực nghiệm của bạn; paper mô tả framework/rubric nhưng không chỉ rõ snapshot judge trong đoạn metric này. Alias `gpt-4o` có thể thay đổi; output lưu model thực tế API trả về trong response metadata.

Mặc định chỉ chấm target–adversarial để tính **ASR (%)** và **AvgSim** như paper; `include_diagnostics: true` thêm source–adversarial/source–target, không đổi metric chính. `judge_metadata.json` tách rõ judge và caption model. Caption v1 có thể chấm bằng judge v2, nhưng code cảnh báo/ghi metadata khi prompt không khớp hoặc không kiểm chứng được; để tái lập caption protocol paper cần sinh caption bằng config v2. Không trộn kết quả hai prompt khi báo cáo.

Chỉ còn **một config evaluation hoạt động: `evaluate_gpt4o_judge_v2.yaml`**. Các config evaluation đặt theo model captioning và protocol evaluation v1 đã được xóa khỏi thư mục config hoạt động; bản sao phục hồi nằm trong [archives/evaluation_configs_legacy_v1.zip](archives/evaluation_configs_legacy_v1.zip), không được Hydra load. Nhánh dispatch evaluation v1 cũng đã được loại bỏ. `caption_*_v1` vẫn là config **sinh caption**, không phải config judge, và được giữ làm base cho caption v2.

Không có trường chọn `caption_model` trong config evaluation: model sinh caption chỉ được ghi vào output để truy xuất nguồn dữ liệu và chặn input trộn nhiều run. Mọi lời gọi API chấm điểm dùng `gpt-4o`; code từ chối model khác. Entry point baseline cũ, class `GPTScorer`, các hàm attack gốc không bị sửa. Muốn tái lập pipeline baseline gốc, dùng các entry point/config gốc; phiên bản Kaggle mới chỉ hỗ trợ GPT-4o judge với protocol paper **score > 0.5**. Khi so sánh baseline/proposed, chạy cả hai bằng cùng protocol v2.

Cần diễn giải đây là độ giống ngữ nghĩa so với **caption target do model sinh**, không phải ground truth annotation hay chứng minh model nhìn đúng ảnh.

Metric gọi GPT-Score trong repo là **LLM-as-judge semantic similarity**, không phải GPTScore log-likelihood trong một số paper. `scores.jsonl` lưu điểm target–adv (và diagnostics nếu bật) cùng metadata response; `scores.csv` tiện phân tích; `summary.json` báo `ASR_percent_valid`, `AvgSim`, số lỗi và ASR lower bound trên toàn tập. Caption/judge lỗi không bị gán score=0. Nếu có lỗi, báo denominator và không so ASR hợp lệ đơn độc. API temperature=0 vẫn không đảm bảo determinism tuyệt đối.

## Lệnh tương đương ngoài notebook

```bash
python src/FOA-Attack/utils.py setup --config-name attack_foa_smoke_v1
python src/FOA-Attack/utils.py run --config-name attack_foa_smoke_v1 --source /path/source --target /path/target
python src/FOA-Attack/utils.py export --config-name attack_foa_smoke_v1
python src/FOA-Attack/utils.py setup --config-name caption_qwen3vl_8b_v2
python src/FOA-Attack/utils.py run --config-name caption_qwen3vl_8b_v2 --input /path/one_attack_run
python src/FOA-Attack/utils.py setup --config-name evaluate_gpt4o_judge_v2
python src/FOA-Attack/utils.py run --config-name evaluate_gpt4o_judge_v2 --input /path/one_caption_run
```

`show --config-name NAME` in config đã compose để kiểm tra trước khi tải weights/gọi API. Tên config không có `.yaml`. Mặc định output `/kaggle/working/foa_outputs`; có thể dùng `--output`. Setup chỉ cài package từ config; không tải weights hay gọi API.

Nguồn API/checkpoint: [OpenAI GPT-5 mini](https://developers.openai.com/api/docs/models/gpt-5-mini), [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking), [LLaVA](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/llava), [InternVL3 HF](https://huggingface.co/OpenGVLab/InternVL3-8B-hf), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
