# My-Attack-V3 — cấu trúc giống MPCAttack

Giữ cách tổ chức của `src/MPCAttack/`: attack, sinh caption, rồi chấm điểm là ba script độc lập. Phần bổ sung là **10 negative captions** bằng cách thay đúng một object của caption source trong notebook v6.4.1. Model weights frozen; vòng lặp attack chỉ tối ưu ảnh.

Hướng dẫn clone, cài môi trường và chạy bằng `python` trên Molab: [RUN_MOLAB.md](RUN_MOLAB.md).

```text
My-Attack-V3/
├── generate_adversarial_examples_MPCAttack.py
├── blackbox_text_generation.py
├── gpt_evaluate.py
├── config_schema.py
├── utils.py
├── negatives.py
├── model_lock.py
├── model_revisions.json
├── surrogates/                  # Encoder MPC; giữ cả lựa chọn L336
├── vlmeval/                     # Kit đi kèm MPCAttack
├── tests/
└── molab/legacy_v1/             # Bảo toàn pipeline/recovery cũ
```

## Đọc và sửa code

| File | Đối chiếu MPC | Nơi cần sửa |
|---|---|---|
| `config_schema.py` | DataConfig, OptimConfig, ModelConfig, MainConfig | Default paths, số mẫu, steps, epsilon, encoder; thêm NegativeConfig mặc định10 |
| `generate_adversarial_examples_MPCAttack.py` | `get_args → main → run_attack`, ImageFolderWithPaths, build_clip_models, extract_feature | Vòng lặp crop → feature → loss → gradient → MI-FGSM nằm trực tiếp trong file |
| `negatives.py` | Phần bổ sung từ notebook bạn đưa | Prompt, tách object, thay một object, lọc trùng, sinh/cache bank |
| `utils.py` | Cùng vai trò utils của MPC | `info_nce_loss`, lưu PNG, verify epsilon, file caption và helpers |
| `blackbox_text_generation.py` | ImageDescriptionGenerator, generate_descriptions, get_args/main | Prompt caption, model victim, decoding, text output và cache từng ảnh |
| `gpt_evaluate.py` | GPTScorer, read_descriptions, save_scores, get_args/main | Rubric MPC, threshold0.5, targeted/untargeted scores |

Các block `# Added:` trong file attack đánh dấu negative samples. Giữ CLIP B16/B32/Laion + InternVL3-1B + DINOv2, fusion image/text0.6/0.4, random crop và MI-FGSM. Negative anchors giữ image source ở mọi encoder; chỉ thay phần text CLIP bằng caption counterfactual. Maximize negative CE với một positive và K negatives như chiều tối ưu MPC; K1 khớp objective gốc.

Default giữ theo pilot V3:2mẫu,seed2026,caption Qwen3B; chúng khác một số default MPC gốc. Đây là tương thích cấu trúc/CLI, không phải tuyên bố replication byte-identical. `--ensemble True/False` và `--no-ensemble` đều dùng được; L336 giữ là lựa chọn tùy chọn với revision pin, không tự thêm vào ensemble mặc định. Seed theo image ID để cùng mẫu không đổi khi chia shard.

Sửa kỹ thuật được ghi rõ, không coi là cải thiện attack: ghép source/target theo ID thay vì zip danh sách sort độc lập, dùng `args.steps`, sửa dataclass mutable defaults, projection RGB và verify PNG sau khi lưu. Cache caption/GPT theo từng record; thiếu kết quả giữ PENDING thay vì điền0. Không dùng caption/judge để tune gradient hoặc chọn checkpoint.

## CLI giống MPC

Chạy `--help` không load weights hoặc gọi API:

```bash
cd src/My-Attack-V3
python generate_adversarial_examples_MPCAttack.py --help
python blackbox_text_generation.py --help
python gpt_evaluate.py --help
```

Các ví dụ thực thi dưới đây dùng **plan đã được user duyệt cho code/config mới**. Refactor hiện tại chỉ kiểm tra synthetic; source ZIP/approval của run cũ không tự áp dụng cho executable mới. `--approved_plan` ghi nhận plan được duyệt, không phải tên file tự cấp quyền.

```bash
python generate_adversarial_examples_MPCAttack.py \
  --cle_data_path /data/images/bigscale \
  --tgt_data_path /data/images/target_images \
  --output /outputs/My-Attack-V3 \
  --num_samples 2 --num_negatives 10 --steps 300 \
  --approved_plan /path/to/reviewed-and-approved-plan.md
```

File attack tự gọi `blackbox_text_generation.generate_descriptions` sau saved-PNG verification, rồi gọi `gpt_evaluate.main` nếu có `--scorer_model` và budget judge trong plan. Mặc định không gọi API judge; caption diagnostic được review riêng trước khi tiếp tục. Logic hai bước này vẫn nằm trong hai file độc lập như MPC, có thể chỉnh hoặc chạy lại trên PNG đã lưu mà không attack lại.

Script chờ review sau bank, sau update1 và sau hậu xử lý của từng mẫu. Reviewer phải ghi đủ7mục trước khi chuyển sang mẫu tiếp theo. Chạy riêng script caption cho PNG có sẵn:

```bash
python blackbox_text_generation.py \
  --output /outputs/My-Attack-V3 \
  --src_data_path /data/images/bigscale \
  --tgt_data_path /data/images/target_images \
  --des_path_src_tgt /outputs/clean_descriptions \
  --model_name Qwen2.5-VL-3B-Instruct \
  --approved_plan /path/to/reviewed-and-approved-plan.md
```

Chấm điểm GPT chỉ khi scope/budget API của plan đã được duyệt:

```bash
python gpt_evaluate.py \
  --output /outputs/My-Attack-V3 \
  --des_path_src_tgt /outputs/clean_descriptions \
  --model_name Qwen2.5-VL-3B-Instruct \
  --scorer_model gpt-4o-mini --success_threshold 0.5 \
  --approved_plan /path/to/reviewed-and-approved-plan.md
```

Kết quả rubric caption là diagnostic theo MPC, không tự gắn nhãn official ASR đã xác minh. API keys dùng environment hoặc `api_keys.yaml/json`; endpoint tùy chọn qua `--base_url`, không hardcode proxy.

Nếu hậu xử lý bị ngắt sau khi đã có `attack_complete.json`, chạy lại cùng config/plan sẽ kiểm tra hash PNG và chỉ tiếp tục caption/evaluation còn thiếu; không load lại attack models hay tối ưu lại ảnh. Attack bị ngắt giữa chừng vẫn được giữ riêng, không tự retry từ đầu.

## Output giống MPC

```text
<output>/img/<source_class>/<id>.png
<output>/description/adversarial_<model>.txt
<des_path_src_tgt>/source_<model>.txt
<des_path_src_tgt>/target_<model>.txt
<output>/description/scores_<model>.txt
<output>/description/scores_<model>_untargeted.txt
<output>/checkpoints/<id>/               # bank, crop, gradient, PNG bounds, history
```

Sửa default trong `config_schema.py`; CLI flags ghi đè các giá trị tương ứng. Pins nằm trong `model_revisions.json`. Dependencies core: `requirements.txt`; SDK API tùy chọn: `requirements-api.txt`. Evaluator khác của VLMEvalKit cần dependencies theo model như MPC; Qwen3B mặc định dùng loader pinned trực tiếp, không buộc Flash Attention.

`--max_seconds` bao phủ toàn CLI bằng timer trên Linux/Molab, bao gồm model loading và bank generation. Windows/programmatic callers dùng thêm kiểm tra thời gian giữa từng call/step. Caption cache chỉ reuse khi có JSON provenance khớp ảnh/protocol; cache text cũ không có metadata bị từ chối thay vì được coi là hợp lệ.

## Kiểm tra offline

```bash
python -X utf8 -m unittest discover -s src/My-Attack-V3/tests -p "test_*.py" -v
```

Refactor này không chạy inference/GPU attack/external evaluator. Source và artifacts run cũ giữ nguyên; xem `molab/README.md`.
