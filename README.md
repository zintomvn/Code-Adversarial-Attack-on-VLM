# Code-Adversarial-Attack-on-VLM

My-Attack-V3 extends [MPCAttack](https://github.com/LiYuanBoJNU/MPCAttack) with ten counterfactual negative captions per source image. Attack, caption generation and evaluation run separately.

## Requirements

Run in the **Molab Terminal**, from `src/My-Attack-V3` (or `/marimo/My-Attack-V3` if uploaded):

```bash
uv venv --python 3.11.15 /marimo/v3_env
uv pip install --python /marimo/v3_env/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
source /marimo/v3_env/bin/activate


pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U transformers
pip install hydra-core pytorch-lightning opencv-python scipy nltk timm==1.0.1 pandas
pip install git+https://github.com/openai/CLIP.git

```

```bash
cd M-Atack-V3
pip install -r requirements.txt
```

Use this separate environment to avoid the CUDA/NCCL conflict observed in Molab's default environment. Use V3 requirements, not the original MPC freeze (editable Git entry, old Torch/CUDA/Transformers pins). Requirements include dataset download, OpenAI SDK and common VLMEvalKit dependencies. Individual local models may require additional packages, depending on their loader. The bundled Qwen loader defaults to PyTorch SDPA; Flash Attention is optional via its constructor. Video-only `decord` is excluded from the image-attack requirements because its wheel failed the Molab platform check.

## Quick Start

### 1. Prepare Data

Set `DATA_ROOT` to the printed directory containing `images/`.

```bash
DATA_ROOT=/marimo/datasets/resources
OUTPUT=/marimo/outputs/experiment
CAPTIONS="$OUTPUT/descriptions"
```

### 2. Generate Adversarial Examples

Run one sample with ten negatives and 300 steps:

```bash
python -u generate_adversarial_examples_MPCAttack.py --cle_data_path "$DATA_ROOT/images/bigscale" --tgt_data_path "$DATA_ROOT/images/target_images" --output "$OUTPUT" --num_samples 1 --num_negatives 10 --steps 300 --epsilon 16 --alpha 1 --device cuda:0
```

For six samples, set `--num_samples 6`; the current filename order selects IDs `0, 1, 10, 100, 101, 102`. Use a new output directory for a new attack. PNGs are saved under `$OUTPUT/img/`; checkpoints and loss history under `$OUTPUT/checkpoints/`. Negatives are generated once before optimization.

### 3. Generate Captions

Use the MPC scripts in `src/My-Attack-V3`. Supported local aliases: `Qwen2.5-VL-7B-Instruct`, `InternVL3-8B`, `llava_v1.5_7b`, `glm-4.1v-9b-thinking`; Qwen3B is not in the copied MPC script. `--device` and `--approved_plan` are not arguments of this caption script.

```bash
python -u blackbox_text_generation.py --output "$OUTPUT" --src_data_path "$DATA_ROOT/images/bigscale" --tgt_data_path "$DATA_ROOT/images/target_images" --des_path_src_tgt "$CAPTIONS" --model_name Qwen2.5-VL-7B-Instruct
```

Source/target captions are saved in `$CAPTIONS`; adversarial captions in `$OUTPUT/description/`. MPC skips existing source/target caption files; adversarial captions are regenerated. Use a fresh caption directory for a fresh evaluation.

### 4. Evaluation

Create `api_keys.json` in the working directory, alongside `gpt_evaluate.py`. Do not commit this file:

```json
{ "gpt-4o-mini": "YOUR_API_KEY", "gpt4o": "YOUR_CAPTION_API_KEY_IF_USED" }
```

```bash
python -u gpt_evaluate.py --output "$OUTPUT" --des_path_src_tgt "$CAPTIONS" --model_name Qwen2.5-VL-7B-Instruct --scorer_model gpt-4o-mini --success_threshold 0.5
```

Results are saved in `$OUTPUT/description/`. This is the original MPC caption-similarity rubric; reported success rates refer to that protocol. Evaluation uses paid API calls.

### 5. Download Results
