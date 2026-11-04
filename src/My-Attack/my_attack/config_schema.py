"""Configuration defaults exported from the review notebook; no inference on import."""
from pathlib import Path
from types import SimpleNamespace
import json

from pathlib import Path

# Execution switches (GPU experiment and CPU-only checks)
INSTALL_PACKAGES = False
RUN_ATTACK = False
RUN_EVALUATION = False
RUN_VISUALIZATION = False
RUN_VALIDATION = False
RUN_SMOKE_TESTS = False
REUSE_COMPLETED_RESULTS = False  # Allow evaluation/plots without re-running the attack
VISUALIZATION_SAMPLE_ID = None   # None -> first completed sample
NUM_SAMPLES = 2
START_INDEX = 0
SEED = 2026

# Dataset: attach https://www.kaggle.com/datasets/zintom/blackbox-adversarial-attack-dataset
DATA_ROOT = Path("/kaggle/input/blackbox-adversarial-attack-dataset")
SOURCE_DIR = None                       # Auto-detect a directory named bigscale
TARGET_DIR = None                       # Auto-detect a directory named target_images
PAIR_MANIFEST = None                    # Optional CSV with columns source,target
OUTPUT_DIR = Path("/kaggle/working/cara_v6_4")

# Attack hyperparameters (pixel space [0, 255]; do not silently normalize)
K = 5
STEPS = 300
XAI_METHOD = "grad_eclip"               # grad_eclip | qwen_attention
EPSILON = 16.0                          # In [0, 255] pixel coordinates
ALPHA = 1.0
MOMENTUM = 0.9
TEMPERATURE = 0.2
OMEGA = 2.0
CROP_AREA = 0.5                        # Fixed XAI crop at each optimization step
ALPHA_GLOBAL = ALPHA                    # Independent global proposal, pixels
ALPHA_CROP = ALPHA                      # Independent crop proposal, pixels
SAVE_BRANCH_IMAGES = True               # Audit global/crop proposals every SAVE_EVERY
SAVE_EVERY = 30
ATTENTION_LAYER = 29
NEGATIVE_RETRIES = 10
CAPTION_ATTEMPTS = 40

# Models
IMAGE_SIZE = 224
CLIP_G14_ID = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
CLIP_B16_ID = "openai/clip-vit-base-patch16"
CLIP_B32_ID = "openai/clip-vit-base-patch32"
QWEN_ID = "Qwen/Qwen3-VL-8B-Instruct"
EVALUATION_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CLIP_DEVICES = {"G14": "cuda:1", "B16": "cuda:0", "B32": "cuda:0"}
ATTACK_DEVICE = "cuda:0"
QWEN_MAX_MEMORY_GIB = {0: 12, 1: 12}  # Adjust for your Kaggle GPUs
IMAGE_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
CAPTION_TEMPERATURE = 0.8
CAPTION_TOP_P = 0.9

# Evaluation is a caption-based proxy until human labels are completed.
EVALUATION_PROMPT = "Describe this image in one concise sentence, no longer than 20 words."

# V53 component — Review configuration; no model/API/experiment runs by default.
RUN_V53 = False
V53_STAGE = "prepare"                 # prepare | attack | postprocess
V53_RUN_ID = "v53-diffusion-regions-review01"
V53_ROOT = Path("/workspace/outputs/v53_diffusion_regions")
V53_SOURCE_DIR = Path("/workspace/data/bigscale/1")
V53_TARGET_DIR = Path("/workspace/data/target_images/1")
V53_PAIR_IDS = ["1", "10"]            # exact filenames, never sorted positions
V53_PAIR_HASHES = {sample_id: {"source": "", "target": ""} for sample_id in V53_PAIR_IDS}
V53_VARIANTS = ["negatives_10", "negative_1"]
V53_APPROVAL = ""                     # fill only after explicit scope approval
V53_APPROVED_CONFIG_SHA256 = ""        # review config fingerprint before launch
V53_MAX_WALL_SECONDS = None            # REQUIRED reviewed aggregate limit
V53_MAX_GPU_HOURS = None               # REQUIRED reviewed limit (one GPU)
V53_MAX_API_USD = None                 # REQUIRED reviewed limit
V53_DEVICE = "cuda:0"                  # MoLab RTX 6000, one GPU, sequential stages
V53_EXPECTED_GPU = "6000"
V53_K = 10
V53_STEPS = 300
V53_EPSILON = 16.0                     # pixel units [0,255]
V53_ALPHA = 1.0
V53_MOMENTUM = 0.9
V53_TEMPERATURE = 0.2
V53_OMEGA = 2.0
V53_SEED = 2026
V53_CROPS = 5
V53_CROP_SCALE = (0.5, 0.9)            # area fraction, not side length
V53_CROP_RATIO = (0.75, 4 / 3)
V53_SIZE = 224
V53_LOG_SECONDS = 30
V53_SAVE_EVERY = 30                    # steps; independent of heartbeat seconds
V53_IMAGE_WEIGHT = 0.6
V53_TEXT_WEIGHT = 0.4
V53_DIFFUSION_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
V53_DIFFUSION_STRENGTH = 0.9
V53_DIFFUSION_GUIDANCE = 7.5
V53_DIFFUSION_STEPS = 40
V53_DIFFUSION_SEED = 3
V53_CAPTION_PROMPT = "Describe this image in one concise sentence, no longer than 20 words."
V53_CAPTION_TEMPERATURE = 0.8
V53_CAPTION_TOP_P = 0.9
V53_CAPTION_TOKENS = 50
V53_VICTIM_TOKENS = 64
V53_CLIP_IDS = [CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID]
V53_VICTIMS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "internvl": "OpenGVLab/InternVL3-8B-hf",
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "gpt": "gpt-5-mini-2025-08-07",
}
# Pin all HF model AND processor revisions before review/run. No automatic main resolution.
V53_REVISIONS = {model_id: "" for model_id in
                 [*V53_CLIP_IDS, V53_DIFFUSION_ID, *list(V53_VICTIMS.values())[:3]]}
V53_GPT_MODEL = V53_VICTIMS["gpt"]       # same model requested for GPT-score
V53_ASR_THRESHOLD = 0.5                 # success iff score >= 0.5 (MPC convention)
V53_API_MAX_CALLS = 24                  # 8 image captions + 16 target/adv scores
V53_API_OUTPUT_TOKENS = 1024
V53_API_INPUT_TOKEN_CAP = 8192          # reserve conservative input bound/request
V53_API_INPUT_USD_PER_M = 0.25          # official docs checked 2026-10-28; review before run
V53_API_OUTPUT_USD_PER_M = 2.0

class AttackConfig(SimpleNamespace):
    """Mutable runtime configuration, configured once before model loading."""

cfg = AttackConfig(**{key: value for key, value in globals().copy().items() if key.isupper()})
cfg.INTERACTIVE_REVIEW = False
cfg.V53_MAX_WALL_SECONDS = None
cfg.V53_MAX_GPU_HOURS = None
cfg.V53_MAX_API_USD = None


def load_config(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in record.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown configuration key: {key}")
        if key in {"V53_ROOT", "V53_SOURCE_DIR", "V53_TARGET_DIR"}:
            value = Path(value)
        setattr(cfg, key, value)
    return cfg
