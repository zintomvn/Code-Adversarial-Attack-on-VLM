from pathlib import Path

# Execution
INSTALL_PACKAGES = False
RUN_ATTACK = False
RUN_EVALUATION = True
RUN_VISUALIZATION = True
NUM_SAMPLES = 2
START_INDEX = 0
SEED = 2026

# Dataset: attach https://www.kaggle.com/datasets/zintom/blackbox-adversarial-attack-dataset
DATA_ROOT = Path("/kaggle/input/blackbox-adversarial-attack-dataset")
SOURCE_DIR = None                       # Auto-detect a directory named bigscale
TARGET_DIR = None                       # Auto-detect a directory named target_images
PAIR_MANIFEST = None                    # Optional CSV with columns source,target
OUTPUT_DIR = Path("/kaggle/working/cara_v6_4_1")


K = 5
STEPS = 300
XAI_METHOD = "qwen_attention"               # grad_eclip | qwen_attention
EPSILON = 16.0                          # In [0, 255] pixel coordinates
ALPHA = 1.0
MOMENTUM = 0.9
TEMPERATURE = 0.2
OMEGA = 2.0
CROP_AREA = 0.5                        # Fixed XAI crop for ALL 300 updates
ALPHA_GLOBAL = ALPHA                    # Independent global proposal, pixels
ALPHA_CROP = ALPHA                      # Independent crop proposal, pixels
SAVE_BRANCH_IMAGES = True               # Audit global/crop proposals every SAVE_EVERY
SAVE_EVERY = 30
SAVE_CROP_EVERY = SAVE_EVERY             # Also save update1 for exact crop inspection
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

RUN_ID = "CARA641-20261027-N2-4V-v1"
MODEL_REVISIONS = {
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K": "4b0305adc6802b2632e11cbe6606a9bdd43d35c9",
    "openai/clip-vit-base-patch16": "57c216476eefef5ab752ec549e440a49ae4ae5f3",
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "llava-hf/llava-1.5-7b-hf": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "OpenGVLab/InternVL3-8B-hf": "259a3b64a14623c0ec91a045cb43f7c5af5fa6af",
}
