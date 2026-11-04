"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
class CLIPG14:
    def __init__(self, device):
        from transformers import CLIPModel, AutoTokenizer
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(CLIP_G14_ID, revision=MODEL_REVISIONS[CLIP_G14_ID]).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_G14_ID, revision=MODEL_REVISIONS[CLIP_G14_ID])

    def encode_image(self, image):
        pixels = F.interpolate(image.to(self.device).float(), (224, 224), mode="bicubic", align_corners=False)
        pixels = (pixels / 255).clamp(0, 1)
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        return l2_normalize(self.model.get_image_features(pixel_values=(pixels - mean) / std))

    @torch.no_grad()
    def encode_text(self, captions):
        tokens = self.tokenizer(captions, padding=True, return_tensors="pt").to(self.device)
        if tokens.input_ids.shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Caption exceeds CLIP token length")
        return l2_normalize(self.model.get_text_features(**tokens))


class CLIPB16:
    def __init__(self, device):
        from transformers import CLIPModel, AutoTokenizer
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(CLIP_B16_ID, revision=MODEL_REVISIONS[CLIP_B16_ID]).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_B16_ID, revision=MODEL_REVISIONS[CLIP_B16_ID])

    def encode_image(self, image):
        pixels = F.interpolate(image.to(self.device).float(), (224, 224), mode="bicubic", align_corners=False)
        pixels = (pixels / 255).clamp(0, 1)
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        return l2_normalize(self.model.get_image_features(pixel_values=(pixels - mean) / std))

    @torch.no_grad()
    def encode_text(self, captions):
        tokens = self.tokenizer(captions, padding=True, return_tensors="pt").to(self.device)
        if tokens.input_ids.shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Caption exceeds CLIP token length")
        return l2_normalize(self.model.get_text_features(**tokens))


class CLIPB32:
    def __init__(self, device):
        from transformers import CLIPModel, AutoTokenizer
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(CLIP_B32_ID, revision=MODEL_REVISIONS[CLIP_B32_ID]).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_B32_ID, revision=MODEL_REVISIONS[CLIP_B32_ID])

    def encode_image(self, image):
        pixels = F.interpolate(image.to(self.device).float(), (224, 224), mode="bicubic", align_corners=False)
        pixels = (pixels / 255).clamp(0, 1)
        mean = pixels.new_tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
        std = pixels.new_tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
        return l2_normalize(self.model.get_image_features(pixel_values=(pixels - mean) / std))

    @torch.no_grad()
    def encode_text(self, captions):
        tokens = self.tokenizer(captions, padding=True, return_tensors="pt").to(self.device)
        if tokens.input_ids.shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Caption exceeds CLIP token length")
        return l2_normalize(self.model.get_text_features(**tokens))


from transformers import AutoModelForImageTextToText, AutoProcessor

class QwenVL:
    def __init__(self, model_id=QWEN_ID, attention=False):
        self.processor = AutoProcessor.from_pretrained(model_id, revision=MODEL_REVISIONS[model_id])
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Qwen requires CUDA GPUs")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        memory = {}
        for gpu in range(device_count):
            free_bytes, _ = torch.cuda.mem_get_info(gpu)
            free_gib = free_bytes / 2 ** 30
            limit = QWEN_MAX_MEMORY_GIB.get(gpu, int(free_gib))
            available = min(float(limit), free_gib - 1.0)
            if available > 1:
                memory[gpu] = f"{available:.1f}GiB"
        if not memory:
            raise RuntimeError("Insufficient CUDA memory to load Qwen")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=MODEL_REVISIONS[model_id],
            torch_dtype=dtype,
            attn_implementation="eager" if attention else "sdpa",
            device_map="auto",
            max_memory=memory,
        ).eval()
        self.model.requires_grad_(False)

    def prepare(self, messages, images=None):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=[text], images=images, return_tensors="pt")
        return batch.to(self.model.device)

    @torch.no_grad()
    def generate(self, text_or_messages, image=None, sample=False, max_new_tokens=80):
        if isinstance(text_or_messages, str):
            batch = self.processor(text=[text_or_messages], return_tensors="pt").to(self.model.device)
        else:
            batch = self.prepare(text_or_messages, images=[image] if image is not None else None)
        kwargs = {"do_sample": sample, "max_new_tokens": max_new_tokens}
        if sample:
            kwargs.update(temperature=CAPTION_TEMPERATURE, top_p=CAPTION_TOP_P)
        output = self.model.generate(**batch, **kwargs)
        generated = output[:, batch["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def caption(self, image, sample=True):
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": EVALUATION_PROMPT}],
        }]
        return self.generate(messages, image=image, sample=sample, max_new_tokens=50)

    def attention_inputs(self, image):
        from qwen_vl_utils import process_vision_info
        # Same Qwen vision-token preprocessing concept as SAGA, using a PIL image directly.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image, no longer than 25 words."},
            ],
        }]
        images, videos, _ = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.processor(text=text, images=images, videos=videos,
                               do_resize=False, return_tensors="pt")
        return batch.to(self.model.device)


class CLIPEnsemble:
    def __init__(self):
        self.device = torch.device(ATTACK_DEVICE)
        self.models = [
            CLIPG14(CLIP_DEVICES["G14"]),
            CLIPB16(CLIP_DEVICES["B16"]),
            CLIPB32(CLIP_DEVICES["B32"]),
        ]

    def encode_image(self, image):
        features = [model.encode_image(image).to(self.device) for model in self.models]
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def encode_text(self, captions):
        return torch.cat([model.encode_text(captions).to(self.device)
                          for model in self.models], dim=-1)

    @torch.no_grad()
    def anchors(self, image, captions):
        features = []
        for model in self.models:
            image_feature = model.encode_image(image).to(self.device)
            text_features = model.encode_text(captions).to(self.device)
            fused = IMAGE_WEIGHT * image_feature + TEXT_WEIGHT * text_features
            features.append(fused)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def build_targets(self, source, target, bank):
        target_bank = self.anchors(target, bank["target_captions"])
        negative_bank = self.anchors(source, bank["negative_captions"])
        distances = (target_bank - target_bank.mean(dim=0, keepdim=True)).square().sum(dim=-1)
        selected_index = int(distances.argmin())
        positive = target_bank[selected_index:selected_index + 1]
        if negative_bank.shape[0] != K:
            raise ValueError("The loss requires K negative anchors")
        return positive.detach(), negative_bank.detach(), selected_index
