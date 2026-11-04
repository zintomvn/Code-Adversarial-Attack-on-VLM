from ..common import *
from ..config_schema import cfg

class CLIPG14:
    def __init__(self, device):
        from transformers import CLIPModel, AutoTokenizer
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(cfg.CLIP_G14_ID).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.CLIP_G14_ID)

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
        self.model = CLIPModel.from_pretrained(cfg.CLIP_B16_ID).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.CLIP_B16_ID)

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
        self.model = CLIPModel.from_pretrained(cfg.CLIP_B32_ID).eval().to(self.device)
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.CLIP_B32_ID)

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


class CLIPEnsemble:
    def __init__(self):
        self.device = torch.device(cfg.ATTACK_DEVICE)
        self.models = [
            CLIPG14(cfg.CLIP_DEVICES["G14"]),
            CLIPB16(cfg.CLIP_DEVICES["B16"]),
            CLIPB32(cfg.CLIP_DEVICES["B32"]),
        ]

    def encode_image(self, image):
        features = [model.encode_image(image).to(self.device) for model in self.models]
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def encode_text(self, captions):
        return torch.cat([model.encode_text(captions).to(self.device) for model in self.models], dim=-1)

    @torch.no_grad()
    def anchors(self, image, captions):
        features = []
        for model in self.models:
            image_feature = model.encode_image(image).to(self.device)
            text_features = model.encode_text(captions).to(self.device)
            fused = cfg.IMAGE_WEIGHT * image_feature + cfg.TEXT_WEIGHT * text_features
            features.append(fused)
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def build_targets(self, source, target, bank):
        target_bank = self.anchors(target, bank["target_captions"])
        negative_bank = self.anchors(source, bank["negative_captions"])
        distances = (target_bank - target_bank.mean(dim=0, keepdim=True)).square().sum(dim=-1)
        selected_index = int(distances.argmin())
        positive = target_bank[selected_index:selected_index + 1]
        if negative_bank.shape[0] != cfg.K:
            raise ValueError("The loss requires K negative anchors")
        return positive.detach(), negative_bank.detach(), selected_index


class V53CLIPEnsemble(CLIPEnsemble):
    _instance = None

    def __new__(cls):
        if torch.cuda.get_device_properties(cfg.V53_DEVICE).total_memory >= 80 * 2**30:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
        return super().__new__(cls)

    def __init__(self):
        if getattr(self, '_ready', False):
            return
        from transformers import CLIPModel, AutoTokenizer
        self.device = torch.device(cfg.V53_DEVICE)
        self.models = []
        for adapter, model_id in zip([CLIPG14, CLIPB16, CLIPB32], cfg.V53_CLIP_IDS):
            instance = adapter.__new__(adapter)
            instance.device = self.device
            instance.model = CLIPModel.from_pretrained(
                model_id, revision=cfg.V53_REVISIONS[model_id]).eval().to(self.device)
            instance.model.requires_grad_(False)
            instance.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=cfg.V53_REVISIONS[model_id])
            self.models.append(instance)
        self._ready = True

    @torch.no_grad()
    def text_blocks(self, captions):
        return [model.encode_text(captions).float().cpu() for model in self.models]

    @torch.no_grad()
    def fused(self, image, captions):
        return torch.cat([
            cfg.V53_IMAGE_WEIGHT*model.encode_image(image).to(self.device)
            + cfg.V53_TEXT_WEIGHT*model.encode_text(captions).to(self.device)
            for model in self.models], dim=-1)
