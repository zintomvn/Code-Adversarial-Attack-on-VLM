from .common import *
from .config_schema import cfg

class V53DiffusionTarget:
    _pipeline = None
    @staticmethod
    def generate_with_fallback(generator, source, target, prompt, seed):
        try:
            image = generator(source, prompt, seed)
            if not isinstance(image, Image.Image):
                raise TypeError("Diffusion must return one PIL image")
            return image.convert("RGB"), {"effective_target": "diffusion", "error": None,
                                           "prompt": prompt, "seed": seed, "attempts": 1}
        except Exception as error:
            # User-approved fallback, never disguised as successful diffusion.
            return target.copy(), {"effective_target": "original_target_fallback",
                                   "error": f"{type(error).__name__}: {error}",
                                   "prompt": prompt, "seed": seed, "attempts": 1}

    @staticmethod
    def generate(source, prompt, seed):
        from diffusers import StableDiffusionImg2ImgPipeline
        pipeline = None
        try:
            pipeline = V53DiffusionTarget._pipeline
            if pipeline is None:
                pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                    cfg.V53_DIFFUSION_ID, revision=cfg.V53_REVISIONS[cfg.V53_DIFFUSION_ID],
                    torch_dtype=torch.float16).to(cfg.V53_DEVICE)
                if torch.cuda.get_device_properties(cfg.V53_DEVICE).total_memory >= 80 * 2**30:
                    V53DiffusionTarget._pipeline = pipeline
            pipeline.enable_attention_slicing()
            result = pipeline(
                prompt=prompt, image=source.resize((512, 512), Image.Resampling.BICUBIC),
                strength=cfg.V53_DIFFUSION_STRENGTH, guidance_scale=cfg.V53_DIFFUSION_GUIDANCE,
                num_inference_steps=cfg.V53_DIFFUSION_STEPS,
                generator=torch.Generator(device=cfg.V53_DEVICE).manual_seed(seed))
            if result.nsfw_content_detected and any(result.nsfw_content_detected):
                raise RuntimeError("Diffusion returned filtered output")
            return result.images[0]
        finally:
            del pipeline
            release_memory()
