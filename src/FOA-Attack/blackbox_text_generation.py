import os
import requests
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra # lib to manage configuration files
import torch
import torchvision


from omegaconf import OmegaConf
from tqdm import tqdm # lib to show progress bar
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from google import genai
import openai
from openai import OpenAI
import anthropic

# Utilities functions -> những hàm phụ trợ
from utils import (
    get_api_key,
    hash_training_config,
    setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
)

# Define valid image extensions
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


def setup_gemini(api_key: str):
    return genai.Client(api_key=api_key)


def setup_claude(api_key: str):
    return anthropic.Anthropic(api_key=api_key)


def setup_gpt4o(api_key: str):
    return OpenAI(
        api_key="api_key",
    )


def get_media_type(image_path: str) -> str:
    """Get the correct media type based on file extension."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


# Image description generator class
class ImageDescriptionGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Get API key for the model
        api_key = get_api_key(model_name)

        if model_name == "gemini":
            self.client = setup_gemini(api_key)
        elif model_name == "claude":
            self.client = setup_claude(api_key)
        elif model_name == "gpt4o":
            self.client = setup_gpt4o(api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str) -> str:
        if self.model_name == "gemini":
            return self._generate_gemini(image_path)
        elif self.model_name == "claude":
            return self._generate_claude(image_path)
        elif self.model_name == "gpt4o":
            return self._generate_gpt4o(image_path)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gemini(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # Initialize wandb using shared utility
    setup_wandb(cfg)
    print(cfg)
    # Get config hash and setup paths
    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    # Get output paths using shared utility
    paths = get_output_paths(cfg, config_hash)
    ensure_dir(paths["desc_output_dir"])

    try:
        # Initialize description generator
        generator = ImageDescriptionGenerator(model_name=cfg.blackbox.model_name)

        # Process original and adversarial images
        tgt_descriptions = []
        adv_descriptions = []

        # Walk through the output directory for adversarial images
        print("Processing images...")
        for root, _, files in os.walk(paths["output_dir"]):
            for file in tqdm(files):
                # Check if file has valid image extension
                if any(
                    file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                ):
                    try:
                        # Get paths
                        adv_path = os.path.join(root, file)
                        # Extract just the filename without extension
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        # Try each valid extension for target image
                        target_found = False
                        for ext in VALID_IMAGE_EXTENSIONS:
                            tgt_path = os.path.join(
                                cfg.data.tgt_data_path, "1", filename_base + ext
                            )
                            if os.path.exists(tgt_path):
                                target_found = True
                                break

                        if target_found:
                            # Generate descriptions
                            tgt_desc = generator.generate_description(tgt_path)
                            adv_desc = generator.generate_description(adv_path)

                            tgt_descriptions.append((file, tgt_desc))
                            adv_descriptions.append((file, adv_desc))

                            # Log to wandb
                            wandb.log(
                                {
                                    f"descriptions/{file}/target": tgt_desc,
                                    f"descriptions/{file}/adversarial": adv_desc,
                                }
                            )

                        else:
                            print(
                                f"Target image not found for {filename_base} with any valid extension, skip it."
                            )

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # Save descriptions
        save_descriptions(
            tgt_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"target_{cfg.blackbox.model_name}.txt"
            ),
        )
        save_descriptions(
            adv_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"adversarial_{cfg.blackbox.model_name}.txt"
            ),
        )

        print(f"Descriptions saved to {paths['desc_output_dir']}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    finally:
        wandb.finish()


from utils import ExperimentRunner, KaggleRuntime


class VersionedImageDescriptionGenerator(ImageDescriptionGenerator):
    """Common caption prompt; modern adapters without changing legacy generators."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.caption.model_id
        self.metadata = {}
        if cfg.caption.provider == 'openai':
            self.client = OpenAI(api_key=KaggleRuntime.secret('OPENAI_API_KEY'), timeout=cfg.caption.timeout, max_retries=3)
        elif cfg.caption.provider == 'gemini':
            from google.genai import types
            self.client = genai.Client(api_key=KaggleRuntime.secret('GEMINI_API_KEY'),
                                       http_options=types.HttpOptions(timeout=int(cfg.caption.timeout * 1000)))
        elif cfg.caption.provider == 'huggingface':
            self._load_local()
        else:
            raise ValueError(f'Unknown provider: {cfg.caption.provider}')

    def _load_local(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('Local VLM inference requires a Kaggle GPU.')
        torch.manual_seed(self.cfg.experiment.seed)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        options = dict(device_map={'': 0}, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager',
                       revision=self.cfg.caption.revision, trust_remote_code=False)
        if self.cfg.caption.quantization == 'nf4':
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        elif self.cfg.caption.quantization != 'none':
            raise ValueError('Use quantization=nf4 or none.')
        self.processor = AutoProcessor.from_pretrained(self.model_name, revision=self.cfg.caption.revision, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **options).eval()
        self.metadata = {'resolved_revision': getattr(self.model.config, '_commit_hash', None),
                         'dtype': str(dtype), 'quantization': self.cfg.caption.quantization,
                         'gpu': torch.cuda.get_device_name(0)}

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3), reraise=True)
    def generate_description(self, image_path):
        cfg = self.cfg.caption
        self.last_response = {}
        if cfg.provider == 'openai':
            response = self.client.responses.create(
                model=self.model_name, store=False, reasoning={'effort': cfg.reasoning_effort},
                max_output_tokens=cfg.max_output_tokens,
                input=[{'role': 'user', 'content': [
                    {'type': 'input_text', 'text': cfg.prompt},
                    {'type': 'input_image', 'image_url': f'data:image/png;base64,{encode_image(image_path)}', 'detail': 'high'},
                ]}])
            if response.status != 'completed':
                raise ValueError(f'Incomplete OpenAI response: {response.status}')
            text = response.output_text
            self.last_response = {'response_id': response.id, 'model': response.model,
                                  'usage': response.usage.model_dump(mode='json') if response.usage else None}
        elif cfg.provider == 'gemini':
            from google.genai import types
            with open(image_path, 'rb') as stream:
                part = types.Part.from_bytes(data=stream.read(), mime_type='image/png')
            response = self.client.models.generate_content(model=self.model_name, contents=[cfg.prompt, part],
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=cfg.max_output_tokens,
                                                   thinking_config=types.ThinkingConfig(thinking_budget=0)))
            if not response.candidates or 'STOP' not in str(response.candidates[0].finish_reason):
                raise ValueError('Gemini output blocked or truncated.')
            text = response.text
            self.last_response = {'model': getattr(response, 'model_version', self.model_name),
                                  'usage': response.usage_metadata.model_dump(mode='json') if response.usage_metadata else None}
        else:
            messages = [{'role': 'user', 'content': [{'type': 'image', 'url': os.path.abspath(image_path)},
                                                    {'type': 'text', 'text': cfg.prompt}]}]
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                         return_dict=True, return_tensors='pt')
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            with torch.inference_mode():
                tokens = self.model.generate(**inputs, do_sample=False, max_new_tokens=cfg.max_new_tokens)
            generated = tokens[0, inputs['input_ids'].shape[1]:]
            if len(generated) >= cfg.max_new_tokens:
                raise ValueError('Local caption reached max_new_tokens; raise the limit in a new config version.')
            text = self.processor.decode(generated, skip_special_tokens=True)
        if not text or not text.strip():
            raise ValueError('Empty caption; refusal/failure is not a valid description.')
        return text.strip()


class CaptionExperimentRunner(ExperimentRunner):
    def run(self):
        import shutil
        input_root, samples = self.read_input('manifest.jsonl')
        generator = VersionedImageDescriptionGenerator(self.cfg)
        self.write_json('model_metadata.json', {'model_id': generator.model_name, **generator.metadata})
        cache, failures = {}, 0
        for sample in tqdm(samples, desc=generator.model_name):
            for role in ('source', 'target', 'adversarial'):
                path = (input_root / sample['images'][role]).resolve()
                if not path.is_relative_to(input_root.resolve()) or not path.is_file():
                    raise ValueError(f'Unsafe or missing image: {path}')
                if self.sha256(path) != sample['image_sha256'][role]:
                    raise ValueError(f'Image checksum mismatch: {path}')
            row = dict(sample)
            row.update(caption_experiment=self.cfg.experiment.name, caption_model=generator.model_name,
                       caption_config_sha256=self.fingerprint, captions={}, caption_responses={}, errors={})
            for role in ('source', 'target', 'adversarial'):
                checksum = sample['image_sha256'][role]
                try:
                    if checksum not in cache:
                        caption = generator.generate_description(str(input_root / sample['images'][role]))
                        cache[checksum] = (caption, generator.last_response)
                    row['captions'][role], row['caption_responses'][role] = cache[checksum]
                except Exception as error:
                    # Keep a failure distinct from a zero score, without exposing API keys/messages.
                    row['errors'][role] = type(error).__name__
                    failures += 1
            row['status'] = 'ok' if not row['errors'] else 'error'
            self.append('captions.jsonl', row)
        shutil.copy2(input_root / 'manifest.jsonl', self.root / 'manifest.jsonl')
        self.write_json('status.json', {'status': 'complete' if not failures else 'partial', 'failed_captions': failures, 'samples': len(samples)})


if __name__ == "__main__":
    main()
