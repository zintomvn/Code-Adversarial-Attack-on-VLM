"""MPC-shaped caption script. Providers are loaded only for approved generation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

PROMPT = 'Describe this image in one concise sentence, no longer than 20 words.'
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
CLOSED_MODELS = ('gpt4o', 'gpt5', 'gemini', 'claude', 'claude4_5')
PROVIDER_MODELS = {'gpt4o': 'gpt-4o', 'gpt5': 'gpt-5', 'gemini': 'gemini-2.0-flash',
                   'claude': 'claude-3-5-sonnet-20241022', 'claude4_5': 'claude-sonnet-4-5-20250929'}


def setup_gpt4o(api_key, base_url=None):
    from openai import OpenAI
    options = {'api_key': api_key, 'max_retries': 0}
    endpoint = base_url or os.environ.get('OPENAI_BASE_URL')
    if endpoint:
        options['base_url'] = endpoint
    return OpenAI(**options)


def setup_gpt5(api_key, base_url=None):
    return setup_gpt4o(api_key, base_url)


def setup_gemini(api_key, base_url=None):
    from google import genai
    options = {'api_key': api_key}
    endpoint = base_url or os.environ.get('GEMINI_BASE_URL')
    if endpoint:
        options['http_options'] = {'base_url': endpoint}
    return genai.Client(**options)


def setup_claude(api_key, base_url=None):
    import anthropic
    options = {'api_key': api_key, 'max_retries': 0}
    endpoint = base_url or os.environ.get('ANTHROPIC_BASE_URL')
    if endpoint:
        options['base_url'] = endpoint
    return anthropic.Anthropic(**options)


def get_media_type(image_path):
    ext = Path(image_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        return 'image/jpeg'
    if ext == '.png':
        return 'image/png'
    raise ValueError(f'Unsupported image extension: {ext}')


class ImageDescriptionGenerator:
    """Readable provider dispatch; construction never loads weights or API clients."""
    def __init__(self, model_name, base_url=None, approved_plan=None, device='auto'):
        self.model_name = model_name
        self.base_url = base_url
        self.approved_plan = approved_plan
        self.device = device
        self.client = None
        self.model = None
        self.processor = None
        self.direct_hf = False
        self.last_response = None

    def _load_local_model(self):
        # Keep MPC's registry for its aliases. The default small Qwen uses a
        # direct loader because the kit does not forward revision pins and
        # forces Flash Attention. This also works with kits lacking its alias.
        if self.model_name != 'Qwen2.5-VL-3B-Instruct':
            from vlmeval.config import supported_VLM
            if self.model_name not in supported_VLM:
                raise ValueError(f'Model {self.model_name} not supported by local vlmeval.')
            self.model = supported_VLM[self.model_name]()
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        model_id = 'Qwen/Qwen2.5-VL-3B-Instruct'
        revisions = json.loads(Path(__file__).with_name('model_revisions.json').read_text(encoding='utf-8'))
        revision = revisions[model_id]
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype='auto', device_map=self.device)
        self.model.eval()
        self.direct_hf = True

    def generate_description(self, image_path):
        from utils import require_experiment_approval, get_api_key
        require_experiment_approval(self.approved_plan)
        if self.model_name in CLOSED_MODELS:
            if self.client is None:
                setup = {'gpt4o': setup_gpt4o, 'gpt5': setup_gpt5, 'gemini': setup_gemini,
                         'claude': setup_claude, 'claude4_5': setup_claude}[self.model_name]
                self.client = setup(get_api_key(self.model_name), self.base_url)
            return getattr(self, '_generate_' + self.model_name)(image_path)
        if self.model is None:
            self._load_local_model()
        if self.direct_hf:
            import torch
            from PIL import Image
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image}, {'type': 'text', 'text': PROMPT}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[image], padding=True, return_tensors='pt')
                inputs = inputs.to(self.model.device)
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=64)
                tokens = outputs[:, inputs['input_ids'].shape[1]:]
                caption = self.processor.batch_decode(tokens, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=False)[0]
                self.last_response = {'text': caption, 'generated_token_ids': tokens.detach().cpu().tolist(),
                                      'token_limit_reached': tokens.shape[-1] >= 64}
        else:
            caption = self.model.generate([str(image_path), PROMPT])
            if self.model_name == 'glm-4.1v-9b-thinking':
                match = re.search(r'<answer>(.*?)</answer>', caption, re.DOTALL)
                if match:
                    caption = match.group(1).strip()
        if not self.direct_hf:
            self.last_response = caption
        return caption

    def _generate_gpt4o(self, image_path):
        from utils import encode_image
        response = self.client.chat.completions.create(
            model=PROVIDER_MODELS[self.model_name], messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT}, {'type': 'image_url', 'image_url': {
                    'url': f'data:{get_media_type(image_path)};base64,{encode_image(image_path)}'}}]}],
            max_tokens=100, temperature=0)
        self.last_response = response.model_dump(mode='json')
        return response.choices[0].message.content

    def _generate_gpt5(self, image_path):
        from utils import encode_image
        response = self.client.responses.create(model='gpt-5', input=[{'role': 'user', 'content': [
            {'type': 'input_text', 'text': PROMPT}, {'type': 'input_image',
             'image_url': f'data:{get_media_type(image_path)};base64,{encode_image(image_path)}'}]}])
        self.last_response = response.model_dump(mode='json')
        return response.output_text

    def _generate_gemini(self, image_path):
        from google.genai import types
        response = self.client.models.generate_content(model=PROVIDER_MODELS['gemini'],
            contents=[PROMPT, types.Part.from_bytes(data=Path(image_path).read_bytes(),
                                                   mime_type=get_media_type(image_path))],
            config={'temperature': 0, 'max_output_tokens': 100})
        self.last_response = response.model_dump(mode='json')
        return response.text

    def _generate_claude(self, image_path):
        from utils import encode_image
        response = self.client.messages.create(model=PROVIDER_MODELS[self.model_name],
            max_tokens=300, temperature=0, messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': PROMPT}, {'type': 'image', 'source': {
                    'type': 'base64', 'media_type': get_media_type(image_path),
                    'data': encode_image(image_path)}}]}])
        self.last_response = response.model_dump(mode='json')
        return response.content[0].text

    def _generate_claude4_5(self, image_path):
        return self._generate_claude(image_path)


def save_descriptions(descriptions, output_file):
    """MPC's filename: caption text format, atomically persisted by shared utils."""
    from utils import save_descriptions as write_descriptions
    write_descriptions(descriptions, output_file)


def _description_rows(path):
    from utils import read_descriptions
    rows = read_descriptions(path)
    stems = [Path(name).stem for name, _ in rows]
    if len(set(stems)) != len(stems):
        raise ValueError(f'Duplicate description IDs in {path}')
    return rows


def _adversarial_images(output):
    """Saved PNGs define the denominator, with exactly one path per stem."""
    root = Path(output) / 'img'
    images = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if path.suffix.lower() != '.png':
            raise ValueError(f'Expected saved adversarial PNG: {path}')
        if path.stem in images:
            raise ValueError(f'Duplicate adversarial ID: {path.stem}')
        images[path.stem] = path
    if not images:
        raise ValueError(f'No saved adversarial PNGs in {root}')
    return images


def generate_descriptions(args, generator=None):
    from utils import find_image, atomic_write_json, require_experiment_approval
    injected = generator is not None
    # Gate before any credential access, registry import or weight loading.
    approval = None if injected else require_experiment_approval(getattr(args, 'approved_plan', None))
    images = _adversarial_images(args.output)
    pairs = [(adv, find_image(args.tgt_data_path, stem), find_image(args.src_data_path, stem))
             for stem, adv in images.items()]
    clean = Path(args.des_path_src_tgt)
    paths = [clean / f'target_{args.model_name}.txt', clean / f'source_{args.model_name}.txt',
             Path(args.output) / 'description' / f'adversarial_{args.model_name}.txt']
    rows = [_description_rows(path) if path.exists() else [] for path in paths]
    cached = [{Path(name).stem: caption for name, caption in group} for group in rows]
    protocol = {'model_name': args.model_name, 'prompt': PROMPT, 'synthetic': injected,
                'approval': approval, 'base_url': getattr(args, 'base_url', None) or os.environ.get(
                    'GEMINI_BASE_URL' if args.model_name == 'gemini' else
                    'ANTHROPIC_BASE_URL' if args.model_name.startswith('claude') else 'OPENAI_BASE_URL'),
                'loader': ('pinned_direct_hf' if args.model_name == 'Qwen2.5-VL-3B-Instruct'
                           else 'provider_api' if args.model_name in CLOSED_MODELS else 'local_vlmeval_registry'),
                'device': getattr(args, 'device', 'auto'),
                'decoding': ({'do_sample': False, 'max_new_tokens': 64}
                             if args.model_name == 'Qwen2.5-VL-3B-Instruct' else {'provider_or_registry_defaults': True}),
                'model_revisions': json.loads(Path(__file__).with_name('model_revisions.json').read_text(encoding='utf-8'))}
    if generator is None:
        generator = ImageDescriptionGenerator(args.model_name, getattr(args, 'base_url', None),
                                               args.approved_plan, getattr(args, 'device', 'auto'))
    print('Using model:', args.model_name)
    print('Processing images...')
    for adv, target, source in pairs:
        for index, image in enumerate((target, source, adv)):
            metadata_path = paths[index].with_suffix('.cache.json')
            metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            previous = metadata.get(adv.stem)
            if adv.stem in cached[index] and not previous:
                raise ValueError(f'Caption text has no verified image/protocol provenance: {paths[index]} / {adv.stem}')
            if previous and (previous['image_sha256'] != image_hash or previous['protocol'] != protocol):
                raise ValueError(f'Stale caption cache for {image}; use a separately reviewed output path')
            if previous:
                if previous.get('state') == 'STARTED':
                    raise RuntimeError(f'Unresolved prior caption request for {image}; no automatic retry')
                if adv.stem in cached[index] and cached[index][adv.stem] != previous['caption']:
                    raise ValueError(f'Caption text conflicts with JSON checkpoint: {image}')
                if adv.stem not in cached[index]:
                    rows[index].append((adv.name, previous['caption']))
                    save_descriptions(rows[index], paths[index])
                    cached[index][adv.stem] = previous['caption']
            if adv.stem in cached[index]:
                continue
            if not injected:
                metadata[adv.stem] = {'state': 'STARTED', 'image_sha256': image_hash,
                                      'protocol': protocol, 'image_path': str(image.resolve())}
                atomic_write_json(metadata_path, metadata)
            caption = generator.generate_description(str(image))
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f'Empty or invalid caption for {image}')
            caption = ' '.join(caption.split())
            metadata[adv.stem] = {'state': 'COMPLETE', 'image_path': str(image.resolve()), 'image_sha256': image_hash,
                                  'protocol': protocol, 'caption': caption,
                                  'raw_response': getattr(generator, 'last_response', caption)}
            # JSON first: a failed text export can be recovered without provider calls.
            atomic_write_json(metadata_path, metadata)
            rows[index].append((adv.name, caption))
            save_descriptions(rows[index], paths[index])
            cached[index][adv.stem] = caption
            print(('TGT:', 'SRC:', 'ADV:')[index], caption)
    print('\nDescriptions finished.')
    return {role: str(path) for role, path in zip(('target', 'source', 'adversarial'), paths)}


def get_args(argv=None):
    from config_schema import MainConfig
    cfg = MainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=cfg.data.output)
    parser.add_argument('--src_data_path', default=cfg.data.cle_data_path)
    parser.add_argument('--tgt_data_path', default=cfg.data.tgt_data_path)
    parser.add_argument('--des_path_src_tgt', default='descriptions_src_tgt')
    parser.add_argument('--model_name', default=cfg.blackbox_open_source.model_name,
                        help='Local supported_VLM alias or gpt4o, gpt5, gemini, claude, claude4_5')
    parser.add_argument('--approved_plan', help='Reviewed plan required for real generation')
    parser.add_argument('--base_url', default=None, help='Optional provider endpoint; otherwise provider environment/default')
    parser.add_argument('--device', default=cfg.model.device, help='HF device map for the pinned Qwen fallback')
    return parser.parse_args(argv)


def main(args=None):
    return generate_descriptions(get_args() if args is None else args)


if __name__ == '__main__':
    main()
