"""MPCAttack entry point; the marked additions provide counterfactual negatives."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from config_schema import MainConfig
from negatives import NegativeBank, NegativeCaptionGenerator
from utils import (atomic_write_json, ensure_dir, info_nce_loss, require_experiment_approval,
                   save_image, verify_perturbation, wait_for_review, sample_seed, execution_deadline)


CLIP_BACKBONE_MAP = {
    'L336': 'ClipL336FeatureExtractor',
    'B16': 'ClipB16FeatureExtractor',
    'B32': 'ClipB32FeatureExtractor',
    'Laion': 'ClipLaionFeatureExtractor',
}


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    image = torch.from_numpy(np.array(pic.convert('RGB'), dtype=np.uint8, copy=True))
    return image.permute(2, 0, 1).contiguous().float()


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, self.samples[index][0]


def match_source_target_paths(source_paths, target_paths):
    """Retain ImageFolder loading while pairing IDs rather than relying on sort order."""
    source_ids = [Path(path).stem for path in source_paths]
    target_ids = [Path(path).stem for path in target_paths]
    if len(set(source_ids)) != len(source_ids) or len(set(target_ids)) != len(target_ids):
        raise ValueError('Source/target image IDs must be unique across class folders')
    targets = dict(zip(target_ids, target_paths))
    missing = set(source_ids) - targets.keys()
    if missing:
        raise ValueError(f'Missing target images for source IDs: {sorted(missing)}')
    return [(source, targets[Path(source).stem]) for source in source_paths]

def build_clip_models(args):
    # Lazy import keeps --help and synthetic tests independent of pretrained loaders.
    import surrogates
    if not args.ensemble and len(args.clip_backbone) != 1:
        raise ValueError('When ensemble=False, exactly one backbone is required')
    models = []
    for name in args.clip_backbone:
        model = getattr(surrogates, CLIP_BACKBONE_MAP[name])().eval().to(args.device)
        model.requires_grad_(False)
        models.append(model)
    # MPC's dictionary interface is retained even for a one-model ensemble.
    return surrogates.EnsembleFeatureExtractor_our(models), models

def load_models(args):
    from surrogates import DINOv2FeatureExtractor, InternVL3_1B_FeatureExtractor
    clip_models, _ = build_clip_models(args)
    internvl_model = InternVL3_1B_FeatureExtractor().eval().to(args.device).requires_grad_(False)
    dino_model = DINOv2FeatureExtractor().eval().to(args.device).requires_grad_(False)
    return clip_models, internvl_model, dino_model


def extract_feature(img, clip_models, internvl_model, dino_model):
    clip_feats = torch.cat(list(clip_models(img).values()), dim=1)
    intern_feat = internvl_model(img).mean(dim=1).float()
    dino_feat = dino_model(img)
    return torch.cat([clip_feats, intern_feat, dino_feat], dim=1)


def negative_bank_identity(args, image_org, source_path, target_path, seed):
    from model_lock import MODEL_REVISIONS
    import negatives
    source_hash = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    target_hash = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
    return {'source_sha256': source_hash, 'target_sha256': target_hash, 'seed': seed,
            'source_pixels_sha256': hashlib.sha256(image_org.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            'input_res': args.input_res, 'preprocess': 'Resize bicubic / CenterCrop / RGB pixel0..255',
            'num_negatives': args.num_negatives, 'model_id': args.negative_model,
            'model_revision': MODEL_REVISIONS[args.negative_model],
            'generator_source_sha256': hashlib.sha256(Path(negatives.__file__).read_bytes()).hexdigest(),
            'caption_attempts': args.caption_attempts, 'negative_retries': args.negative_retries,
            'decoding': {'temperature': .8, 'top_p': .9, 'caption_max_tokens': 50,
                         'object_max_tokens': 80, 'num_beams': 1}}


def prepare_negative_bank(args, image_org, source_path, target_path, sample_dir, seed=None, deadline=None):
    """Added: the notebook's source-only caption/object-replacement bank."""
    seed = args.seed if seed is None else seed
    identity = negative_bank_identity(args, image_org, source_path, target_path, seed)
    bank_path = (Path(args.negative_bank_path) / f'{Path(source_path).stem}.json'
                 if args.negative_bank_path else sample_dir / 'negative_bank.json')
    if bank_path.exists():
        bank = json.loads(bank_path.read_text(encoding='utf-8'))
        if bank.get('identity') != identity:
            raise ValueError(f'Negative bank does not match this pair/configuration: {bank_path}')
    else:
        devices = [torch.device(args.device).index or 0] if str(args.device).startswith('cuda') else []
        # Bank generation must not consume the attack's initialization/crop RNG.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            generator = NegativeCaptionGenerator(args.negative_model, sample_dir / 'bank_generations',
                                                  max_calls=args.caption_attempts + args.num_negatives * (1 + args.negative_retries),
                                                  device=args.device, deadline=deadline)
            builder = NegativeBank(generator, seed=seed, k=args.num_negatives,
                                   caption_attempts=args.caption_attempts, negative_retries=args.negative_retries)
            try:
                pixels = image_org[0].permute(1, 2, 0).byte().cpu().numpy()
                from PIL import Image
                bank = builder.create(Image.fromarray(pixels))
                bank['identity'] = identity
                atomic_write_json(bank_path, bank)
            finally:
                atomic_write_json(sample_dir / 'bank_provenance.json', builder.provenance)
                generator.close()
    captions = bank.get('negative_captions', [])
    if len(captions) != args.num_negatives or len({' '.join(text.casefold().split()) for text in captions}) != len(captions):
        raise ValueError('The negative bank must contain the configured number of distinct captions')
    atomic_write_json(sample_dir / 'negative_bank.json', bank)
    return bank


def build_negative_features(args, source_clip, negative_text, source_internvl, source_dino):
    """Added: K source-image/negative-text CLIP anchors plus source InternVL/DINO."""
    clip_negatives = torch.cat([
        args.lam * source_clip[index] + (1 - args.lam) * negative_text[index]
        for index in source_clip
    ], dim=1)
    return torch.cat([clip_negatives, source_internvl.expand(args.num_negatives, -1),
                      source_dino.expand(args.num_negatives, -1)], dim=1).detach()


def postprocess_saved_images(args):
    """Chain the same two editable evaluation scripts after saved-PNG verification."""
    from blackbox_text_generation import generate_descriptions
    descriptions = args.des_path_src_tgt or str(Path(args.output) / 'descriptions_src_tgt')
    caption_args = argparse.Namespace(output=args.output, src_data_path=args.cle_data_path,
        tgt_data_path=args.tgt_data_path, des_path_src_tgt=descriptions, model_name=args.model_name,
        approved_plan=args.approved_plan, base_url=args.base_url, device=args.device)
    generate_descriptions(caption_args)
    if args.scorer_model:
        from gpt_evaluate import main as evaluate
        evaluate(argparse.Namespace(output=args.output, des_path_src_tgt=descriptions,
            model_name=args.model_name, scorer_model=args.scorer_model, success_threshold=args.success_threshold,
            approved_plan=args.approved_plan, base_url=args.base_url))


def run_attack(args):
    approval = require_experiment_approval(args.approved_plan)

    if args.batch_size != 1:
        raise ValueError('Use batch_size=1 so each image has its own negative bank')
    if not (0 <= args.lam <= 1) or args.steps < 1 or args.num_samples < 1 or args.num_negatives < 1:
        raise ValueError('Invalid fusion weight, sample count, steps or negative count')
    if (args.start_index < 0 or args.input_res < 1 or args.save_every < 1 or args.max_seconds < 1
            or args.caption_attempts < args.num_negatives or args.negative_retries < 1
            or not 0 < args.crop_scale[0] <= args.crop_scale[1] <= 1
            or any(not math.isfinite(value) or value <= 0
                   for value in (args.alpha, args.epsilon, args.tau, args.omega))):
        raise ValueError('Invalid index, checkpoint/budget, crop range or optimization parameter')
    
    deadline = time.time() + args.max_seconds
    ensure_dir(args.output)
    review_dir = Path(args.review_dir or Path(args.output) / 'reviews')
    ensure_dir(review_dir)

    # tranform function
    transform_fn = transforms.Compose([
        transforms.Resize(args.input_res, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.input_res),
        transforms.Lambda(lambda image: to_tensor(image)),
    ])

    source_crop = transforms.RandomResizedCrop(args.input_res, scale=args.crop_scale)
    source_dataset = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    target_dataset = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)

    pairs = match_source_target_paths([row[0] for row in source_dataset.samples],
                                      [row[0] for row in target_dataset.samples])
    pairs = pairs[args.start_index:args.start_index + args.num_samples]

    if len(pairs) != args.num_samples:
        raise ValueError('Dataset contains fewer pairs than the configured cohort')
    source_indices = {path: index for index, (path, _) in enumerate(source_dataset.samples)}
    target_indices = {path: index for index, (path, _) in enumerate(target_dataset.samples)}

    # Same model construction as MPC, deferred so caption-only resume loads no attack models.
    models = None
    for source_path, target_path in pairs:
        if time.time() >= deadline:
            raise TimeoutError('Reviewed execution time limit exhausted')
        sample_id = Path(source_path).stem
        sample_dir = Path(args.output) / 'checkpoints' / sample_id
        ensure_dir(sample_dir)
        configuration = {'args': vars(args), 'approval': approval, 'source': source_path, 'target': target_path,
                         'source_sha256': hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                         'target_sha256': hashlib.sha256(Path(target_path).read_bytes()).hexdigest()}
        complete_path = sample_dir / 'attack_complete.json'
        if complete_path.exists():
            previous_config = json.loads((sample_dir / 'config.json').read_text(encoding='utf-8'))
            if previous_config != json.loads(json.dumps(configuration, default=str)):
                raise ValueError('Saved attack config/data differ; do not reuse outputs under a changed protocol')
            complete = json.loads(complete_path.read_text(encoding='utf-8'))
            image_path = Path(complete['adversarial_path'])
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != complete['image_sha256']:
                raise ValueError('Saved adversarial PNG changed before post-processing resume')
            postprocess_saved_images(args)
            wait_for_review(review_dir, f'{sample_id}_complete', complete_path, deadline, complete=True)
            continue
        if (sample_dir / 'attack_started.json').exists():
            raise RuntimeError('This attack was already attempted; use saved outputs for post-processing')
        
        image_org, _, _ = source_dataset[source_indices[source_path]]
        image_tgt, _, _ = target_dataset[target_indices[target_path]]
        image_org, image_tgt = image_org.unsqueeze(0), image_tgt.unsqueeze(0)
        current_seed = sample_seed(args.seed, source_path)
        set_seed(current_seed)
        save_image(image_org, sample_dir / 'source.png')
        save_image(image_tgt, sample_dir / 'target.png')
        atomic_write_json(sample_dir / 'config.json', configuration)

        # Added: prepare and inspect the K-negative bank before optimization.
        bank = prepare_negative_bank(args, image_org, source_path, target_path, sample_dir,
                                     seed=current_seed, deadline=deadline)
        wait_for_review(review_dir, f'{sample_id}_data_bank', sample_dir / 'negative_bank.json', deadline)
        if models is None:
            models = load_models(args)
        
        clip_models, internvl_model, dino_model = models
        src_text = internvl_model.generate_description(image_org)
        tar_text = internvl_model.generate_description(image_tgt)
        
        atomic_write_json(sample_dir / 'mpc_descriptions.json', {'source': src_text, 'target': tar_text})
        img_src, img_tgt = image_org.to(args.device), image_tgt.to(args.device)
        delta = (16 / 255 * torch.randn_like(image_org)).to(args.device).requires_grad_(True)
        with torch.no_grad():
            text_feature_src = clip_models.get_text_features(src_text)
            text_feature_tar = clip_models.get_text_features(tar_text)
            src_feats, tgt_feats = clip_models(img_src), clip_models(img_tgt)
            internvl_src = internvl_model(img_src).mean(dim=1).float()
            internvl_tgt = internvl_model(img_tgt).mean(dim=1).float()
            dino_src, dino_tgt = dino_model(img_src), dino_model(img_tgt)
            clip_src = torch.cat([args.lam * src_feats[index] + (1 - args.lam) * text_feature_src[index]
                                  for index in src_feats], dim=1)
            clip_tgt = torch.cat([args.lam * tgt_feats[index] + (1 - args.lam) * text_feature_tar[index]
                                  for index in tgt_feats], dim=1)
            z_src = torch.cat([clip_src, internvl_src, dino_src], dim=1)
            z_tgt = torch.cat([clip_tgt, internvl_tgt, dino_tgt], dim=1)
            # Added: ten negative text features replace the single-source denominator.
            negative_text = clip_models.get_text_features(bank['negative_captions'])
            z_negatives = build_negative_features(args, src_feats, negative_text, internvl_src, dino_src)
        torch.save({'target': z_tgt.cpu(), 'negatives': z_negatives.cpu()}, sample_dir / 'anchors.pt')
        atomic_write_json(sample_dir / 'attack_started.json', {'attempt': 1})
        momentum = torch.zeros_like(delta)
        history = []
        for epoch in tqdm(range(args.steps), desc=f'Attack {sample_id}'):
            if time.time() >= deadline:
                raise TimeoutError('Reviewed execution time limit exhausted')
            adv_img = img_src + delta
            # Record the actual RandomResizedCrop parameters without sampling twice.
            top, left, height, width = source_crop.get_params(adv_img, source_crop.scale, source_crop.ratio)
            local_cropped = transforms.functional.resized_crop(
                adv_img, top, left, height, width, [args.input_res, args.input_res],
                interpolation=source_crop.interpolation, antialias=source_crop.antialias)
            z_adv = extract_feature(local_cropped, clip_models, internvl_model, dino_model)
            loss = info_nce_loss(z_adv, z_tgt, z_negatives, temperature=args.tau, omega=args.omega)
            grad, = torch.autograd.grad(loss, delta)
            if not torch.isfinite(loss) or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                raise FloatingPointError('Nonfinite objective/gradient or disconnected adversarial image')
            # Original MPC MI-FGSM update; valid-pixel projection is shared safety logic.
            with torch.no_grad():
                momentum = momentum * 0.9 + grad
                delta = (delta + args.alpha * momentum.sign()).clamp(-args.epsilon, args.epsilon)
                delta = torch.maximum(torch.minimum(delta, 255 - img_src), -img_src)
            delta = delta.detach().requires_grad_(True)
            step = epoch + 1
            record = {'step': step, 'negative_ce': float(loss.detach()), 'cross_entropy': float(-loss.detach()),
                      'source_similarity': float(F.cosine_similarity(z_adv.detach(), z_src).mean()),
                      'target_similarity': float(F.cosine_similarity(z_adv.detach(), z_tgt).mean()),
                      'negative_count': len(z_negatives), 'crop': [top, left, height, width],
                      'query_shape': list(z_adv.shape), 'negative_shape': list(z_negatives.shape),
                      'gradient_l1': float(grad.abs().sum()), 'tau': args.tau, 'omega': args.omega,
                      'direction': 'ascent_negative_cross_entropy', 'reduction': 'mean_cross_entropy'}
            history.append(record)
            if step == 1 or step % args.save_every == 0 or step == args.steps:
                checkpoint = sample_dir / f'step_{step:03d}.png'
                save_image(img_src + delta, checkpoint)
                atomic_write_json(checkpoint.with_suffix('.json'), record)
                torch.save({'delta': delta.detach().cpu(), 'momentum': momentum.cpu(), 'step': step,
                            'cpu_rng': torch.get_rng_state(),
                            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                           checkpoint.with_suffix('.pt'))
            if step == 1:
                save_image(local_cropped, sample_dir / 'actual_crop_step001.png')
                wait_for_review(review_dir, f'{sample_id}_update1',
                                sample_dir / 'step_001.json', deadline)
        # Save/verify the PNG before any victim caption or GPT judgment.
        adv_path = Path(args.output) / 'img' / Path(source_path).parent.name / f'{sample_id}.png'
        save_image(img_src + delta, adv_path)
        verification = verify_perturbation(adv_path, image_org, args.epsilon)
        atomic_write_json(sample_dir / 'history.json', history)
        atomic_write_json(sample_dir / 'attack_complete.json', {**verification, 'official_asr': None,
                          'caption_status': 'PENDING', 'adversarial_path': str(adv_path)})
        postprocess_saved_images(args)
        atomic_write_json(sample_dir / 'caption_complete.json', {'model_name': args.model_name,
                          'judge': args.scorer_model, 'official_asr': None,
                          'diagnostic_status': 'MANUAL_REVIEW_PENDING' if not args.scorer_model else 'ADAPTED_GPT_RUBRIC'})
        # Inspect all seven required items before moving to the next source-target pair.
        wait_for_review(review_dir, f'{sample_id}_complete', sample_dir / 'attack_complete.json', deadline, complete=True)


def get_args(argv=None):
    cfg = MainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    # Data — original MPC argument names.
    parser.add_argument('--cle_data_path', default=cfg.data.cle_data_path)
    parser.add_argument('--tgt_data_path', default=cfg.data.tgt_data_path)
    parser.add_argument('--output', default=cfg.data.output)
    parser.add_argument('--batch_size', type=int, default=cfg.data.batch_size)
    parser.add_argument('--num_samples', type=int, default=cfg.data.num_samples)
    parser.add_argument('--start_index', type=int, default=0)
    # Model.
    parser.add_argument('--input_res', type=int, default=cfg.model.input_res)
    parser.add_argument('--crop_scale', nargs=2, type=float, default=cfg.model.crop_scale)
    def boolean(value):
        if value.lower() not in ('true', 'false'):
            raise argparse.ArgumentTypeError('Expected True or False')
        return value.lower() == 'true'
    parser.add_argument('--ensemble', nargs='?', const=True, type=boolean, default=cfg.model.ensemble)
    parser.add_argument('--no-ensemble', action='store_false', dest='ensemble')
    parser.add_argument('--device', default=cfg.model.device)
    parser.add_argument('--clip_backbone', nargs='+', choices=CLIP_BACKBONE_MAP, default=cfg.model.backbone)
    # Optim / MPCAttack parameters.
    parser.add_argument('--alpha', type=float, default=cfg.optim.alpha)
    parser.add_argument('--epsilon', type=float, default=cfg.optim.epsilon)
    parser.add_argument('--steps', type=int, default=cfg.optim.steps)
    parser.add_argument('--lam', type=float, default=cfg.lam)
    parser.add_argument('--tau', type=float, default=cfg.tau)
    parser.add_argument('--omega', type=float, default=cfg.omega)
    parser.add_argument('--seed', type=int, default=cfg.seed)
    # Added: negative samples from the supplied notebook.
    parser.add_argument('--num_negatives', type=int, default=cfg.negative.num_negatives)
    parser.add_argument('--negative_model', default=cfg.negative.model_id)
    parser.add_argument('--negative_bank_path', default=cfg.negative.bank_path)
    parser.add_argument('--caption_attempts', type=int, default=cfg.negative.caption_attempts)
    parser.add_argument('--negative_retries', type=int, default=cfg.negative.negative_retries)
    # Same standalone caption/judge scripts are also chained for approved runs.
    parser.add_argument('--model_name', default=cfg.blackbox_open_source.model_name)
    parser.add_argument('--des_path_src_tgt')
    parser.add_argument('--scorer_model', default=None, help='Optional GPT judge; requires an approved API budget')
    parser.add_argument('--success_threshold', type=float, default=.5)
    parser.add_argument('--base_url')
    # Experiment audit; these do not modify MPC optimization.
    parser.add_argument('--approved_plan')
    parser.add_argument('--review_dir')
    parser.add_argument('--max_seconds', type=int, default=4500)
    parser.add_argument('--save_every', type=int, default=30)
    return parser.parse_args(argv)


def main():
    args = get_args()
    set_seed(args.seed)
    with execution_deadline(args.max_seconds):
        run_attack(args)


if __name__ == '__main__':
    main()
