"""Run verified author entry points with I/O-only checkpoint hooks.

No attack, loss, gradient, crop, attention, precision or optimizer is reimplemented.
The dataset hook removes a completed prefix and restores the exact post-sample RNG
after DataLoader iterator initialization, before the next sample is transformed.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

from baseline_runtime import Drive, read_json, sha, verify_sources, write_json


def rng_state():
    import random
    import numpy as np
    import torch
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
        torch=torch.get_rng_state().tolist(), cuda=[s.tolist() for s in torch.cuda.get_rng_state_all()])


def restore_rng(state):
    import random
    import numpy as np
    import torch
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.setstate(tuples(state['python']))
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(torch.tensor(state['torch'], dtype=torch.uint8))
    torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in state['cuda']])


def pin_model_downloads(revisions):
    """Select exact HF commits at the cache resolver; author model code stays intact."""
    import transformers.utils.hub as hub
    original = hub.cached_files
    def cached_files(path_or_repo_id, *args, **kwargs):
        revision = revisions.get(str(path_or_repo_id))
        if revision:
            kwargs['revision'] = revision
        return original(path_or_repo_id, *args, **kwargs)
    hub.cached_files = cached_files


class Checkpoints:
    def __init__(self, root, method):
        self.root, self.method = Path(root), method
        self.cfg = read_json(self.root / 'config.json')
        self.run_id = read_json(self.root / 'identity.json')['fingerprint']
        self.pairs = read_json(self.root / 'selected_pairs.json')
        self.by_stem = {p['stem']: p for p in self.pairs}
        self.samples = self.root / 'results' / method / 'samples'
        self.samples.mkdir(parents=True, exist_ok=True)
        self.drive = Drive(self.cfg['drive_parent_id']) if self.cfg['backup_drive'] else None
        self.remote = self.cfg['drive_prefix'].strip('/') + '/' + self.cfg['run_name']
        self.completed = {}
        self.environment = self.env()
        for marker in self.samples.glob('*/complete.json'):
            record = read_json(marker)
            if record['run_id'] != self.run_id or record['environment'] != self.environment:
                raise ValueError('Checkpoint identity/environment mismatch')
            for name, digest in record['artifacts_sha256'].items():
                if sha(marker.parent / name) != digest:
                    raise ValueError('Corrupt checkpoint artifact')
            self.completed[record['stem']] = record
        self.resume_state = None
        order = sorted(self.by_stem)
        pending_seen = False
        for stem in order:
            if stem in self.completed:
                if pending_seen:
                    raise ValueError('Completed samples must be a contiguous prefix in author order')
                self.resume_state = read_json(self.samples / f'{self.by_stem[stem]["index"]:05d}' / 'rng.json')
            else:
                pending_seen = True
        # Re-upload local commits after a transient Drive failure before advancing RNG.
        for archive in sorted((self.root / 'checkpoints' / method).glob('*.zip')):
            if self.drive:
                self.drive.put(archive, self.remote + '/checkpoints/' + method + '/' + archive.name)

    @staticmethod
    def env():
        from importlib.metadata import version
        import platform
        import torch
        report = dict(python=platform.python_version(), gpu=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)), cuda=torch.version.cuda,
            packages={p: version(p) for p in ('torch', 'torchvision', 'transformers', 'numpy', 'pillow')})
        if 'kmeans_pytorch' in sys.modules:
            report['kmeans_source_sha256'] = sha(sys.modules['kmeans_pytorch'].__file__)
        return report

    def before_sample(self):
        if time.time() >= float(os.environ.get('BASELINE_DEADLINE', 'inf')):
            raise SystemExit(75)
        if self.resume_state is not None:
            restore_rng(self.resume_state)
            self.resume_state = None
            print('RNG_RESTORED_AT_SAMPLE_BOUNDARY', flush=True)

    def commit(self, stem, source, target, adversarial, author_files=()):
        from PIL import Image
        import numpy as np
        saved_rng = rng_state()
        record = self.by_stem[stem]
        sample = self.samples / f'{record["index"]:05d}'
        sample.mkdir(exist_ok=True)
        for name, value in [('source.png', source), ('target.png', target), ('adversarial.png', adversarial)]:
            if isinstance(value, (str, Path)):
                shutil.copyfile(value, sample / name)
            else:
                value.save(sample / name)
        with Image.open(sample / 'source.png') as a, Image.open(sample / 'adversarial.png') as b:
            linf = int(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max())
        if linf > int(np.ceil(self.cfg['epsilon'])):
            raise ValueError(f'PNG perturbation exceeds epsilon: {linf}')
        write_json(sample / 'rng.json', saved_rng)
        done = dict(index=record['index'], stem=stem, run_id=self.run_id,
            environment=self.environment, saved_png_linf_0_255=linf,
            source_input_sha256=record['source_sha256'], target_input_sha256=record['target_sha256'],
            saga_target_text=record['saga_target'] if self.method == 'saga' else None,
            artifacts_sha256={p.name: sha(p) for p in sample.iterdir() if p.name in
                ('source.png', 'target.png', 'adversarial.png', 'rng.json')})
        write_json(sample / 'complete.json', done)
        checkpoint = self.root / 'checkpoints' / self.method / (sample.name + '.zip')
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix('.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in [*sample.iterdir(), *map(Path, author_files)]:
                z.write(p, p.relative_to(self.root).as_posix())
        temporary.replace(checkpoint)
        if self.drive:
            self.drive.put(checkpoint, self.remote + '/checkpoints/' + self.method + '/' + checkpoint.name)
        self.completed[stem] = done
        self.manifest()
        print('COMMITTED', self.method, sample.name, len(self.completed), '/', len(self.pairs), 'linf', linf, flush=True)
        # Network retry jitter must not perturb the author's Python/NumPy/Torch RNG stream.
        restore_rng(saved_rng)

    def manifest(self):
        manifest = self.samples.parent / 'manifest.jsonl'
        rows = []
        for stem, record in sorted(self.completed.items(), key=lambda x: x[1]['index']):
            base = f'samples/{record["index"]:05d}/'
            rows.append(dict(index=record['index'], method=self.method,
                source=base + 'source.png', target=base + 'target.png', adversarial=base + 'adversarial.png'))
        temp = manifest.with_suffix('.tmp')
        temp.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        temp.replace(manifest)
        if self.drive:
            self.drive.put(manifest, self.remote + '/results/' + self.method + '/manifest.jsonl')


def run_image_attack(root, method, state):
    import torch
    import torchvision
    from torchvision import transforms
    from PIL import Image
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    module_name = ('generate_adversarial_samples_foa_attack' if method == 'foa'
        else 'generate_adversarial_examples_MPCAttack')
    module = importlib.import_module(module_name)
    original_dataset = module.ImageFolderWithPaths
    original_save = torchvision.utils.save_image
    current = {}

    class Dataset(original_dataset):
        def __init__(self, directory, *args, **kwargs):
            super().__init__(directory, *args, **kwargs)
            self.role = 'source' if Path(directory).resolve() == (root / 'inputs/source').resolve() else 'target'
            expected = set(state.by_stem)
            if {Path(p).stem for p, _ in self.samples} != expected:
                raise ValueError('Dataset differs from frozen cohort')
            self.samples = [(p, y) for p, y in self.samples if Path(p).stem not in state.completed]
            self.imgs = self.samples
            self.targets = [y for _, y in self.samples]

        def __getitem__(self, index):
            if self.role == 'source':
                state.before_sample()
            tensor, label, path = super().__getitem__(index)
            current[self.role] = (Path(path).stem, tensor)
            return tensor, label, path

    def save_image(tensor, filename, *args, **kwargs):
        path = Path(filename)
        tmp = path.with_name(path.stem + '.atomic.png')
        original_save(tensor, tmp, *args, **kwargs)
        tmp.replace(path)
        source_stem, source_tensor = current['source']
        target_stem, target_tensor = current['target']
        if source_stem != target_stem:
            raise ValueError('Author loader source/target pairing differs')
        # Use the exact author serializer for canonical evaluation images.
        from io import BytesIO
        images = []
        for value in (source_tensor, target_tensor):
            buffer = BytesIO()
            original_save((value / 255).clamp(0, 1), buffer, format='PNG')
            with Image.open(BytesIO(buffer.getvalue())) as im:
                images.append(im.copy())
        state.commit(source_stem, images[0], images[1], path)

    module.ImageFolderWithPaths = Dataset
    torchvision.utils.save_image = save_image
    cfg = state.cfg
    if method == 'foa':
        config = OmegaConf.load(root / 'author/foa/config/ensemble_3models.yaml')
        config.data.cle_data_path = str(root / 'inputs/source')
        config.data.tgt_data_path = str(root / 'inputs/target')
        config.data.output = str(root / 'author_output/foa')
        config.data.num_samples = cfg['n']
        config.data.batch_size = 1
        config.optim.steps, config.optim.epsilon, config.optim.alpha = cfg['steps'], cfg['epsilon'], cfg['alpha']
        config.wandb.entity = 'offline'
        module.main.__wrapped__(config)
    else:
        args = SimpleNamespace(cle_data_path=str(root / 'inputs/source'),
            tgt_data_path=str(root / 'inputs/target'), output=str(root / 'author_output/mpc'),
            batch_size=1, input_res=224, crop_scale=[0.5, 0.9], ensemble=True,
            device='cuda:0', clip_backbone=['B16', 'B32', 'Laion'], alpha=cfg['alpha'],
            epsilon=cfg['epsilon'], steps=cfg['steps'], **cfg['mpc'])
        module.set_seed(2023)
        module.run_attack(args)


def run_saga(root, state):
    import saga.pipeline as pipeline
    from saga.cli import parse_args
    from saga.data import load_pairs
    from PIL import Image
    from torchvision import transforms
    cfg, args = state.cfg, state.cfg['saga']
    cli = parse_args(['--image_dir', str(root / 'inputs/source/images'),
        '--target_text_jsonl', str(root / 'author/saga/input/coco.jsonl'),
        '--num_images', str(cfg['n']), '--save_path', str(root / 'author_output'),
        '--note', 'saga', '--resume', '--attack_steps', str(cfg['steps']),
        '--attack_epsilon', str(cfg['epsilon']), '--attack_alpha', str(cfg['alpha']),
        '--seed', str(args['seed']), '--progressive_num_phases', str(args['phases']),
        '--hotspot_top_k', str(args['top_k']), '--layer_idx', str(args['layer'])])
    pairs = load_pairs(cli.image_dir, cli.target_text_jsonl, None, cli.num_images, cli.seed)
    original_attention, original_write = pipeline.extract_attention, pipeline.write_json
    def attention(*a, **kw):
        state.before_sample()
        return original_attention(*a, **kw)
    def write(path, value):
        original_write(path, value)
        if value.get('completed'):
            sample = Path(path).parent
            stem = sample.name
            target_file = root / 'data' / state.by_stem[stem]['target']
            with Image.open(target_file) as image:
                target = transforms.CenterCrop(224)(transforms.Resize(224,
                    interpolation=transforms.InterpolationMode.BICUBIC)(image.convert('RGB')))
            state.commit(stem, sample / 'original.png', target, sample / 'adversarial.png',
                author_files=[sample / name for name in ('original.png', 'adversarial.png', 'metadata.json')])
    pipeline.extract_attention, pipeline.write_json = attention, write
    pipeline.run(cli, pairs)


def main(root, method):
    root = Path(root).resolve()
    verify_sources(root)
    os.chdir(root / 'author' / method)
    sys.path.insert(0, str(root / 'author' / method))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU first')
    if method in ('foa', 'mpc'):
        import inspect
        import kmeans_pytorch
        if 'iter_limit' not in inspect.signature(kmeans_pytorch.kmeans).parameters:
            raise RuntimeError('Install the pinned official kmeans dependency; PyPI 0.3 lacks iter_limit.')
    state = Checkpoints(root, method)
    print('HARDWARE', json.dumps(state.environment), flush=True)
    pin_model_downloads(state.cfg['model_revisions'])
    if method == 'saga':
        run_saga(root, state)
    else:
        run_image_attack(root, method, state)
    if len(state.completed) != state.cfg['n']:
        raise RuntimeError('Author entry point returned before all samples completed')
    state.manifest()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
