"""Run source, data, checkpoint and original-FGSM resume checks on CPU.

python verify_baselines.py --data /path/to/attack_data_n200.zip
GPU execution and API results must be checked separately in deployment_status.json.
"""
import argparse
import ast
import io
import json
import random
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

from baseline_runtime import extract, prepare_data, read_json, rebuild_manifests, sha, verify_sources, write_json

HERE = Path(__file__).resolve().parent


def verify(data_archive=None):
    checks = {}
    with tempfile.TemporaryDirectory(prefix='verification_', dir=HERE) as temporary:
        root = Path(temporary).resolve()
        assert root.is_relative_to(HERE.resolve())
        extract(HERE / 'baseline_code.zip', root)
        lock = verify_sources(root)
        checks['author_files_verified'] = sum(len(info['files']) for info in lock.values())
        namespace = {}
        notebook = read_json(HERE / 'official_baselines_kaggle.ipynb')
        exec(''.join(notebook['cells'][1]['source']), namespace)
        cfg = namespace['CONFIG']
        if data_archive:
            cfg['data_archive'] = str(Path(data_archive).resolve())
            rows = prepare_data(root, cfg, None)
            checks['pairs_validated'] = len(rows)
            checks['unique_sources'] = len({row['stem'] for row in rows})
        for path in HERE.glob('*.ipynb'):
            for cell in read_json(path)['cells']:
                if cell['cell_type'] == 'code':
                    ast.parse(''.join(cell['source']))
        checks['notebook_cells_parse'] = True
        for name, expected in read_json(HERE / 'evaluation_source.json').items():
            if name == 'protocol_sha256':
                assert sha(HERE / 'eval_protocol_v32.py') == expected
        checks['evaluation_protocol_hash_matches'] = True
        write_json(root / 'selected_pairs.json', [{'index': 0}])
        sample = root / 'results/foa/samples/00000'
        sample.mkdir(parents=True)
        for name in ('source.png', 'target.png', 'adversarial.png', 'rng.json'):
            (sample / name).write_bytes(b'integrity fixture')
        write_json(sample / 'complete.json', dict(index=0, run_id='test',
            artifacts_sha256={p.name: sha(p) for p in sample.iterdir()}))
        rebuild_manifests(root, {'methods': ['foa']}, 'test')
        assert (root / 'results/foa/manifest.jsonl').is_file()
        checks['restored_manifest_reconstructed'] = True
        (sample / 'adversarial.png').write_bytes(b'corrupt')
        try:
            rebuild_manifests(root, {'methods': ['foa']}, 'test')
        except ValueError:
            checks['corrupt_checkpoint_rejected'] = True
        else:
            raise AssertionError('Corrupt checkpoint accepted')
        malicious = io.BytesIO()
        with zipfile.ZipFile(malicious, 'w') as z:
            z.writestr('../escape.txt', b'bad')
        malicious.seek(0)
        try:
            extract(malicious, root / 'zip_test')
        except ValueError:
            checks['zip_traversal_rejected'] = True
        else:
            raise AssertionError('ZIP traversal accepted')
        checks.update(verify_author_fgsm_rng(root))
    return checks


def verify_author_fgsm_rng(root):
    """Use the author's actual FGSM function with a small differentiable test surrogate."""
    import numpy as np
    import torch
    from torchvision import transforms
    from author_worker import restore_rng, rng_state

    source = (root / 'author/foa/generate_adversarial_samples_foa_attack.py').read_text(encoding='utf-8')
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'fgsm_attack')
    program = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    namespace = dict(torch=torch, log_metrics=lambda *a, **kw: None, tqdm=lambda steps, **kw: steps)
    exec(compile(ast.fix_missing_locations(program), '<original-author-fgsm>', 'exec'), namespace)

    class Extractor:
        def __call__(self, tensor):
            return tensor, tensor.mean(dim=(-1, -2), keepdim=True)

    class Objective:
        def set_ground_truth(self, target):
            self.target = target

        def __call__(self, global_feature, local_feature):
            return -((global_feature - self.target) ** 2).mean() + local_feature.mean() * 0.1

    cfg = SimpleNamespace(optim=SimpleNamespace(steps=7, alpha=1.0, epsilon=16),
        model=SimpleNamespace(use_source_crop=True))
    clean = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32) % 255
    target = torch.flip(clean, [-1])
    crop = transforms.RandomResizedCrop(32, scale=(0.5, 0.9))

    def attack():
        return namespace['fgsm_attack'](cfg, Extractor(), Objective(), crop, crop, 0, clean, target)

    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    attack()
    boundary = json.loads(json.dumps(rng_state()))
    uninterrupted = attack()
    random.random()
    np.random.random(100)
    torch.rand(1000)
    restore_rng(boundary)
    resumed = attack()
    assert torch.equal(uninterrupted, resumed), 'Resume changed the original FGSM result'
    assert (resumed * 255 - clean).abs().max() <= 16.001
    return dict(original_author_fgsm_resume_exact_cpu=True, rng_json_roundtrip=True, perturbation_budget=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.data), indent=2))
