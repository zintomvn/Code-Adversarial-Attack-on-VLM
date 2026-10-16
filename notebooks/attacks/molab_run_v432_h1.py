"""Run the original v4.3.2-h1 notebook sequentially on one Molab GPU.

Only configuration is overridden. Execute this launcher in a separate Python
process, not by importing it into a reactive marimo notebook.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['check', 'setup', 'preflight', 'smoke', 'attack'], required=True)
    parser.add_argument('--notebook', type=Path, default=Path(__file__).with_name('kltn-attack-v4.3.2-h1.ipynb'))
    parser.add_argument('--dataset-root', type=Path, default=Path('data'))
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--target-dir', type=Path)
    parser.add_argument('--pair-manifest', type=Path)
    parser.add_argument('--work-root', type=Path, default=Path('work'))
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--sample-start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--variants', nargs='+', default=['hybrid_dynamic'])
    parser.add_argument('--resume-run-dir', type=Path)
    parser.add_argument('--cpu-saved-tensors', action='store_true', help='Offload autograd saved tensors to host RAM.')
    parser.add_argument('--offload-clip', action='store_true', help='Move CLIP to host RAM during Qwen phases.')
    args = parser.parse_args()
    notebook = args.notebook.resolve()
    payload = json.loads(notebook.read_text(encoding='utf-8'))
    cells = [(i, ''.join(c['source'])) for i, c in enumerate(payload['cells']) if c['cell_type'] == 'code']
    if not cells or not cells[0][1].startswith('# Cell 1 — User configuration.'):
        raise ValueError('Expected the original v4.3.2-h1 configuration cell.')
    # Validate every cell before installing packages or executing anything.
    compiled = [(i, source, compile(source, f'{notebook.name}:cell_{i}', 'exec')) for i, source in cells]
    work = args.work_root.resolve()
    overrides = dict(
        DATASET_ROOT=args.dataset_root.resolve(), WORK_ROOT=work,
        OUTPUT_ROOT=work / 'kltn_attack_v4_3_2_h1_molab_outputs',
        SOURCE_DIR=args.source_dir.resolve() if args.source_dir else None,
        TARGET_DIR=args.target_dir.resolve() if args.target_dir else None,
        PAIR_MANIFEST=args.pair_manifest.resolve() if args.pair_manifest else None,
        ATTACK_DEVICE='cuda:0', ATTENTION_DEVICE='cuda:0', CAPTION_DEVICE='cuda:0',
        RUN_ATTACK=args.mode == 'attack', INSTALL_DEPENDENCIES=False, SETUP_ONLY=False,
        RUN_SELF_CHECKS=True, GPU_SMOKE=args.mode == 'smoke',
        NUM_SAMPLES=args.num_samples, SAMPLE_START=args.sample_start, STEPS=args.steps,
        RUN_VARIANTS=args.variants,
        RESUME_RUN_DIR=args.resume_run_dir.resolve() if args.resume_run_dir else None,
        SAVED_TENSORS_ON_CPU=args.cpu_saved_tensors,
        OFFLOAD_CLIP_DURING_QWEN=args.offload_clip,
        RUN_PREVIEW=False, EXPORT_PARTIAL=False,
    )
    namespace = {'__name__': '__main__', '__file__': str(notebook)}
    exec(compiled[0][2], namespace)
    namespace.update(overrides)
    namespace['validate_configuration']()
    if args.mode == 'check':
        print(json.dumps({'notebook': str(notebook), 'sha256': hashlib.sha256(notebook.read_bytes()).hexdigest(),
                          'code_cells_compiled': len(compiled), 'configuration': overrides}, default=str, indent=2))
        return
    if args.mode == 'setup':
        # Read the notebook's pins without importing ML libraries or loading weights.
        tree = ast.parse(compiled[1][1])
        pins = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'PINNED_PACKAGES' for t in node.targets))
        subprocess.run([sys.executable, '-m', 'pip', 'install', *pins, 'numpy', 'Pillow'], check=True)
        print('Setup complete. Run preflight in a NEW Python process.')
        return
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(work / 'hf_cache'))
    for position, (index, source, code) in enumerate(compiled[1:], start=1):
        print(f'Executing notebook cell {index}', flush=True)
        exec(code, namespace)
        if position == 1:
            # In plain Python the notebook otherwise falls back to build-time hashes.
            # Record current executed source hashes so code edits invalidate resume.
            labels = namespace['BUILD_CELL_FIRST_LINES']
            namespace['BUILD_CELL_HASHES'] = {
                labels[s.strip().splitlines()[0]]: hashlib.sha256(s.strip().encode()).hexdigest()
                for _, s, _ in compiled if s.strip().splitlines()[0] in labels
            }
            if args.mode in ('preflight', 'smoke', 'attack'):
                torch = namespace['torch']
                print(f'Python={sys.version.split()[0]} Torch={torch.__version__} CUDA={torch.version.cuda}', flush=True)
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable. Attach the Molab GPU first.')
                print(f'GPU={torch.cuda.get_device_name(0)}; count={torch.cuda.device_count()}; '
                      f'VRAM={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB', flush=True)
                probe = torch.randn(64, 64, device='cuda:0', requires_grad=True)
                probe.square().mean().backward()
                torch.cuda.synchronize()
                del probe
    if args.mode != 'attack':
        # The original preflight prints dataset errors without raising them.
        # Require valid input pairs and device placement in this launcher.
        namespace['verify_devices_and_runtime']()
        pairs = namespace['prepare_pairs']()
        print(f'PREFLIGHT PASSED: {len(pairs)} image pair(s). No pretrained weights downloaded.')
    else:
        run_dir = Path(namespace['RUN_DIR'])
        status = json.loads((run_dir / 'status.json').read_text(encoding='utf-8'))
        print(f'RUN_DIR={run_dir}', flush=True)
        if status['status'] != 'complete':
            raise RuntimeError(f'Attack has incomplete variants. Inspect {run_dir / "summary.json"} and errors.jsonl.')
        print(f'ATTACK COMPLETE. Download {run_dir}.zip before shutdown.', flush=True)


if __name__ == '__main__':
    main()
