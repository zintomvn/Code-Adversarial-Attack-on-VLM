"""Explicit commands; importing the package never launches experiments."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['remote', 'paid'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--env', type=Path)
    parser.add_argument('--sample', type=int, choices=[0, 1])
    args = parser.parse_args()
    if args.command == 'remote':
        from .pipeline import run_remote
        run_remote(args.root, args.package)
    else:
        from .evaluation import paid_evaluation
        from .state import atomic_json
        import json
        import zipfile
        folder = args.root / f'{args.sample:05d}'
        result = paid_evaluation(folder, args.root / 'api', args.env)
        from .state import fingerprint
        result.update(run_identity=fingerprint(json.loads((args.root / 'identity.json').read_text())), sample=args.sample)
        atomic_json(folder / 'evaluation.json', result)
        with zipfile.ZipFile(args.root / f'paid_{args.sample:05d}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(folder / 'evaluation.json', 'evaluation.json')


if __name__ == '__main__':
    main()
