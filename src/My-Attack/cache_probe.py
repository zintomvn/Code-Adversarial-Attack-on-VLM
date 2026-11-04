import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Synthetic native-cache process-boundary check; no models/data/API calls."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
from my_attack.persistence import StagePersistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prime', 'verify'])
    parser.add_argument('--root', type=Path, default=Path('/marimo/MyAttack-cache-probe'))
    parser.add_argument('--cache', type=Path, default=Path('/marimo/__marimo__/cache/MyAttack-probe'))
    args = parser.parse_args()
    persistence = StagePersistence(args.root, args.cache)
    artifact = args.root / 'stage' / 'sentinel.bin'
    sentinel = b'My-Attack native cache sentinel 2026-10-28' * 16
    if args.stage == 'prime':
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(sentinel)
        print(persistence.save('synthetic', artifact.parent), flush=True)
        # Remove only the exact synthetic scratch ZIP, forcing the child to hit native cache.
        for entry in persistence.entries.values():
            Path(entry['scratch']).unlink()
        artifact.unlink()
        subprocess.run([sys.executable, __file__, 'verify', '--root', str(args.root),
                        '--cache', str(args.cache)], check=True)
    else:
        persistence.restore()
        assert artifact.read_bytes() == sentinel
        print('NATIVE_CACHE_NEW_PROCESS_PASS', hashlib.sha256(sentinel).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
