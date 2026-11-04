"""Verify real Drive storage with synthetic bytes, using a fresh local directory/process."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import tempfile
from my_attack.drive_persistence import DriveStagePersistence, GoogleDriveTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prime', 'restore'])
    parser.add_argument('--folder-id', default='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI')
    parser.add_argument('--prefix', default='My-Attack/storage-verification-20261029')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    transport = GoogleDriveTransport(args.folder_id, args.prefix)
    sentinel = b'My-Attack external Drive checkpoint verification 2026-10-29\n' * 128
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        persistence = DriveStagePersistence(root, transport)
        if args.stage == 'prime':
            folder = root / '200/negatives_10'
            folder.mkdir(parents=True)
            (folder / 'checkpoint.pt').write_bytes(sentinel)
            result = persistence.save('synthetic-checkpoint', folder)
            print(json.dumps(result), flush=True)
        else:
            persistence.restore()
            actual = (root / '200/negatives_10/checkpoint.pt').read_bytes()
            if actual != sentinel:
                raise AssertionError('Fresh-directory Drive restoration mismatch')
            result = {'status': 'DRIVE_FRESH_DIRECTORY_RESTORE_PASS',
                      'folder_id': transport.folder_id,
                      'sha256': hashlib.sha256(actual).hexdigest(), 'bytes': len(actual),
                      'models_called': 0, 'attack_steps': 0, 'gpt_requests': 0}
            persistence.verified_put('verification_restore.json', persistence.encoded(result))
            print(json.dumps(result), flush=True)
        if args.report:
            args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
