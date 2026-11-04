import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Prepare pinned data/model identity, then optionally launch direct Python execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


class MoLabBootstrap:
    def __init__(self, args):
        self.args = args
        self.project = Path(__file__).resolve().parent

    def configure(self):
        import kagglehub
        from huggingface_hub import HfApi
        from my_attack.config_schema import cfg

        data = Path(kagglehub.dataset_download('zintom/blackbox-adversarial-attack-dataset/versions/1'))
        source = data / 'images/bigscale/nips17'
        target = data / 'images/target_images/1'
        ids = self.args.ids.split(',') if self.args.ids else [str(i) for i in range(200, 400)]
        if len(ids) > 202 or len(set(ids)) != len(ids):
            raise ValueError('At most202 distinct approved pairs')
        if self.args.identity_lock:
            lock = json.loads(self.args.identity_lock.read_text())
            locked = lock.get('identity.json', lock).get('config', lock.get('config', lock))
            revisions = locked['V53_REVISIONS']
        else:
            api = HfApi()
            revisions = {model: api.model_info(model).sha for model in cfg.V53_REVISIONS}
        hashes = {}
        for sample in ids:
            hashes[sample] = {}
            for role, path in [('source', source / f'{sample}.png'), ('target', target / f'{sample}.jpg')]:
                hashes[sample][role] = hashlib.sha256(path.read_bytes()).hexdigest()
                if self.args.identity_lock and hashes[sample][role] != locked['V53_PAIR_HASHES'][sample][role]:
                    raise RuntimeError('Dataset changed relative to the reviewed R1 identity')
        record = {
            'V53_RUN_ID': self.args.run_id, 'V53_ROOT': str(self.args.output),
            'V53_SOURCE_DIR': str(source), 'V53_TARGET_DIR': str(target),
            'V53_PAIR_IDS': ids, 'V53_PAIR_HASHES': hashes, 'V53_REVISIONS': revisions,
            'V53_API_MAX_CALLS': len(ids) * 12,
            'V53_APPROVAL': 'User2026-10-29 explicitly selected rerun200 with full verified Google Drive storage; prior manual review waiver retained',
            'V53_MAX_WALL_SECONDS': self.args.max_hours * 3600 if self.args.max_hours else None,
            'V53_MAX_GPU_HOURS': self.args.max_hours,
            'V53_MAX_API_USD': self.args.max_api_usd}
        path = self.project / 'configs' / f'{self.args.run_id}.json'
        path.parent.mkdir(exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != record:
            raise RuntimeError('Existing config changed; do not overwrite a run identity')
        path.write_text(json.dumps(record, indent=2), encoding='utf-8')
        # Source files survive MoLab reclamation; retain the exact configuration without secrets.
        retained = self.project / 'configs' / f'{self.args.run_id}_retained.py'
        retained.write_text('CONFIG_JSON = ' + repr(json.dumps(record, sort_keys=True)) + '\n', encoding='utf-8')
        print(json.dumps({'config': str(path), 'pairs': len(ids), 'model_revisions': revisions}), flush=True)
        return path

    def run(self):
        config = self.configure()
        if self.args.launch:
            command = [sys.executable, '-u', str(self.project / 'run_pipeline.py'), '--config', str(config),
                       '--drive-folder-id', self.args.drive_folder_id,
                       '--drive-prefix', self.args.drive_prefix]
            if self.args.restore:
                command.append('--restore')
            subprocess.run(command, cwd=self.project, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='MYATTACK-20261029-DRIVE-N200-R2')
    parser.add_argument('--ids', help='Explicit IDs; default confirmation200..399 (exactly200 pairs)')
    parser.add_argument('--output', type=Path, default=Path('/marimo/outputs/My-Attack'))
    parser.add_argument('--cache-root', type=Path, default=Path('/marimo/__marimo__/cache/My-Attack'))
    parser.add_argument('--drive-folder-id', default='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI')
    parser.add_argument('--drive-prefix', default='My-Attack')
    parser.add_argument('--identity-lock', type=Path)
    parser.add_argument('--max-hours', type=float)
    parser.add_argument('--max-api-usd', type=float)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--restore', action='store_true')
    MoLabBootstrap(parser.parse_args()).run()


if __name__ == '__main__':
    main()
