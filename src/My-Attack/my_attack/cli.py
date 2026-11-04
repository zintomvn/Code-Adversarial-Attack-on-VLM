"""Lightweight CLI; --help does not import torch or load any model."""
import argparse
from pathlib import Path


def main(default_stage='all'):
    parser = argparse.ArgumentParser(description='My-Attack V53: attack, captions and GPT-score')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['all', 'attack', 'caption', 'evaluate'], default=default_stage)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--drive-folder-id')
    parser.add_argument('--drive-prefix', default='My-Attack')
    parser.add_argument('--allow-local-only', action='store_true', help='Explicit development-only override')
    parser.add_argument('--restore', action='store_true', help='Restore retained stages; cache miss never runs models')
    parser.add_argument('--drive-sync-every-samples', type=int, default=1)
    parser.add_argument('--resume-local', action='store_true',
                        help='Explicitly resume existing local checkpoints without replacing them from Drive')
    args = parser.parse_args()
    from .config_schema import load_config
    from .pipeline import MyAttackPipeline
    cfg = load_config(args.config)
    transport = None
    if args.drive_folder_id:
        from .drive_persistence import GoogleDriveTransport
        transport = GoogleDriveTransport(args.drive_folder_id, args.drive_prefix.rstrip('/') + '/' + cfg.V53_RUN_ID)
        print('Drive checkpoint folder ID:', transport.folder_id, flush=True)
    elif not args.allow_local_only:
        parser.error('--drive-folder-id is required for experiments; no silent local-only checkpoints')
    pipeline = MyAttackPipeline(args.cache_root, transport)
    if transport:
        pipeline.persistence.sync_every_samples = args.drive_sync_every_samples
        if args.drive_sync_every_samples < 1:
            parser.error('--drive-sync-every-samples must be positive')
    if args.resume_local:
        if not (pipeline.root / 'identity.json').exists():
            parser.error('--resume-local requires an existing identity.json')
        pipeline.persistence.restore = lambda: ['existing local checkpoints preserved; no remote overwrite']
    pipeline.run(args.stage, restore=args.restore or args.resume_local)
