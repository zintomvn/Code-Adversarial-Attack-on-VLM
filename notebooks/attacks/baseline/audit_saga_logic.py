"""Read-only provenance and hotspot-schedule audit; does not run/change attacks."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import types
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
UPSTREAM = REPO / 'tmp/baseline-upstream/SAGA'


def audit():
    lock = json.loads((HERE / 'SOURCE_LOCK.json').read_text())['saga']
    remote = subprocess.check_output(
        ['git', '-C', str(UPSTREAM), 'ls-remote', 'origin', 'HEAD'], text=True
    ).split()[0]
    files = []
    with zipfile.ZipFile(HERE / 'baseline_code.zip') as bundle:
        for name, expected in lock['files'].items():
            original = subprocess.check_output(
                ['git', '-C', str(UPSTREAM), 'show', lock['commit'] + ':' + name]
            )
            packaged = bundle.read('author/saga/' + name)
            local = (REPO / 'src/SAGA' / name).read_bytes()
            normal = lambda data: data.replace(b'\r\n', b'\n')
            row = dict(file=name, locked_sha256=expected,
                git_matches_lock=hashlib.sha256(original).hexdigest() == expected,
                bundle_matches_git=packaged == original,
                local_matches_git_normalized=normal(local) == normal(original))
            if name.endswith('.py'):
                row['local_ast_matches_git'] = (
                    ast.dump(ast.parse(local.decode('utf8')), include_attributes=False)
                    == ast.dump(ast.parse(original.decode('utf8')), include_attributes=False)
                )
            files.append(row)
        hotspots = types.ModuleType('official_saga_hotspots')
        exec(compile(bundle.read('author/saga/saga/hotspots.py'),
                     '<frozen official saga/hotspots.py>', 'exec'), hotspots.__dict__)

    cases = []
    for size in (7, 14, 24):
        for kind in ('uniform', 'random_seed_43'):
            attention = np.ones((size, size)) if kind == 'uniform' else np.random.default_rng(43).random((size, size))
            schedule = hotspots.build_schedule(attention, 10, 3, 0.3)
            stage_counts = dict(sorted(Counter(r['phase_group'] for r in schedule).items()))
            overlaps = []
            for stage in range(1, 11):
                boxes = hotspots.get_top_k_boxes_fixed_area(attention, stage / 10, k=3, iou_threshold=0.3)
                values = [float(hotspots.compute_box_iou(a, b)) for i, a in enumerate(boxes) for b in boxes[i+1:]]
                if any(v > 0.3 for v in values):
                    overlaps.append(dict(stage=stage, max_pair_iou=max(values)))
            block = 300 // len(schedule)
            allocation = Counter(min(step // block, len(schedule) - 1) for step in range(300))
            cases.append(dict(grid=[size, size], map=kind, schedule_items=len(schedule),
                per_stage_counts=stage_counts, steps_per_item=block,
                last_item_steps=allocation[len(schedule)-1],
                iou_cap_violating_stages=overlaps))

    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
        upstream_repository=lock['repository'], frozen_commit=lock['commit'],
        remote_head=remote, remote_head_matches_frozen=remote == lock['commit'],
        file_count=len(files), files=files, schedule_examples=cases,
        caveat='Synthetic maps demonstrate release behavior; actual run schedules may vary by image. No GPU rerun or remote runtime rehash in this audit.')
    assert all(r['git_matches_lock'] and r['bundle_matches_git'] for r in files)
    assert all(r.get('local_ast_matches_git', True) for r in files)
    (HERE / 'saga_logic_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    audit()
