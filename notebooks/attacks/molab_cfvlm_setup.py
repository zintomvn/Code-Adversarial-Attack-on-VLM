"""Restore factual captions/data and start an isolated CF-VLM-policy campaign."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

PARENT = 'KLTN_Molab_Hypotheses_n200_v432_h1'
ATTACK_PREFIX = PARENT + '/cfvlm_v2/results'
EVAL_PREFIX = PARENT + '/cfvlm_v2/evaluation'
DATA_SHA = '3518b43ac953db8192a31698105cee4f277cc543351bf28e1eec6d7452987103'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_frozen(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Frozen file changed: {path}; use a new campaign namespace')
    path.write_bytes(data)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_frozen(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError(f'Existing code differs: {destination}')
    shutil.copy2(source, destination)


def initialize(package, root, eval_root):
    from gdrive_fsspec import GoogleDriveFileSystem
    fs = GoogleDriveFileSystem(token='cache', skip_instance_cache=True)
    root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    checksums = read(package / 'checksums.json')
    for name, expected in checksums.items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Package checksum mismatch: {name}')
    for name in ('molab_hypothesis_campaign.py', 'molab_remaining_tasks.py', 'molab_overnight.py',
                 'kltn-attack-v4.3.3-h1-cfvlm.ipynb', 'text_gen_eval_v3.1.ipynb'):
        copy_frozen(package / name, root / name)
    for name in ('molab_eval_completed.py', 'text_gen_eval_v3.1.ipynb'):
        copy_frozen(package / name, eval_root / name)
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    seed_file = provenance / 'legacy_primary_root.zip'
    if not seed_file.exists():
        fs.get_file(PARENT + '/results/primary_ablations/root_latest.zip', str(seed_file))
    with ZipFile(seed_file) as archive:
        if archive.testzip():
            raise ValueError('Legacy seed archive CRC failure')
        texts = json.loads(archive.read('text_records.json'))
        previous_config = json.loads(archive.read('run_config.json'))
        for variant in ('foa_reference', 'mpc_clip'):
            status = json.loads(archive.read(variant + '/status.json'))
            manifest = [json.loads(line) for line in archive.read(variant + '/manifest.jsonl').decode().splitlines() if line.strip()]
            if status.get('completed') != 200 or status.get('status') != 'complete' or [r['index'] for r in manifest] != list(range(200)):
                raise ValueError('Both legacy baseline cohorts must be complete')
    if len(texts) != 200 or any(not row.get('source') or not row.get('target') or row.get('caption_error')
                                for row in texts.values()):
        raise ValueError('Require all 200 factual source/target captions from the old campaign')
    factual = {sid: {key: row[key] for key in ('source', 'target')} for sid, row in texts.items()}
    # Never import previous negatives or their 13-image hybrid attack prefix.
    write_frozen(provenance / 'factual_captions.json', factual)
    plan = read(package / 'original_plan.json')
    if any(previous_config['model_revisions'].get(k) != v for k, v in plan['attack_model_revisions'].items()):
        raise ValueError('Legacy model revisions do not match the prespecified campaign')
    plan['experiment'] = 'kltn_v433_h1_cfvlm_audited_v2_n200_seed2023'
    plan['attack_notebook'] = 'kltn-attack-v4.3.3-h1-cfvlm.ipynb'
    plan['attack_jobs'][0]['variants'] = plan['attack_jobs'][0]['variants'][2:]
    plan['attack_jobs'][0]['overrides']['MANUAL_TEXT_OVERRIDES'] = factual
    plan['evaluation']['local_batch_size'] = 64
    policy = load(package / 'cfvlm_negatives.py', 'cfpolicy')
    plan['negative_policy'] = {'version': policy.NEGATIVE_POLICY_VERSION,
                               'prompt_sha256': policy.CF_PROMPT_SHA256, 'bank_size': 4,
                               'selected_negatives': 2, 'generation_batch_size': 16,
                               'audit_prompt_sha256': policy.CF_AUDIT_SHA256,
                               'semantic_audit': 'second_image_conditioned_Qwen_pass_reject_uncertain',
                               'audit_model': 'Qwen/Qwen3-VL-8B-Instruct'}
    plan['legacy_provenance'] = {
        'source_prefix': PARENT + '/results',
        'metadata_archive_sha256': hashlib.sha256(seed_file.read_bytes()).hexdigest(),
        'factual_captions_sha256': hashlib.sha256((provenance / 'factual_captions.json').read_bytes()).hexdigest(),
        'baseline_variants': ['foa_reference', 'mpc_clip'],
        'legacy_hybrid_samples_imported': 0,
        'original_bank_imported': False,
    }
    plan['notes'][0] = 'Eight corrected ablations x 200 pairs = 1600 attack samples. Two prior baselines retained separately.'
    write_frozen(root / 'molab_hypotheses_n200.json', plan)
    write_frozen(eval_root / 'molab_hypotheses_n200.json', plan)
    data = root / 'attack_data_n200.zip'
    if not data.exists():
        fs.get_file(PARENT + '/attack_data_n200.zip', str(data))
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('Original data archive checksum mismatch')
    if not (root / 'data/pairs_200.jsonl').exists():
        with ZipFile(data) as archive:
            for item in archive.infolist():
                if not (root / 'data' / item.filename).resolve().is_relative_to((root / 'data').resolve()):
                    raise ValueError('Unsafe data ZIP path')
            archive.extractall(root / 'data')
    driver = load(root / 'molab_hypothesis_campaign.py', 'campaign')
    driver.validate_package(root, plan)
    if not (root / 'campaign_state.json').exists():
        if fs.exists(ATTACK_PREFIX + '/primary_ablations/campaign_state.json'):
            driver.restore_campaign(root, plan, ATTACK_PREFIX)
        else:
            driver.write_json(root / 'campaign_state.json', {'attack_runs': {}, 'evaluations': {}, 'eval_model_revisions': {}})
    fs.makedirs(PARENT + '/cfvlm_v2/code', exist_ok=True)
    for local in (root / 'molab_hypotheses_n200.json', provenance / 'factual_captions.json'):
        remote = PARENT + '/cfvlm_v2/code/' + local.name
        fs.put_file(str(local), remote)
        with fs.open(remote, 'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest() != hashlib.sha256(local.read_bytes()).hexdigest():
                raise ValueError('Plan/provenance backup checksum mismatch')
    print('CF-VLM package/data validated; factual captions locked; legacy negative bank excluded.', flush=True)
    return plan


def setup_environments(root, eval_root, plan):
    driver = load(root / 'molab_hypothesis_campaign.py', 'campaign_setup')
    executable = root / '.venvs/attack/bin/python'
    if not executable.exists():
        base_python = getattr(sys, '_base_executable', sys.executable)
        subprocess.run([base_python, '-m', 'venv', '--system-site-packages', str(executable.parent.parent)], check=True)
    tree = ast.parse(driver.notebook_cells(root / plan['attack_notebook'])[1][1])
    pins = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'PINNED_PACKAGES' for t in node.targets))
    subprocess.run([str(executable), '-m', 'pip', 'install', *pins, 'numpy', 'Pillow', 'gdrive-fsspec', 'python-dotenv'], check=True)
    load(eval_root / 'molab_eval_completed.py', 'eval_setup').setup(eval_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--root', type=Path, default=Path('/marimo/cfvlm_n200_v2'))
    parser.add_argument('--eval-root', type=Path, default=Path('/marimo/eval_cfvlm_n200_v2'))
    parser.add_argument('--control-root', type=Path, default=Path('/marimo/overnight_cfvlm_n200_v2'))
    parser.add_argument('--max-hours', type=float, default=7.5)
    args = parser.parse_args()
    plan = initialize(args.package, args.root, args.eval_root)
    setup_environments(args.root, args.eval_root, plan)
    evaluator = load(args.eval_root / 'molab_eval_completed.py', 'eval_restore')
    evaluator.prepare(args.eval_root, args.root, ATTACK_PREFIX)
    evaluator.restore_evaluations(args.eval_root, EVAL_PREFIX)
    command = [sys.executable, '-u', str(args.root / 'molab_overnight.py'),
               '--root', str(args.root), '--eval-root', str(args.eval_root), '--control-root', str(args.control_root),
               '--max-hours', str(args.max_hours), '--attack-workers', '2', '--caption-batch-size', '64',
               '--backup-prefix', ATTACK_PREFIX, '--eval-backup-prefix', EVAL_PREFIX]
    print('Starting attack -> immediate GPU caption -> GPT evaluation controller.', flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
