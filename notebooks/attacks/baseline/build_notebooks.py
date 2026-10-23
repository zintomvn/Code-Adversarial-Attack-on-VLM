"""Rebuild portable notebooks and a small, deterministic, verified author bundle."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import pprint
import subprocess
import textwrap
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def archive_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return buffer.getvalue()


def build():
    # These files were audited against git object bytes, not copied from a modified working tree.
    lock_path = HERE / 'SOURCE_LOCK.json'
    if not lock_path.exists():
        lock_path = REPO / 'tmp/baseline_validation/author/SOURCE_LOCK.json'
    lock = json.loads(lock_path.read_text())
    repos = dict(foa=REPO / 'tmp/baseline-upstream/FOA-Attack',
        mpc=REPO / 'tmp/baseline-mpc-verified', saga=REPO / 'tmp/baseline-upstream/SAGA')
    sources, audit = {}, {}
    prior_sources = {}
    if (HERE / 'baseline_code.zip').exists():
        with zipfile.ZipFile(HERE / 'baseline_code.zip') as prior:
            prior_sources = {name: prior.read(name) for name in prior.namelist() if name.startswith('author/')}
    local_dirs = dict(foa='FOA-Attack', mpc='MPCAttack', saga='SAGA')
    for method, entry in lock.items():
        audit[method] = {}
        for relative, expected in entry['files'].items():
            if (repos[method] / '.git').exists():
                data = subprocess.check_output(['git', '-C', str(repos[method]), 'show', entry['commit'] + ':' + relative])
            else:
                data = prior_sources['author/' + method + '/' + relative]
            assert hashlib.sha256(data).hexdigest() == expected, (method, relative)
            sources['author/' + method + '/' + relative] = data
            local = REPO / 'src' / local_dirs[method] / relative
            audit[method][relative] = dict(upstream_sha256=expected,
                local_sha256=hashlib.sha256(local.read_bytes()).hexdigest() if local.exists() else None,
                local_identical=local.exists() and local.read_bytes() == data,
                normalized_lf_identical=local.exists() and local.read_bytes().replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n'))
    sources['author/SOURCE_LOCK.json'] = lock_path.read_bytes()
    (HERE / 'SOURCE_LOCK.json').write_bytes(lock_path.read_bytes())
    (HERE / 'source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    notebook_path = REPO / 'notebooks/text_gen_eval_v3.2.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    protocol = '# Extracted verbatim from text_gen_eval_v3.2.ipynb code cells 5, 6, 7.\n'
    protocol += '\n\n'.join(''.join(notebook['cells'][i]['source']) for i in (5, 6, 7))
    ast.parse(protocol)
    (HERE / 'eval_protocol_v32.py').write_text(protocol, encoding='utf-8')
    evaluation_lock = dict(notebook='notebooks/text_gen_eval_v3.2.ipynb',
        notebook_sha256=hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        extracted_cells=[5, 6, 7], protocol_sha256=hashlib.sha256((HERE / 'eval_protocol_v32.py').read_bytes()).hexdigest())
    (HERE / 'evaluation_source.json').write_text(json.dumps(evaluation_lock, indent=2) + '\n')
    for name in ('baseline_runtime.py', 'author_worker.py', 'eval_worker.py', 'notebook_session.py',
                 'eval_protocol_v32.py', 'evaluation_source.json'):
        sources[name] = (HERE / name).read_bytes()
    bundle = archive_bytes(sources)
    (HERE / 'baseline_code.zip').write_bytes(bundle)
    bundle_sha = hashlib.sha256(bundle).hexdigest()
    revisions = {'openai/clip-vit-base-patch16': '57c216476eefef5ab752ec549e440a49ae4ae5f3',
        'openai/clip-vit-base-patch32': '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
        'laion/CLIP-ViT-G-14-laion2B-s12B-b42K': '4b0305adc6802b2632e11cbe6606a9bdd43d35c9',
        'facebook/dinov2-base': 'f9e44c814b77203eaa57a6bdbbd535f21ede1415',
        'OpenGVLab/InternVL3-1B-hf': '014c0583a0d4bedf29fbe2dbff4f865eb998e171',
        'Qwen/Qwen3-VL-8B-Instruct': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        'llava-hf/llava-1.5-7b-hf': 'b234b804b114d9e37bb655e11cbbb5f5e971b7a9',
        'OpenGVLab/InternVL3-8B-hf': '259a3b64a14623c0ec91a045cb43f7c5af5fa6af'}
    model_settings = {
        'LLaVA-1.5-7B': dict(provider='huggingface', model_id='llava-hf/llava-1.5-7b-hf', quantization='nf4'),
        'InternVL3-8B': dict(provider='huggingface', model_id='OpenGVLab/InternVL3-8B-hf', quantization='nf4'),
        'Qwen3-VL-8B': dict(provider='huggingface', model_id='Qwen/Qwen3-VL-8B-Instruct', quantization='nf4'),
        'Gemini 2.5 Flash': dict(provider='gemini', model_id='gemini-2.5-flash', quantization='none'),
        'GPT-5 mini': dict(provider='openai', model_id='gpt-5-mini-2025-08-07', quantization='none')}
    config = dict(run_name='official_n200_e16_s300_drive_v6', methods=['foa', 'mpc', 'saga'],
        n=200, epsilon=16, steps=300, alpha=1.0, cpu_threads=4, session_hours=7.5,
        data_archive='', data_drive_file_id='1V51d_Zk74JY84uJhsUIX-A1y1wfJrDJN',
        data_sha256='3518b43ac953db8192a31698105cee4f277cc543351bf28e1eec6d7452987103',
        backup_drive=True, drive_parent_id='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI',
        drive_prefix='official_baselines', model_revisions=revisions,
        mpc=dict(lam=0.6, tau=0.2, omega=2.0),
        saga=dict(seed=43, phases=10, top_k=3, layer=29), run_evaluation=True,
        evaluation=dict(models=list(model_settings), model_settings=model_settings,
            prompt='Describe this image in one concise sentence, no longer than 20 words.',
            max_new_tokens=64, local_batch_size=8, api_workers=4, run_judge=True,
            judge_model='gpt-4o', judge_workers=4, judge_rpm=12, judge_tpm=8000,
            success_threshold=0.3, success_operator='>='))
    config_cell = '''# Edit this cell only when changing account, data, or hyperparameters.
# epsilon is in pixel units [0,255]. Original MPC requires steps=300 and batch=1.
# Use the SAME RUN_NAME/config to resume; choose a NEW RUN_NAME for a new experiment.
# On another account: change drive_parent_id; share/copy the dataset or set data_archive.
from pathlib import Path
import os
import marimo as mo
CONFIG = ''' + pprint.pformat(config, width=110, sort_dicts=False) + '''
WORK_ROOT = Path('/kaggle/working' if Path('/kaggle').exists() else '/marimo' if Path('/marimo').exists() else '.') / 'baselines' / CONFIG['run_name']
'''
    encoded = base64.b64encode(bundle).decode()
    boot = '''# Frozen author sources and orchestration; no dataset or credentials embedded.
import base64 as _b64, hashlib as _hash, io as _io, zipfile as _zip
import subprocess as _sp, sys as _sys, shutil as _shutil, importlib.util as _iu
for _module, _package in [('googleapiclient','google-api-python-client'), ('pydata_google_auth','pydata-google-auth'), ('dotenv','python-dotenv')]:
    if _iu.find_spec(_module) is None:
        _uv = _shutil.which('uv')
        _sp.run([_uv,'pip','install','--python',_sys.executable,_package] if _uv else [_sys.executable,'-m','pip','install',_package], check=True)
_bundle = _b64.b64decode(''' + repr(encoded) + ''')
assert _hash.sha256(_bundle).hexdigest() == ''' + repr(bundle_sha) + '''
WORK_ROOT.mkdir(parents=True, exist_ok=True)
with _zip.ZipFile(_io.BytesIO(_bundle)) as _z:
    for _entry in _z.infolist():
        _target = (WORK_ROOT / _entry.filename).resolve()
        if not _target.is_relative_to(WORK_ROOT.resolve()): raise ValueError('Unsafe source archive')
        _data = _z.read(_entry)
        if _target.exists() and _target.read_bytes() != _data: raise ValueError('Run code differs; select a new RUN_NAME')
        _target.parent.mkdir(parents=True, exist_ok=True)
        if not _target.exists(): _target.write_bytes(_data)
if str(WORK_ROOT.resolve()) not in _sys.path: _sys.path.insert(0, str(WORK_ROOT.resolve()))
from baseline_runtime import verify_sources as _verify_sources
_verify_sources(WORK_ROOT)
# Explicit import path for isolated child Python environments.
os.environ['PYTHONPATH'] = str(WORK_ROOT.resolve()) + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
from notebook_session import Session as _Session
session = _Session(WORK_ROOT, CONFIG)
auth_panel = session.auth_ui()
start_button = mo.ui.run_button(label='Chạy / tiếp tục baseline + evaluation')
refresh = mo.ui.refresh(options=['10s','30s','1m'], default_interval='30s')
mo.vstack([auth_panel, start_button])
'''
    launch = '''if start_button.value:
    session.launch()
else:
    print('Kết nối Drive, chọn GPU, rồi bấm Chạy / tiếp tục. API keys chỉ cần cho evaluation.')
'''
    monitor = '''refresh.value
mo.vstack([refresh, session.monitor()])
'''
    intro = '''# Official baselines · FOA-Attack / MPC-Attack / SAGA

N=200 · ε=16 pixel · 300 steps. Author code is locked by commit and SHA-256.

1. Select RTX Pro 6000 in **Configure compute** (Kaggle: enable GPU + Internet).
2. Edit CONFIG: account folder, dataset path/Drive ID, hyperparameters and evaluation models.
3. Run cells, connect Drive using the private OAuth form, optionally enter API keys.
4. Click **Chạy / tiếp tục**. Keep the same RUN_NAME after a session expires.

Checkpoints upload to Drive after each completed attack image. An interrupted image
restarts from the preceding image's saved RNG state. Captions and GPT scores resume
from durable batches. No credentials are stored in notebook code or result archives.

FOA: original fixed K=10, FGSM, source/target crops; MPC: original 300-step MI-FGSM;
SAGA: original released text targets, Qwen3-VL attention and region schedule.
Targets and preprocessing differ by author; details are in README_vi.md.
GPU runtime uses PyTorch 2.8 CUDA 12.8 for Blackwell. Attack precision stays unchanged.
Runs use one method at a time to fit the available 32 GiB system RAM.

Evaluation reuses v3.2 CaptionGenerator and GPTScorer verbatim: LLaVA-1.5-7B,
InternVL3-8B, Qwen3-VL-8B, Gemini 2.5 Flash, GPT-5 mini. The judge is GPT-4o,
temperature=0, max_tokens=100; default ASR uses score >= 0.3. API usage is billed.
'''
    cells = [('markdown', intro), ('code', config_cell), ('code', boot), ('code', launch), ('code', monitor)]
    def cell(kind, source):
        result = dict(cell_type=kind, id=hashlib.sha256(source.encode()).hexdigest()[:12],
            metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result
    metadata = dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11'), baseline_bundle_sha256=bundle_sha)
    output = dict(cells=[cell(*c) for c in cells], metadata=metadata, nbformat=4, nbformat_minor=5)
    (HERE / 'official_baselines_n200.ipynb').write_text(json.dumps(output, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    # Jupyter's last two cells replace Marimo's reactive button/monitor with sequential calls.
    jupyter = json.loads(json.dumps(output))
    jupyter['cells'][1] = cell('code', config_cell.replace('import marimo as mo\n', ''))
    jupyter['cells'][2] = cell('code', boot[:boot.index('auth_panel = session.auth_ui()')] +
        "_url = session.auth_url() if CONFIG['backup_drive'] else None\n"
        "if _url:\n    from IPython.display import display, Markdown\n    display(Markdown('[Connect Drive](' + _url + ')'))\n"
        "    import getpass\n    session.connect(getpass.getpass('OAuth callback URL: '))\n"
        "for _key in ('OPENAI_API_KEY','GEMINI_API_KEY'):\n"
        "    if CONFIG['run_evaluation'] and not os.environ.get(_key):\n"
        "        import getpass\n        _value=getpass.getpass(_key+' (empty to defer): ')\n"
        "        if _value: session.set_secret(_key,_value)\n")
    jupyter['cells'][3] = cell('code', 'session.launch()\n')
    jupyter['cells'][4] = cell('code', 'session.status()  # Re-run this cell for current progress.\n')
    (HERE / 'official_baselines_kaggle.ipynb').write_text(json.dumps(jupyter, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    native = '# /// script\n# requires-python = ">=3.11"\n# dependencies = ["marimo", "google-api-python-client", "pydata-google-auth", "python-dotenv"]\n# ///\nimport marimo\n\n__generated_with = "0.25.2"\napp = marimo.App()\n\n'
    for args, content, returns in [
        ('mo', 'mo.md(' + repr(intro) + ')', ''),
        ('', config_cell, 'CONFIG, WORK_ROOT, mo'),
        ('CONFIG, WORK_ROOT, mo', boot, 'session, auth_panel, start_button, refresh'),
        ('start_button, session', launch, ''),
        ('refresh, session, mo', monitor, '')]:
        native += '@app.cell\ndef _(' + args + '):\n' + textwrap.indent(content.rstrip(), '    ')
        native += '\n    return' + (' ' + returns if returns else '') + '\n\n'
    native += 'if __name__ == "__main__":\n    app.run()\n'
    ast.parse(native)
    (HERE / 'official_baselines_molab.py').write_text(native, encoding='utf-8')
    # Small code cell used for browser deployment through the editor, without file uploads.
    editor = config_cell + '\n' + boot
    (HERE / 'molab_editor_cell.py').write_text(editor, encoding='utf-8')
    write = dict(bundle_sha256=bundle_sha, bundle_bytes=len(bundle), author_files=sum(len(e['files']) for e in lock.values()))
    (HERE / 'build_manifest.json').write_text(json.dumps(write, indent=2) + '\n')
    print(json.dumps(write))


if __name__ == '__main__':
    build()
