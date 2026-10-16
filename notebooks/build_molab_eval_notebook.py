"""Build the self-contained Molab evaluation notebook from reviewed source files."""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build():
    repo = Path(__file__).resolve().parents[1]
    paths = {
        'molab_eval_completed.py': repo / 'notebooks/molab_eval_completed.py',
        'molab_eval_schedule.py': repo / 'notebooks/molab_eval_schedule.py',
        'text_gen_eval_v3.1.ipynb': repo / 'notebooks/text_gen_eval_v3.1.ipynb',
        'molab_hypotheses_n200.json': repo / 'notebooks/attacks/molab_hypotheses_n200.json',
    }
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as z:
        for name, path in paths.items():
            z.writestr(name, path.read_bytes())
        z.writestr('bundle_checksums.json', json.dumps(
            {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}, indent=2))
    payload = base64.b64encode(buffer.getvalue()).decode()
    cells = []

    def add(kind, source):
        c = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            c.update(execution_count=None, outputs=[])
        cells.append(c)

    add('markdown', '''# Đánh giá các cấu hình attack đã hoàn tất — Molab n200

Notebook độc lập, không cần chờ đủ 10 cấu hình attack. Chỉ nhận cấu hình có đủ 200 cặp và checksum hợp lệ; hiển thị riêng phần còn thiếu. Hiện có thể đánh giá FOA reference và MPC-CLIP trước.

Dùng nguyên protocol `text_gen_eval_v3.1.ipynb`: LLaVA-1.5-7B, InternVL3-8B, GPT-4o, success `score >= 0.3`. InternVL3-8B khác checkpoint nhưng cùng họ với captioner attack 1B.

Molab: bật GPU, import notebook này, chạy các cell theo thứ tự. Không chạy attack đồng thời khi bắt đầu evaluation. Batch 64 đã smoke trên RTX PRO 6000 cho LLaVA và InternVL, không phải mức tối ưu đã benchmark; OOM tự chia nhỏ. T4 nên bắt đầu batch 2–4.

Thêm `OPENAI_API_KEY` vào environment/Secrets hoặc `/marimo/.env`; không dán key vào cell hay chat. Thiếu key vẫn tạo caption GPU, rồi resume GPT judge sau. GPT judge phát sinh phí API; mặc định giữ quota 12 RPM / 8000 TPM của campaign.
''')
    add('code', '''from pathlib import Path
EVAL_ROOT = Path('/marimo/eval_completed_n200')
ATTACK_ROOT = Path('/marimo/hypotheses_n200')
# Kaggle sau này: đổi EVAL_ROOT thành /kaggle/working/eval_completed_n200
# và ATTACK_ROOT thành thư mục campaign đã tải; kết nối Drive hoặc dùng input local.
CONFIG_IDS = None  # None = mọi cấu hình đủ 200; hoặc ['primary_ablations__foa_reference']
VICTIM_MODELS = ['LLaVA-1.5-7B', 'InternVL3-8B']
CAPTION_BATCH_SIZE = 64  # đã smoke trên RTX PRO 6000; T4 dùng 2–4; tự chia khi OOM
RUN_GPT_JUDGE = True
BACKUP_TO_DRIVE = True
MAX_HOURS = 8  # nếu phiên Molab đã dùng ~9 giờ: đặt 1–2, hoặc mở phiên GPU mới
''')
    add('code', '''# Bootstrap: code + notebook v3.1 được nhúng, không tải code từ URL thay đổi.
import base64 as _base64, hashlib as _hashlib, io as _io, json as _json
import subprocess as _subprocess, sys as _sys, importlib.util as _importlib_util, shutil as _shutil
from zipfile import ZipFile as _ZipFile
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
_embedded_bundle = "''' + payload + '''"
with _ZipFile(_io.BytesIO(_base64.b64decode(_embedded_bundle))) as _bundle:
    _checks = _json.loads(_bundle.read('bundle_checksums.json'))
    for _name, _digest in _checks.items():
        _data = _bundle.read(_name)
        assert _hashlib.sha256(_data).hexdigest() == _digest
        _dest = EVAL_ROOT / _name
        if _dest.exists() and _dest.read_bytes() != _data and (EVAL_ROOT / 'evaluation_state.json').exists():
            _old_hash = _hashlib.sha256(_dest.read_bytes()).hexdigest()
            if _name == 'molab_eval_completed.py' and _old_hash == 'cc9de38eb2debb89c991bf4d1cd20d5db8ee9adf9899952cfe54a5efda30cc5a':
                (EVAL_ROOT / 'molab_eval_completed.pre_quota_guard.py').write_bytes(_dest.read_bytes())
            else:
                raise RuntimeError('Package changed during resume; use the original notebook or a new EVAL_ROOT.')
        _dest.write_bytes(_data)
if _importlib_util.find_spec('gdrive_fsspec') is None:
    _install = (['uv', 'pip', 'install', '--python', _sys.executable] if _shutil.which('uv')
                else [_sys.executable, '-m', 'pip', 'install'])
    _subprocess.run([*_install, 'gdrive-fsspec'], check=True)

def eval_stage(mode):
    executable = EVAL_ROOT / '.venv/bin/python'
    python = str(executable) if executable.is_file() and mode != 'setup' else _sys.executable
    script = 'molab_eval_schedule.py' if mode == 'run' else 'molab_eval_completed.py'
    command = [python, '-u', str(EVAL_ROOT / script), mode,
               '--root', str(EVAL_ROOT), '--source-root', str(ATTACK_ROOT),
               '--batch-size', str(CAPTION_BATCH_SIZE), '--max-hours', str(MAX_HOURS),
               '--models', *VICTIM_MODELS]
    if CONFIG_IDS:
        command += ['--only', *CONFIG_IDS]
    if not RUN_GPT_JUDGE:
        command.append('--captions-only')
    if not BACKUP_TO_DRIVE:
        command.append('--no-backup')
    log_path = EVAL_ROOT / ('stage_' + mode + '.log')
    with log_path.open('a', encoding='utf-8') as log:
        with _subprocess.Popen(command, stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT,
                               text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
    if code == 75:
        print('PAUSED/PARTIAL: checkpoint saved; rerun run after fixing any reported errors.')
    elif code:
        raise RuntimeError(f'{mode} exited {code}; inspect {log_path}')
    return code

eval_stage('setup')
''')
    add('markdown', '''## Kết nối Drive

Trong Molab có thể dùng Files → Remote Storage → Google Drive. Cell sau dùng cùng cơ chế OAuth; giữ quyền riêng tư của thư mục. Nếu token còn hiệu lực thì dùng lại. Nếu không dùng Drive và đã có đủ input local, có thể bỏ cell này và đặt `BACKUP_TO_DRIVE=False`.
''')
    add('code', '''if BACKUP_TO_DRIVE:
    from gdrive_fsspec import GoogleDriveFileSystem
    eval_drive = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                      auth_kwargs={'use_local_webserver': False})
    print('Drive connected; credentials are not printed.')
''')
    add('code', '''# Dùng output local nếu còn, nếu thiếu thì tải checkpoint từ Drive.
# Chỉ trích PNG + complete markers; không cần tải model weights hay optimizer checkpoints.
eval_stage('prepare')
''')
    add('code', '''# Phục hồi evaluation cũ nếu có; lần đầu không có checkpoint thì không làm gì.
if BACKUP_TO_DRIVE:
    eval_stage('restore')
''')
    add('code', '''# Ưu tiên đánh giá ngay các cấu hình đã đủ 200 ảnh.
# Sinh caption GPU cho mọi task sẵn sàng trước, rồi mới chạy GPT judge.
# Tự resume captions/scores, không gọi lại API cho score hợp lệ đã lưu.
eval_stage('run')
''')
    add('code', '''eval_stage('report')
from IPython.display import display, FileLink
display(FileLink(str(EVAL_ROOT / 'summary_completed.json')))
display(FileLink(str(EVAL_ROOT / 'inventory.json')))
''')
    add('markdown', '''## Chạy tiếp và đọc kết quả

Mỗi cấu hình × victim có caption checkpoint, `evaluation/scores.csv`, `summary.json` và trace riêng. `ASR` toàn bộ cohort chỉ được báo khi đủ 200 score hợp lệ; không biến 13 ảnh của hybrid thành kết quả 200 ảnh.

Timeout: mở lại đúng notebook với cùng EVAL_ROOT và package, kết nối Drive, chạy prepare → restore → run. Đừng đổi model revision, rubric hoặc ảnh đầu vào trong một run. Khi thêm key sau caption-only, chạy lại run; caption hợp lệ được giữ lại.

Khi 8 cấu hình còn lại hoàn thành: chạy lại prepare (CONFIG_IDS=None) → restore → run. Chỉ các cấu hình đủ cohort được nhận, các task evaluation đã hoàn tất được bỏ qua. Đây là workflow evaluation mới; không thay đổi campaign attack cũ.

Nguồn: [OpenAI authentication](https://developers.openai.com/api/reference/overview#authentication), [rate limits](https://developers.openai.com/api/docs/guides/rate-limits), [LLaVA batching](https://huggingface.co/docs/transformers/v4.57.0/model_doc/llava).
''')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'language_info': {'name': 'python', 'version': '3.13'}},
                'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'molab-eval-{i:02d}'
    out = repo / 'notebooks/text_gen_eval_molab_completed_v1.ipynb'
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    release = repo / 'outputs/molab_eval_completed_v1'
    release.mkdir(parents=True, exist_ok=True)
    shutil_paths = {**paths, out.name: out}
    with ZipFile(release / 'molab_eval_completed_v1.zip', 'w', ZIP_DEFLATED) as z:
        for name, path in shutil_paths.items():
            z.writestr(name, path.read_bytes())
    print(out)


if __name__ == '__main__':
    build()
