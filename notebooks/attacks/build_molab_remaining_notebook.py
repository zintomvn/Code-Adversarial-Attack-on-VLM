"""Package a separate optional launcher for remaining attack jobs."""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def build():
    base = Path(__file__).resolve().parent
    files = {name: base / name for name in ['molab_remaining_tasks.py', 'molab_hypothesis_campaign.py',
             'molab_hypotheses_n200.json', 'kltn-attack-v4.3.2-h1.ipynb']}
    files['text_gen_eval_v3.1.ipynb'] = base.parent / 'text_gen_eval_v3.1.ipynb'
    memory = io.BytesIO()
    with ZipFile(memory, 'w', ZIP_DEFLATED) as z:
        for name, path in files.items():
            z.writestr(name, path.read_bytes())
        z.writestr('checksums.json', json.dumps({n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}))
    encoded = base64.b64encode(memory.getvalue()).decode()
    cells = []

    def add(kind, text):
        cell = {'cell_type': kind, 'metadata': {}, 'id': f'attack-remaining-{len(cells):02d}', 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# Molab — chạy tiếp attack sau evaluation

Ưu tiên chạy notebook evaluation trước. Không chạy launcher này cùng GPU với evaluation hoặc launcher attack cũ.

Giữ nguyên code, hash resume, thứ tự ảnh và trạng thái FOA của campaign cũ. Job `primary_ablations` tiếp tục từ hybrid_plain đang dở, bỏ qua FOA/MPC đã xong; hai controls H4 có worker/state riêng và dùng lại caption đã có.

`JOB_WORKERS=2` thử chạy hai job độc lập trên GPU >=80 GiB, không phải batch hai ảnh trong cùng chuỗi FOA. Có thể tranh CPU/VRAM và chậm hơn; bắt đầu 1 nếu chưa benchmark. Không cam kết GPU utilization 100%.

Backup qua hàng đợi giới hạn, ZIP bất biến, checksum readback. Dòng LOCAL SAMPLE là đã lưu local; DRIVE BACKUP DRAINED AND VERIFIED mới xác nhận upload hàng đợi hoàn tất. Dừng tự động theo ngân sách giờ sẽ flush; tắt cưỡng bức runtime có thể cần chạy lại các mẫu chưa durable.
''')
    add('code', '''from pathlib import Path
ATTACK_ROOT = Path('/marimo/hypotheses_n200')
JOB_WORKERS = 2  # thử 2 trên RTX PRO 6000; giảm 1 nếu OOM hoặc thông lượng giảm
MAX_HOURS = 8  # đặt theo thời gian còn lại của phiên, không vượt giới hạn Molab
RUN_REMAINING_ATTACK = False  # chỉ bật sau khi evaluation đã dừng/hoàn tất
''')
    add('code', '''import base64 as _b64, io as _io, json as _json, hashlib as _hash
import subprocess as _sp, sys as _sys, importlib.util as _iu, shutil as _shutil
from zipfile import ZipFile as _Zip
ATTACK_ROOT.mkdir(parents=True, exist_ok=True)
_payload = "''' + encoded + '''"
with _Zip(_io.BytesIO(_b64.b64decode(_payload))) as _bundle:
    for _name, _digest in _json.loads(_bundle.read('checksums.json')).items():
        _data = _bundle.read(_name)
        assert _hash.sha256(_data).hexdigest() == _digest
        _p = ATTACK_ROOT / _name
        if _p.exists() and _p.read_bytes() != _data:
            raise RuntimeError('Existing package differs; preserve the original campaign files before resume.')
        _p.write_bytes(_data)
if _iu.find_spec('gdrive_fsspec') is None:
    _install = (['uv', 'pip', 'install', '--python', _sys.executable] if _shutil.which('uv')
                else [_sys.executable, '-m', 'pip', 'install'])
    _sp.run([*_install, 'gdrive-fsspec'], check=True)

def attack_stage(mode, original=False):
    exe = ATTACK_ROOT / '.venvs/attack/bin/python'
    python = str(exe) if exe.is_file() and mode != 'setup' else _sys.executable
    script = 'molab_hypothesis_campaign.py' if original else 'molab_remaining_tasks.py'
    command = [python, '-u', str(ATTACK_ROOT / script), mode, '--root', str(ATTACK_ROOT), '--max-hours', str(MAX_HOURS)]
    if not original:
        command += ['--workers', str(JOB_WORKERS)]
    with _sp.Popen(command, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
            code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=60)
    if code not in (0, 75):
        raise RuntimeError(f'Attack stage failed: {mode} ({code}); inspect parallel_tasks logs')
    return code
''')
    add('code', '''from gdrive_fsspec import GoogleDriveFileSystem
attack_drive = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                    auth_kwargs={'use_local_webserver': False})
if not (ATTACK_ROOT / 'data/pairs_200.jsonl').exists():
    _archive = ATTACK_ROOT / 'attack_data_n200.zip'
    attack_drive.get_file('KLTN_Molab_Hypotheses_n200_v432_h1/attack_data_n200.zip', str(_archive))
    assert _hash.sha256(_archive.read_bytes()).hexdigest() == '3518b43ac953db8192a31698105cee4f277cc543351bf28e1eec6d7452987103'
    with _Zip(_archive) as _data_zip:
        for _item in _data_zip.infolist():
            if not (ATTACK_ROOT / 'data' / _item.filename).resolve().is_relative_to((ATTACK_ROOT / 'data').resolve()):
                raise ValueError('Unsafe data ZIP path')
        _data_zip.extractall(ATTACK_ROOT / 'data')
# Nếu còn đầy đủ workspace cũ thì không cần tải lại hàng trăm checkpoint.
if not (ATTACK_ROOT / 'campaign_state.json').exists():
    attack_stage('restore', original=True)
attack_stage('setup', original=True)
attack_stage('list')
''')
    add('code', '''if RUN_REMAINING_ATTACK:
    attack_stage('run')
else:
    print('READY. Finish/pause priority evaluation, then set RUN_REMAINING_ATTACK=True.')
''')
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    out = base / 'molab_attack_remaining_parallel_v1.ipynb'
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    release = base.parents[1] / 'outputs/molab_eval_completed_v1'
    release.mkdir(parents=True, exist_ok=True)
    with ZipFile(release / 'molab_attack_remaining_parallel_v1.zip', 'w', ZIP_DEFLATED) as z:
        for name, path in {**files, out.name: out}.items():
            z.writestr(name, path.read_bytes())
    print(out)


if __name__ == '__main__':
    build()
