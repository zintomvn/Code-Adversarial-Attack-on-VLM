"""Package the corrected attack and immediate evaluation controller."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo


def notebook(base, bundle_sha):
    cells = []
    def add(kind, source):
        cell = {'cell_type': kind, 'metadata': {}, 'id': f'cfvlm-overnight-{len(cells):02d}',
                'source': source.splitlines(keepends=True)}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', '''# Molab: CF-VLM negative captions + attack → evaluation

Chạy các cell theo thứ tự. Sinh negative caption theo một thuộc tính hoặc một
quan hệ nhân quả của CF-VLM; thay nguyên nhân cùng hệ quả liên quan được phép.
Bank mới gồm 4 ứng viên, mine/select 2 như trước. Giữ nguyên hai baseline cũ;
8 ablation dùng root/plan riêng, không nhập 13 mẫu hybrid hoặc bank cũ.

Controller chạy tối đa 2 job attack độc lập. Khi một cấu hình đủ 200 ảnh,
nhường GPU ở ranh giới mẫu đã lưu → caption hai victim batch 64 (OOM chia nhỏ)
→ GPT judge ngay. Judge có thể chạy cùng attack tiếp theo. Backup Drive có
checksum. Phiên mới cần kết nối Drive lại và thêm OPENAI_API_KEY vào Secrets
hoặc /marimo/.env; không đặt key vào notebook. Molab vẫn có thể timeout trước
khi xong cả 8 cấu hình; rerun cùng package/root để resume checkpoint.

[Paper](https://arxiv.org/abs/2506.17267) · [Code chính thức](https://github.com/HCP-AI-Research-Lab/CF-VLM).
Đây là áp dụng chính sách chỉnh sửa caption vào attack, không tái hiện toàn bộ
CF-VLM fine-tuning/SDXL. JSON validator kiểm tra cấu trúc; cần kiểm tra ngữ nghĩa
và tính sai với ảnh qua raw output/trace được lưu.
''')
    add('code', '''from pathlib import Path
CF_PACKAGE = Path('/marimo/cfvlm_code_v2')
CF_MAX_HOURS = 5.8  # đặt nhỏ hơn thời gian còn lại của phiên Molab
CF_RUN_OVERNIGHT = True
''')
    add('code', '''def cf_connect_drive():
    import importlib.util, shutil, subprocess, sys
    required = [p for p, module in [('gdrive-fsspec','gdrive_fsspec'),('python-dotenv','dotenv')]
                if importlib.util.find_spec(module) is None]
    if required:
        command = ['uv','pip','install','--python',sys.executable] if shutil.which('uv') else [sys.executable,'-m','pip','install']
        subprocess.run([*command, *required], check=True)
    from gdrive_fsspec import GoogleDriveFileSystem
    return GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True,
                                auth_kwargs={'use_local_webserver': False})
cf_drive = cf_connect_drive()
''')
    add('code', '''def cf_download_package():
    import hashlib, json
    from zipfile import ZipFile
    from gdrive_fsspec import GoogleDriveFileSystem
    CF_PACKAGE.mkdir(parents=True, exist_ok=True)
    archive = CF_PACKAGE / 'molab_cfvlm_overnight_v2.zip'
    # A fresh client also avoids reusing stale HTTP connections after OAuth.
    storage = GoogleDriveFileSystem(token='cache', use_listings_cache=False, skip_instance_cache=True)
    storage.get_file('KLTN_Molab_Hypotheses_n200_v432_h1/molab_cfvlm_overnight_v2.zip', str(archive))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != 'BUNDLE_SHA':
        raise ValueError('Package differs from the pinned notebook version')
    with ZipFile(archive) as z:
        for name, expected in json.loads(z.read('checksums.json')).items():
            destination = (CF_PACKAGE / name).resolve()
            if not destination.is_relative_to(CF_PACKAGE.resolve()):
                raise ValueError('Unsafe bundle path')
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Package member checksum mismatch')
            destination.write_bytes(data)
        (CF_PACKAGE / 'checksums.json').write_bytes(z.read('checksums.json'))
    print('Code package verified.')
cf_download_package()
'''.replace('BUNDLE_SHA', bundle_sha))
    add('code', '''def cf_start_overnight():
    import os, subprocess, sys
    if not CF_RUN_OVERNIGHT:
        print('Set CF_RUN_OVERNIGHT=True to run.')
        return
    pid_file = CF_PACKAGE / 'setup.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text())
        state = subprocess.run(['ps','-p',str(pid),'-o','stat='], capture_output=True, text=True).stdout.strip()
        if state and not state.startswith('Z'):
            print('Setup/controller already running, PID', pid)
            return
    with (CF_PACKAGE / 'overnight_setup.log').open('a') as log:
        child = subprocess.Popen([sys.executable,'-u',str(CF_PACKAGE/'molab_cfvlm_setup.py'),
                                  '--max-hours',str(CF_MAX_HOURS)], stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
    pid_file.write_text(str(child.pid))
    print('Setup/controller launched, PID',child.pid)
cf_start_overnight()
''')
    add('code', '''def cf_show_progress():
    import json, subprocess
    control = Path('/marimo/overnight_cfvlm_n200_v2/controller_state.json')
    if control.exists():
        state = json.loads(control.read_text())
        print('Event:',state['event'],'Hours left:',state['remaining_hours'])
        print('Attacks:',state['active_attacks'],'GPT:',state['active_judge'])
        for row in state['configurations']:
            print(row['id'],row['attack_completed'],'/200; captions=',row['captions_complete'],'judge=',row['judge_complete'])
    log = CF_PACKAGE / 'overnight_setup.log'
    if log.exists():
        print(log.read_text(errors='replace')[-2500:])
    for log in Path('/marimo/cfvlm_n200_v2/parallel_tasks').glob('*/overnight_attack.log'):
        print(log.parent.name,log.read_text(errors='replace')[-1500:])
    print(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                   '--format=csv'],text=True))
cf_show_progress()
''')
    value = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
             'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'cell{i}', 'exec')
    (base / 'molab_cfvlm_overnight_v2.ipynb').write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')


def build():
    base = Path(__file__).resolve().parent
    files = {name: base / name for name in ('molab_cfvlm_setup.py', 'molab_hypothesis_campaign.py',
             'molab_remaining_tasks.py', 'molab_overnight.py', 'cfvlm_negatives.py',
             'kltn-attack-v4.3.3-h1-cfvlm.ipynb')}
    files['original_plan.json'] = base / 'molab_hypotheses_n200.json'
    files['molab_eval_completed.py'] = base.parent / 'molab_eval_completed.py'
    files['text_gen_eval_v3.1.ipynb'] = base.parent / 'text_gen_eval_v3.1.ipynb'
    destination = base / 'molab_cfvlm_overnight_v2.zip'
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        def put(name, data):
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
        for name, path in files.items():
            put(name, path.read_bytes())
        put('checksums.json', json.dumps({name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for name, path in files.items()}, sort_keys=True).encode())
    notebook(base, hashlib.sha256(destination.read_bytes()).hexdigest())
    print(json.dumps({'file': str(destination), 'bytes': destination.stat().st_size,
                      'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}))
    return destination


if __name__ == '__main__':
    build()
