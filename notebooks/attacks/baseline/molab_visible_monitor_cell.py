# Visible progress and log panel, including in kiosk/presentation mode.
refresh.value
import subprocess as _live_sp
import json as _live_json
_live_status = session.status()
_live_panels = [
    mo.md('## Tiến độ baseline\nTrạng thái: **' + _live_status['stage'].get('stage', 'ready') + '** · ' + _live_status['stage'].get('model', _live_status['stage'].get('method', '')) + '\n\nTự cập nhật mỗi 30 giây.'),
    refresh,
    mo.ui.table(_live_status['attacks'], selection=None),
]
_live_eval_rows = []
for _live_eval_dir in sorted((WORK_ROOT/'evaluation').glob('*')):
    if not _live_eval_dir.is_dir():
        continue
    _live_eval_row = {'model': _live_eval_dir.name, 'captions_ok': 0, 'scores_ok': 0, 'errors': 0}
    for _live_kind in ['captions', 'scores']:
        _live_result = _live_eval_dir/(_live_kind+'.json')
        if _live_result.exists():
            _live_records = _live_json.loads(_live_result.read_text())
            _live_eval_row[_live_kind+'_ok'] = sum(r.get('status') == 'ok' for r in _live_records.values())
            _live_eval_row['errors'] += sum(r.get('status') == 'error' for r in _live_records.values())
    _live_eval_rows.append(_live_eval_row)
if _live_eval_rows:
    _live_panels += [mo.md('### Đánh giá VLM / GPT scores'), mo.ui.table(_live_eval_rows, selection=None)]
_live_gpu = _live_sp.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'], capture_output=True, text=True)
_live_panels += [mo.md('### GPU'), mo.plain_text(_live_gpu.stdout.strip())]
for _live_name in ['evaluation.log', 'saga.log', 'controller.log', 'storage_watchdog.log']:
    _live_path = WORK_ROOT/'logs'/_live_name
    if _live_path.exists():
        with _live_path.open('rb') as _live_file:
            _live_file.seek(max(0, _live_path.stat().st_size-2200))
            _live_tail = _live_file.read().decode('utf8', errors='replace')
        _live_tail = '\n'.join(line for line in _live_tail.replace('\r','\n').splitlines() if line.strip())[-2200:]
        _live_panels += [mo.md('### ' + _live_name), mo.plain_text(_live_tail)]
mo.vstack(_live_panels)
