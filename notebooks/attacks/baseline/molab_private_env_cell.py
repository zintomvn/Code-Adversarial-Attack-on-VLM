# Private .env transfer: payload enters only through a password form, never notebook source.
import io as _env_io, base64 as _env_b64, json as _env_json, tempfile as _env_tmp
from pathlib import Path as _EnvPath
from dotenv import dotenv_values as _env_values, load_dotenv as _env_load

def _accept_private_env(payload):
    if not payload: return
    raw=_env_b64.b64decode(payload,validate=True)
    values=_env_values(stream=_env_io.StringIO(raw.decode('utf-8-sig')))
    if not values.get('OPENAI_API_KEY'): raise ValueError('OPENAI_API_KEY is missing or empty')
    dest=_EnvPath('/marimo/.env')
    fd,temp=_env_tmp.mkstemp(prefix='.env-transfer-',dir=str(dest.parent))
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,'wb') as stream: stream.write(raw)
        os.replace(temp,dest)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    _env_load(dest,override=True)
    report={'remote_file':str(dest),'file_bytes_match_upload':dest.read_bytes()==raw,'permissions':oct(dest.stat().st_mode & 0o777),'kernel_key_matches_env':os.environ.get('OPENAI_API_KEY')==values['OPENAI_API_KEY'],'processes':{}}
    for role,pointer in [('controller','controller_pid.json'),('watchdog','storage_watchdog_pid.json')]:
        path=WORK_ROOT/pointer
        if not path.exists(): continue
        saved=_env_json.loads(path.read_text())
        proc=_EnvPath('/proc')/str(saved['pid'])
        try:
            fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
            live=fields[0]!='Z' and fields[19]==saved['start_ticks']
            entries=(proc/'environ').read_bytes().split(b'\x00')
            key=next((item.split(b'=',1)[1].decode() for item in entries if item.startswith(b'OPENAI_API_KEY=')),'')
            report['processes'][role]={'alive':live,'key_present':bool(key),'key_matches_env':key==values['OPENAI_API_KEY']}
        except OSError:
            report['processes'][role]={'alive':False}
    (WORK_ROOT/'secret_transfer_status.json').write_text(_env_json.dumps(report,indent=2))
    print(_env_json.dumps(report,indent=2))

env_transfer_form=mo.ui.text(kind='password',label='Nạp file .env vào phiên Molab').form(submit_button_label='Lưu .env riêng tư',clear_on_submit=True,on_change=_accept_private_env)
mo.vstack([mo.md('Nạp cấu hình bí mật; không ghi khóa vào code hoặc kết quả thí nghiệm.'),env_transfer_form])

