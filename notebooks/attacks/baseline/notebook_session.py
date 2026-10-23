"""UI and background launch shared by the Molab and Jupyter notebooks."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from baseline_runtime import credentials, read_json, write_json


class Session:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.flow = None
        self.root.mkdir(parents=True, exist_ok=True)
        from dotenv import load_dotenv
        load_dotenv('/marimo/.env', override=False)

    def auth_url(self):
        try:
            credentials()
            return None
        except Exception:
            pass
        from pydata_google_auth import auth
        from google_auth_oauthlib.flow import Flow
        client = {'web': dict(client_id=auth.WEBAPP_CLIENT_ID,
            client_secret=auth.WEBAPP_CLIENT_SECRET, auth_uri=auth.GOOGLE_AUTH_URI,
            token_uri=auth.GOOGLE_TOKEN_URI, redirect_uris=[auth.WEBAPP_REDIRECT_URI])}
        self.flow = Flow.from_client_config(client, scopes=['https://www.googleapis.com/auth/drive'],
            autogenerate_code_verifier=True)
        self.flow.redirect_uri = auth.WEBAPP_REDIRECT_URI
        url, self.state = self.flow.authorization_url(access_type='offline', prompt='consent')
        return url

    def connect(self, value):
        if not value:
            return
        if self.flow is None:
            raise RuntimeError('Run the authentication cell again to create a fresh OAuth flow.')
        code = value.strip()
        if code.startswith(('https://', 'http://')):
            query = parse_qs(urlparse(code).query)
            if query.get('state', [None])[0] != self.state:
                raise ValueError('OAuth state mismatch')
            code = query.get('code', [''])[0]
        self.flow.fetch_token(code=code)
        from pydata_google_auth.cache import ReadWriteCredentialsCache
        ReadWriteCredentialsCache().save(self.flow.credentials)
        self.flow = None
        print('Google Drive connected. Credentials stay outside the notebook and result archives.')

    @staticmethod
    def set_secret(name, value):
        if value and value.strip():
            os.environ[name] = value.strip()
            print(name + ': available (value hidden).')

    def auth_ui(self):
        import marimo as mo
        url = self.auth_url() if self.config['backup_drive'] else None
        panel = []
        if url:
            panel += [mo.md(f'[Kết nối Google Drive]({url}) — cấp quyền rồi dán URL kết quả vào ô bên dưới.'),
                mo.ui.text(kind='password', label='OAuth callback URL').form(
                    submit_button_label='Kết nối Drive', clear_on_submit=True, on_change=self.connect)]
        else:
            panel += [mo.md('**Drive đã kết nối**' if self.config['backup_drive'] else '**Drive backup disabled**')]
        for name in ('OPENAI_API_KEY', 'GEMINI_API_KEY'):
            if not os.environ.get(name):
                panel.append(mo.ui.text(kind='password', label=name).form(clear_on_submit=True,
                    submit_button_label='Lưu khóa cho phiên này',
                    on_change=lambda value, key=name: self.set_secret(key, value)))
            else:
                panel.append(mo.md(f'`{name}`: available'))
        return mo.vstack(panel)

    def launch(self):
        if self.config['backup_drive']:
            credentials()
        import json
        pid_file = self.root / 'controller_pid.json'
        if pid_file.exists():
            old = read_json(pid_file)
            try:
                raw = Path(f'/proc/{old["pid"]}/stat').read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                if fields[0] != 'Z' and fields[19] == old['start_ticks']:
                    print('Controller already running:', old['pid'])
                    return old['pid']
            except OSError:
                pass
        probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True)
        if probe.returncode:
            raise RuntimeError('Enable GPU in Configure compute first.')
        print(probe.stdout.strip())
        config_file = self.root / 'config.json'
        if config_file.exists():
            previous = read_json(config_file)
            immutable = ('n', 'epsilon', 'steps', 'alpha', 'methods', 'mpc', 'saga', 'model_revisions')
            if any(previous[k] != self.config[k] for k in immutable):
                raise ValueError('Hyperparameters changed. Choose a new RUN_NAME.')
        write_json(config_file, self.config)
        logs = self.root / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / 'controller.log').open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(self.root / 'baseline_runtime.py'), str(config_file)],
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        raw = Path(f'/proc/{child.pid}/stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        write_json(pid_file, dict(pid=child.pid, start_ticks=fields[19]))
        print('Started/resumed controller:', child.pid)
        return child.pid

    def status(self):
        status = self.root / 'status.json'
        rows = []
        for method in self.config['methods']:
            rows.append(dict(method=method, completed=len(list(
                (self.root / 'results' / method / 'samples').glob('*/complete.json'))), expected=self.config['n']))
        return dict(stage=read_json(status) if status.exists() else {'stage': 'ready'}, attacks=rows)

    def monitor(self):
        import marimo as mo
        data = self.status()
        panels = [mo.md(f'**Trạng thái:** `{data["stage"].get("stage")}`'),
            mo.ui.table(data['attacks'], selection=None)]
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], capture_output=True, text=True)
        if gpu.returncode == 0:
            panels.append(mo.plain_text(gpu.stdout.strip()))
        log_panels = {}
        for log in sorted((self.root / 'logs').glob('*.log')):
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 3000))
                log_panels[log.name] = mo.plain_text(stream.read().decode('utf-8', errors='replace'))
        if log_panels:
            panels.append(mo.accordion(log_panels))
        panels.append(mo.md('**Drive:** [Mở thư mục checkpoint](https://drive.google.com/drive/folders/'
            + self.config['drive_parent_id'] + ')'))
        return mo.vstack(panels)
