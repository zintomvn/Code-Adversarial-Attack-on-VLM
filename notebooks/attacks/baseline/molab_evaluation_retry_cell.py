# Resume evaluation with bounded Drive transport retries.
import hashlib as _retry_hash, json as _retry_json
_retry_dir = WORK_ROOT/'transport_support'
_retry_dir.mkdir(exist_ok=True)
_retry_source = "\"\"\"Optional bounded retries for Drive transport failures; no algorithm changes.\"\"\"\nimport os\n\nif os.environ.get('BASELINE_DRIVE_TRANSPORT_RETRY') == '1':\n    import functools\n    import http.client\n    import ssl\n    import time\n\n    import httplib2\n    from google_auth_httplib2 import AuthorizedHttp\n    from googleapiclient.discovery import build\n    import baseline_runtime\n\n    _transport_errors = (TimeoutError, ConnectionError, ssl.SSLError,\n                         http.client.RemoteDisconnected, httplib2.ServerNotFoundError)\n\n    def _wrap_drive_method(original):\n        @functools.wraps(original)\n        def retry(self, *args, **kwargs):\n            for attempt in range(4):\n                try:\n                    return original(self, *args, **kwargs)\n                except _transport_errors as exc:\n                    if attempt == 3:\n                        raise\n                    print('DRIVE_TRANSPORT_RETRY', original.__name__,\n                          attempt + 1, type(exc).__name__, flush=True)\n                    time.sleep(min(5 * 2 ** attempt, 20))\n                    self.api = build('drive', 'v3', cache_discovery=False,\n                        http=AuthorizedHttp(baseline_runtime.credentials(),\n                                            http=httplib2.Http(timeout=180)))\n        return retry\n\n    for _method in ('children', 'put', 'download'):\n        setattr(baseline_runtime.Drive, _method,\n                _wrap_drive_method(getattr(baseline_runtime.Drive, _method)))\n\r\n"
(_retry_dir/'sitecustomize.py').write_text(_retry_source, encoding='utf8')
os.environ['BASELINE_DRIVE_TRANSPORT_RETRY']='1'
_retry_paths=[str(_retry_dir.resolve()),str(WORK_ROOT.resolve())]
os.environ['PYTHONPATH']=os.pathsep.join(_retry_paths+[p for p in os.environ.get('PYTHONPATH','').split(os.pathsep) if p and p not in _retry_paths])
print('Transport hook SHA256:', _retry_hash.sha256((_retry_dir/'sitecustomize.py').read_bytes()).hexdigest())
for _retry_model_dir in sorted((WORK_ROOT/'evaluation').glob('*')):
    _retry_summary=_retry_model_dir/'summary.json'
    if _retry_summary.exists():
        print(_retry_model_dir.name, _retry_json.loads(_retry_summary.read_text()))
session.launch()
