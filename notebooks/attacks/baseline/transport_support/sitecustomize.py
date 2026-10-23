"""Optional bounded retries for Drive transport failures; no algorithm changes."""
import os

if os.environ.get('BASELINE_DRIVE_TRANSPORT_RETRY') == '1':
    import functools
    import http.client
    import ssl
    import time

    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build
    import baseline_runtime

    _transport_errors = (TimeoutError, ConnectionError, ssl.SSLError,
                         http.client.RemoteDisconnected, httplib2.ServerNotFoundError)

    def _wrap_drive_method(original):
        @functools.wraps(original)
        def retry(self, *args, **kwargs):
            for attempt in range(4):
                try:
                    return original(self, *args, **kwargs)
                except _transport_errors as exc:
                    if attempt == 3:
                        raise
                    print('DRIVE_TRANSPORT_RETRY', original.__name__,
                          attempt + 1, type(exc).__name__, flush=True)
                    time.sleep(min(5 * 2 ** attempt, 20))
                    self.api = build('drive', 'v3', cache_discovery=False,
                        http=AuthorizedHttp(baseline_runtime.credentials(),
                                            http=httplib2.Http(timeout=180)))
        return retry

    for _method in ('children', 'put', 'download'):
        setattr(baseline_runtime.Drive, _method,
                _wrap_drive_method(getattr(baseline_runtime.Drive, _method)))
