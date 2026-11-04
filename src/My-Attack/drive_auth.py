"""One-time standard gdrive-fsspec user OAuth for a headless MoLab terminal."""
import builtins
import getpass
import json
from pathlib import Path


def main():
    # Same pydata-google-auth flow as gdrive-fsspec, with a headless browser redirect.
    # Never log the authorization code or cached OAuth credentials.
    original_print = builtins.print

    def auth_print(*args, **kwargs):
        text = ' '.join(str(arg) for arg in args)
        marker = 'Please visit this URL to authorize this application: '
        if marker in text:
            url = text.split(marker, 1)[1].strip()
            Path('/marimo/myattack_oauth_link.json').write_text(json.dumps({'authorization_url': url}), encoding='utf-8')
        original_print(*args, **kwargs)

    builtins.print = auth_print
    builtins.input = lambda prompt='': getpass.getpass(prompt)
    try:
        from gdrive_fsspec import GoogleDriveFileSystem
        fs = GoogleDriveFileSystem(root_file_id='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI',
                                  token='cache', skip_instance_cache=True,
                                  auth_kwargs={'use_local_webserver': False})
        metadata = fs.files.get(fileId='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI',
                                fields='id,name,capabilities(canAddChildren)').execute()
        original_print('DRIVE_AUTHENTICATED', json.dumps(metadata), flush=True)
    finally:
        builtins.print = original_print


if __name__ == '__main__':
    main()
