"""Small MoLab connection/status notebook; the attack runs in the separate Python CLI."""
import marimo

app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    return mo, Path


@app.cell
def _(mo):
    mo.md('''# My-Attack · Drive-backed N200

    The Python CLI runs attack → saved-PNG verification → four victim captions → GPT-score.
    Every stage commits checksum-verified output to Google Drive before proceeding.
    Folder: **KLTN_Molab_Hypotheses_n200_v432_h1 / My-Attack**.
    This dashboard does not launch model inference automatically.
    ''')
    return


@app.cell
def _():
    from gdrive_fsspec import GoogleDriveFileSystem
    drive_fs = GoogleDriveFileSystem(root_file_id='1daramuahKwdAVGQtsDZTzy1v0OSG-kiI',
                                    token='cache', skip_instance_cache=True)
    return (drive_fs,)


@app.cell
def _(Path, mo):
    status_path = Path('/marimo/outputs/My-Attack/MYATTACK-20261029-DRIVE-N200-R2/heartbeat.json')
    mo.plain_text(status_path.read_text() if status_path.exists() else 'Storage preflight/setup pending. No inference launched by dashboard.')
    return


if __name__ == '__main__':
    app.run()
