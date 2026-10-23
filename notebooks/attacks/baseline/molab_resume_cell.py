# Run after the existing bootstrap and Drive authentication cells.
# Molab's isolated Python environments need an explicit project import path.
# This changes only the launch environment; author code and run identity stay fixed.
import os

_project_path = str(WORK_ROOT.resolve())
_existing_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
if _project_path not in _existing_paths:
    os.environ['PYTHONPATH'] = os.pathsep.join(
        [_project_path] + [part for part in _existing_paths if part]
    )
session.launch()
