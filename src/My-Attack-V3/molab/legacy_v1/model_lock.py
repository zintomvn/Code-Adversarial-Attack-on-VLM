"""Immutable model identities shared by the unchanged MPC surrogate classes."""
import json
from pathlib import Path

MODEL_REVISIONS = json.loads((Path(__file__).parent / 'model_revisions.json').read_text())
