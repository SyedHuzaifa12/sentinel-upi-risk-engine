"""Loads the Phase 0 synthetic UPI event stream for Phase 2 training.

Kept separate from load.py (the legacy CSV loader for the RandomForest
path) -- that module and its callers are untouched by this phase.
"""
import json

from feature_lib.event import UPIEvent

from ..utils.paths import SYNTHETIC_EVENTS_PATH


def load_synthetic_events(path=SYNTHETIC_EVENTS_PATH) -> list:
    """Returns UPIEvents in ascending timestamp order (as written by the
    generator -- not re-sorted here, so a corrupt/reordered file would
    surface as a temporal-integrity assertion failure downstream rather
    than being silently masked)."""
    events = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            events.append(UPIEvent(**json.loads(line)))
    return events
