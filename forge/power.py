"""Keep Windows from sleeping while a long run is in progress.

A real run once recorded a single model call lasting 9,310 seconds, consistent with the laptop sleeping mid-run.
SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) asks Windows not to sleep while this process runs; it
does not keep the display on and is released when the process exits. No-op on other systems."""
from __future__ import annotations

import contextlib
import os

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake(set_state=None):
    if set_state is None:
        if os.name != "nt":
            yield False
            return
        import ctypes
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
    try:
        set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        yield True
    finally:
        set_state(ES_CONTINUOUS)
