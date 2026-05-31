"""reader.py — the Reader (C). Value head + boundary head over frozen features.

The Reader CONSTITUTES texts (segments) and confers value. It does NOT merely
filter whole clips. If you ever reduce it to a yes/no gate on raw decodes, the
design has collapsed into plain D.

M4: value head + cold→warm gate.
M5: boundary head (segmentation).
"""

from __future__ import annotations


def _todo_m4() -> None:
    raise NotImplementedError("reader.py lands in M4 (value head) and M5 (boundary head)")
