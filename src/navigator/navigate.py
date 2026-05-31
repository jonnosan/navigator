"""navigate.py — code-space operations (M2 + M6).

Pure code-space ops. No audio in/out here — those are codec.py's job.

M2: noise, interpolate, splice, diversity sampling.
M6: reader_gradient_step — preference-weighted move toward high-value codes.
    This is the synthesis: human gestures (training the Reader) ALSO point a
    direction in code space because every candidate's Code is known.
"""

from __future__ import annotations


def _todo_m2() -> None:
    raise NotImplementedError("navigate.py lands in M2 (and M6 for reader_gradient_step)")
