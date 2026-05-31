"""loop.py — the core loop (M5 + M6 + M8).

D supplies the moves, C supplies the judgement, and the Reader's judgement
becomes a navigable gradient over D's code space.

Modes:
  cold:  diverse codes -> decode -> forced A/B reading -> preference pairs
  warm:  encode anchor -> navigate (+ reader_gradient) -> decode candidates
         -> Reader.segment + score -> rank -> return Reader-carved SLICES
         -> human gestures train Reader AND set next direction in code space
"""

from __future__ import annotations


def _todo_m5() -> None:
    raise NotImplementedError("loop.py lands in M5 (warm path) and M6 (reader_gradient wiring)")
