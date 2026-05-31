"""codec.py — frozen DAC/EnCodec adapter (M1).

The invertible map. Must expose `encode(audio) -> Code`, `decode(Code) -> audio`,
and `latents(Code) -> features`. Round-trip ≈ identity is testable proof the
foundation is sound; see test_codec.py.

NEVER fine-tune. If a future need pushes you to, stop and re-read the README.
"""

from __future__ import annotations


def _todo_m1() -> None:
    raise NotImplementedError("codec.py lands in M1")
