"""store.py — SQLite + content-addressed audio + code cache (M1).

Schema mirrors schemas.py. Audio at audio_dir/<sha256[:2]>/<sha256>.wav.
Codes at code_cache_dir/<code_id>.npz (or .safetensors).
"""

from __future__ import annotations


def _todo_m1() -> None:
    raise NotImplementedError("store.py lands in M1")
