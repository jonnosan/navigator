"""scripts/encode_audio.py — CLI: audio → Code (cached, stored).

Idempotent under repeat: the audio file and the resulting Code are content-
addressed by sha256, so re-encoding the same wav is a no-op after the first
run.

Run:
    uv run python scripts/encode_audio.py /path/to/sample.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import soundfile as sf

# Ensure src/ is importable when running as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from navigator.codec import Codec, resolve_device  # noqa: E402
from navigator.settings import Settings, default_config_path  # noqa: E402
from navigator.store import Store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Encode a wav file into a Code.")
    parser.add_argument("audio_path", type=Path, help="Input audio file (wav).")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    settings = Settings.from_yaml(args.config)
    device = resolve_device(settings.device)
    store = Store(settings.paths)
    codec = Codec(settings.codec, device=device)

    audio, sr = sf.read(str(args.audio_path), dtype="float32", always_2d=False)
    clip = store.put_audio(audio, sr, source="input")
    code_tensor = codec.encode(audio, sr)
    code = store.put_code(code_tensor, clip_id=clip.id)

    print(f"clip={clip.id}  code={code.id}  frames={code.shape}  fr={code.frame_rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
