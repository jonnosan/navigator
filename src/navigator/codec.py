"""codec.py — frozen codec adapter (M1).

The invertible map. Exposes `encode(audio) -> CodeTensor`,
`decode(CodeTensor) -> audio`, `latents(CodeTensor) -> features`. Round-trip
≈ identity is testable proof the foundation is sound — see test_codec.py.

Two backends are supported behind the same interface:

- **encodec** (default): Facebook's EnCodec via Hugging Face `transformers`.
  Light dep tree, runs anywhere torch does.
- **dac**: Descript Audio Codec — higher fidelity, heavier dep tree
  (pulls `descript-audiotools` + ipython transitively). Optional install:
  ``uv sync --extra dac``.

NEVER fine-tune. If a future need pushes that way, stop and re-read the
README. The same constraint applies to the optional generator (M9).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from navigator.settings import CodecCfg, DeviceCfg

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CodeTensor — in-memory companion to schemas.Code
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeTensor:
    """In-memory output of `Codec.encode`.

    The pydantic `schemas.Code` is the on-disk record (where this lives, what
    it's named, what produced it). `CodeTensor` is the data itself.

    Per the philosophy: this is durable machinery — meaningful, manipulable,
    invertible — but carries NO musical value on its own. The Reader is what
    confers value, after decode.
    """

    tokens: np.ndarray  # int64 RVQ codes; shape varies per backend (see below)
    frame_rate: float
    codec_name: str
    sample_rate: int  # the codec's native sample rate (decode output rate)
    dtype: str = "int64"

    def content_hash(self) -> str:
        """sha256 over (codec_name, frame_rate, tokens). Stable across runs."""
        h = hashlib.sha256()
        h.update(self.codec_name.encode())
        h.update(np.float64(self.frame_rate).tobytes())
        h.update(self.tokens.astype(np.int64, copy=False).tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(cfg: DeviceCfg) -> str:
    """Pick a torch device string from config. `auto` → cuda > mps > cpu."""
    pref = cfg.preference
    if pref != "auto":
        return pref
    import torch  # local import — torch is M1+ only

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Codec — the frozen invertible map (dispatching wrapper)
# ---------------------------------------------------------------------------


class Codec:
    """Frozen codec adapter. One instance per process.

    The codec is the ONLY component that maps audio↔code. No other module
    encodes or decodes audio. The model is loaded in `.eval()` mode and never
    has `requires_grad` flipped on.
    """

    def __init__(self, cfg: CodecCfg, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        if cfg.name == "encodec":
            self._impl: _Backend = _EncodecBackend(cfg, device)
        elif cfg.name == "dac":
            self._impl = _DacBackend(cfg, device)
        else:
            raise NotImplementedError(f"codec.name={cfg.name!r} not supported")
        log.info(
            "codec loaded: %s on %s (sample_rate=%d, frame_rate=%.2f)",
            cfg.name,
            device,
            self.sample_rate,
            self.frame_rate,
        )

    @property
    def sample_rate(self) -> int:
        return self._impl.sample_rate

    @property
    def frame_rate(self) -> float:
        return self._impl.frame_rate

    def encode(self, audio: np.ndarray, sample_rate: int) -> CodeTensor:
        """audio (mono or stereo, any length) → CodeTensor.

        Audio is converted to mono float32 at the codec's native sample rate
        before encoding.
        """
        audio = _to_mono(audio)
        if sample_rate != self.sample_rate:
            audio = _resample(audio, sample_rate, self.sample_rate)
        return self._impl.encode_mono_at_native_sr(audio)

    def decode(self, code: CodeTensor) -> tuple[np.ndarray, int]:
        """CodeTensor → (mono float32 audio, sample_rate)."""
        return self._impl.decode(code), self.sample_rate

    def latents(self, code: CodeTensor) -> np.ndarray:
        """CodeTensor → continuous feature tensor `[B, dim, T_frames]`.

        Used by the Reader (M4) as its feature input. Going through codes
        (rather than raw audio) makes features deterministic given a code,
        so reading is reproducible.
        """
        return self._impl.latents(code)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class _Backend:
    """Internal interface. Each backend owns its frozen model + sample rate."""

    sample_rate: int
    frame_rate: float

    def encode_mono_at_native_sr(self, audio: np.ndarray) -> CodeTensor: ...
    def decode(self, code: CodeTensor) -> np.ndarray: ...
    def latents(self, code: CodeTensor) -> np.ndarray: ...


class _EncodecBackend(_Backend):
    """EnCodec via Hugging Face transformers.

    Tokens have shape `[1, n_codebooks, T_frames]`. We carry exactly one
    "chunk" so we can pass them straight back to `model.decode` (which wants
    `[n_chunks, B, n_q, T]`).
    """

    def __init__(self, cfg: CodecCfg, device: str) -> None:
        import torch
        from transformers import AutoProcessor, EncodecModel

        self.device = device
        self.model = EncodecModel.from_pretrained(cfg.model).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        self.sample_rate = int(self.model.config.sampling_rate)
        # transformers exposes frame_rate on most checkpoints; fall back to derived.
        fr = getattr(self.model.config, "frame_rate", None)
        if fr is None:
            # EnCodec's hop is sample_rate / frame_rate; derive from hop if needed.
            hop = getattr(self.model.config, "chunk_length_s", None)
            fr = self.sample_rate / hop if hop else 75.0
        self.frame_rate = float(fr)
        self._torch = torch

    def encode_mono_at_native_sr(self, audio: np.ndarray) -> CodeTensor:
        inputs = self.processor(
            raw_audio=audio.astype(np.float32),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        with self._torch.no_grad():
            out = self.model.encode(input_values, padding_mask)
        # audio_codes: [n_chunks, B, n_q, T]. We keep a single-chunk view.
        codes = out.audio_codes.detach().cpu().numpy().astype(np.int64)
        # Store as [B, n_q, T] for compactness; reshape on decode.
        tokens = codes[0]  # first (only) chunk
        # Stash scales on the array's dtype isn't possible — keep them on the
        # backend instance for this process. They are constant per checkpoint.
        self._last_scales = out.audio_scales
        self._last_padding_mask = padding_mask
        return CodeTensor(
            tokens=tokens,
            frame_rate=self.frame_rate,
            codec_name="encodec",
            sample_rate=self.sample_rate,
        )

    def decode(self, code: CodeTensor) -> np.ndarray:
        codes_t = self._torch.from_numpy(code.tokens).to(self.device)
        if codes_t.ndim == 3:
            codes_t = codes_t.unsqueeze(0)  # [1, B, n_q, T]
        with self._torch.no_grad():
            audio_values = self.model.decode(
                codes_t,
                getattr(self, "_last_scales", [None]),
                getattr(self, "_last_padding_mask", None),
            )[0]
        return audio_values.squeeze().detach().cpu().numpy().astype(np.float32)

    def latents(self, code: CodeTensor) -> np.ndarray:
        """Embed each codebook index, sum across codebooks → [B, dim, T]."""
        codes_t = self._torch.from_numpy(code.tokens).to(self.device)  # [B, n_q, T]
        with self._torch.no_grad():
            # transformers' EncodecResidualVectorQuantizer.decode takes [B, n_q, T]
            # and returns [B, dim, T] (the summed codebook embeddings).
            embeddings = self.model.quantizer.decode(codes_t)
        return embeddings.detach().cpu().numpy()


class _DacBackend(_Backend):
    """Descript Audio Codec (44/24/16 kHz).

    Tokens have shape `[1, n_codebooks, T_frames]`.
    """

    def __init__(self, cfg: CodecCfg, device: str) -> None:
        import torch

        try:
            import dac
        except ImportError as e:  # pragma: no cover — install hint
            raise ImportError(
                "DAC backend requires the optional 'dac' extra: `uv sync --extra dac`. "
                "Or switch codec.name to 'encodec' in config.yaml."
            ) from e

        model_type = {44100: "44khz", 24000: "24khz", 16000: "16khz"}.get(cfg.sample_rate)
        if model_type is None:
            raise ValueError(
                f"DAC has no model at sample_rate={cfg.sample_rate}; expected 16000/24000/44100"
            )
        weights = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(str(weights)).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.sample_rate = int(self.model.sample_rate)
        self.frame_rate = float(self.model.sample_rate) / float(self.model.hop_length)
        self._torch = torch

    def encode_mono_at_native_sr(self, audio: np.ndarray) -> CodeTensor:
        x = (
            self._torch.from_numpy(audio.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        x = self.model.preprocess(x, self.sample_rate)
        with self._torch.no_grad():
            _, codes, _, _, _ = self.model.encode(x)
        tokens = codes.detach().cpu().numpy().astype(np.int64)
        return CodeTensor(
            tokens=tokens,
            frame_rate=self.frame_rate,
            codec_name="dac",
            sample_rate=self.sample_rate,
        )

    def decode(self, code: CodeTensor) -> np.ndarray:
        codes_t = self._torch.from_numpy(code.tokens).to(self.device)
        if codes_t.ndim == 2:
            codes_t = codes_t.unsqueeze(0)
        with self._torch.no_grad():
            z, _, _ = self.model.quantizer.from_codes(codes_t)
            y = self.model.decode(z)
        return y.squeeze().detach().cpu().numpy().astype(np.float32)

    def latents(self, code: CodeTensor) -> np.ndarray:
        codes_t = self._torch.from_numpy(code.tokens).to(self.device)
        if codes_t.ndim == 2:
            codes_t = codes_t.unsqueeze(0)
        with self._torch.no_grad():
            z, _, _ = self.model.quantizer.from_codes(codes_t)
        return z.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # soundfile returns [T, C]; some sources give [C, T]. Detect by shape.
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(axis=0)
        return audio.mean(axis=1)
    raise ValueError(f"unsupported audio shape {audio.shape}")


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)
