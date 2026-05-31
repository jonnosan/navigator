"""test_codec.py — M1 acceptance.

Two acceptance claims:

1. The codec is the invertible map: `encode → decode ≈ input` by a
   log-spectral correlation threshold (`config.codec.roundtrip_corr_min` —
   0.90 for EnCodec, raise to 0.95 for DAC). If this falls, NAVIGATOR's
   foundation is unfounded.

2. The store round-trips Clip + Code + ReadingEvent with full provenance
   (and code_id references on every event — the load-bearing field).

These tests download real codec weights on first run (cached afterwards).
If the network or the weights repo are unavailable, the codec test is
skipped with a clear reason rather than failing silently.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from navigator.schemas import Code, ReadingEvent, Segment
from navigator.settings import PathsCfg, Settings, default_config_path
from navigator.store import Store

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_music_like_audio(duration_s: float = 1.5, sr: int = 44100, seed: int = 7) -> np.ndarray:
    """A musical-ish test signal: mixed harmonics + bandlimited noise + envelope.

    Neural codecs trained on music handle this well; pure sines are
    out-of-distribution and would give a misleading round-trip number.
    """
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    rng = np.random.RandomState(seed)
    chord = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.20 * np.sin(2 * np.pi * 277.18 * t)  # C#4 (major third)
        + 0.15 * np.sin(2 * np.pi * 329.63 * t)  # E4 (fifth)
        + 0.10 * np.sin(2 * np.pi * 440.0 * t)  # A4 (octave)
    )
    # Bandlimited noise — gives the codec broadband content to chew on.
    noise = 0.05 * rng.randn(len(t)).astype(np.float32)
    # Linear fade in/out so there's no click at the boundaries.
    fade_n = int(0.02 * sr)
    env = np.ones_like(t)
    env[:fade_n] = np.linspace(0, 1, fade_n)
    env[-fade_n:] = np.linspace(1, 0, fade_n)
    return ((chord + noise) * env).astype(np.float32)


def _spectral_correlation(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Pearson correlation of log-magnitude STFTs. Robust to small time shifts."""
    from scipy.signal import stft

    nperseg = 2048
    _, _, S1 = stft(a, fs=sr, nperseg=nperseg)
    _, _, S2 = stft(b, fs=sr, nperseg=nperseg)
    n = min(S1.shape[1], S2.shape[1])
    M1 = np.log(np.abs(S1[:, :n]).flatten() + 1e-8)
    M2 = np.log(np.abs(S2[:, :n]).flatten() + 1e-8)
    return float(np.corrcoef(M1, M2)[0, 1])


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings.from_yaml(default_config_path())


@pytest.fixture
def tmp_store(tmp_path: Path, settings: Settings) -> Store:
    paths = PathsCfg(
        data_dir=tmp_path,
        audio_dir=tmp_path / "audio",
        code_cache_dir=tmp_path / "codes",
        db_path=tmp_path / "navigator.sqlite",
    )
    s = Store(paths)
    yield s
    s.close()


@pytest.fixture(scope="session")
def codec(settings: Settings):
    """Real codec; skips if torch or the configured codec library is unavailable."""
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed (M1 deps)")
    backend = settings.codec.name
    required = {"encodec": "transformers", "dac": "dac"}.get(backend)
    if required and importlib.util.find_spec(required) is None:
        pytest.skip(f"{required} not installed (codec.name={backend}; M1 deps)")
    from navigator.codec import Codec

    try:
        return Codec(settings.codec, device="cpu")
    except Exception as e:  # weights unreachable / other one-off setup failure
        pytest.skip(f"codec could not be loaded: {e}")


# ---------------------------------------------------------------------------
# store round-trip — no codec weights needed
# ---------------------------------------------------------------------------


def test_store_round_trips_clip(tmp_store: Store) -> None:
    audio = _make_music_like_audio(duration_s=0.5, sr=44100)
    clip = tmp_store.put_audio(audio, 44100, source="input")
    loaded = tmp_store.get_clip(clip.id)
    assert loaded == clip
    audio_back, sr_back = tmp_store.load_audio(clip.id)
    assert sr_back == 44100
    assert audio_back.shape == audio.shape
    np.testing.assert_allclose(audio_back, audio, atol=1e-5)


def test_store_put_audio_is_idempotent(tmp_store: Store) -> None:
    audio = _make_music_like_audio(duration_s=0.5, sr=44100)
    a = tmp_store.put_audio(audio, 44100, source="input")
    b = tmp_store.put_audio(audio, 44100, source="input")
    assert a.id == b.id


def test_store_round_trips_reading_event(tmp_store: Store) -> None:
    audio = _make_music_like_audio(duration_s=0.5, sr=44100)
    clip = tmp_store.put_audio(audio, 44100, source="input")
    # Fake code row so the FK is satisfied — round-trips work without a real codec.
    fake_code = Code(
        id="code_fake_1",
        clip_id=clip.id,
        codec_name="dac",
        tokens_path="fa/code_fake_1.npz",
        shape=[1, 9, 43],
        dtype="int64",
        frame_rate=86.13,
    )
    tmp_store.conn.execute(
        "INSERT INTO codes "
        "(id, clip_id, codec_name, tokens_path, shape_json, dtype, frame_rate, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fake_code.id,
            fake_code.clip_id,
            fake_code.codec_name,
            fake_code.tokens_path,
            "[1,9,43]",
            fake_code.dtype,
            fake_code.frame_rate,
            fake_code.created_at.isoformat(),
        ),
    )
    tmp_store.conn.commit()

    event = ReadingEvent(
        id="ev_t1",
        clip_id=clip.id,
        code_id=fake_code.id,
        gesture="more_like_this",
        segment=Segment(start_s=0.1, end_s=0.4),
        source="human",
    )
    tmp_store.put_reading_event(event)
    back = tmp_store.get_reading_event(event.id)
    assert back is not None
    assert back.code_id == fake_code.id  # the load-bearing field survives
    assert back.segment is not None and back.segment.end_s == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# the invertible map — needs codec weights
# ---------------------------------------------------------------------------


def test_codec_round_trip_passes_threshold(codec, settings: Settings) -> None:
    """encode → decode → spectral correlation ≥ config threshold."""
    audio = _make_music_like_audio(duration_s=1.5, sr=codec.sample_rate)
    code_tensor = codec.encode(audio, codec.sample_rate)
    audio_back, sr_back = codec.decode(code_tensor)
    assert sr_back == codec.sample_rate

    # Trim to common length (decode may add a few samples of padding).
    n = min(len(audio), len(audio_back))
    corr = _spectral_correlation(audio[:n], audio_back[:n], codec.sample_rate)
    threshold = settings.codec.roundtrip_corr_min
    assert corr >= threshold, (
        f"spectral correlation {corr:.4f} < threshold {threshold:.4f} — "
        "if this fails, NAVIGATOR's invertible-map foundation is unfounded; "
        "investigate before lowering the threshold."
    )


def test_codec_strength_zero_is_identity(codec) -> None:
    """encode→decode of a code is deterministic and stable across calls.

    A precursor to M2's strength-0 = identity invariant.
    """
    audio = _make_music_like_audio(duration_s=0.5, sr=codec.sample_rate)
    c1 = codec.encode(audio, codec.sample_rate)
    c2 = codec.encode(audio, codec.sample_rate)
    # Same input → same tokens.
    assert np.array_equal(c1.tokens, c2.tokens)
    # Same code → same decode.
    a1, _ = codec.decode(c1)
    a2, _ = codec.decode(c1)
    np.testing.assert_allclose(a1, a2, atol=1e-6)


def test_codec_latents_have_expected_shape(codec) -> None:
    """`latents()` returns continuous features for the Reader (M4)."""
    audio = _make_music_like_audio(duration_s=0.5, sr=codec.sample_rate)
    code = codec.encode(audio, codec.sample_rate)
    feats = codec.latents(code)
    # [B, dim, T_frames] — same time axis as the codes.
    assert feats.ndim == 3
    assert feats.shape[0] == code.tokens.shape[0]  # batch
    assert feats.shape[-1] == code.tokens.shape[-1]  # time


def test_codec_store_full_pipeline(codec, tmp_store: Store) -> None:
    """encode → put_code → load_code_tensor → decode round-trips through disk."""
    audio = _make_music_like_audio(duration_s=0.5, sr=codec.sample_rate)
    clip = tmp_store.put_audio(audio, codec.sample_rate, source="input")
    code_tensor = codec.encode(audio, codec.sample_rate)
    code = tmp_store.put_code(code_tensor, clip_id=clip.id)

    # Idempotent.
    code2 = tmp_store.put_code(code_tensor, clip_id=clip.id)
    assert code.id == code2.id

    # Reload from disk + decode.
    reloaded = tmp_store.load_code_tensor(code.id, sample_rate=codec.sample_rate)
    assert np.array_equal(reloaded.tokens, code_tensor.tokens)
    audio_back, _ = codec.decode(reloaded)
    assert audio_back.shape[0] > 0
