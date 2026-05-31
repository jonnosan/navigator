"""Schemas — the data model where NAVIGATOR's philosophy is enforced.

Re-read the README's 6-point stance before editing. In particular:

- `Code` is the DURABLE INVERTIBLE OBJECT. It is meaningful machinery, but it
  carries NO musical value on its own. Two readings of the same Code may yield
  different songs.

- `ReadingEvent.code_id` is the load-bearing link: every human (or model) gesture
  is anchored to the *code* whose decoded audio was being read. This is what
  turns "I like this part" into a navigable direction in code space.

- `GenerationResult.clip` is the Reader-CARVED SLICE, not the raw decode. The
  raw decode is preserved in `source_clip` for provenance; the Reader's
  segmentation and per-segment scores live alongside it. If you ever find
  yourself returning the source_clip as the song, you've collapsed the design.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------


class Segment(BaseModel):
    """A [start, end) time range on an audio clip, in seconds."""

    model_config = ConfigDict(frozen=True)

    start_s: NonNegativeFloat
    end_s: NonNegativeFloat

    def duration_s(self) -> float:
        return float(self.end_s) - float(self.start_s)


# ---------------------------------------------------------------------------
# Code — the durable invertible object (D's heart)
# ---------------------------------------------------------------------------

PerturbOpType = Literal[
    "noise",  # add scaled noise to tokens/latents
    "interpolate",  # blend two codes
    "mask_regen",  # blank a span; refill (requires generator)
    "splice",  # paste a span from another code
    "corpus_sample",  # draw from optional corpus-code model
    "reader_gradient",  # move toward preferred / away from dispreferred codes
    "identity",  # no-op (used by encode path / strength-0)
]


class PerturbOp(BaseModel):
    """An atomic move in code space.

    A PerturbOp records exactly how a `produced_code_id` came to be. Chained
    via `Lineage.perturb_op_ids`, these form the full provenance of every
    candidate. NEVER edit a stored PerturbOp — append a new one instead.
    """

    id: str  # content hash over (type, params, anchor_code_ids)
    type: PerturbOpType
    params: dict = Field(default_factory=dict)  # strength, span, k, etc.
    anchor_code_ids: list[str] = Field(default_factory=list)
    produced_code_id: str
    created_at: datetime = Field(default_factory=_utcnow)


class Code(BaseModel):
    """A code (tokens / latents) in the frozen codec's space.

    The DURABLE INVERTIBLE OBJECT — meaningful machinery, but carrying no
    musical value of its own. `clip_id` is None for candidate codes that have
    not yet been decoded; it is set once a decoded `Clip` exists.
    """

    id: str  # content hash of the token/latent tensor
    clip_id: str | None = None
    codec_name: str  # e.g. "dac" / "encodec"
    tokens_path: str  # on-disk reference (npz/safetensors), relative to code_cache_dir
    shape: list[int]
    dtype: str  # e.g. "int64", "float16"
    frame_rate: float  # codec frames per second
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Clip — audio on disk, with full lineage
# ---------------------------------------------------------------------------

ClipSource = Literal["input", "generated", "imported"]


class Lineage(BaseModel):
    """Provenance chain for a Clip.

    For an "input" clip (user-supplied audio), all fields are empty. For a
    "generated" clip, `parent_code_ids` holds the code(s) that seeded the
    perturbation chain, and `perturb_op_ids` holds the operations applied in
    order. `parent_clip_ids` records human-meaningful anchors (e.g. the clips
    selected via `--like` or `--interpolate`).
    """

    parent_clip_ids: list[str] = Field(default_factory=list)
    parent_code_ids: list[str] = Field(default_factory=list)
    perturb_op_ids: list[str] = Field(default_factory=list)


class Clip(BaseModel):
    """An audio file on disk (content-addressed)."""

    id: str  # sha256 of the audio file
    path: str  # relative to audio_dir
    duration_s: float
    sample_rate: int
    channels: int = 1
    source: ClipSource
    genre_tags: list[str] = Field(default_factory=list)  # optional weak feature only
    lineage: Lineage = Field(default_factory=Lineage)
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Reading — the act that constitutes value
# ---------------------------------------------------------------------------

Gesture = Literal[
    "play",  # opened/played (weak signal)
    "keep",  # explicit "save this"
    "skip",  # explicit "not this"
    "more_like_this",  # may carry a Segment payload
    "thumb_up",
    "thumb_down",
    "ab_choice",  # forced-choice cold-start vote (preferred=this)
    "scrub",  # cursor movement / focus (weak segment-level signal)
]

ReaderMode = Literal["cold", "warm"]
EventSource = Literal["human", "reader_model"]


class ReadingEvent(BaseModel):
    """A single act of reading.

    THE HEART of NAVIGATOR. Every gesture is anchored to:

    - `clip_id`: the audio the reader was reading.
    - `code_id`: **the code whose decoded audio is being read** — load-bearing.
      This is what turns the reader's judgement into a point in code space,
      and is what `reader_gradient_step` later uses to compute directions.

    Optional `segment` localises the gesture in time (essential for
    "more_like_this" and boundary-head training).
    """

    id: str
    clip_id: str
    code_id: str  # the code that produced the candidate being read — load-bearing
    segment: Segment | None = None
    gesture: Gesture
    source: EventSource = "human"
    reader_mode_at_event: ReaderMode | None = None  # snapshot of system mode
    reader_version: str | None = None  # which Reader checkpoint, if model-sourced
    counterpart_event_id: str | None = None  # for ab_choice / thumb pairs
    notes: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class PreferencePair(BaseModel):
    """A derived (preferred, dispreferred) pair used to train the Reader.

    Pairs record BOTH sides' codes — that is what lets `reader_gradient_step`
    move candidate codes toward preferred regions of the space.
    """

    id: str
    preferred_event_id: str
    dispreferred_event_id: str
    preferred_clip_id: str
    dispreferred_clip_id: str
    preferred_code_id: str
    dispreferred_code_id: str
    derived_from: Literal["ab_choice", "thumb_pair", "more_like_this", "keep_skip"]
    weight: float = 1.0
    created_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# Generation request / result
# ---------------------------------------------------------------------------

GenerationMode = Literal["like", "interpolate", "corpus", "free"]


class GenerationRequest(BaseModel):
    mode: GenerationMode
    anchor_clip_ids: list[str] = Field(default_factory=list)
    strength: float = 0.30  # 0 ≈ identity decode
    count: int = 4
    candidate_multiplier: int = 8
    use_generator: bool | None = None  # None = follow config
    use_reader_gradient: bool = True
    corpus_dir: str | None = None  # for mode="corpus"
    notes: str | None = None


class ScoredSegment(BaseModel):
    """A boundary the Reader carved, with its value score."""

    segment: Segment
    value_score: float


class GenerationResult(BaseModel):
    """A returned "song" — the Reader-CARVED SLICE plus full provenance.

    `clip` is the slice the Reader chose to return (the song). `source_clip`
    is the raw decoded audio it was carved from (kept for provenance and
    re-reading). `code` is the candidate code that decoded into source_clip
    — and is what subsequent reading events will reference.
    """

    request: GenerationRequest
    clip: Clip  # the carved slice — THE song
    source_clip: Clip  # the full decoded candidate
    code: Code  # the candidate code that produced source_clip
    perturb_op_chain: list[PerturbOp]  # how the code was reached
    reader_segmentation: list[ScoredSegment]  # all boundaries the Reader carved
    overall_score: float
    reader_mode: ReaderMode
    reader_version: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
