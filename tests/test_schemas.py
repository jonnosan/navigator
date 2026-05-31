"""test_schemas.py — locks the load-bearing parts of schemas.py.

These tests exist less to catch typos than to PIN the design:

  - A ReadingEvent must reference both clip_id AND code_id (the link from
    judgement to a point in code space).
  - PreferencePair carries both sides' codes (so reader_gradient_step can
    compute a direction).
  - GenerationResult separates the Reader-carved slice (`clip`) from the raw
    decoded candidate (`source_clip`).
  - Genre tags are optional everywhere — pipeline must validate with none.
  - PerturbOp is immutable in spirit: round-tripping JSON preserves it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from navigator import schemas
from navigator.schemas import (
    Clip,
    Code,
    GenerationRequest,
    GenerationResult,
    Lineage,
    PerturbOp,
    PreferencePair,
    ReadingEvent,
    ScoredSegment,
    Segment,
)


def test_module_import() -> None:
    """`from navigator import schemas` works (M0 acceptance)."""
    assert schemas.__name__.endswith("schemas")


def test_code_has_no_intrinsic_value() -> None:
    """A Code carries machinery, not value. clip_id can be unset for candidates."""
    c = Code(
        id="code_abc",
        clip_id=None,
        codec_name="dac",
        tokens_path="ab/code_abc.npz",
        shape=[1, 9, 512],
        dtype="int64",
        frame_rate=86.13,
    )
    assert c.clip_id is None
    dumped = c.model_dump()
    # No "value", "score", "quality" fields on Code — by design.
    assert not any(k in dumped for k in ("value", "score", "quality"))


def test_clip_default_lineage_is_empty() -> None:
    """Input clips have no lineage; the field is present but empty."""
    clip = Clip(
        id="clip_a",
        path="ab/clip_a.wav",
        duration_s=3.2,
        sample_rate=44100,
        source="input",
    )
    assert clip.lineage.parent_clip_ids == []
    assert clip.lineage.parent_code_ids == []
    assert clip.lineage.perturb_op_ids == []
    assert clip.genre_tags == []  # tags optional


def test_generated_clip_carries_full_provenance() -> None:
    """A generated clip must reference parents AND the op chain."""
    clip = Clip(
        id="clip_gen",
        path="cd/clip_gen.wav",
        duration_s=4.0,
        sample_rate=44100,
        source="generated",
        lineage=Lineage(
            parent_clip_ids=["clip_a"],
            parent_code_ids=["code_a"],
            perturb_op_ids=["op_1", "op_2"],
        ),
    )
    assert clip.lineage.perturb_op_ids == ["op_1", "op_2"]


def test_reading_event_anchors_judgement_to_code() -> None:
    """The load-bearing invariant: a ReadingEvent links clip AND code.

    Without code_id, "I like this" cannot become a direction in code space.
    """
    ev = ReadingEvent(
        id="ev_1",
        clip_id="clip_gen",
        code_id="code_cand_42",  # the candidate code whose decode we were reading
        gesture="more_like_this",
        segment=Segment(start_s=4.5, end_s=9.0),
    )
    assert ev.code_id == "code_cand_42"
    assert ev.segment is not None and ev.segment.duration_s() == pytest.approx(4.5)

    # code_id is REQUIRED — removing it must fail validation.
    with pytest.raises(ValidationError):
        ReadingEvent(  # type: ignore[call-arg]
            id="ev_bad",
            clip_id="clip_gen",
            gesture="keep",
        )


def test_preference_pair_carries_both_codes() -> None:
    """A pair without both codes can't seed reader_gradient_step."""
    pair = PreferencePair(
        id="pair_1",
        preferred_event_id="ev_pref",
        dispreferred_event_id="ev_disp",
        preferred_clip_id="clip_p",
        dispreferred_clip_id="clip_d",
        preferred_code_id="code_p",
        dispreferred_code_id="code_d",
        derived_from="ab_choice",
    )
    assert pair.preferred_code_id and pair.dispreferred_code_id


def test_perturb_op_records_anchors_and_produced_code() -> None:
    op = PerturbOp(
        id="op_xyz",
        type="interpolate",
        params={"alpha": 0.4},
        anchor_code_ids=["code_a", "code_b"],
        produced_code_id="code_mid",
    )
    # Round-trip preserves the op exactly — provenance must be stable.
    again = PerturbOp.model_validate_json(op.model_dump_json())
    assert again == op


def test_generation_result_separates_slice_from_raw_decode() -> None:
    """The carved slice IS the song. The raw decode is provenance only."""
    req = GenerationRequest(mode="like", anchor_clip_ids=["clip_anchor"], strength=0.3)
    raw = Clip(
        id="clip_raw",
        path="ef/clip_raw.wav",
        duration_s=10.0,
        sample_rate=44100,
        source="generated",
    )
    carved = Clip(
        id="clip_slice",
        path="ef/clip_slice.wav",
        duration_s=3.5,
        sample_rate=44100,
        source="generated",
        lineage=Lineage(parent_clip_ids=["clip_raw"], parent_code_ids=["code_cand"]),
    )
    code = Code(
        id="code_cand",
        clip_id="clip_raw",
        codec_name="dac",
        tokens_path="ef/code_cand.npz",
        shape=[1, 9, 860],
        dtype="int64",
        frame_rate=86.13,
    )
    res = GenerationResult(
        request=req,
        clip=carved,
        source_clip=raw,
        code=code,
        perturb_op_chain=[
            PerturbOp(
                id="op_1",
                type="noise",
                params={"sigma": 0.05},
                anchor_code_ids=["code_anchor"],
                produced_code_id="code_cand",
            ),
        ],
        reader_segmentation=[
            ScoredSegment(segment=Segment(start_s=0.0, end_s=3.5), value_score=0.81),
            ScoredSegment(segment=Segment(start_s=3.5, end_s=10.0), value_score=0.42),
        ],
        overall_score=0.81,
        reader_mode="warm",
    )
    assert res.clip.id != res.source_clip.id
    assert res.clip.duration_s < res.source_clip.duration_s  # it's a slice
    # The candidate code points to the raw decode that came out of it.
    assert res.code.clip_id == res.source_clip.id


def test_genre_tags_are_optional_throughout() -> None:
    """Tags are weak features only — clip validates fine without them."""
    clip = Clip(
        id="c1",
        path="x/c1.wav",
        duration_s=1.0,
        sample_rate=44100,
        source="imported",
    )
    assert clip.genre_tags == []
