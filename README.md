# NAVIGATOR

A Reader that navigates an invertible code space. Personal music generation, one
person can build/run/maintain on one machine.

> The code is durable machinery; value is made at reading.

See [`docs/architecture.md`](docs/architecture.md) for the C4 system
architecture (context, containers, components, key flows).

## The 6-point stance (the philosophy is load-bearing — re-read before design decisions)

1. **The code is a real, durable object — but it is meaningful *machinery*, never
   meaningful *music*.** The frozen codec gives a persistent, invertible,
   manipulable representation (encode/decode). A code, and even its decoded
   audio, carries NO musical value in itself.
2. **Value is constituted only at reading.** A decoded candidate becomes a *song
   with value* only when the Reader reads it — segments it and scores it. Two
   readings of the same decode may yield different songs.
3. **The Reader CONSTITUTES texts; it does not merely filter them.** Boundaries
   plus value confer existence on a song. If the Reader is ever reduced to a
   whole-clip yes/no gate, NAVIGATOR has collapsed into plain D.
4. **The human is the ultimate reader, and their reading is a direction in code
   space.** Every gesture (keep / skip / "more like this part" + time range /
   scrub / thumb) (a) trains the Reader and (b) — because every candidate's code
   is known — points where to move next in code space.
5. **We train almost nothing.** Codec and (optional) generator are frozen,
   downloaded. The only substantial trainable thing is the small Reader (value +
   boundary heads on frozen features). An optional tiny corpus-region model over
   codes is the only other trainable thing. All trainable artefacts fit in
   minutes on CPU/MPS.
6. **Start blank — taste comes into being through use.** No seed corpus of
   "liked" songs. The Reader starts empty and is cold-started via forced-choice
   reading. A corpus, if later imported, is read like anything else.

## The loop

```
                    The code is durable machinery; value is made at reading.

           ┌──────────────── COLD MODE (blank Reader) ────────────────┐
           │  navigate: sample DIVERSE codes (spread across space)     │  [D]
           │  decode → present FORCED CHOICE A/B to human              │  [D decode, human reads]
           │  choice = ReadingEvent + PreferencePair (records codes)   │  [C signal]
           │  → train Reader; when held-out pairwise acc clears gate → WARM
           └───────────────────────────────────────────────────────────┘

           ┌──────────────── WARM MODE ───────────────────────────────┐
 anchor/   │  encode anchor/corpus → code                              │  [D: invertible map]
 corpus ──►│  navigate: perturb/interpolate/mask_regen/corpus_sample   │  [D: K candidate codes]
 audio     │           + reader_gradient_step (move toward high value) │  ← SYNTHESIS
           │  decode each candidate code → audio                       │  [D]
           │  Reader.segment + Reader.score each                       │  [C: CONSTITUTE texts]
           │  rank; returned "song" = Reader-carved SLICE, not raw     │  [C]
           │  → human gestures (keep/skip/more_like_this/scrub/thumb)  │  [C: human reads]
           │       ├─ train Reader            (better judgement)       │
           │       └─ each gesture's CODE → next move in code space    │  ← SYNTHESIS payoff
           └───────────────────────────────────────────────────────────┘
```

## Status — M0 (skeleton)

Done:
- Layout per spec (`src/navigator/`, `tests/`, `web/`, `scripts/`, `data/`).
- `pyproject.toml` (uv-managed), `config.yaml`, `settings.py`.
- `schemas.py`: `Clip`, `Code`, `PerturbOp`, `Lineage`, `Segment`, `ReadingEvent`,
  `PreferencePair`, `GenerationRequest`, `GenerationResult`, `ScoredSegment`.
- Module stubs for codec / navigate / reader / generator / corpus_model /
  bootstrap / loop / store / api (each raises `NotImplementedError` until its
  milestone).
- `tests/test_schemas.py` + `tests/test_settings.py` pass; per-milestone test
  files exist but are `pytest.skip`-marked.

Not yet (per-milestone):
- M1 codec round-trip + store
- M2 code-space navigation
- M3 active-learning cold start
- M4 Reader value head + cold→warm gate
- M5 boundary head + coupled loop
- M6 `reader_gradient_step` (the synthesis)
- M7 interactive surface
- M8 zero-to-personalised walkthrough
- M9 (optional) corpus-code model + generator fill

## Run

```bash
cd /Users/jonno/src/navigator
uv sync --extra dev          # install pydantic, pyyaml, pytest, ruff
uv run pytest                # M0 acceptance: tests pass
uv run python -c "from navigator import schemas; print(schemas.__name__)"
```

## Philosophical stance, in a paragraph

The code is a durable mark on the world — encode is a real operation, the bytes
persist on disk, decode is invertible. But the mark has no intrinsic musical
value; it sits there as machinery, awaiting a reader. Value is congealed prior
reading: the Reader's judgement is exactly the residue of past acts of reading
(human-supplied at cold start, model-supplemented later). When the Reader reads
a fresh decode, it carves boundaries and confers value — that *is* the act that
brings a song into being. NAVIGATOR fuses CODEC (D: the world is a manipulable
representation) with READER (C: the text is constituted at reception) by noting
that, because every candidate's code is known, the Reader's judgement is also a
navigable gradient over the code space. "More like that" becomes a direction to
move in.
