"""store.py — SQLite + content-addressed audio + code cache (M1).

Layout:
    audio_dir/<sha256[:2]>/<sha256>.wav     # content-addressed audio
    code_cache_dir/<code_id[:2]>/<code_id>.npz  # token tensors
    db_path                                  # sqlite metadata

All `put_*` methods are idempotent: re-storing identical bytes is a no-op.
Reading events and preference pairs use caller-supplied IDs; `INSERT OR
IGNORE` makes the call safe to repeat.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from navigator.codec import CodeTensor
from navigator.schemas import (
    Clip,
    ClipSource,
    Code,
    Lineage,
    PerturbOp,
    PreferencePair,
    ReadingEvent,
    Segment,
)
from navigator.settings import PathsCfg

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    duration_s REAL NOT NULL,
    sample_rate INTEGER NOT NULL,
    channels INTEGER NOT NULL,
    source TEXT NOT NULL,
    genre_tags_json TEXT NOT NULL DEFAULT '[]',
    lineage_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS codes (
    id TEXT PRIMARY KEY,
    clip_id TEXT,
    codec_name TEXT NOT NULL,
    tokens_path TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    dtype TEXT NOT NULL,
    frame_rate REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (clip_id) REFERENCES clips(id)
);

CREATE TABLE IF NOT EXISTS perturb_ops (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    params_json TEXT NOT NULL,
    anchor_code_ids_json TEXT NOT NULL,
    produced_code_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (produced_code_id) REFERENCES codes(id)
);

CREATE TABLE IF NOT EXISTS reading_events (
    id TEXT PRIMARY KEY,
    clip_id TEXT NOT NULL,
    code_id TEXT NOT NULL,
    segment_json TEXT,
    gesture TEXT NOT NULL,
    source TEXT NOT NULL,
    reader_mode_at_event TEXT,
    reader_version TEXT,
    counterpart_event_id TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (clip_id) REFERENCES clips(id),
    FOREIGN KEY (code_id) REFERENCES codes(id)
);

CREATE TABLE IF NOT EXISTS preference_pairs (
    id TEXT PRIMARY KEY,
    preferred_event_id TEXT NOT NULL,
    dispreferred_event_id TEXT NOT NULL,
    preferred_clip_id TEXT NOT NULL,
    dispreferred_clip_id TEXT NOT NULL,
    preferred_code_id TEXT NOT NULL,
    dispreferred_code_id TEXT NOT NULL,
    derived_from TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (preferred_event_id) REFERENCES reading_events(id),
    FOREIGN KEY (dispreferred_event_id) REFERENCES reading_events(id)
);

CREATE INDEX IF NOT EXISTS idx_reading_events_code ON reading_events(code_id);
CREATE INDEX IF NOT EXISTS idx_reading_events_clip ON reading_events(clip_id);
CREATE INDEX IF NOT EXISTS idx_codes_clip ON codes(clip_id);
"""


class Store:
    """Persistence: SQLite + content-addressed audio + code cache."""

    def __init__(self, paths: PathsCfg) -> None:
        self.paths = paths
        self.audio_dir = Path(paths.audio_dir)
        self.code_cache_dir = Path(paths.code_cache_dir)
        self.db_path = Path(paths.db_path)
        for d in (self.audio_dir, self.code_cache_dir, self.db_path.parent):
            d.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- hashing ---------------------------------------------------------

    @staticmethod
    def audio_content_hash(audio: np.ndarray, sample_rate: int) -> str:
        h = hashlib.sha256()
        h.update(sample_rate.to_bytes(8, "little"))
        h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        return h.hexdigest()

    # ---- audio / clips ---------------------------------------------------

    def _audio_path(self, clip_id: str) -> Path:
        return self.audio_dir / clip_id[:2] / f"{clip_id}.wav"

    def put_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        source: ClipSource = "input",
        lineage: Lineage | None = None,
        genre_tags: list[str] | None = None,
    ) -> Clip:
        clip_id = self.audio_content_hash(audio, sample_rate)
        path = self._audio_path(clip_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), audio, sample_rate, subtype="FLOAT")
        channels = 1 if audio.ndim == 1 else int(audio.shape[-1])
        clip = Clip(
            id=clip_id,
            path=str(path.relative_to(self.audio_dir)),
            duration_s=float(audio.shape[0] if audio.ndim == 1 else audio.shape[0])
            / float(sample_rate),
            sample_rate=int(sample_rate),
            channels=channels,
            source=source,
            genre_tags=genre_tags or [],
            lineage=lineage or Lineage(),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO clips "
            "(id, path, duration_s, sample_rate, channels, source, "
            " genre_tags_json, lineage_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                clip.id,
                clip.path,
                clip.duration_s,
                clip.sample_rate,
                clip.channels,
                clip.source,
                json.dumps(clip.genre_tags),
                clip.lineage.model_dump_json(),
                clip.created_at.isoformat(),
            ),
        )
        self.conn.commit()
        return clip

    def get_clip(self, clip_id: str) -> Clip | None:
        row = self.conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        return _row_to_clip(row) if row else None

    def load_audio(self, clip_id: str) -> tuple[np.ndarray, int]:
        clip = self.get_clip(clip_id)
        if clip is None:
            raise KeyError(clip_id)
        audio, sr = sf.read(str(self.audio_dir / clip.path), dtype="float32")
        return audio, int(sr)

    # ---- codes -----------------------------------------------------------

    def _code_path(self, code_id: str) -> Path:
        return self.code_cache_dir / code_id[:2] / f"{code_id}.npz"

    def put_code(self, tensor: CodeTensor, *, clip_id: str | None = None) -> Code:
        code_id = tensor.content_hash()
        path = self._code_path(code_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(path), tokens=tensor.tokens)
        code = Code(
            id=code_id,
            clip_id=clip_id,
            codec_name=tensor.codec_name,
            tokens_path=str(path.relative_to(self.code_cache_dir)),
            shape=list(tensor.tokens.shape),
            dtype=tensor.dtype,
            frame_rate=tensor.frame_rate,
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO codes "
            "(id, clip_id, codec_name, tokens_path, shape_json, dtype, frame_rate, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                code.id,
                code.clip_id,
                code.codec_name,
                code.tokens_path,
                json.dumps(code.shape),
                code.dtype,
                code.frame_rate,
                code.created_at.isoformat(),
            ),
        )
        # If we now know which clip this code belongs to, backfill.
        if clip_id is not None:
            self.conn.execute(
                "UPDATE codes SET clip_id = ? WHERE id = ? AND clip_id IS NULL",
                (clip_id, code.id),
            )
        self.conn.commit()
        return code

    def get_code(self, code_id: str) -> Code | None:
        row = self.conn.execute("SELECT * FROM codes WHERE id = ?", (code_id,)).fetchone()
        return _row_to_code(row) if row else None

    def load_code_tensor(self, code_id: str, sample_rate: int) -> CodeTensor:
        """Reload a stored Code as a CodeTensor.

        `sample_rate` must be supplied by the caller (typically from the active
        codec) — sample_rate is not stored per-code, since one codec is assumed.
        """
        code = self.get_code(code_id)
        if code is None:
            raise KeyError(code_id)
        npz = np.load(str(self.code_cache_dir / code.tokens_path))
        return CodeTensor(
            tokens=npz["tokens"],
            frame_rate=code.frame_rate,
            codec_name=code.codec_name,
            sample_rate=sample_rate,
            dtype=code.dtype,
        )

    # ---- perturb ops -----------------------------------------------------

    def put_perturb_op(self, op: PerturbOp) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO perturb_ops "
            "(id, type, params_json, anchor_code_ids_json, produced_code_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                op.id,
                op.type,
                json.dumps(op.params),
                json.dumps(op.anchor_code_ids),
                op.produced_code_id,
                op.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    # ---- reading events / preference pairs -------------------------------

    def put_reading_event(self, event: ReadingEvent) -> None:
        seg_json = (
            json.dumps({"start_s": event.segment.start_s, "end_s": event.segment.end_s})
            if event.segment is not None
            else None
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO reading_events "
            "(id, clip_id, code_id, segment_json, gesture, source, "
            " reader_mode_at_event, reader_version, counterpart_event_id, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.id,
                event.clip_id,
                event.code_id,
                seg_json,
                event.gesture,
                event.source,
                event.reader_mode_at_event,
                event.reader_version,
                event.counterpart_event_id,
                event.notes,
                event.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def get_reading_event(self, event_id: str) -> ReadingEvent | None:
        row = self.conn.execute("SELECT * FROM reading_events WHERE id = ?", (event_id,)).fetchone()
        return _row_to_reading_event(row) if row else None

    def put_preference_pair(self, pair: PreferencePair) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO preference_pairs "
            "(id, preferred_event_id, dispreferred_event_id, "
            " preferred_clip_id, dispreferred_clip_id, "
            " preferred_code_id, dispreferred_code_id, "
            " derived_from, weight, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pair.id,
                pair.preferred_event_id,
                pair.dispreferred_event_id,
                pair.preferred_clip_id,
                pair.dispreferred_clip_id,
                pair.preferred_code_id,
                pair.dispreferred_code_id,
                pair.derived_from,
                pair.weight,
                pair.created_at.isoformat(),
            ),
        )
        self.conn.commit()

    def list_preference_pairs(self) -> list[PreferencePair]:
        rows = self.conn.execute("SELECT * FROM preference_pairs ORDER BY created_at").fetchall()
        return [_row_to_preference_pair(r) for r in rows]


# ---- row → pydantic helpers ----------------------------------------------


def _row_to_clip(row: sqlite3.Row) -> Clip:
    return Clip(
        id=row["id"],
        path=row["path"],
        duration_s=row["duration_s"],
        sample_rate=row["sample_rate"],
        channels=row["channels"],
        source=row["source"],
        genre_tags=json.loads(row["genre_tags_json"]),
        lineage=Lineage.model_validate_json(row["lineage_json"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_code(row: sqlite3.Row) -> Code:
    return Code(
        id=row["id"],
        clip_id=row["clip_id"],
        codec_name=row["codec_name"],
        tokens_path=row["tokens_path"],
        shape=json.loads(row["shape_json"]),
        dtype=row["dtype"],
        frame_rate=row["frame_rate"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_reading_event(row: sqlite3.Row) -> ReadingEvent:
    seg = json.loads(row["segment_json"]) if row["segment_json"] else None
    return ReadingEvent(
        id=row["id"],
        clip_id=row["clip_id"],
        code_id=row["code_id"],
        segment=Segment(**seg) if seg else None,
        gesture=row["gesture"],
        source=row["source"],
        reader_mode_at_event=row["reader_mode_at_event"],
        reader_version=row["reader_version"],
        counterpart_event_id=row["counterpart_event_id"],
        notes=row["notes"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_preference_pair(row: sqlite3.Row) -> PreferencePair:
    return PreferencePair(
        id=row["id"],
        preferred_event_id=row["preferred_event_id"],
        dispreferred_event_id=row["dispreferred_event_id"],
        preferred_clip_id=row["preferred_clip_id"],
        dispreferred_clip_id=row["dispreferred_clip_id"],
        preferred_code_id=row["preferred_code_id"],
        dispreferred_code_id=row["dispreferred_code_id"],
        derived_from=row["derived_from"],
        weight=row["weight"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )
