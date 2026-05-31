"""api.py — FastAPI surface (M7).

Endpoints (planned):
  POST /encode          audio -> Code
  POST /generate        GenerationRequest -> [GenerationResult]
  POST /like-this       clip_id (+ optional segment) -> queues a "like" navigation
  POST /ab/pair         returns the next cold-start A/B pair
  POST /events          record a ReadingEvent
  GET  /status          {mode: cold|warm, confidence, events_count, ...}

NO auth here. A reverse proxy handles it. Bind 127.0.0.1 only.
"""

from __future__ import annotations


def _todo_m7() -> None:
    raise NotImplementedError("api.py lands in M7")
