import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone

import structlog

from app.config import get_settings
from app.telemetry.pricing import compute_cost

log = structlog.get_logger(__name__)

_local = threading.local()


_run_buffer: dict[str, list[dict]] = defaultdict(list)
_buffer_lock = threading.Lock()

_DDL = """
CREATE TABLE IF NOT EXISTS field_extractions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    strategy          TEXT    NOT NULL,
    reranker          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    document_id       TEXT    NOT NULL,
    schema_name       TEXT    NOT NULL,
    field_name        TEXT    NOT NULL,
    predicted_value   TEXT,
    confidence        REAL,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens      INTEGER DEFAULT 0,
    cost_usd          REAL    DEFAULT 0,
    latency_s         REAL    DEFAULT 0,
    n_sources         INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS document_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT    NOT NULL,
    created_at        TEXT    NOT NULL,
    strategy          TEXT    NOT NULL,
    reranker          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    document_id       TEXT    NOT NULL,
    schema_name       TEXT    NOT NULL,
    n_fields          INTEGER DEFAULT 0,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens      INTEGER DEFAULT 0,
    cost_usd          REAL    DEFAULT 0,
    latency_s         REAL    DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_fe_run    ON field_extractions(run_id);
CREATE INDEX IF NOT EXISTS idx_fe_strat  ON field_extractions(strategy, reranker);
CREATE INDEX IF NOT EXISTS idx_dr_run    ON document_runs(run_id);
CREATE INDEX IF NOT EXISTS idx_dr_doc    ON document_runs(document_id);
"""


def _conn() -> sqlite3.Connection:
    if not getattr(_local, "conn", None):
        db_path = get_settings().telemetry_db
        db_path.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(db_path))
        c.row_factory = sqlite3.Row
        c.executescript(_DDL)
        c.commit()
        _local.conn = c
    return _local.conn


@dataclass
class ExtractionContext:
    strategy: str = "dense"
    reranker: str = "none"
    model: str = "gpt-4.1-mini"
    run_id: str = dc_field(default_factory=lambda: str(uuid.uuid4()))
    persist: bool = True


class Tracker:
    def __init__(self, persist: bool = True) -> None:
        self._persist = persist

    def record_field(
        self,
        ctx: ExtractionContext,
        document_id: str,
        schema_name: str,
        field_name: str,
        *,
        predicted_value: str | None,
        confidence: float,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
        n_sources: int,
    ) -> None:
        cost = compute_cost(ctx.model, prompt_tokens, completion_tokens)
        record = {
            "document_id": document_id,
            "schema_name": schema_name,
            "field_name": field_name,
            "predicted_value": predicted_value,
            "confidence": round(confidence, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "latency_s": round(latency_s, 3),
            "n_sources": n_sources,
        }

        with _buffer_lock:
            _run_buffer[ctx.run_id].append(record)

        log.info(
            "field_extracted",
            run_id=ctx.run_id,
            strategy=ctx.strategy,
            reranker=ctx.reranker,
            **record,
        )

        if self._persist and ctx.persist:
            _conn().execute(
                """INSERT INTO field_extractions
                   (run_id, created_at, strategy, reranker, model, document_id,
                    schema_name, field_name, predicted_value, confidence,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost_usd, latency_s, n_sources)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ctx.run_id,
                    _now(),
                    ctx.strategy,
                    ctx.reranker,
                    ctx.model,
                    document_id,
                    schema_name,
                    field_name,
                    predicted_value,
                    confidence,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    cost,
                    latency_s,
                    n_sources,
                ),
            )
            _conn().commit()

    def record_document(
        self,
        ctx: ExtractionContext,
        document_id: str,
        schema_name: str,
        *,
        n_fields: int,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float,
    ) -> None:
        cost = compute_cost(ctx.model, prompt_tokens, completion_tokens)

        log.info(
            "document_extracted",
            run_id=ctx.run_id,
            strategy=ctx.strategy,
            reranker=ctx.reranker,
            document_id=document_id,
            schema_name=schema_name,
            n_fields=n_fields,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6),
            latency_s=round(latency_s, 3),
        )

        if self._persist and ctx.persist:
            _conn().execute(
                """INSERT INTO document_runs
                   (run_id, created_at, strategy, reranker, model, document_id,
                    schema_name, n_fields, prompt_tokens, completion_tokens,
                    total_tokens, cost_usd, latency_s)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ctx.run_id,
                    _now(),
                    ctx.strategy,
                    ctx.reranker,
                    ctx.model,
                    document_id,
                    schema_name,
                    n_fields,
                    prompt_tokens,
                    completion_tokens,
                    prompt_tokens + completion_tokens,
                    cost,
                    latency_s,
                ),
            )
            _conn().commit()

    @staticmethod
    def get_run_records(run_id: str) -> list[dict]:
        with _buffer_lock:
            return list(_run_buffer.get(run_id, []))

    @staticmethod
    def pop_run_records(run_id: str) -> list[dict]:
        with _buffer_lock:
            return _run_buffer.pop(run_id, [])


_tracker: Tracker | None = None


def get_tracker() -> Tracker:
    global _tracker
    if _tracker is None:
        _tracker = Tracker(persist=True)
    return _tracker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
