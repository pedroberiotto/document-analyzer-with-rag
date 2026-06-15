from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / "results" / "eval.db"

_DDL = """
CREATE TABLE IF NOT EXISTS eval_field_results (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_run_id          TEXT    NOT NULL,
    created_at           TEXT    NOT NULL,
    strategy             TEXT    NOT NULL,
    reranker             TEXT    NOT NULL,
    k                    INTEGER,
    document_id          TEXT    NOT NULL,
    schema_name          TEXT    NOT NULL,
    field_name           TEXT    NOT NULL,
    predicted_value      TEXT,
    ground_truth         TEXT,
    confidence           REAL,
    field_match          REAL,
    retrieval_precision  REAL,
    faithfulness         REAL,
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    cost_usd             REAL,
    latency_s            REAL,
    n_sources            INTEGER
);

CREATE TABLE IF NOT EXISTS eval_run_summaries (
    eval_run_id          TEXT    PRIMARY KEY,
    created_at           TEXT    NOT NULL,
    strategy             TEXT    NOT NULL,
    reranker             TEXT    NOT NULL,
    k                    INTEGER,
    n_documents          INTEGER,
    n_fields_total       INTEGER,
    n_fields_annotated   INTEGER,
    field_match          REAL,
    retrieval_precision  REAL,
    faithfulness         REAL,
    avg_confidence       REAL,
    avg_latency_s        REAL,
    total_tokens         INTEGER,
    total_cost_usd       REAL
);

CREATE INDEX IF NOT EXISTS idx_efr_run   ON eval_field_results(eval_run_id);
CREATE INDEX IF NOT EXISTS idx_efr_strat ON eval_field_results(strategy, reranker);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvalDB:
    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_DB_PATH))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def insert_field_result(
        self,
        *,
        eval_run_id: str,
        strategy: str,
        reranker: str,
        k: int,
        document_id: str,
        schema_name: str,
        field_name: str,
        predicted_value: str | None,
        ground_truth: str | None,
        confidence: float,
        field_match: float,
        retrieval_precision: float,
        faithfulness: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_s: float,
        n_sources: int,
    ) -> None:
        self._conn.execute(
            """INSERT INTO eval_field_results
               (eval_run_id, created_at, strategy, reranker, k,
                document_id, schema_name, field_name,
                predicted_value, ground_truth, confidence,
                field_match, retrieval_precision, faithfulness,
                prompt_tokens, completion_tokens, cost_usd, latency_s, n_sources)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eval_run_id,
                _now(),
                strategy,
                reranker,
                k,
                document_id,
                schema_name,
                field_name,
                predicted_value,
                ground_truth,
                confidence,
                field_match,
                retrieval_precision,
                faithfulness,
                prompt_tokens,
                completion_tokens,
                cost_usd,
                latency_s,
                n_sources,
            ),
        )
        self._conn.commit()

    def upsert_summary(self, summary: dict[str, Any]) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO eval_run_summaries
               (eval_run_id, created_at, strategy, reranker, k,
                n_documents, n_fields_total, n_fields_annotated,
                field_match, retrieval_precision, faithfulness,
                avg_confidence, avg_latency_s, total_tokens, total_cost_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                summary["eval_run_id"],
                _now(),
                summary["strategy"],
                summary["reranker"],
                summary["k"],
                summary["n_documents"],
                summary["n_fields_total"],
                summary["n_fields_annotated"],
                summary["field_match"],
                summary["retrieval_precision"],
                summary["faithfulness"],
                summary["avg_confidence"],
                summary["avg_latency_s"],
                summary["total_tokens"],
                summary["total_cost_usd"],
            ),
        )
        self._conn.commit()

    def get_latest_summaries(self) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM eval_run_summaries
               WHERE (strategy, reranker, created_at) IN (
                   SELECT strategy, reranker, MAX(created_at)
                   FROM eval_run_summaries
                   GROUP BY strategy, reranker
               )
               ORDER BY field_match DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_summaries(
        self,
        strategy: str | None = None,
        reranker: str | None = None,
    ) -> list[dict]:
        q = "SELECT * FROM eval_run_summaries"
        params: list = []
        clauses: list[str] = []
        if strategy:
            clauses.append("strategy = ?")
            params.append(strategy)
        if reranker:
            clauses.append("reranker = ?")
            params.append(reranker)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC"
        return [dict(r) for r in self._conn.execute(q, params).fetchall()]

    def get_field_results(self, eval_run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM eval_field_results WHERE eval_run_id = ? ORDER BY document_id, field_name",
            (eval_run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EvalDB:
        return self

    def __exit__(self, *_) -> None:
        self.close()
