import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.extraction import extract_fields, get_llm  # noqa: E402
from app.ingestion import index_document  # noqa: E402
from app.models import ExtractionSchema  # noqa: E402
from app.retrieval import RerankerType, RetrieverStrategy, build_retriever  # noqa: E402
from app.telemetry import configure_logging  # noqa: E402
from app.telemetry.tracker import ExtractionContext, get_tracker  # noqa: E402
from eval.db import EvalDB  # noqa: E402
from eval.metrics import (  # noqa: E402
    compute_faithfulness,
    compute_field_match,
    compute_retrieval_precision,
)

EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR / "dataset"
GROUND_TRUTH_FILE = EVAL_DIR / "ground_truth.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

_PLACEHOLDER = {"FILL_IN", "FILL_IN_MANUALLY", ""}


def _load_ground_truth() -> dict[str, Any]:
    with GROUND_TRUTH_FILE.open() as f:
        return json.load(f)


def _build_schemas(gt: dict) -> dict[str, ExtractionSchema]:
    return {s["name"]: ExtractionSchema(**s) for s in gt["schemas"]}


def _eval_document(
    doc_gt: dict,
    schema: ExtractionSchema,
    strategy: RetrieverStrategy,
    reranker: RerankerType,
    k: int,
    eval_run_id: str,
    db: EvalDB,
) -> dict[str, Any]:
    doc_id = doc_gt["document_id"]

    # index_document caches by document_id: re-embeds only if not already indexed.
    vectorstore = index_document(str(DATASET_DIR / doc_gt["pdf_filename"]), doc_id)

    llm = get_llm()
    retriever = build_retriever(
        vectorstore,
        strategy=strategy,
        reranker=reranker,
        llm=llm if strategy == RetrieverStrategy.AGENTIC else None,
        k=k,
    )

    run_id = f"{eval_run_id}::{doc_id}"
    ctx = ExtractionContext(
        strategy=strategy.value,
        reranker=reranker.value,
        model=get_settings().active_llm_model,
        run_id=run_id,
        persist=False,
    )

    t0 = time.perf_counter()
    result = extract_fields(doc_id, schema, retriever, ctx=ctx)
    doc_latency = time.perf_counter() - t0

    usage = {r["field_name"]: r for r in get_tracker().pop_run_records(run_id)}

    field_rows: list[dict] = []
    for fr in result.fields:
        gt_value = doc_gt["fields"].get(fr.name)
        if isinstance(gt_value, str) and gt_value.upper() in _PLACEHOLDER:
            gt_value = None

        u = usage.get(fr.name, {})
        row = {
            "document_id": doc_id,
            "schema_name": schema.name,
            "field_name": fr.name,
            "predicted_value": fr.value,
            "ground_truth": gt_value,
            "confidence": round(fr.confidence, 4),
            "field_match": compute_field_match(fr.value, gt_value),
            "retrieval_precision": compute_retrieval_precision(fr.sources, gt_value),
            "faithfulness": compute_faithfulness(fr.value, fr.sources),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "cost_usd": u.get("cost_usd", 0.0),
            "latency_s": u.get("latency_s", 0.0),
            "n_sources": len(fr.sources),
        }
        field_rows.append(row)
        db.insert_field_result(
            eval_run_id=eval_run_id, strategy=strategy.value, reranker=reranker.value, k=k, **row
        )

    return {
        "document_id": doc_id,
        "pdf_filename": doc_gt["pdf_filename"],
        "schema": schema.name,
        "latency_s": round(doc_latency, 3),
        "prompt_tokens": sum(r["prompt_tokens"] for r in field_rows),
        "completion_tokens": sum(r["completion_tokens"] for r in field_rows),
        "total_tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in field_rows),
        "total_cost_usd": round(sum(r["cost_usd"] for r in field_rows), 6),
        "fields": field_rows,
    }


def _aggregate(
    doc_results: list[dict],
    strategy: RetrieverStrategy,
    reranker: RerankerType,
    k: int,
    eval_run_id: str,
) -> dict:
    all_fields = [f for d in doc_results for f in d["fields"]]
    annotated = [f for f in all_fields if f["ground_truth"] is not None]

    def _mean(key: str, rows: list[dict]) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    return {
        "eval_run_id": eval_run_id,
        "strategy": strategy.value,
        "reranker": reranker.value,
        "k": k,
        "n_documents": len(doc_results),
        "n_fields_total": len(all_fields),
        "n_fields_annotated": len(annotated),
        "field_match": _mean("field_match", annotated),
        "retrieval_precision": _mean("retrieval_precision", annotated),
        "faithfulness": _mean("faithfulness", all_fields),
        "avg_confidence": _mean("confidence", all_fields),
        "avg_latency_s": round(sum(d["latency_s"] for d in doc_results) / len(doc_results), 3)
        if doc_results
        else 0,
        "total_tokens": sum(d["total_tokens"] for d in doc_results),
        "total_cost_usd": round(sum(d["total_cost_usd"] for d in doc_results), 6),
    }


def run_eval(
    strategy: RetrieverStrategy,
    reranker: RerankerType,
    output_path: Path,
    k: int = 5,
) -> dict[str, Any]:
    gt = _load_ground_truth()
    schemas = _build_schemas(gt)
    eval_run_id = str(uuid.uuid4())
    doc_results: list[dict] = []

    with EvalDB() as db:
        for doc_gt in gt["documents"]:
            pdf_path = DATASET_DIR / doc_gt["pdf_filename"]
            if not pdf_path.exists():
                print(f"  [skip] {doc_gt['pdf_filename']} not in eval/dataset/")
                continue

            schema = schemas.get(doc_gt["schema"])
            if schema is None:
                print(f"  [skip] unknown schema '{doc_gt['schema']}'")
                continue

            print(f"  → {doc_gt['document_id']} ({doc_gt['schema']})")
            doc_results.append(
                _eval_document(doc_gt, schema, strategy, reranker, k, eval_run_id, db)
            )

        if not doc_results:
            print("\nNo documents evaluated. Add PDFs to eval/dataset/ and fill ground_truth.json.")
            sys.exit(1)

        summary = _aggregate(doc_results, strategy, reranker, k, eval_run_id)
        db.upsert_summary(summary)

    output = {"summary": summary, "documents": doc_results}
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n  Results → {output_path}  |  eval_run_id={eval_run_id}")
    return output


def _print_summary(summary: dict) -> None:
    label = f"{summary['strategy']}+{summary['reranker']}"
    print(f"\n{'=' * 55}")
    print(f"  {label}  (k={summary.get('k', '?')})")
    print(f"{'=' * 55}")
    print(f"  Documents evaluated : {summary['n_documents']}")
    print(f"  Annotated fields    : {summary['n_fields_annotated']} / {summary['n_fields_total']}")
    print(f"  Field match         : {summary['field_match']:.3f}")
    print(f"  Retrieval precision : {summary['retrieval_precision']:.3f}")
    print(f"  Faithfulness        : {summary['faithfulness']:.3f}")
    print(f"  Avg confidence      : {summary['avg_confidence']:.3f}")
    print(f"  Avg latency         : {summary['avg_latency_s']:.2f} s")
    print(f"  Total tokens        : {summary['total_tokens']:,}")
    print(f"  Total cost          : ${summary['total_cost_usd']:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one retriever strategy.")
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in RetrieverStrategy],
        default=RetrieverStrategy.DENSE.value,
    )
    parser.add_argument(
        "--reranker", choices=[r.value for r in RerankerType], default=RerankerType.NONE.value
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = _parse_args()
    strategy = RetrieverStrategy(args.strategy)
    reranker = RerankerType(args.reranker)
    output_path = args.output or (RESULTS_DIR / f"{strategy.value}+{reranker.value}.json")

    print(f"\nRunning eval  strategy={strategy.value}  reranker={reranker.value}  k={args.k}")
    result = run_eval(strategy, reranker, output_path, k=args.k)
    _print_summary(result["summary"])
