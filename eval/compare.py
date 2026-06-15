import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import RerankerType, RetrieverStrategy  # noqa: E402
from app.telemetry import configure_logging  # noqa: E402
from eval.db import EvalDB  # noqa: E402
from eval.run_eval import RESULTS_DIR, run_eval  # noqa: E402

_TABLE_COLS = [
    ("field_match", "FieldMatch", 10),
    ("retrieval_precision", "RetPrec", 8),
    ("faithfulness", "Faithful", 9),
    ("avg_confidence", "AvgConf", 8),
    ("avg_latency_s", "Lat(s)", 7),
    ("total_tokens", "Tokens", 9),
    ("total_cost_usd", "Cost($)", 9),
]


def _print_table(summaries: list[dict]) -> None:
    label_w = 30
    sep = "  "
    header = [f"{'Strategy+Reranker':<{label_w}}"] + [f"{c:>{w}}" for _, c, w in _TABLE_COLS]
    divider = ["-" * label_w] + ["-" * w for _, _, w in _TABLE_COLS]
    print("\n" + sep.join(header))
    print(sep.join(divider))
    for s in summaries:
        label = f"{s['strategy']}+{s['reranker']}"
        row = [f"{label:<{label_w}}"]
        for key, _, col_w in _TABLE_COLS:
            val = s.get(key, "—")
            if isinstance(val, float):
                row.append(f"{val:>{col_w}.4f}")
            elif isinstance(val, int):
                row.append(f"{val:>{col_w},}")
            else:
                row.append(f"{str(val):>{col_w}}")
        print(sep.join(row))


def _persist(summaries: list[dict]) -> tuple[Path, Path]:
    csv_path = RESULTS_DIR / "comparison.csv"
    json_path = RESULTS_DIR / "comparison.json"
    if summaries:
        fieldnames = list(summaries[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    return csv_path, json_path


def compare_fresh(
    strategies: list[RetrieverStrategy],
    rerankers: list[RerankerType],
    k: int = 5,
    skip_on_error: bool = True,
) -> list[dict]:
    summaries: list[dict] = []

    for strategy, reranker in product(strategies, rerankers):
        label = f"{strategy.value}+{reranker.value}"
        output_path = RESULTS_DIR / f"{label}.json"
        print(f"\n{'=' * 60}\n  Running: {label}  (k={k})\n{'=' * 60}")
        try:
            result = run_eval(strategy, reranker, output_path, k=k)
            summaries.append(result["summary"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            if not skip_on_error:
                raise

    return summaries


def compare_from_db(
    strategies: list[RetrieverStrategy] | None = None,
    rerankers: list[RerankerType] | None = None,
) -> list[dict]:
    with EvalDB() as db:
        summaries = db.get_latest_summaries()

    if strategies:
        strat_vals = {s.value for s in strategies}
        summaries = [s for s in summaries if s["strategy"] in strat_vals]
    if rerankers:
        rer_vals = {r.value for r in rerankers}
        summaries = [s for s in summaries if s["reranker"] in rer_vals]

    return summaries


def compare(
    strategies: list[RetrieverStrategy],
    rerankers: list[RerankerType],
    k: int = 5,
    skip_on_error: bool = True,
    from_db: bool = False,
) -> list[dict]:
    if from_db:
        print("\nReading latest results from eval DB …")
        summaries = compare_from_db(strategies, rerankers)
        if not summaries:
            print("No results found in eval DB. Run without --from-db first.")
            return []
    else:
        summaries = compare_fresh(strategies, rerankers, k=k, skip_on_error=skip_on_error)

    if not summaries:
        print("No results to display.")
        return summaries

    csv_path, json_path = _persist(summaries)

    print("\n\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)
    _print_table(summaries)
    print(f"\n  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")

    return summaries


def _parse_args() -> argparse.Namespace:
    all_strategies = [s.value for s in RetrieverStrategy]
    all_rerankers = [r.value for r in RerankerType]
    parser = argparse.ArgumentParser(description="Compare all retrieval strategies.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=all_strategies,
        default=all_strategies,
        metavar="STRATEGY",
    )
    parser.add_argument(
        "--rerankers", nargs="+", choices=all_rerankers, default=all_rerankers, metavar="RERANKER"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Read latest results from eval SQLite DB instead of re-running",
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    args = _parse_args()
    compare(
        strategies=[RetrieverStrategy(s) for s in args.strategies],
        rerankers=[RerankerType(r) for r in args.rerankers],
        k=args.k,
        skip_on_error=not args.fail_fast,
        from_db=args.from_db,
    )
