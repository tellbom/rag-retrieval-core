"""Compare two eval reports and detect metric regressions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(baseline: dict, current: dict, threshold: float = 0.02) -> list[str]:
    """Return regression messages. Empty list means no regression."""
    regressions: list[str] = []

    def check(name: str, baseline_value: float, current_value: float) -> None:
        drop = baseline_value - current_value
        if drop > threshold:
            regressions.append(
                f"REGRESSION {name}: {baseline_value:.4f} -> "
                f"{current_value:.4f} (drop={drop:.4f} > threshold={threshold:.4f})"
            )

    check("MRR", baseline.get("avg_mrr", 0.0), current.get("avg_mrr", 0.0))

    for k_value, baseline_recall in baseline.get("avg_recall_at_k", {}).items():
        current_recall = current.get("avg_recall_at_k", {}).get(k_value, 0.0)
        check(f"Recall@{k_value}", baseline_recall, current_recall)

    for k_value, baseline_ndcg in baseline.get("avg_ndcg_at_k", {}).items():
        current_ndcg = current.get("avg_ndcg_at_k", {}).get(k_value, 0.0)
        check(f"NDCG@{k_value}", baseline_ndcg, current_ndcg)

    return regressions


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Compare two eval reports.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    current = _load(args.current)
    regressions = compare(baseline, current, threshold=args.threshold)

    print(f"Baseline: {args.baseline} (config={baseline.get('config_version')})")
    print(f"Current:  {args.current} (config={current.get('config_version')})")
    print(f"Threshold: {args.threshold}")

    if regressions:
        print(f"\n{len(regressions)} regression(s) detected:")
        for regression in regressions:
            print(f"  - {regression}")
        sys.exit(1)

    print("\nNo regressions detected.")


if __name__ == "__main__":
    _cli()
