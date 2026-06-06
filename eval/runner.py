"""Run a golden set against the live query service."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from eval.metrics import mrr, ndcg_at_k, recall_at_k

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return unique ids in first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


@dataclass
class GoldenItem:
    id: str
    query: str
    relevant_doc_ids: list[str]
    relevant_chunk_ids: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ItemResult:
    id: str
    query: str
    success: bool
    retrieved_doc_ids: list[str]
    recall_at_k: dict[str, float]
    mrr: float
    ndcg_at_k: dict[str, float]
    latency_ms: float
    error: str = ""
    iterations: int = 1
    sub_queries: list[str] = field(default_factory=list)
    iterative_enabled: bool = False
    self_eval_sufficient: bool | None = None
    self_eval_confidence: str = ""
    self_eval_missing: str = ""
    topic_absent: bool = False


@dataclass
class EvalReport:
    golden_file: str
    business_type: str
    query_url: str
    config_version: str
    k_values: list[int]
    total_queries: int
    successful_queries: int
    avg_recall_at_k: dict[str, float]
    avg_mrr: float
    avg_ndcg_at_k: dict[str, float]
    avg_latency_ms: float
    item_results: list[ItemResult]
    timestamp: str = ""


class EvalRunner:
    """Run a golden set against the live /query endpoint."""

    def __init__(
        self,
        query_url: str,
        k_values: list[int] | None = None,
        *,
        timeout: float = 60.0,
        enable_rewrite: bool = False,
        enable_iterative: bool = False,
        filter_extra: dict[str, str] | None = None,
    ) -> None:
        self._base_url = query_url.rstrip("/")
        self._url = self._base_url + "/query"
        self._k_values = sorted(k_values or [5, 10, 20])
        self._timeout = timeout
        self._enable_rewrite = enable_rewrite
        self._enable_iterative = enable_iterative
        self._filter_extra = filter_extra or {}

    def run(self, golden_file: str | Path) -> EvalReport:
        golden_path = Path(golden_file)
        data = json.loads(golden_path.read_text(encoding="utf-8"))

        business_type = data.get("business_type", "")
        items = [
            GoldenItem(
                id=item["id"],
                query=item["query"],
                relevant_doc_ids=item.get("relevant_doc_ids", []),
                relevant_chunk_ids=item.get("relevant_chunk_ids", []),
                notes=item.get("notes", ""),
            )
            for item in data.get("items", [])
        ]

        config_version = self._get_config_version()
        item_results: list[ItemResult] = []

        for item in items:
            result = self._eval_item(item, business_type)
            item_results.append(result)
            status = "OK" if result.success else "FAIL"
            recalls = " ".join(
                f"R@{k}={result.recall_at_k.get(str(k), 0):.2f}"
                for k in self._k_values
            )
            print(
                f"  {status} [{item.id}] {recalls} "
                f"MRR={result.mrr:.3f} {item.query[:50]}"
            )

        return self._aggregate(
            item_results,
            golden_file=str(golden_path),
            business_type=business_type,
            config_version=config_version,
        )

    def _eval_item(self, item: GoldenItem, business_type: str) -> ItemResult:
        payload = {
            "query": item.query,
            "business_type": business_type,
            "filters": self._build_filters(business_type),
            "enable_rewrite": self._enable_rewrite,
            "enable_iterative": self._enable_iterative,
        }

        started = time.monotonic()
        try:
            response = httpx.post(self._url, json=payload, timeout=self._timeout)
            latency_ms = (time.monotonic() - started) * 1000
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            latency_ms = (time.monotonic() - started) * 1000
            return ItemResult(
                id=item.id,
                query=item.query,
                success=False,
                retrieved_doc_ids=[],
                recall_at_k={str(k): 0.0 for k in self._k_values},
                mrr=0.0,
                ndcg_at_k={str(k): 0.0 for k in self._k_values},
                latency_ms=latency_ms,
                error=str(exc),
            )

        citations = body.get("citations", [])
        retrieved_doc_ids = [citation["doc_id"] for citation in citations]

        if item.relevant_chunk_ids:
            relevant = set(item.relevant_chunk_ids)
            retrieved_ids = [
                citation.get("chunk_id", citation["doc_id"])
                for citation in citations
            ]
            if len(retrieved_ids) != len(set(retrieved_ids)):
                logger.warning(
                    "Duplicate chunk_id values in response for item %s; "
                    "deduplicating before scoring; check fusion/rerank output",
                    item.id,
                )
                retrieved_ids = _dedupe_preserve_order(retrieved_ids)
        else:
            relevant = set(item.relevant_doc_ids)
            retrieved_ids = _dedupe_preserve_order(retrieved_doc_ids)

        recall = {
            str(k): recall_at_k(relevant, retrieved_ids, k)
            for k in self._k_values
        }
        ndcg = {
            str(k): ndcg_at_k(relevant, retrieved_ids, k)
            for k in self._k_values
        }

        return ItemResult(
            id=item.id,
            query=item.query,
            success=True,
            retrieved_doc_ids=retrieved_doc_ids,
            recall_at_k=recall,
            mrr=mrr(relevant, retrieved_ids, k=max(self._k_values)),
            ndcg_at_k=ndcg,
            latency_ms=latency_ms,
            iterations=body.get("iterations", 1),
            sub_queries=body.get("sub_queries", []),
            iterative_enabled=body.get("iterative_enabled", False),
            self_eval_sufficient=body.get("self_eval_sufficient"),
            self_eval_confidence=body.get("self_eval_confidence", ""),
            self_eval_missing=body.get("self_eval_missing", ""),
            topic_absent=body.get("topic_absent", False),
        )

    def _build_filters(self, business_type: str) -> dict[str, object]:
        filters: dict[str, object] = {}
        if business_type:
            filters["business_type"] = business_type
        if self._filter_extra:
            filters["extra"] = self._filter_extra
        return filters

    def _get_config_version(self) -> str:
        try:
            response = httpx.get(self._base_url + "/health", timeout=5.0)
            return response.json().get("config_version", "unknown")
        except Exception:
            return "unknown"

    def _aggregate(
        self,
        results: list[ItemResult],
        *,
        golden_file: str,
        business_type: str,
        config_version: str,
    ) -> EvalReport:
        successful = [result for result in results if result.success]
        count = len(successful)

        avg_recall = {
            str(k): (
                sum(result.recall_at_k.get(str(k), 0.0) for result in successful)
                / count
                if count
                else 0.0
            )
            for k in self._k_values
        }
        avg_ndcg = {
            str(k): (
                sum(result.ndcg_at_k.get(str(k), 0.0) for result in successful)
                / count
                if count
                else 0.0
            )
            for k in self._k_values
        }
        avg_mrr = sum(result.mrr for result in successful) / count if count else 0.0
        avg_latency = (
            sum(result.latency_ms for result in results) / len(results)
            if results
            else 0.0
        )

        return EvalReport(
            golden_file=golden_file,
            business_type=business_type,
            query_url=self._url,
            config_version=config_version,
            k_values=self._k_values,
            total_queries=len(results),
            successful_queries=count,
            avg_recall_at_k=avg_recall,
            avg_mrr=avg_mrr,
            avg_ndcg_at_k=avg_ndcg,
            avg_latency_ms=avg_latency,
            item_results=results,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Run a golden set against the live query service."
    )
    parser.add_argument("--query-url", default="http://localhost:8002")
    parser.add_argument("--golden", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--output", default=None)
    parser.add_argument("--enable-rewrite", action="store_true")
    parser.add_argument("--enable-iterative", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--filter-extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional exact-match filter pushed via QueryFilters.extra. "
        "May be provided multiple times.",
    )
    args = parser.parse_args()

    filter_extra: dict[str, str] = {}
    for item in args.filter_extra:
        if "=" not in item:
            parser.error(f"--filter-extra must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            parser.error(f"--filter-extra must be KEY=VALUE, got {item!r}")
        filter_extra[key] = value

    runner = EvalRunner(
        query_url=args.query_url,
        k_values=args.k,
        timeout=args.timeout,
        enable_rewrite=args.enable_rewrite,
        enable_iterative=args.enable_iterative,
        filter_extra=filter_extra,
    )
    report = runner.run(args.golden)

    print(f"\nResults: {report.business_type} ({report.successful_queries}/{report.total_queries} ok)")
    print(f"Config: {report.config_version}")
    for k in report.k_values:
        recall = report.avg_recall_at_k.get(str(k), 0.0)
        ndcg = report.avg_ndcg_at_k.get(str(k), 0.0)
        print(f"  Recall@{k}={recall:.4f} NDCG@{k}={ndcg:.4f}")
    print(f"  MRR={report.avg_mrr:.4f} avg_latency={report.avg_latency_ms:.0f}ms")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    _cli()
