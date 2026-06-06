"""
core/serving/rerank.py

HTTP client for TEI reranker service (bge-reranker-v2-m3 cross-encoder).

Design decisions:
- RerankerClient is used exclusively on the query hot path (P1-14).
- Accepts a query + list of candidate texts; returns scores in the same order.
- Graceful degradation: if the reranker is unavailable, the caller receives
  a RerankUnavailableError and can fall back to RRF order, flagging results
  as `reranked=false`. This matches the plan's degradation policy.
- Circuit breaker (simple): after `max_consecutive_failures` consecutive
  failures the client trips open and returns RerankUnavailableError immediately
  without attempting network calls, until `reset_after_seconds` elapses.
  This prevents a slow/dead reranker from adding latency to every query.
- Timeout is kept tight (query path): configurable, default 30 s.

TEI rerank API (POST /rerank):
    Request:  {"query": "...", "texts": ["t1", "t2", ...]}
    Response: [{"index": 0, "score": 0.93}, {"index": 1, "score": 0.12}, ...]
              — sorted by score descending by default.
              We re-order to match the original input order.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

from core.config.models import RerankerConfig

logger = logging.getLogger(__name__)

_RERANK_PATH = "/rerank"

# Circuit breaker defaults
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
_DEFAULT_RESET_AFTER_SECONDS = 60.0


@dataclass
class RerankScore:
    """Score for a single candidate, keyed by original list index."""
    index: int
    score: float


class RerankerClient:
    """
    Wraps one TEI reranker endpoint.

    Usage
    -----
    client = RerankerClient(reranker_cfg)
    scores = client.rerank("query text", ["candidate 0", "candidate 1"])
    # scores: list[RerankScore] in original input order (index 0, 1, ...)
    # sorted by .score descending for convenience

    On failure, raises RerankUnavailableError — callers fall back to RRF order.
    """

    def __init__(
        self,
        config: RerankerConfig,
        *,
        max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
        reset_after_seconds: float = _DEFAULT_RESET_AFTER_SECONDS,
    ) -> None:
        self._config = config
        self._url = config.endpoint.rstrip("/") + _RERANK_PATH
        self._max_batch_size = config.max_batch_size
        self._timeout = httpx.Timeout(
            connect=5.0,
            read=config.timeout_seconds,
            write=5.0,
            pool=2.0,
        )
        # Circuit breaker state
        self._consecutive_failures = 0
        self._max_consecutive_failures = max_consecutive_failures
        self._reset_after_seconds = reset_after_seconds
        self._tripped_at: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(self, query: str, texts: list[str]) -> list[RerankScore]:
        """
        Score each text against the query using the cross-encoder.

        Returns
        -------
        list[RerankScore]
            One entry per input text, sorted by score descending.
            .index refers to the position in the original `texts` list.

        Raises
        ------
        RerankUnavailableError
            When the reranker is down, circuit-breaker is open, or times out.
            Callers must handle this and fall back to RRF order.
        """
        if not texts:
            return []

        self._check_circuit_breaker()

        try:
            scores = self._rerank_batches(query, texts)
            self._on_success()
            return scores
        except RerankUnavailableError:
            self._on_failure()
            raise

    @property
    def circuit_open(self) -> bool:
        """True when the circuit breaker is tripped (reranker not attempted)."""
        if self._tripped_at is None:
            return False
        return (time.monotonic() - self._tripped_at) < self._reset_after_seconds

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rerank_batches(self, query: str, texts: list[str]) -> list[RerankScore]:
        """Call TEI in batches and remap batch-local indices to global indices."""
        if len(texts) <= self._max_batch_size:
            return self._call_tei(query, texts)

        merged: list[RerankScore] = []
        for start in range(0, len(texts), self._max_batch_size):
            batch = texts[start:start + self._max_batch_size]
            for score in self._call_tei(query, batch):
                merged.append(
                    RerankScore(index=start + score.index, score=score.score)
                )

        merged.sort(key=lambda score: score.score, reverse=True)
        return merged

    def _check_circuit_breaker(self) -> None:
        if not self.circuit_open:
            return
        elapsed = time.monotonic() - (self._tripped_at or 0)
        raise RerankUnavailableError(
            f"Reranker circuit breaker is open (tripped {elapsed:.0f}s ago, "
            f"resets after {self._reset_after_seconds:.0f}s). "
            "Results will use RRF order."
        )

    def _on_success(self) -> None:
        if self._consecutive_failures > 0:
            logger.info("Reranker recovered after %d failure(s)", self._consecutive_failures)
        self._consecutive_failures = 0
        self._tripped_at = None

    def _on_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._max_consecutive_failures:
            self._tripped_at = time.monotonic()
            logger.warning(
                "Reranker circuit breaker tripped after %d consecutive failures. "
                "Will retry after %.0fs.",
                self._consecutive_failures,
                self._reset_after_seconds,
            )

    def _call_tei(self, query: str, texts: list[str]) -> list[RerankScore]:
        payload = {"query": query, "texts": texts}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload)
        except httpx.TimeoutException as exc:
            raise RerankUnavailableError(
                f"Reranker request timed out after {self._config.timeout_seconds}s "
                f"(candidates={len(texts)})"
            ) from exc
        except httpx.RequestError as exc:
            raise RerankUnavailableError(
                f"Reranker request failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RerankUnavailableError(
                f"TEI reranker returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()
        if not isinstance(data, list):
            raise RerankUnavailableError(
                f"Unexpected reranker response type: {type(data).__name__}"
            )

        # TEI returns [{"index": i, "score": f}, ...] sorted by score desc.
        # We keep that sort order and just normalise into RerankScore.
        scores: list[RerankScore] = []
        for item in data:
            if not isinstance(item, dict) or "index" not in item or "score" not in item:
                raise RerankUnavailableError(
                    f"Malformed reranker response item: {item!r}"
                )
            scores.append(RerankScore(index=int(item["index"]), score=float(item["score"])))

        logger.debug(
            "Reranked %d candidates (top score=%.4f)",
            len(scores), scores[0].score if scores else 0.0,
        )
        return scores


class RerankUnavailableError(Exception):
    """
    Raised when the reranker service is unavailable or circuit-breaker open.
    Callers must catch this and return RRF-ordered results flagged reranked=false.
    """
