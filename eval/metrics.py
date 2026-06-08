"""Retrieval evaluation metrics."""

from __future__ import annotations

import math


def recall_at_k(relevant_ids: set[str], retrieved: list[str], k: int) -> float:
    """Recall@K = |relevant intersect retrieved[:k]| / |relevant|."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for item_id in retrieved[:k] if item_id in relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(relevant_ids: set[str], retrieved: list[str], k: int) -> float:
    """Precision@K = |relevant intersect retrieved[:k]| / k."""
    if k == 0:
        return 0.0
    hits = sum(1 for item_id in retrieved[:k] if item_id in relevant_ids)
    return hits / k


def mrr(relevant_ids: set[str], retrieved: list[str], k: int = 100) -> float:
    """Reciprocal rank of the first relevant result within k."""
    for rank, item_id in enumerate(retrieved[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevant_ids: set[str], retrieved: list[str], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    gain = 0.0
    for rank, item_id in enumerate(retrieved[:k], start=1):
        if item_id in relevant_ids:
            gain += 1.0 / math.log2(rank + 1)
    return gain


def ndcg_at_k(relevant_ids: set[str], retrieved: list[str], k: int) -> float:
    """Normalized DCG@K with binary relevance."""
    if not relevant_ids:
        return 0.0
    actual_dcg = dcg_at_k(relevant_ids, retrieved, k)
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def negative_sample_score(retrieved: list[str]) -> float:
    """
    Score for a negative sample (expected_empty=true).

    The correct system behaviour is to return no citations.
    Returns 1.0 when retrieved is empty, 0.0 otherwise.

    This is intentionally separate from recall_at_k / mrr / ndcg_at_k because
    those metrics are undefined when relevant_ids is empty (denominator = 0).
    Mixing negative-sample scores into the positive-sample averages would
    distort both directions, so the caller (EvalRunner) keeps them in a
    separate summary bucket.
    """
    return 1.0 if not retrieved else 0.0
