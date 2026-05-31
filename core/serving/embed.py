"""
core/serving/embed.py

HTTP client for TEI embedding service.

Design decisions:
- One EmbeddingClient instance per configured embedding model.
- Batched: caller passes a list of texts; client chunks into batches of
  `batch_size` and calls TEI sequentially (CPU TEI has its own internal
  batching; we honour the configured batch_size to avoid OOM on large corpora).
- Synchronous by default (ingestion is offline/batch). An async variant can
  be added when the query path needs it (P1-12).
- Timeout and retry are explicit, not implicit. A single failed batch raises
  immediately; the Indexer's failed_index_records table handles retry at the
  document level.
- Returns raw float lists; the caller (Embedder, P1-08) is responsible for
  assembling named vectors.

TEI embed API (POST /embed):
    Request:  {"inputs": "text"} or {"inputs": ["t1", "t2", ...]}
    Response: [[float, ...], ...]   — list of vectors, one per input
"""

from __future__ import annotations

import logging
from typing import Sequence

import httpx

from core.config.models import EmbeddingModelConfig

logger = logging.getLogger(__name__)

_EMBED_PATH = "/embed"


class EmbeddingClient:
    """
    Wraps one TEI embedding endpoint as configured by an EmbeddingModelConfig.

    Usage
    -----
    client = EmbeddingClient(model_cfg)
    vectors = client.embed(["sentence one", "sentence two"])
    # vectors: list[list[float]], len == len(texts)
    """

    def __init__(self, config: EmbeddingModelConfig) -> None:
        self._config = config
        self._url = config.endpoint.rstrip("/") + _EMBED_PATH
        self._batch_size = config.batch_size
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,   # CPU inference for a full batch can be slow
            write=10.0,
            pool=5.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._config.id

    @property
    def vector_name(self) -> str:
        return self._config.vector_name

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a list of texts. Returns one vector per text.
        Splits into batches of `config.batch_size` automatically.

        Raises
        ------
        EmbeddingError  on HTTP error, timeout, or unexpected response shape.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        texts_list = list(texts)

        for start in range(0, len(texts_list), self._batch_size):
            batch = texts_list[start : start + self._batch_size]
            vectors = self._call_tei(batch)
            all_vectors.extend(vectors)

        if len(all_vectors) != len(texts_list):
            raise EmbeddingError(
                f"TEI returned {len(all_vectors)} vectors for {len(texts_list)} inputs "
                f"(model={self._config.id})"
            )
        return all_vectors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_tei(self, batch: list[str]) -> list[list[float]]:
        payload = {"inputs": batch}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url, json=payload)
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"Embedding request timed out (model={self._config.id}, "
                f"batch_size={len(batch)})"
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingError(
                f"Embedding request failed (model={self._config.id}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise EmbeddingError(
                f"TEI embed returned HTTP {resp.status_code} "
                f"(model={self._config.id}): {resp.text[:200]}"
            )

        data = resp.json()

        # TEI returns a list of vectors directly
        if not isinstance(data, list):
            raise EmbeddingError(
                f"Unexpected TEI embed response type: {type(data).__name__} "
                f"(model={self._config.id})"
            )

        logger.debug(
            "Embedded %d texts → %d vectors (model=%s)",
            len(batch), len(data), self._config.id,
        )
        return data


class EmbeddingError(Exception):
    """Raised on any embedding service failure."""
