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

_EMBED_PATH    = "/embed"
_EMBED_ALL_PATH = "/embed_all"
_TOKENIZE_PATH  = "/tokenize"


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
        self._url        = config.endpoint.rstrip("/") + _EMBED_PATH
        self._url_all    = config.endpoint.rstrip("/") + _EMBED_ALL_PATH
        self._url_tok    = config.endpoint.rstrip("/") + _TOKENIZE_PATH
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

    def embed_all(self, text: str) -> list[list[float]]:
        """
        Call TEI /embed_all for a single text and return the token-level
        vector matrix.

        TEI /embed_all API:
            Request:  {"inputs": "text"}          — single string only
            Response: [[[float, ...], ...]]        — list[list[list[float]]]
                      outer list has length 1 (one input),
                      inner list has length == number of tokens,
                      innermost list has length == embedding dimension.

        Returns
        -------
        list[list[float]] of shape [num_tokens × dimension].
        Raises EmbeddingError on HTTP error, timeout, or unexpected response.
        """
        payload = {"inputs": text}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url_all, json=payload)
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"/embed_all timed out (model={self._config.id})"
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingError(
                f"/embed_all request failed (model={self._config.id}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise EmbeddingError(
                f"TEI /embed_all returned HTTP {resp.status_code} "
                f"(model={self._config.id}): {resp.text[:200]}"
            )

        data = resp.json()
        # data shape: [[[float]]] — one outer entry per input
        if (
            not isinstance(data, list)
            or len(data) != 1
            or not isinstance(data[0], list)
        ):
            raise EmbeddingError(
                f"Unexpected /embed_all response shape (model={self._config.id}): "
                f"expected list of length 1, got {type(data).__name__} "
                f"len={len(data) if isinstance(data, list) else '?'}"
            )

        token_vectors: list[list[float]] = data[0]
        logger.debug(
            "/embed_all: %d token vectors dim=%d (model=%s)",
            len(token_vectors),
            len(token_vectors[0]) if token_vectors else 0,
            self._config.id,
        )
        return token_vectors

    def tokenize(self, text: str) -> list[tuple[int, int]]:
        """
        Call TEI /tokenize for a single text and return per-token character
        offsets as a list of (char_start, char_end) tuples.

        TEI /tokenize API:
            Request:  {"inputs": "text", "add_special_tokens": true}
            Response: [[{"id": int, "text": str, "start": int, "stop": int,
                         "special": bool}, ...]]
                      outer list has length 1 (one input).

        Special tokens (e.g. [CLS], [SEP]) are included with their `start`
        and `stop` both set to 0 (or equal), which the pooling logic treats
        as zero-width and skips.

        Returns
        -------
        list of (char_start, char_end) per token, including special tokens.
        Raises EmbeddingError on HTTP error, timeout, or unexpected response.
        """
        payload = {"inputs": text, "add_special_tokens": True}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(self._url_tok, json=payload)
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"/tokenize timed out (model={self._config.id})"
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingError(
                f"/tokenize request failed (model={self._config.id}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise EmbeddingError(
                f"TEI /tokenize returned HTTP {resp.status_code} "
                f"(model={self._config.id}): {resp.text[:200]}"
            )

        data = resp.json()
        if not isinstance(data, list) or len(data) != 1:
            raise EmbeddingError(
                f"Unexpected /tokenize response shape (model={self._config.id})"
            )

        token_list = data[0]
        offsets: list[tuple[int, int]] = []
        for tok in token_list:
            start = tok.get("start", 0)
            stop  = tok.get("stop", 0)
            if start is None or stop is None:
                start = stop = 0
            offsets.append((start, stop))

        logger.debug(
            "/tokenize: %d tokens (model=%s)",
            len(offsets), self._config.id,
        )
        return offsets

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
