"""
core/ingestion/embedder.py

Embedder: assigns dense vectors to every Chunk in a ChunkingResult.

Responsibilities
----------------
1. For every configured embedding model, call the TEI embedding service
   with the chunk's `context_text` (the padded, index-ready text).
2. Write the resulting vector into `chunk.named_vectors[vector_name]`.
3. Stamp `chunk.embedding_model_versions` with `{model_id: version}` so
   every stored chunk carries full provenance — which model version produced
   its vectors.  A model version change without a re-index is detectable.
4. Batch chunks across all models: inner loop = models, outer loop = chunks
   batched per model's configured batch_size.

Adding a new embedding model
-----------------------------
Add one entry to `models.embeddings` in the JSON config.  No code changes.
The Embedder reads the model list from AppConfig at construction time.

Error handling
--------------
- If one model's embedding call fails for a batch, EmbeddingError propagates
  to the caller (ingestion pipeline / Indexer).
- The Indexer (P1-09) writes failed chunks to `failed_index_records` for retry.
- Partial vector sets are NOT written — a chunk gets vectors from ALL models
  or none, preventing silent mixed-version state.

What gets embedded
------------------
`chunk.context_text` — not `chunk.text`.  context_text = context prefix +
text, which is what the retrieval path will compare against query embeddings.
Embedding the wider context_text is the "contextual retrieval" improvement
that improves recall for parent-padded chunks.

Public API
----------
    embedder = Embedder.from_config(cfg, serving_registry)
    result = embedder.embed(chunking_result)   # ChunkingResult → ChunkingResult
    # Each chunk now has named_vectors and embedding_model_versions populated.

Exposed config (JSON, via AppConfig.models.embeddings):
    models:
      embeddings:
        - id:          "bge_base"
          name:        "BAAI/bge-base-zh-v1.5"
          version:     "1.5.0"
          endpoint:    "http://localhost:8080"
          vector_name: "bge_base"
          dimension:   768
          batch_size:  32
          max_seq_len: 512
          normalize:   true

Any field above is changeable by editing the JSON config.
A `version` change triggers a rebuild (P1-10); other fields change behaviour
at next ingestion run.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from core.config.models import AppConfig, EmbeddingModelConfig
from core.ingestion.chunk import Chunk, ChunkingResult
from core.ingestion.late_chunking_utils import (
    build_group_text,
    find_token_range,
    l2_normalize,
    mean_pool,
)
from core.serving.embed import EmbeddingClient, EmbeddingError
from core.serving.registry import ServingRegistry

logger = logging.getLogger(__name__)


class Embedder:
    """
    Assigns named vectors to all chunks in a ChunkingResult.

    Parameters
    ----------
    clients:
        Ordered list of (EmbeddingModelConfig, EmbeddingClient) pairs.
        Order matches the order declared in `models.embeddings` config.
    """

    def __init__(
        self,
        clients: list[tuple[EmbeddingModelConfig, EmbeddingClient]],
    ) -> None:
        if not clients:
            raise ValueError("Embedder requires at least one embedding client")
        self._clients = clients

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        registry: ServingRegistry,
    ) -> "Embedder":
        """
        Build an Embedder from AppConfig + a warmed-up ServingRegistry.
        Called once at service startup.
        """
        clients: list[tuple[EmbeddingModelConfig, EmbeddingClient]] = []
        for emb_cfg in cfg.models.embeddings:
            client = registry.get_embedding_client(emb_cfg.id)
            clients.append((emb_cfg, client))
        return cls(clients)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, result: ChunkingResult) -> ChunkingResult:
        """
        Embed all chunks in `result`.

        For each configured embedding model:
        - use_late_chunking=False (default): call /embed per chunk (existing behaviour).
        - use_late_chunking=True: group chunks by lc_group_id, call /embed_all once
          per group, pool token vectors by char offset, L2-normalise.
          Any chunk that cannot be pooled falls back to ordinary /embed.

        Each chunk receives:
          - `named_vectors`:            {vector_name: [float, ...]}
          - `embedding_model_versions`: {model_id: version_string}

        Returns a new ChunkingResult; input chunks are not mutated.
        Raises EmbeddingError if any ordinary /embed model call fails.
        """
        if not result.chunks:
            return result

        chunks = result.chunks

        # Accumulate vectors per chunk index
        vectors_by_idx: dict[int, dict[str, list[float]]] = {
            i: {} for i in range(len(chunks))
        }
        versions_by_idx: dict[int, dict[str, str]] = {
            i: {} for i in range(len(chunks))
        }

        for emb_cfg, client in self._clients:
            logger.debug(
                "Embedding %d chunks with model=%s (vector=%s, late_chunking=%s)",
                len(chunks), emb_cfg.id, emb_cfg.vector_name, emb_cfg.use_late_chunking,
            )

            if emb_cfg.use_late_chunking:
                self._embed_late_chunking(
                    chunks, emb_cfg, client, vectors_by_idx, versions_by_idx
                )
            else:
                texts = [c.context_text for c in chunks]
                all_vectors = client.embed(texts)

                if len(all_vectors) != len(chunks):
                    raise EmbeddingError(
                        f"Model {emb_cfg.id} returned {len(all_vectors)} vectors "
                        f"for {len(chunks)} chunks"
                    )

                for i, vec in enumerate(all_vectors):
                    if len(vec) != emb_cfg.dimension:
                        raise EmbeddingError(
                            f"Model {emb_cfg.id} returned vector dimension {len(vec)} "
                            f"for chunk index {i}; expected {emb_cfg.dimension}"
                        )
                    vectors_by_idx[i][emb_cfg.vector_name] = vec
                    versions_by_idx[i][emb_cfg.id] = emb_cfg.version

        # Write vectors back onto (immutable-style) copies of chunks
        embedded_chunks: list[Chunk] = []
        for i, chunk in enumerate(chunks):
            embedded_chunks.append(
                _stamp_vectors(chunk, vectors_by_idx[i], versions_by_idx[i])
            )

        logger.info(
            "Embedded %d chunks across %d model(s) for doc_id=%s",
            len(embedded_chunks), len(self._clients), result.doc_id,
        )
        return ChunkingResult(
            doc_id=result.doc_id,
            chunks=embedded_chunks,
            needs_split_count=result.needs_split_count,
        )

    def _embed_late_chunking(
        self,
        chunks: list[Chunk],
        emb_cfg: EmbeddingModelConfig,
        client: EmbeddingClient,
        vectors_by_idx: dict[int, dict[str, list[float]]],
        versions_by_idx: dict[int, dict[str, str]],
    ) -> None:
        """
        Late Chunking path for one embedding model.

        Algorithm per lc_group_id:
          1. Collect sibling chunks (lc_group_id not None) sorted by position.
          2. Build group_text = build_group_text([c.text for c in siblings]).
          3. Check group_text token count against model max_seq_len; if over →
             fallback all siblings to ordinary /embed.
          4. Call /tokenize(group_text) → token char offsets.
          5. Call /embed_all(group_text) → token vectors.
          6. For each sibling: map (char_start, char_end) → token range →
             mean_pool → l2_normalize → write to vectors_by_idx.
          7. Any chunk that fails mapping falls back to ordinary /embed.

        Chunks with lc_group_id=None always go through ordinary /embed.
        """
        # --- Partition chunks: eligible vs ineligible ---
        # chunk_index_map: original index in `chunks` list
        eligible: dict[str, list[tuple[int, Chunk]]] = {}   # group_id → [(idx, chunk)]
        fallback_indices: list[int] = []

        for idx, chunk in enumerate(chunks):
            if chunk.lc_group_id is None or chunk.char_start is None or chunk.char_end is None:
                fallback_indices.append(idx)
            else:
                eligible.setdefault(chunk.lc_group_id, []).append((idx, chunk))

        # --- Process each Late Chunking group ---
        additional_fallbacks: list[int] = []

        for group_id, group_entries in eligible.items():
            # Sort by position to match build_group_text order
            group_entries.sort(key=lambda t: t[1].position)
            group_texts = [c.text for _, c in group_entries]
            group_text  = build_group_text(group_texts)

            # Rough token-count guard before calling TEI
            # Use a char-based estimate: 1 token ≈ 2 chars for Chinese
            # The precise check happens at TEI; this avoids an obvious oversize call.
            max_chars = emb_cfg.max_seq_len * 4  # conservative upper bound
            if len(group_text) > max_chars:
                logger.debug(
                    "Late chunking group %s exceeds max_seq_len estimate "
                    "(%d chars > %d char limit), falling back to /embed",
                    group_id, len(group_text), max_chars,
                )
                additional_fallbacks.extend(idx for idx, _ in group_entries)
                continue

            # Call /tokenize and /embed_all
            try:
                token_offsets  = client.tokenize(group_text)
                token_vectors  = client.embed_all(group_text)
            except EmbeddingError as exc:
                logger.warning(
                    "Late chunking TEI call failed for group %s (model=%s): %s — "
                    "falling back to /embed for %d chunk(s)",
                    group_id, emb_cfg.id, exc, len(group_entries),
                )
                additional_fallbacks.extend(idx for idx, _ in group_entries)
                continue

            if len(token_offsets) != len(token_vectors):
                logger.warning(
                    "Late chunking token count mismatch for group %s: "
                    "/tokenize=%d /embed_all=%d — falling back",
                    group_id, len(token_offsets), len(token_vectors),
                )
                additional_fallbacks.extend(idx for idx, _ in group_entries)
                continue

            # Pool each chunk's vector
            for idx, chunk in group_entries:
                token_range = find_token_range(
                    chunk.char_start,  # type: ignore[arg-type]
                    chunk.char_end,    # type: ignore[arg-type]
                    token_offsets,
                )
                if token_range is None:
                    logger.debug(
                        "Late chunking: could not map char[%d:%d] for chunk %s — fallback",
                        chunk.char_start, chunk.char_end, chunk.chunk_id,
                    )
                    additional_fallbacks.append(idx)
                    continue

                t_start, t_end = token_range
                pooled = mean_pool(token_vectors, t_start, t_end)
                if pooled is None:
                    logger.debug(
                        "Late chunking: mean_pool failed for chunk %s — fallback",
                        chunk.chunk_id,
                    )
                    additional_fallbacks.append(idx)
                    continue

                if emb_cfg.normalize:
                    pooled = l2_normalize(pooled)

                if len(pooled) != emb_cfg.dimension:
                    logger.warning(
                        "Late chunking: pooled vector dim %d ≠ expected %d for "
                        "chunk %s — fallback",
                        len(pooled), emb_cfg.dimension, chunk.chunk_id,
                    )
                    additional_fallbacks.append(idx)
                    continue

                vectors_by_idx[idx][emb_cfg.vector_name] = pooled
                versions_by_idx[idx][emb_cfg.id] = emb_cfg.version

        # --- Fallback: ordinary /embed for ineligible + failed chunks ---
        all_fallback = list(set(fallback_indices + additional_fallbacks))
        if not all_fallback:
            return

        fallback_chunks = [chunks[i] for i in all_fallback]
        fallback_texts  = [c.context_text for c in fallback_chunks]
        logger.debug(
            "Late chunking fallback: %d chunk(s) → /embed (model=%s)",
            len(fallback_chunks), emb_cfg.id,
        )

        try:
            fb_vectors = client.embed(fallback_texts)
        except EmbeddingError:
            # Re-raise: ordinary /embed failure is not recoverable
            raise

        for local_i, orig_idx in enumerate(all_fallback):
            vec = fb_vectors[local_i]
            if len(vec) != emb_cfg.dimension:
                raise EmbeddingError(
                    f"Model {emb_cfg.id} fallback /embed returned dimension "
                    f"{len(vec)} for chunk index {orig_idx}; expected {emb_cfg.dimension}"
                )
            vectors_by_idx[orig_idx][emb_cfg.vector_name] = vec
            versions_by_idx[orig_idx][emb_cfg.id] = emb_cfg.version

    def embed_batch(
        self,
        results: list[ChunkingResult],
    ) -> list[ChunkingResult]:
        """
        Embed multiple ChunkingResults (one per document).
        Processes documents sequentially; each document's chunks are
        batched together for efficient GPU/CPU utilisation per model.
        """
        return [self.embed(r) for r in results]

    @property
    def model_ids(self) -> list[str]:
        """Ordered list of configured model IDs."""
        return [cfg.id for cfg, _ in self._clients]

    @property
    def vector_names(self) -> list[str]:
        """Ordered list of Qdrant named vector keys."""
        return [cfg.vector_name for cfg, _ in self._clients]

    def iter_clients(self) -> list[tuple[EmbeddingModelConfig, EmbeddingClient]]:
        """Return configured embedding clients in config order."""
        return list(self._clients)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _stamp_vectors(
    chunk: Chunk,
    named_vectors: dict[str, list[float]],
    model_versions: dict[str, str],
) -> Chunk:
    """
    Return a new Chunk with named_vectors and embedding_model_versions filled in.
    Uses dataclasses.replace to avoid mutating the original.
    """
    return replace(
        chunk,
        named_vectors={**chunk.named_vectors, **named_vectors},
        embedding_model_versions={**chunk.embedding_model_versions, **model_versions},
    )
