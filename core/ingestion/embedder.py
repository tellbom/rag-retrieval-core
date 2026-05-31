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

        Each chunk receives:
          - `named_vectors`:            {vector_name: [float, ...]}
          - `embedding_model_versions`: {model_id: version_string}

        Returns a new ChunkingResult; input chunks are not mutated.
        Raises EmbeddingError if any model call fails.
        """
        if not result.chunks:
            return result

        # Collect texts once; embed per model
        chunks = result.chunks
        texts = [c.context_text for c in chunks]

        # Accumulate vectors per chunk index: {chunk_idx: {vector_name: [float]}}
        vectors_by_idx: dict[int, dict[str, list[float]]] = {
            i: {} for i in range(len(chunks))
        }
        versions_by_idx: dict[int, dict[str, str]] = {
            i: {} for i in range(len(chunks))
        }

        for emb_cfg, client in self._clients:
            logger.debug(
                "Embedding %d chunks with model=%s (vector=%s)",
                len(chunks), emb_cfg.id, emb_cfg.vector_name,
            )
            all_vectors = client.embed(texts)   # raises EmbeddingError on failure

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
