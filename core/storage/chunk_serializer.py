"""
core/storage/chunk_serializer.py

Converts a Chunk into the wire formats expected by ES 7.x and Qdrant.

ES document body
----------------
All standard_fields from the Chunk are written as flat key→value pairs.
The `text` field maps to `context_text` (the padded, index-ready text).
System fields (_enhanced, _config_version, _embedding_model_versions) are
always written regardless of what's in standard_fields.

Qdrant PointStruct
-------------------
- `id`:       UUID derived deterministically from chunk_id (Qdrant requires UUID or uint64).
- `vectors`:  {vector_name: [float, ...]} — all named vectors from the chunk.
- `payload`:  flat dict of filter fields + text + metadata for context building
              without an ES round-trip.

Neither serializer validates that the Chunk has vectors — the caller
(Indexer) is responsible for only calling these after embedding.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from qdrant_client.http import models as qmodels

from core.ingestion.chunk import Chunk


# ---------------------------------------------------------------------------
# ES document
# ---------------------------------------------------------------------------

def chunk_to_es_doc(chunk: Chunk, enhanced: bool = False) -> dict[str, Any]:
    """
    Build the Elasticsearch document body for a chunk.

    The document id used for upsert is `chunk.chunk_id` (passed separately
    to the ES client as the `_id` field, not included in the body).
    """
    doc: dict[str, Any] = {
        # Identity
        "chunk_id":        chunk.chunk_id,
        "doc_id":          chunk.doc_id,
        "parent_id":       chunk.parent_id,
        "hierarchy_level": chunk.hierarchy_level,
        "position":        chunk.position,

        # Text — context_text is what gets indexed and highlighted
        "text":            chunk.context_text,

        # Business metadata
        "business_type":   chunk.business_type,

        # Provenance
        "config_version":           chunk.config_version,
        "embedding_model_versions": _versions_str(chunk.embedding_model_versions),

        # System flags
        "_enhanced":                enhanced,
        "_config_version":          chunk.config_version,
        "_embedding_model_versions": _versions_str(chunk.embedding_model_versions),
    }

    # Flatten source_metadata into the doc (caller controls the keys)
    for k, v in (chunk.source_metadata or {}).items():
        # Avoid overwriting core fields
        if k not in doc:
            doc[k] = v

    return doc


# ---------------------------------------------------------------------------
# Qdrant point
# ---------------------------------------------------------------------------

def chunk_to_qdrant_point(chunk: Chunk, enhanced: bool = False) -> qmodels.PointStruct:
    """
    Build a Qdrant PointStruct for a chunk.

    Qdrant point id is a UUID derived from chunk_id so it is:
    - Deterministic (same chunk_id → same UUID → idempotent upsert)
    - Valid UUID format required by Qdrant
    """
    point_id = _chunk_id_to_uuid(chunk.chunk_id)

    payload: dict[str, Any] = {
        # Identity (for dedup and parent backfill)
        "chunk_id":        chunk.chunk_id,
        "doc_id":          chunk.doc_id,
        "parent_id":       chunk.parent_id,
        "hierarchy_level": chunk.hierarchy_level,
        "position":        chunk.position,

        # Text stored in payload so context can be built without ES round-trip
        "text":            chunk.context_text,

        # Filter fields (payload-indexed in Qdrant provisioner)
        "business_type":   chunk.business_type,

        # Provenance
        "config_version":            chunk.config_version,
        "embedding_model_versions":  _versions_str(chunk.embedding_model_versions),

        # System flags
        "_enhanced": enhanced,
    }

    # Flatten source_metadata (filter-able fields like category, created_time)
    for k, v in (chunk.source_metadata or {}).items():
        if k not in payload:
            payload[k] = v

    return qmodels.PointStruct(
        id=point_id,
        vector=chunk.named_vectors,   # {vector_name: [float, ...]}
        payload=payload,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministic UUID v5 from chunk_id (namespace = DNS)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _versions_str(versions: dict[str, str]) -> str:
    """Compact string: 'model_a=1.0.0,model_b=2.0.0' — for keyword field storage."""
    return ",".join(f"{k}={v}" for k, v in sorted(versions.items()))
