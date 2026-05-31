"""
core/storage/original_text_store.py

OriginalTextStore: stores and retrieves the original (pre-cleaning) text
of every ingested document.

Why this exists
---------------
Rebuild (P1-10) requires re-running the full ingestion pipeline from scratch
on every document when chunking config or embedding models change.  The
original texts must be retrievable without going back to the source business
system.  This store is the authoritative source for rebuild.

For CRUD update: the store holds the latest version of each document's raw
text so a single doc can be re-ingested without touching others.

Storage layout
--------------
Each document is stored as a UTF-8 JSON file:
    {base_dir}/{doc_id[:2]}/{doc_id}.json

The two-char prefix shards documents across 256 subdirectories to avoid
inode exhaustion on large corpora.

File format:
    {
      "doc_id":        "...",
      "raw_text":      "...",      ← the original un-cleaned text
      "business_type": "...",
      "source_metadata": {...},
      "stored_at":     "ISO-8601"
    }

Configuration
-------------
Base directory is controlled by RAG_ORIGINAL_TEXT_DIR env var
(default: ./data/originals).

Public API
----------
    store = OriginalTextStore(base_dir)
    store.put(doc_id, raw_text, business_type, source_metadata)
    entry = store.get(doc_id)      # → OriginalTextEntry | None
    store.delete(doc_id)
    doc_ids = store.list_all()     # → list[str]  (for rebuild)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path("data/originals")


@dataclass
class OriginalTextEntry:
    doc_id: str
    raw_text: str
    business_type: str
    source_metadata: dict
    stored_at: str  # ISO-8601 UTC


class OriginalTextStore:
    """
    Filesystem-backed store for original document texts.

    Parameters
    ----------
    base_dir:
        Root directory for the store.  Created on first use.
    """

    def __init__(self, base_dir: str | Path = _DEFAULT_BASE_DIR) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str = "",
        source_metadata: dict | None = None,
    ) -> None:
        """
        Store (or overwrite) the original text for `doc_id`.
        Overwrites silently — for update, the caller has already replaced
        the text in the business system.
        """
        path = self._path_for(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "doc_id":          doc_id,
            "raw_text":        raw_text,
            "business_type":   business_type,
            "source_metadata": source_metadata or {},
            "stored_at":       _utcnow(),
        }
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("OriginalTextStore.put: doc_id=%s (%d chars)", doc_id, len(raw_text))

    def get(self, doc_id: str) -> OriginalTextEntry | None:
        """
        Retrieve the stored entry for `doc_id`.
        Returns None if the document has not been stored.
        """
        path = self._path_for(doc_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return OriginalTextEntry(
                doc_id=data["doc_id"],
                raw_text=data["raw_text"],
                business_type=data.get("business_type", ""),
                source_metadata=data.get("source_metadata", {}),
                stored_at=data.get("stored_at", ""),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "OriginalTextStore: corrupt entry for doc_id=%s: %s", doc_id, exc
            )
            return None

    def delete(self, doc_id: str) -> bool:
        """
        Remove the stored original for `doc_id`.
        Returns True if deleted, False if not found.
        """
        path = self._path_for(doc_id)
        if path.exists():
            path.unlink()
            logger.debug("OriginalTextStore.delete: doc_id=%s", doc_id)
            return True
        return False

    def list_all(self) -> list[str]:
        """
        Return doc_ids of all stored originals.
        Used by RebuildService to enumerate documents for full re-ingest.
        """
        doc_ids: list[str] = []
        for json_file in self._base.rglob("*.json"):
            doc_ids.append(json_file.stem)
        return sorted(doc_ids)

    def iter_all(self) -> Iterator[OriginalTextEntry]:
        """
        Iterate over all stored entries (memory-efficient for large corpora).
        Skips corrupt or unreadable files with a warning.
        """
        for doc_id in self.list_all():
            entry = self.get(doc_id)
            if entry is not None:
                yield entry

    def count(self) -> int:
        """Return number of stored originals."""
        return sum(1 for _ in self._base.rglob("*.json"))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _path_for(self, doc_id: str) -> Path:
        """Two-level sharding: {base}/{doc_id[:2]}/{doc_id}.json"""
        shard = doc_id[:2] if len(doc_id) >= 2 else "xx"
        return self._base / shard / f"{doc_id}.json"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
