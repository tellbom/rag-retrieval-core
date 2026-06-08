"""
core/ingestion/audit_scheduler.py

Incremental background scheduler that periodically audits un-reviewed chunks.

Behaviour
---------
On each tick the scheduler:
  1. Calls ChunkReviewStore.list_pending_doc_ids() to find doc_ids present in
     the chunk index that have no review document yet.
  2. Processes up to `batch_size` doc_ids per tick (oldest-first ordering is
     delegated to list_pending_doc_ids).
  3. For each pending doc_id: calls ChunkQualityAuditor.audit() via
     asyncio.to_thread() so the synchronous LLM HTTP call does not block the
     event loop, then saves the report via ChunkReviewStore.save().
  4. Sleeps for `interval_seconds` before the next tick.

The scheduler runs as a single asyncio Task created in the FastAPI lifespan
context.  It is cancelled cleanly on service shutdown.

Configuration (environment variables)
--------------------------------------
RAG_AUDIT_INTERVAL_SECONDS  — seconds between ticks (default: 3600 = 1 hour)
RAG_AUDIT_BATCH_SIZE        — max doc_ids audited per tick (default: 20)
RAG_AUDIT_ENABLED           — set to "0" to disable the scheduler entirely
                              (audit API endpoints remain available)

Design constraints
------------------
- No new dependencies: uses only asyncio, the existing LLM client, and the
  two audit classes from CQR-02.
- asyncio.to_thread() wraps every synchronous audit call.  The event loop
  remains responsive during LLM HTTP waits.
- Errors on individual doc_ids are logged and skipped; they do not stop the
  scheduler.
- The scheduler is intentionally simple: no distributed locking, no
  persistent job table.  It is designed for a single ingestion service
  instance (which matches the air-gapped, single-node deployment model).

Public API
----------
    scheduler = AuditScheduler(auditor, review_store, chunk_index)
    task = asyncio.create_task(scheduler.run())
    # ... on shutdown:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    # Status (for /health endpoint)
    info = scheduler.status()
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.ingestion.chunk_quality_auditor import ChunkQualityAuditor
from core.storage.chunk_review_store import ChunkReviewStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable names and defaults
# ---------------------------------------------------------------------------

_ENV_INTERVAL = "RAG_AUDIT_INTERVAL_SECONDS"
_ENV_BATCH    = "RAG_AUDIT_BATCH_SIZE"
_ENV_ENABLED  = "RAG_AUDIT_ENABLED"

_DEFAULT_INTERVAL = 3600   # 1 hour
_DEFAULT_BATCH    = 20


def _read_env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        parsed = int(val)
        if parsed < 1:
            raise ValueError("must be >= 1")
        return parsed
    except ValueError as exc:
        logger.warning(
            "Invalid value for %s=%r (%s); using default %d",
            name, val, exc, default,
        )
        return default


# ---------------------------------------------------------------------------
# Scheduler state (returned by .status())
# ---------------------------------------------------------------------------

@dataclass
class SchedulerStatus:
    enabled: bool
    running: bool
    interval_seconds: int
    batch_size: int
    ticks_completed: int
    last_tick_at: str | None        # ISO-8601 UTC, or None if never run
    last_tick_audited: int          # doc_ids audited on the last tick
    last_tick_failed: int           # doc_ids that failed on the last tick
    total_audited: int              # cumulative since startup
    total_failed: int               # cumulative since startup


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AuditScheduler:
    """
    Incremental audit scheduler.

    Parameters
    ----------
    auditor:
        ChunkQualityAuditor instance.
    review_store:
        ChunkReviewStore instance.
    chunk_index:
        ES alias/index name for the chunk data (used by list_pending_doc_ids).
    interval_seconds:
        Seconds to sleep between ticks.  Reads RAG_AUDIT_INTERVAL_SECONDS if
        not supplied explicitly.
    batch_size:
        Maximum doc_ids to audit per tick.  Reads RAG_AUDIT_BATCH_SIZE if not
        supplied explicitly.
    """

    def __init__(
        self,
        auditor: ChunkQualityAuditor,
        review_store: ChunkReviewStore,
        chunk_index: str,
        *,
        interval_seconds: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._auditor = auditor
        self._review_store = review_store
        self._chunk_index = chunk_index
        self._interval = interval_seconds if interval_seconds is not None \
            else _read_env_int(_ENV_INTERVAL, _DEFAULT_INTERVAL)
        self._batch = batch_size if batch_size is not None \
            else _read_env_int(_ENV_BATCH, _DEFAULT_BATCH)

        # Runtime counters
        self._running = False
        self._ticks = 0
        self._last_tick_at: str | None = None
        self._last_tick_audited = 0
        self._last_tick_failed = 0
        self._total_audited = 0
        self._total_failed = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main loop.  Runs until cancelled.

        Designed to be launched as an asyncio Task:
            task = asyncio.create_task(scheduler.run())

        On CancelledError the loop exits cleanly; the exception is NOT
        re-raised so asyncio.gather(..., return_exceptions=True) sees None.
        """
        self._running = True
        logger.info(
            "AuditScheduler started: interval=%ds batch=%d chunk_index=%s",
            self._interval, self._batch, self._chunk_index,
        )

        try:
            while True:
                await self._tick()
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            logger.info("AuditScheduler cancelled; shutting down cleanly.")
        finally:
            self._running = False

    def status(self) -> SchedulerStatus:
        """Return a snapshot of the scheduler's runtime state."""
        return SchedulerStatus(
            enabled=True,
            running=self._running,
            interval_seconds=self._interval,
            batch_size=self._batch,
            ticks_completed=self._ticks,
            last_tick_at=self._last_tick_at,
            last_tick_audited=self._last_tick_audited,
            last_tick_failed=self._last_tick_failed,
            total_audited=self._total_audited,
            total_failed=self._total_failed,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        """Run one sweep: fetch pending, audit each, save results."""
        tick_audited = 0
        tick_failed = 0

        try:
            pending = await asyncio.to_thread(
                self._review_store.list_pending_doc_ids,
                chunk_index=self._chunk_index,
                limit=self._batch,
            )
        except Exception as exc:
            logger.error("AuditScheduler: failed to fetch pending doc_ids: %s", exc)
            self._record_tick(0, 0)
            return

        if not pending:
            logger.debug("AuditScheduler tick: no pending doc_ids.")
            self._record_tick(0, 0)
            return

        logger.info(
            "AuditScheduler tick: %d pending doc_id(s) to audit.", len(pending)
        )

        for doc_id in pending:
            success = await self._audit_one(doc_id)
            if success:
                tick_audited += 1
            else:
                tick_failed += 1

        self._record_tick(tick_audited, tick_failed)
        logger.info(
            "AuditScheduler tick complete: audited=%d failed=%d",
            tick_audited, tick_failed,
        )

    async def _audit_one(self, doc_id: str) -> bool:
        """
        Audit a single doc_id and persist the result.

        Returns True on success, False if either the audit or the save failed.
        The scheduler continues regardless of the return value.
        """
        try:
            # audit() is synchronous (httpx LLM call inside); run in thread
            report = await asyncio.to_thread(
                self._auditor.audit, doc_id
            )
        except Exception as exc:
            logger.error(
                "AuditScheduler: unexpected error auditing doc_id=%s: %s",
                doc_id, exc,
            )
            return False

        try:
            await asyncio.to_thread(self._review_store.save, report)
        except Exception as exc:
            logger.error(
                "AuditScheduler: failed to save report for doc_id=%s: %s",
                doc_id, exc,
            )
            return False

        logger.debug(
            "AuditScheduler: saved report for doc_id=%s summary=%s",
            doc_id, report.get("audit_summary"),
        )
        return True

    def _record_tick(self, audited: int, failed: int) -> None:
        self._ticks += 1
        self._last_tick_at = datetime.now(timezone.utc).isoformat()
        self._last_tick_audited = audited
        self._last_tick_failed = failed
        self._total_audited += audited
        self._total_failed += failed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def scheduler_enabled() -> bool:
    """Return False if RAG_AUDIT_ENABLED=0, True otherwise."""
    return os.environ.get(_ENV_ENABLED, "1").strip() != "0"
