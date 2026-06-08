"""
core/ingestion/chunk_quality_auditor.py

Offline chunk quality auditor.

Fetches all chunks for a doc_id from Elasticsearch, groups them by
lc_group_id (Phase 2) or parent_id (Phase 1 fallback), then calls the
intranet LLM once per group to audit semantic boundary quality.

Design constraints
------------------
- Read-only: no indexed data is modified, no rebuild triggered.
- No online pipeline is touched.
- lc_group_id is the primary grouping key.  After CQR-01 lands,
  Phase 2 chunks will carry this field in ES.  For Phase 1 data
  (lc_group_id absent), the auditor falls back to parent_id grouping,
  which carries the same structural-sibling semantic.
- LLM call failures degrade gracefully: verdict="audit_failed", no exception
  propagated out of audit().  Matches the Enhancer degradation pattern.
- JSON parse failures follow the same fallback.
- This module exposes only ChunkQualityAuditor.  CLI entry-point and
  FastAPI wiring live in separate files (CQR-03+).

Grouping
--------
Primary:  lc_group_id — set by StructuralChunker / SemanticChunker (Phase 2).
          Chunks with the same lc_group_id were cut from the same "group text"
          (built by late_chunking_utils.build_group_text) and their embeddings
          were produced from the same token matrix.  Auditing them together
          gives the LLM the full context in which the boundary decision was made.

Fallback: parent_id — Phase 1 chunks do not carry lc_group_id in ES.
          Siblings sharing the same parent_id were cut from the same structural
          unit (heading / clause / paragraph).  Root chunks (parent_id=None)
          are each treated as their own single-chunk group.

The grouping_key_type field in the report ("lc_group_id" vs
"parent_id_fallback") tells the operator which strategy was used.

ES scroll
---------
Chunks are fetched with a term query on doc_id + scroll API.  Page size is
500 documents.  The scroll context is always cleared in a finally block.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from elasticsearch import Elasticsearch

from core.ingestion.llm_client import LLMCallError, LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_VERDICTS = frozenset({"ok", "should_merge", "should_split", "boundary_issue"})
_VALID_CONFIDENCE = frozenset({"high", "medium", "low"})

# Characters allocated per chunk text inside the audit prompt.
# Total group text budget = _CHARS_PER_CHUNK * chunks_in_group,
# capped at _MAX_GROUP_CHARS to stay within the LLM context window.
_CHARS_PER_CHUNK = 800
_MAX_GROUP_CHARS = 4000

_ES_PAGE_SIZE = 500
_ES_SCROLL_TTL = "2m"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _ChunkHit:
    """Minimal ES hit representation used internally by the auditor."""
    chunk_id: str
    doc_id: str
    parent_id: str | None
    hierarchy_level: int
    position: int
    text: str           # ES "text" field = context_text from ingestion
    lc_group_id: str | None


@dataclass
class GroupAuditResult:
    """Audit result for one sibling group."""
    group_key: str              # lc_group_id value or parent_id fallback key
    grouping_key_type: str      # "lc_group_id" | "parent_id_fallback"
    chunk_ids: list[str]
    verdict: str                # ok | should_merge | should_split | boundary_issue
                                # | audit_failed | skipped_no_group_key
    confidence: str | None      # high | medium | low | None
    reason: str
    affected_chunk_ids: list[str]


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一名企业知识库切块质量审计员。你的任务是评估给定的文档切块是否具有合理的语义边界。

【输出规则】
- 只输出合法 JSON，无任何前缀、markdown 围栏或解释文字。
- 输出对象只包含以下字段：verdict、confidence、reason、affected_chunk_ids。
- verdict 枚举值（严格限定，不得使用其他值）：
  - "ok"：各 chunk 语义完整，边界合理，token 分布均匀。
  - "should_merge"：相邻 chunk 语义高度连续，人为切断了同一概念，建议合并。
  - "should_split"：单个 chunk 内包含多个不相关主题，建议拆分。
  - "boundary_issue"：切断了句子或关键实体，导致上下文丢失。
- confidence 枚举值："high" | "medium" | "low"
- reason：一句话，语言与文档语言保持一致，描述判断依据。禁止包含修改建议的正文文本。
- affected_chunk_ids：有问题的 chunk_id 列表；若 verdict="ok" 则为空列表。"""

_USER_PROMPT_TEMPLATE = """\
以下是同一{group_context}下的 {n} 个相邻 chunk，请审计其切块质量。

{chunk_blocks}

请输出 JSON 审计结果。"""

_CHUNK_BLOCK_TEMPLATE = "--- chunk {idx} / chunk_id={chunk_id} ---\n{text}"

_GROUP_CONTEXT_LC = "Late Chunking 分组（lc_group_id={gk}）"
_GROUP_CONTEXT_PARENT = "结构性父节点（parent_id={gk}）"
_GROUP_CONTEXT_ROOT = "文档根节点"


def _build_user_prompt(
    hits: list[_ChunkHit],
    group_key: str,
    grouping_type: str,
) -> str:
    per_chunk = min(_CHARS_PER_CHUNK, _MAX_GROUP_CHARS // max(len(hits), 1))

    blocks = "\n\n".join(
        _CHUNK_BLOCK_TEMPLATE.format(
            idx=i + 1,
            chunk_id=h.chunk_id,
            text=h.text[:per_chunk],
        )
        for i, h in enumerate(hits)
    )

    if grouping_type == "lc_group_id":
        ctx = _GROUP_CONTEXT_LC.format(gk=group_key)
    elif group_key.startswith("root:"):
        ctx = _GROUP_CONTEXT_ROOT
    else:
        ctx = _GROUP_CONTEXT_PARENT.format(gk=group_key)

    return _USER_PROMPT_TEMPLATE.format(
        group_context=ctx,
        n=len(hits),
        chunk_blocks=blocks,
    )


# ---------------------------------------------------------------------------
# ES fetch
# ---------------------------------------------------------------------------

def _fetch_chunks(
    es: Elasticsearch,
    index: str,
    doc_id: str,
) -> list[_ChunkHit]:
    """
    Fetch all chunks for doc_id from ES using scroll.
    Returns hits sorted by (hierarchy_level, position).
    """
    query: dict[str, Any] = {
        "query": {"term": {"doc_id": doc_id}},
        "_source": [
            "chunk_id", "doc_id", "parent_id",
            "hierarchy_level", "position",
            "text", "lc_group_id",
        ],
        "sort": [{"hierarchy_level": "asc"}, {"position": "asc"}],
        "size": _ES_PAGE_SIZE,
    }

    hits: list[_ChunkHit] = []
    resp = es.search(index=index, body=query, scroll=_ES_SCROLL_TTL)
    scroll_id: str = resp["_scroll_id"]

    try:
        while True:
            batch = resp["hits"]["hits"]
            if not batch:
                break
            for h in batch:
                src = h["_source"]
                hits.append(_ChunkHit(
                    chunk_id=src.get("chunk_id", h["_id"]),
                    doc_id=src.get("doc_id", doc_id),
                    parent_id=src.get("parent_id"),
                    hierarchy_level=int(src.get("hierarchy_level", 0)),
                    position=int(src.get("position", 0)),
                    text=src.get("text", ""),
                    lc_group_id=src.get("lc_group_id"),
                ))
            resp = es.scroll(scroll_id=scroll_id, scroll=_ES_SCROLL_TTL)
    finally:
        try:
            es.clear_scroll(scroll_id=scroll_id)
        except Exception:
            pass  # best-effort cleanup

    return hits


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_hits(
    hits: list[_ChunkHit],
) -> tuple[list[tuple[str, str, list[_ChunkHit]]], list[_ChunkHit]]:
    """
    Group hits by lc_group_id when present, otherwise by parent_id.

    Returns
    -------
    (groups, ungroupable)

    groups      : list of (group_key, grouping_type, sorted_hits)
                  grouping_type = "lc_group_id" | "parent_id_fallback"
    ungroupable : hits with no assignable key (empty in normal operation)
    """
    has_lc = any(h.lc_group_id is not None for h in hits)
    grouping_type = "lc_group_id" if has_lc else "parent_id_fallback"

    buckets: dict[str, list[_ChunkHit]] = {}
    ungroupable: list[_ChunkHit] = []

    for h in hits:
        if has_lc:
            key = h.lc_group_id
        else:
            key = h.parent_id if h.parent_id is not None else f"root:{h.chunk_id}"

        if key is None:
            ungroupable.append(h)
            continue

        buckets.setdefault(key, []).append(h)

    groups = [
        (
            gk,
            grouping_type,
            sorted(members, key=lambda h: (h.hierarchy_level, h.position)),
        )
        for gk, members in buckets.items()
    ]
    groups.sort(key=lambda t: (t[2][0].hierarchy_level, t[2][0].position))

    return groups, ungroupable


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(
    raw: str,
    valid_chunk_ids: set[str],
) -> tuple[str, str | None, str, list[str]]:
    """
    Parse the LLM JSON response.

    Returns (verdict, confidence, reason, affected_chunk_ids).
    On any parse or validation error returns ("audit_failed", None, msg, []).
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(l for l in lines if not l.startswith("```")).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        msg = f"JSON 解析失败: {exc}. 响应片段: {raw[:200]!r}"
        logger.warning("Audit LLM parse error: %s", msg)
        return "audit_failed", None, msg, []

    if not isinstance(data, dict):
        msg = f"LLM 返回非 object 类型: {type(data).__name__}"
        logger.warning("Audit LLM non-dict response: %s", msg)
        return "audit_failed", None, msg, []

    verdict = str(data.get("verdict", ""))
    if verdict not in _VALID_VERDICTS:
        msg = f"verdict 超出枚举范围: {verdict!r}"
        logger.warning("Audit verdict out of enum: %s", msg)
        return "audit_failed", None, msg, []

    confidence = data.get("confidence")
    if confidence not in _VALID_CONFIDENCE:
        logger.warning("Audit confidence %r out of enum; set to None", confidence)
        confidence = None

    reason = str(data.get("reason", "")).strip() or "（LLM 未提供 reason）"

    raw_affected = data.get("affected_chunk_ids", [])
    affected: list[str] = []
    if isinstance(raw_affected, list):
        affected = [str(x) for x in raw_affected if str(x) in valid_chunk_ids]

    return verdict, confidence, reason, affected


# ---------------------------------------------------------------------------
# Per-group audit
# ---------------------------------------------------------------------------

def _audit_group(
    llm: LLMClient,
    group_key: str,
    grouping_type: str,
    hits: list[_ChunkHit],
    *,
    dry_run: bool = False,
) -> GroupAuditResult:
    """Run the LLM audit for one sibling group."""
    chunk_ids = [h.chunk_id for h in hits]
    user_prompt = _build_user_prompt(hits, group_key, grouping_type)

    if dry_run:
        logger.info(
            "[dry-run] group=%s type=%s chunks=%d",
            group_key, grouping_type, len(hits),
        )
        logger.debug("[dry-run] system_prompt:\n%s", _SYSTEM_PROMPT)
        logger.debug("[dry-run] user_prompt:\n%s", user_prompt)
        return GroupAuditResult(
            group_key=group_key,
            grouping_key_type=grouping_type,
            chunk_ids=chunk_ids,
            verdict="dry_run",
            confidence=None,
            reason="dry-run 模式，未实际调用 LLM",
            affected_chunk_ids=[],
        )

    try:
        raw = llm.chat(_SYSTEM_PROMPT, user_prompt, temperature=0.0)
    except LLMCallError as exc:
        logger.warning("LLM call failed for group=%s: %s", group_key, exc)
        return GroupAuditResult(
            group_key=group_key,
            grouping_key_type=grouping_type,
            chunk_ids=chunk_ids,
            verdict="audit_failed",
            confidence=None,
            reason=f"LLM 调用失败: {exc}",
            affected_chunk_ids=[],
        )

    verdict, confidence, reason, affected = _parse_llm_response(
        raw, valid_chunk_ids=set(chunk_ids)
    )
    return GroupAuditResult(
        group_key=group_key,
        grouping_key_type=grouping_type,
        chunk_ids=chunk_ids,
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        affected_chunk_ids=affected,
    )


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _build_report(
    doc_id: str,
    results: list[GroupAuditResult],
    ungroupable: list[_ChunkHit],
) -> dict[str, Any]:
    _COUNTABLE = {"ok", "should_merge", "should_split", "boundary_issue", "audit_failed"}
    summary: dict[str, int] = {}
    for r in results:
        if r.verdict in _COUNTABLE:
            summary[r.verdict] = summary.get(r.verdict, 0) + 1

    skipped_entries = [
        {
            "lc_group_id": None,
            "grouping_key_type": "skipped_no_group_key",
            "chunk_ids": [h.chunk_id],
            "verdict": "skipped_no_group_key",
            "confidence": None,
            "reason": "chunk 无 lc_group_id 且无可用分组键，跳过审计",
            "affected_chunk_ids": [],
        }
        for h in ungroupable
    ]

    groups_json = [
        {
            "lc_group_id": r.group_key,
            "grouping_key_type": r.grouping_key_type,
            "chunk_ids": r.chunk_ids,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "reason": r.reason,
            "affected_chunk_ids": r.affected_chunk_ids,
        }
        for r in results
    ] + skipped_entries

    return {
        "doc_id": doc_id,
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "total_groups": len(results) + len(ungroupable),
        "skipped_groups": len(ungroupable),
        "audit_summary": summary,
        "groups": groups_json,
    }


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class ChunkQualityAuditor:
    """
    Audits chunk quality for a single doc_id.

    Callable by:
    - FastAPI router (CQR-03): POST /audit/doc/{doc_id}
    - Scheduler (CQR-04): background incremental sweep
    - CLI wrapper (CQR-03): python -m core.ingestion.chunk_quality_auditor

    Parameters
    ----------
    es:
        Raw elasticsearch.Elasticsearch client.
    llm:
        LLMClient instance.
    chunk_index:
        ES alias/index name for the main chunk data (e.g. "rag_chunks").
    """

    def __init__(
        self,
        es: Elasticsearch,
        llm: LLMClient,
        chunk_index: str = "rag_chunks",
    ) -> None:
        self._es = es
        self._llm = llm
        self._chunk_index = chunk_index

    def audit(
        self,
        doc_id: str,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Fetch, group, and audit all chunks for doc_id.

        Returns the report dict (JSON-serialisable).  Never raises — any
        per-group LLM failure is captured as verdict="audit_failed" in the
        report.  ES fetch failures produce a minimal failed report.

        Parameters
        ----------
        doc_id:
            The document to audit.
        dry_run:
            If True, build and log prompts but do not call the LLM.
        """
        logger.info(
            "ChunkQualityAuditor.audit: doc_id=%s index=%s dry_run=%s",
            doc_id, self._chunk_index, dry_run,
        )

        try:
            hits = _fetch_chunks(self._es, self._chunk_index, doc_id)
        except Exception as exc:
            logger.error("Failed to fetch chunks for doc_id=%s: %s", doc_id, exc)
            return {
                "doc_id": doc_id,
                "audited_at": datetime.now(timezone.utc).isoformat(),
                "total_groups": 0,
                "skipped_groups": 0,
                "audit_summary": {"audit_failed": 1},
                "groups": [{
                    "lc_group_id": None,
                    "grouping_key_type": "fetch_error",
                    "chunk_ids": [],
                    "verdict": "audit_failed",
                    "confidence": None,
                    "reason": f"ES 拉取失败: {exc}",
                    "affected_chunk_ids": [],
                }],
            }

        if not hits:
            logger.warning("No chunks found for doc_id=%s", doc_id)
            return _build_report(doc_id, [], [])

        logger.info(
            "Fetched %d chunks for doc_id=%s; grouping...", len(hits), doc_id
        )

        groups, ungroupable = _group_hits(hits)
        logger.info(
            "doc_id=%s: %d groups, %d ungroupable",
            doc_id, len(groups), len(ungroupable),
        )

        results: list[GroupAuditResult] = []
        for group_key, grouping_type, group_hits in groups:
            result = _audit_group(
                self._llm,
                group_key,
                grouping_type,
                group_hits,
                dry_run=dry_run,
            )
            results.append(result)
            logger.debug(
                "Group %s → verdict=%s confidence=%s",
                group_key, result.verdict, result.confidence,
            )

        return _build_report(doc_id, results, ungroupable)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
# Usage:
#   python -m core.ingestion.chunk_quality_auditor \
#       --doc-id <doc_id> \
#       --config <path_to_app_config.json> \
#       [--index <es_alias>] \
#       [--es-host http://localhost:9200] \
#       [--output <report.json>] \
#       [--dry-run] \
#       [--log-level DEBUG|INFO|WARNING|ERROR]

def _cli() -> None:
    import argparse
    import json
    import os
    import sys

    from elasticsearch import Elasticsearch

    from core.config.loader import ConfigLoadError, load_config
    from core.ingestion.llm_client import LLMCallError, LLMClient

    parser = argparse.ArgumentParser(
        prog="python -m core.ingestion.chunk_quality_auditor",
        description="Offline chunk quality audit. Read-only; no data is modified.",
    )
    parser.add_argument("--doc-id", required=True, help="Document ID to audit.")
    parser.add_argument(
        "--config", required=True,
        help="Path to AppConfig JSON (used for LLM endpoint settings).",
    )
    parser.add_argument(
        "--index", default=None,
        help="ES alias/index to query. Defaults to AppConfig storage base_name.",
    )
    parser.add_argument(
        "--es-host", default=None,
        help="ES host URL. Defaults to AppConfig storage hosts[0].",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write JSON report to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log prompts without calling the LLM; report is NOT saved.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args.config)
    except ConfigLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    llm_cfg = cfg.models.enhancement_llm
    if llm_cfg is None:
        print(
            "ERROR: models.enhancement_llm is not configured. "
            "The auditor requires an LLM endpoint.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        llm = LLMClient.from_config(llm_cfg)
    except LLMCallError as exc:
        print(f"ERROR: LLM client init failed: {exc}", file=sys.stderr)
        sys.exit(1)

    storage = getattr(cfg, "storage", None)
    if args.es_host:
        es_host = args.es_host
    elif storage is not None:
        es_host = storage.elasticsearch.hosts[0]
    else:
        es_host = os.environ.get("RAG_ES_HOSTS", "http://localhost:9200").split(",")[0].strip()

    if args.index:
        index = args.index
    elif storage is not None:
        index = storage.base_name
    else:
        index = os.environ.get("RAG_STORAGE_BASE_NAME", "rag_chunks")

    timeout = storage.elasticsearch.timeout_seconds if storage else 30
    max_retries = storage.elasticsearch.max_retries if storage else 3

    es = Elasticsearch(
        [es_host],
        timeout=timeout,
        max_retries=max_retries,
        retry_on_timeout=True,
        http_compress=True,
    )

    auditor = ChunkQualityAuditor(es=es, llm=llm, chunk_index=index)

    try:
        report = auditor.audit(args.doc_id, dry_run=args.dry_run)
    except Exception as exc:
        print(f"ERROR: Audit failed unexpectedly: {exc}", file=sys.stderr)
        sys.exit(1)

    report_json = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report_json)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report_json)


if __name__ == "__main__":
    _cli()
