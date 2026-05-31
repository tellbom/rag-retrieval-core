"""
core/query/rewriter.py

QueryRewriter: optionally rewrites a normalised query using the intranet LLM.

Behaviour
---------
- If `enabled=False` (from config): returns None immediately, no LLM call.
- If LLM call fails for any reason: logs a warning, returns None (degraded).
  The preprocessor uses normalized_query as effective_query in that case.
  The query pipeline NEVER blocks or errors because of a rewrite failure.

Rewrite prompt goal
--------------------
The LLM rewrites the user's query to be more explicit and retrieval-friendly:
- Expand abbreviations and acronyms.
- Make implicit intent explicit.
- Remove conversational filler.
- Preserve the original language (Chinese in, Chinese out).

The rewrite is a single short string — NOT an expansion into multiple queries
(that is HyDE/multi-query, a Phase 2 feature).

Reuse of LLMClient
------------------
QueryRewriter uses the same LLMClient as the ingestion Enhancer, since both
call the intranet LLM over the OpenAI-compatible API.  The client is
constructed from `models.enhancement_llm` config (same endpoint).

Public API
----------
    rewriter = QueryRewriter.from_config(cfg)
    rewritten = rewriter.rewrite("用户查询")   # str | None
"""

from __future__ import annotations

import logging

from core.config.models import AppConfig
from core.ingestion.llm_client import LLMCallError, LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a search query optimisation assistant for an enterprise knowledge base.
Rewrite the user's query to be clearer, more explicit, and better suited for \
document retrieval. Rules:
- Output ONLY the rewritten query text. No explanation, no quotes, no preamble.
- Preserve the original language (Chinese queries → Chinese output).
- Keep it concise (one sentence or phrase, max 100 characters).
- Expand abbreviations. Make implicit intent explicit.
- If the query is already clear and explicit, return it unchanged.\
"""


class QueryRewriter:
    """
    Optionally rewrites queries using the intranet LLM.

    Parameters
    ----------
    llm_client:  LLMClient instance. None → always disabled (no LLM call).
    enabled:     Global rewrite switch. False → skip even if client is set.
    timeout:     Per-call timeout (seconds). Short by default — query path
                 is latency-sensitive; rewrite failure must degrade quickly.
    """

    def __init__(
        self,
        llm_client: LLMClient | None,
        *,
        enabled: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self._client = llm_client
        self._enabled = enabled
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, normalized_query: str) -> str | None:
        """
        Attempt to rewrite the query.

        Returns
        -------
        str   — the rewritten query if successful.
        None  — if disabled, LLM unavailable, or rewrite is nonsensical.

        Never raises — all failures degrade to None.
        """
        if not self._enabled or self._client is None:
            return None

        if not normalized_query.strip():
            return None

        try:
            rewritten = self._client.chat(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=normalized_query,
                temperature=0.0,
            )
            rewritten = rewritten.strip()

            # Sanity checks: if the LLM returns something unusable, degrade
            if not rewritten:
                logger.debug("QueryRewriter: empty response, using original")
                return None
            if len(rewritten) > 500:
                logger.debug(
                    "QueryRewriter: response too long (%d chars), using original",
                    len(rewritten),
                )
                return None

            logger.debug(
                "QueryRewriter: %r → %r", normalized_query[:60], rewritten[:60]
            )
            return rewritten

        except LLMCallError as exc:
            logger.warning(
                "QueryRewriter: LLM call failed (degrading to original): %s", exc
            )
            return None
        except Exception as exc:
            logger.warning(
                "QueryRewriter: unexpected error (degrading to original): %s", exc
            )
            return None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: AppConfig, *, enabled: bool | None = None) -> "QueryRewriter":
        """
        Build a QueryRewriter from AppConfig.

        Parameters
        ----------
        cfg:     AppConfig.
        enabled: Capability switch for this rewriter instance. Per-request
                 use is still controlled by QueryPreprocessor.process().
        """
        llm_cfg = cfg.models.enhancement_llm
        if llm_cfg is None:
            logger.debug("QueryRewriter: no LLM configured, rewrite disabled")
            return cls(llm_client=None, enabled=False)

        client = LLMClient.from_config(llm_cfg)
        return cls(
            llm_client=client,
            enabled=enabled if enabled is not None else True,
        )

    @classmethod
    def disabled(cls) -> "QueryRewriter":
        """Return a rewriter that is permanently disabled (no LLM calls ever)."""
        return cls(llm_client=None, enabled=False)
