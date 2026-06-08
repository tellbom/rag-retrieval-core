"""Iterative retrieval wrapper for the online query pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from core.ingestion.llm_client import LLMCallError, LLMClient
from core.pipeline.query_pipeline import QueryPipeline, QueryPipelineResult
from core.query.context_builder import ContextBuilder, Citation
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate

logger = logging.getLogger(__name__)

_MAX_ROUNDS_HARD_CAP = 3
_DEFAULT_MAX_ROUNDS = 2
_TOPIC_ABSENT_SENTINEL = "TOPIC_ABSENT"
_FACTUAL_NEGATION_SENTINEL = "FACTUAL_NEGATION"

_SELFEVAL_SYSTEM_PROMPT = """\
You are a retrieval quality evaluator for an enterprise knowledge base.

Output rules
------------
- LANGUAGE RULE (highest priority): every text field you produce
  (missing_aspects and sub_queries) MUST be written in the same language
  as the original question. If the question is Chinese, output Chinese.
- Return one JSON object only. No preamble, no markdown fences.
- Use exactly these fields and do not add extra fields:
{
  "sufficient": true,
  "confidence": "high",
  "missing_aspects": "",
  "sub_queries": []
}

Field rules:
- sufficient is the only stop/continue signal.
- confidence must be one of "high", "medium", "low".
- missing_aspects is a plain string. Use the exact sentinel values below
  when they apply; otherwise write a short description of what is missing.
- sub_queries is an array of strings in the same language as the question.
- If sufficient=false, provide 1 to 2 focused follow-up retrieval queries.
- Each sub_query MUST contain at least one distinguishing term that does
  not appear in the original question: a product name, organisation name,
  date, technical identifier, financial term, or close synonym for the
  missing fact.
- If sufficient=true, sub_queries must be [].

Decision rules:
1. LANGUAGE: Always match the original question language in text fields.

2. TOPIC_ABSENT: If the retrieved context clearly confirms that the topic
   is not covered by the knowledge base at all, set:
     sufficient=true, missing_aspects="TOPIC_ABSENT", sub_queries=[].

3. FACTUAL_NEGATION: If the question asserts a specific fact (e.g. "Did X
   do Y?", "Is it true that P equals Q?") AND the retrieved context shows
   the opposite is true (i.e. the answer contradicts the question's
   premise), set:
     sufficient=true, missing_aspects="FACTUAL_NEGATION", sub_queries=[].
   Examples of FACTUAL_NEGATION:
   - Question: "Did Microsoft say Fairwater would use millions of gallons
     of water?" — Context says Fairwater uses restaurant-level water (far
     less). → FACTUAL_NEGATION.
   - Question: "Has Anthropic said Claude never contributed to merged
     code?" — Context says Claude contributed to >80% of merged code.
     → FACTUAL_NEGATION.
   - Question: "Was xAI Series E only $2 billion, below a $15B target?" —
     Context shows the actual amount was much higher. → FACTUAL_NEGATION.
   Do NOT use FACTUAL_NEGATION when the question is neutral or open-ended
   and the answer simply says "no additional information found".

4. INSUFFICIENT_RESPONSE: If the answer says there is not enough
   information, decide whether the topic is absent (use TOPIC_ABSENT) or
   likely exists but was missed (set sufficient=false and create focused
   sub_queries).

5. PARTIAL_ANSWER: If the answer is on-topic but misses a material fact
   such as a number, date, name, filing, or technical parameter, set
   sufficient=false and generate sub_queries for that specific gap.

6. OVER-TRIGGERING GUARD: Do not mark sufficient=false merely because a
   longer answer is possible. Only mark false when an important aspect is
   concretely missing.
"""

_SELFEVAL_USER_TEMPLATE = """\
Original question:
{query}

Retrieved answer:
{answer}

Context blocks used ({block_count} blocks):
{context_summary}

Evaluate whether the answer sufficiently addresses the question.
Return the JSON object now.
"""


@dataclass
class IterativeRetrievalResult:
    query: str
    processed_query_text: str
    answer_text: str
    grounded: bool
    reranked: bool
    citations: list[Citation]
    context_blocks_used: int
    llm_model: str
    retriever_candidate_counts: dict[str, int]
    fused_count: int
    rerank_input_count: int
    iterations: int = 1
    sub_queries: list[str] = field(default_factory=list)
    iterative_enabled: bool = False
    self_eval_sufficient: bool | None = None
    self_eval_confidence: str = ""
    self_eval_missing: str = ""
    topic_absent: bool = False
    factual_negation: bool = False


@dataclass
class _SelfEvalResult:
    sufficient: bool
    confidence: str
    missing_aspects: str
    sub_queries: list[str]
    parse_ok: bool


class SelfEvaluator:
    """LLM-based answer sufficiency evaluator with conservative fallback."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._client = llm_client

    def evaluate(self, query: str, result: QueryPipelineResult) -> _SelfEvalResult:
        context_summary = _summarise_citations(result.answer.citations)
        user_prompt = _SELFEVAL_USER_TEMPLATE.format(
            query=query,
            answer=result.answer.answer,
            block_count=result.answer.context_used,
            context_summary=context_summary,
        )
        try:
            raw = self._client.chat(
                system_prompt=_SELFEVAL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )
        except LLMCallError as exc:
            logger.warning(
                "IterativeRetriever: self-eval LLM failed, using round-1 result: %s",
                exc,
            )
            return _fallback_sufficient()

        return self._parse(raw)

    def _parse(self, raw: str) -> _SelfEvalResult:
        text = _strip_markdown_fence(raw.strip())
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("IterativeRetriever: self-eval returned invalid JSON")
            return _fallback_sufficient()

        if not isinstance(parsed, dict):
            logger.warning("IterativeRetriever: self-eval JSON was not an object")
            return _fallback_sufficient()

        sufficient = parsed.get("sufficient")
        if not isinstance(sufficient, bool):
            logger.warning("IterativeRetriever: self-eval missing boolean sufficient")
            return _fallback_sufficient()

        confidence = parsed.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            logger.warning("IterativeRetriever: self-eval invalid confidence")
            return _fallback_sufficient()

        missing_aspects = parsed.get("missing_aspects")
        if not isinstance(missing_aspects, str):
            logger.warning("IterativeRetriever: self-eval missing string missing_aspects")
            return _fallback_sufficient()

        sub_queries_raw = parsed.get("sub_queries")
        if not isinstance(sub_queries_raw, list):
            logger.warning("IterativeRetriever: self-eval missing list sub_queries")
            return _fallback_sufficient()
        sub_queries = [
            item.strip()
            for item in sub_queries_raw
            if isinstance(item, str) and item.strip()
        ][:2]

        if not sufficient and not sub_queries:
            logger.warning(
                "IterativeRetriever: self-eval insufficient but no sub_queries"
            )
            return _fallback_sufficient()

        return _SelfEvalResult(
            sufficient=sufficient,
            confidence=confidence,
            missing_aspects=missing_aspects,
            sub_queries=[] if sufficient else sub_queries,
            parse_ok=True,
        )


class IterativeRetriever:
    """Run the base query pipeline, self-evaluate, and optionally search again."""

    def __init__(
        self,
        *,
        pipeline: QueryPipeline,
        self_evaluator: SelfEvaluator,
        context_builder: ContextBuilder,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> None:
        self._pipeline = pipeline
        self._self_evaluator = self_evaluator
        self._context_builder = context_builder
        self._max_rounds = max(1, min(max_rounds, _MAX_ROUNDS_HARD_CAP))

    def run(
        self,
        raw_query: str,
        *,
        filters: QueryFilters | None = None,
        business_type: str = "",
        enable_rewrite: bool = False,
    ) -> IterativeRetrievalResult:
        try:
            r1 = self._pipeline.run(
                raw_query,
                filters=filters,
                business_type=business_type,
                enable_rewrite=enable_rewrite,
            )
        except Exception as exc:
            logger.warning(
                "IterativeRetriever: round-1 pipeline failed (%s), returning empty",
                exc,
            )
            return self._empty_result(raw_query)

        eval_result = self._self_evaluator.evaluate(raw_query, r1)
        if eval_result.sufficient or self._max_rounds < 2:
            topic_absent = (
                eval_result.sufficient
                and eval_result.missing_aspects.strip() == _TOPIC_ABSENT_SENTINEL
            )
            factual_negation = (
                eval_result.sufficient
                and eval_result.missing_aspects.strip() == _FACTUAL_NEGATION_SENTINEL
            )
            if topic_absent:
                logger.info(
                    "IterativeRetriever: TOPIC_ABSENT detected; "
                    "clearing citations for query=%r",
                    raw_query[:80],
                )
                return self._empty_result_from(
                    r1, eval_result=eval_result,
                    iterations=1, topic_absent=True, factual_negation=False,
                )
            if factual_negation:
                logger.info(
                    "IterativeRetriever: FACTUAL_NEGATION detected; "
                    "clearing citations for query=%r",
                    raw_query[:80],
                )
                return self._empty_result_from(
                    r1, eval_result=eval_result,
                    iterations=1, topic_absent=False, factual_negation=True,
                )
            return self._from_pipeline_result(
                r1,
                iterations=1,
                sub_queries=[],
                eval_result=eval_result,
                enabled=True,
                topic_absent=False,
                factual_negation=False,
            )

        r2_results: list[QueryPipelineResult] = []
        for sub_query in eval_result.sub_queries:
            try:
                r2_results.append(
                    self._pipeline.run(
                        sub_query,
                        filters=filters,
                        business_type=business_type,
                        enable_rewrite=enable_rewrite,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "IterativeRetriever: round-2 pipeline failed (%s), "
                    "using round-1 result",
                    exc,
                )
                return self._from_pipeline_result(
                    r1,
                    iterations=1,
                    sub_queries=eval_result.sub_queries,
                    eval_result=eval_result,
                    enabled=True,
                    topic_absent=False,
                    factual_negation=False,
                )

        merged = _merge_candidates(
            _extract_candidates(r1),
            *[_extract_candidates(result) for result in r2_results],
        )
        if not merged:
            return self._from_pipeline_result(
                r1,
                iterations=1,
                sub_queries=eval_result.sub_queries,
                eval_result=eval_result,
                enabled=True,
                topic_absent=False,
                factual_negation=False,
            )

        try:
            context = self._context_builder.build(
                raw_query,
                merged,
                reranked=any(result.reranked for result in [r1, *r2_results]),
                qdrant=self._pipeline._qdrant,
                qdrant_collection=self._pipeline._qdrant_collection,
            )
            answer = self._pipeline._answer_generator.generate(raw_query, context)
        except Exception as exc:
            logger.warning(
                "IterativeRetriever: final context/answer failed (%s), "
                "using round-1 result",
                exc,
            )
            return self._from_pipeline_result(
                r1,
                iterations=1,
                sub_queries=eval_result.sub_queries,
                eval_result=eval_result,
                enabled=True,
                topic_absent=False,
                factual_negation=False,
            )

        retriever_counts: dict[str, int] = dict(r1.retriever_candidate_counts)
        for result in r2_results:
            for key, value in result.retriever_candidate_counts.items():
                retriever_counts[key] = retriever_counts.get(key, 0) + value

        return IterativeRetrievalResult(
            query=raw_query,
            processed_query_text=r1.processed_query_text,
            answer_text=answer.answer,
            grounded=answer.grounded,
            reranked=answer.reranked,
            citations=answer.citations,
            context_blocks_used=answer.context_used,
            llm_model=answer.llm_model,
            retriever_candidate_counts=retriever_counts,
            fused_count=len(merged),
            rerank_input_count=len(merged),
            iterations=2,
            sub_queries=eval_result.sub_queries,
            iterative_enabled=True,
            self_eval_sufficient=eval_result.sufficient,
            self_eval_confidence=eval_result.confidence,
            self_eval_missing=eval_result.missing_aspects,
            topic_absent=False,
            factual_negation=False,
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: QueryPipeline,
        context_builder: ContextBuilder,
        llm_client: LLMClient,
        *,
        max_rounds: int = _DEFAULT_MAX_ROUNDS,
    ) -> "IterativeRetriever":
        return cls(
            pipeline=pipeline,
            self_evaluator=SelfEvaluator(llm_client),
            context_builder=context_builder,
            max_rounds=max_rounds,
        )

    @staticmethod
    def _from_pipeline_result(
        r: QueryPipelineResult,
        *,
        iterations: int,
        sub_queries: list[str],
        eval_result: _SelfEvalResult | None,
        enabled: bool,
        topic_absent: bool = False,
        factual_negation: bool = False,
    ) -> IterativeRetrievalResult:
        return IterativeRetrievalResult(
            query=r.query,
            processed_query_text=r.processed_query_text,
            answer_text=r.answer.answer,
            grounded=r.answer.grounded,
            reranked=r.reranked,
            citations=r.answer.citations,
            context_blocks_used=r.answer.context_used,
            llm_model=r.answer.llm_model,
            retriever_candidate_counts=r.retriever_candidate_counts,
            fused_count=r.fused_count,
            rerank_input_count=r.rerank_input_count,
            iterations=iterations,
            sub_queries=sub_queries,
            iterative_enabled=enabled,
            self_eval_sufficient=(
                eval_result.sufficient if eval_result is not None else None
            ),
            self_eval_confidence=(
                eval_result.confidence if eval_result is not None else ""
            ),
            self_eval_missing=(
                eval_result.missing_aspects if eval_result is not None else ""
            ),
            topic_absent=topic_absent,
            factual_negation=factual_negation,
        )

    @staticmethod
    def _empty_result_from(
        r: QueryPipelineResult,
        *,
        eval_result: _SelfEvalResult,
        iterations: int,
        topic_absent: bool,
        factual_negation: bool,
    ) -> IterativeRetrievalResult:
        """Return a result with empty citations.

        Used when TOPIC_ABSENT or FACTUAL_NEGATION is detected so that
        the API response carries no doc references and the eval runner
        sees an empty retrieved_doc_ids list.
        """
        from core.query.answer_generator import _DEFAULT_INSUFFICIENT_CONTEXT_ZH

        return IterativeRetrievalResult(
            query=r.query,
            processed_query_text=r.processed_query_text,
            answer_text=_DEFAULT_INSUFFICIENT_CONTEXT_ZH,
            grounded=False,
            reranked=r.reranked,
            citations=[],            # ← cleared
            context_blocks_used=0,   # ← cleared
            llm_model=r.answer.llm_model,
            retriever_candidate_counts=r.retriever_candidate_counts,
            fused_count=r.fused_count,
            rerank_input_count=r.rerank_input_count,
            iterations=iterations,
            sub_queries=[],
            iterative_enabled=True,
            self_eval_sufficient=eval_result.sufficient,
            self_eval_confidence=eval_result.confidence,
            self_eval_missing=eval_result.missing_aspects,
            topic_absent=topic_absent,
            factual_negation=factual_negation,
        )

    @staticmethod
    def _empty_result(raw_query: str) -> IterativeRetrievalResult:
        from core.query.answer_generator import _DEFAULT_INSUFFICIENT_CONTEXT_ZH

        return IterativeRetrievalResult(
            query=raw_query,
            processed_query_text=raw_query,
            answer_text=_DEFAULT_INSUFFICIENT_CONTEXT_ZH,
            grounded=False,
            reranked=False,
            citations=[],
            context_blocks_used=0,
            llm_model="",
            retriever_candidate_counts={},
            fused_count=0,
            rerank_input_count=0,
            iterations=0,
            sub_queries=[],
            iterative_enabled=True,
            topic_absent=False,
            factual_negation=False,
        )


def _fallback_sufficient() -> _SelfEvalResult:
    return _SelfEvalResult(
        sufficient=True,
        confidence="low",
        missing_aspects="",
        sub_queries=[],
        parse_ok=False,
    )


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _summarise_citations(citations: list[Citation]) -> str:
    if not citations:
        return "(no citations)"
    lines: list[str] = []
    for citation in citations[:8]:
        title = citation.title or citation.doc_id
        lines.append(f"[{citation.index}] {title} ({citation.doc_id})")
    return "\n".join(lines)


def _extract_candidates(r: QueryPipelineResult) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for citation in r.answer.citations:
        candidates.append(
            RetrievalCandidate(
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                text=citation.extra.get("text", ""),
                payload={
                    **dict(citation.extra),
                    "title": citation.title,
                    "source": citation.source,
                },
                bm25_score=citation.bm25_score,
                dense_scores=dict(citation.dense_scores),
                rrf_score=citation.rrf_score,
                rerank_score=citation.rerank_score,
            )
        )
    return candidates


def _merge_candidates(
    first_round: list[RetrievalCandidate],
    *later_rounds: list[RetrievalCandidate],
) -> list[RetrievalCandidate]:
    by_id: dict[str, RetrievalCandidate] = {}
    for candidate in first_round:
        by_id.setdefault(candidate.chunk_id, candidate)
    for round_candidates in later_rounds:
        for candidate in round_candidates:
            by_id.setdefault(candidate.chunk_id, candidate)

    return sorted(
        by_id.values(),
        key=lambda candidate: (
            candidate.rerank_score
            if candidate.rerank_score is not None
            else float("-inf"),
            candidate.rrf_score
            if candidate.rrf_score is not None
            else float("-inf"),
            candidate.primary_dense_score()
            if candidate.primary_dense_score() is not None
            else float("-inf"),
        ),
        reverse=True,
    )
