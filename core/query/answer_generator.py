"""Generate grounded answers from built retrieval context."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.config.models import AppConfig
from core.ingestion.llm_client import LLMCallError, LLMClient
from core.query.context_builder import BuiltContext, Citation

logger = logging.getLogger(__name__)

_DEFAULT_INSUFFICIENT_CONTEXT_ZH = "\u6839\u636e\u73b0\u6709\u8d44\u6599\u65e0\u6cd5\u56de\u7b54\u8be5\u95ee\u9898"
_DEFAULT_INSUFFICIENT_CONTEXT_EN = (
    "The provided context does not contain sufficient information "
    "to answer this question."
)

_DEFAULT_SYSTEM_PROMPT = f"""\
You are an enterprise knowledge base assistant. Answer the user's question
using ONLY the provided reference documents. Rules:
1. Base your answer strictly on the provided context. Do not use outside knowledge.
2. Cite sources using [N] notation, where N matches the reference block number.
3. If the context does not contain enough information to answer the question,
say exactly: "{_DEFAULT_INSUFFICIENT_CONTEXT_ZH}" (or in English:
"{_DEFAULT_INSUFFICIENT_CONTEXT_EN}")
4. Be concise and precise. Answer in the same language as the question.
5. Never fabricate facts or invent citations.\
"""

_USER_PROMPT_TEMPLATE = """\
Question: {query}

Reference documents:
{context_text}

Please answer the question based on the reference documents above.\
"""


@dataclass
class GeneratedAnswer:
    """Output of AnswerGenerator.generate()."""

    answer: str
    citations: list[Citation]
    grounded: bool
    context_used: int = 0
    reranked: bool = False
    llm_model: str = ""

    def summary(self) -> str:
        return (
            f"grounded={self.grounded} "
            f"context_used={self.context_used} "
            f"answer_len={len(self.answer)} "
            f"citations={len(self.citations)}"
        )


class AnswerGenerationError(Exception):
    """Raised when the LLM call fails and the caller must handle the error."""


class AnswerGenerator:
    """Generate a grounded answer from BuiltContext using the intranet LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        insufficient_context_response: str = _DEFAULT_INSUFFICIENT_CONTEXT_ZH,
        temperature: float = 0.0,
    ) -> None:
        self._client = llm_client
        self._system_prompt = system_prompt
        self._insufficient_response = insufficient_context_response
        self._temperature = temperature

    def generate(
        self,
        query: str,
        context: BuiltContext,
    ) -> GeneratedAnswer:
        """Generate an answer grounded in the provided context."""
        if context.is_empty:
            logger.debug(
                "AnswerGenerator: empty context for query=%r, returning fallback",
                query[:60],
            )
            return GeneratedAnswer(
                answer=self._insufficient_response,
                citations=[],
                grounded=False,
                context_used=0,
                reranked=context.reranked,
                llm_model=self._client._model,
            )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            query=query,
            context_text=context.context_text,
        )

        try:
            answer_text = self._client.chat(
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                temperature=self._temperature,
            )
        except LLMCallError as exc:
            raise AnswerGenerationError(
                f"LLM call failed for query={query!r}: {exc}"
            ) from exc

        answer_text = answer_text.strip()

        logger.debug(
            "AnswerGenerator: generated %d chars for query=%r (context_blocks=%d)",
            len(answer_text),
            query[:60],
            context.candidate_count,
        )

        return GeneratedAnswer(
            answer=answer_text,
            citations=list(context.citations),
            grounded=True,
            context_used=context.candidate_count,
            reranked=context.reranked,
            llm_model=self._client._model,
        )

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        *,
        insufficient_context_response: str = _DEFAULT_INSUFFICIENT_CONTEXT_ZH,
    ) -> "AnswerGenerator":
        llm_cfg = cfg.models.enhancement_llm
        if llm_cfg is None:
            raise ValueError(
                "AnswerGenerator requires models.enhancement_llm to be configured. "
                "Add an enhancement_llm entry to the config."
            )
        client = LLMClient.from_config(llm_cfg)
        return cls(
            llm_client=client,
            insufficient_context_response=insufficient_context_response,
        )
