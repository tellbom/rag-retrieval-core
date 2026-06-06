"""
core/ingestion/enhancer.py

Enhancer: generates LLM-derived fields for a CleanedDocument and returns
an EnhancedDocument.

Invariants enforced here
------------------------
1. `mutate_source = False` — the canonical `text` field is NEVER touched.
   Derived fields are ADDED, not substituted.
2. Global switch: if `enhancement.enabled = False`, return an EnhancedDocument
   with enhanced=False and all derived fields as None immediately (no LLM call).
3. Per-field switches: only fields enabled in `derived_fields` config are requested.
4. Degradation policy:
   - `rules_only_and_flag`:  on LLM failure → return EnhancedDocument with
     enhanced=False, enhancement_error set, all derived fields None.
     Ingestion continues unblocked.
   - `fail_fast`:  on LLM failure → re-raise as EnhancementError.
     Caller decides whether to abort or skip the document.
5. Single LLM call per document: all enabled fields are requested together
   in one structured JSON prompt to minimise latency and token cost.

Prompt design
-------------
The system prompt instructs the model to respond with a JSON object only
(no preamble, no markdown fences).  The keys match the enabled field names.
The user prompt is the document text prefixed by a brief instruction.
A strict JSON parser is used; on parse failure the degradation policy applies.

Output JSON schema (subset of enabled fields):
    {
      "summary":             "2-5 sentence summary...",
      "keywords":            ["kw1", "kw2", ...],
      "entities":            ["entity1", "entity2", ...],
      "potential_questions": ["q1", "q2", ...],
      "context_padding":     "1-2 sentence context blurb..."
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.config.models import AppConfig, DerivedFieldsConfig, EnhancementConfig
from core.ingestion.cleaning_record import CleanedDocument
from core.ingestion.enhanced_document import EnhancedDocument
from core.ingestion.llm_client import LLMCallError, LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Per-field instruction lines included in the system prompt.
# Only lines for enabled fields are included at build time.
#
# Design rules for prompt wording
# --------------------------------
# - String fields (summary, context_padding): use inline prose description,
#   never bullet lists.  Bullet lists visually resemble array fields and cause
#   some LLMs to return a list instead of a string.  Every string field ends
#   with an explicit "Return as a plain string." anchor.
# - Array fields (keywords, entities, potential_questions): use bullet
#   sub-rules, and end with an explicit "Return as a JSON array of strings."
#   anchor.  This keeps the type contract unambiguous regardless of LLM.
_FIELD_INSTRUCTIONS: dict[str, str] = {
    "summary": (
        '"summary": Write 2 to 5 concise sentences that describe the document '
        "purpose, main content, and important conclusions. "
        "Do not copy large sections of the source text verbatim. "
        "Preserve the original document language. "
        "Return as a plain string."
    ),
    "keywords": (
        '"keywords":\n'
        "- 5 to 15 items.\n"
        "- Prefer domain-specific phrases: business terms, technical terms, "
        "policy names, equipment names, process names, standard names, fault "
        "codes, model numbers, and document identifiers.\n"
        '- Avoid generic words such as "management", "work", "system", '
        '"process" unless they form part of a formal term.\n'
        "- Preserve the original document language.\n"
        "- Return as a JSON array of strings."
    ),
    "entities": (
        '"entities":\n'
        "- Extract only entities explicitly mentioned in the document.\n"
        "- Include: person names, organisation names, location names, product "
        "names, equipment names, project names, standard numbers, document "
        "numbers, model numbers, fault codes, and process identifiers.\n"
        "- Do not infer or generate entities not present in the source text.\n"
        "- Return as a JSON array of strings."
    ),
    "potential_questions": (
        '"potential_questions":\n'
        "- 3 to 8 questions this document could answer.\n"
        "- Base every question on information explicitly present in the document.\n"
        "- Preserve the original document language.\n"
        "- Return as a JSON array of strings."
    ),
    "context_padding": (
        '"context_padding": Write a single sentence of at most 30 words that '
        "describes what this document is about, suitable for prepending to a "
        "chunk as context. "
        "Preserve the original document language. "
        "Return as a plain string."
    ),
}

_USER_PROMPT_TEMPLATE = """Document text (language: auto-detect):
---
{text}
---
Return the JSON object now."""

# Maximum text length sent to the LLM (characters).
# Keeps prompt within the model's context window; text is already cleaned.
_MAX_TEXT_CHARS = 6000


def _build_system_prompt(fields: list[str]) -> str:
    """
    Build a system prompt that only mentions the requested fields.

    The prompt is structured in three layers:
    1. Role + output format constraint (JSON only, no markdown).
    2. Type contract: explicitly names which fields are plain strings and which
       are JSON arrays — this prevents LLMs from returning a list for `summary`.
    3. Per-field instructions from _FIELD_INSTRUCTIONS.
    """
    # Partition fields by expected return type for the type contract line
    _str_fields  = {"summary", "context_padding"}
    _list_fields = {"keywords", "entities", "potential_questions"}

    string_keys = [f for f in fields if f in _str_fields]
    array_keys  = [f for f in fields if f in _list_fields]

    type_contract_lines: list[str] = []
    if string_keys:
        keys = ", ".join(f'"{k}"' for k in string_keys)
        type_contract_lines.append(f"- {keys}: plain string (NOT an array).")
    if array_keys:
        keys = ", ".join(f'"{k}"' for k in array_keys)
        type_contract_lines.append(f"- {keys}: JSON array of strings (NOT a plain string).")
    type_contract = "\n".join(type_contract_lines)

    field_lines = "\n\n".join(
        _FIELD_INSTRUCTIONS[f] for f in fields if f in _FIELD_INSTRUCTIONS
    )
    field_list = ", ".join(f'"{f}"' for f in fields)

    return (
        "You are a document analysis assistant for an enterprise knowledge base.\n"
        f"Analyse the provided document text and return a JSON object with ONLY "
        f"the following keys: {field_list}.\n\n"
        "Output rules:\n"
        "- Respond with valid JSON only. No preamble, no markdown fences, no explanation.\n"
        "- Omit any key not in the requested list. Do not include null values.\n"
        "- Required value types (strictly enforced):\n"
        f"{type_contract}\n\n"
        "Field instructions:\n"
        f"{field_lines}"
    )


# ---------------------------------------------------------------------------
# Enhancer
# ---------------------------------------------------------------------------

class EnhancementError(Exception):
    """
    Raised when degradation_policy == "fail_fast" and the LLM call fails.
    Callers should skip or abort the document.
    """


class Enhancer:
    """
    Generates LLM-derived fields for a CleanedDocument.

    Parameters
    ----------
    enhancement_cfg:
        The `enhancement` section of AppConfig.
    llm_client:
        An LLMClient instance.  Pass None to run in disabled mode
        (equivalent to enhancement.enabled = False).
    """

    def __init__(
        self,
        enhancement_cfg: EnhancementConfig,
        llm_client: LLMClient | None,
    ) -> None:
        self._cfg = enhancement_cfg
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, cleaned: CleanedDocument) -> EnhancedDocument:
        """
        Enhance one CleanedDocument.

        Returns
        -------
        EnhancedDocument
            enhanced=True if all requested fields were generated.
            enhanced=False if disabled or LLM failed (degradation policy applied).

        Raises
        ------
        EnhancementError
            Only when degradation_policy == "fail_fast" and LLM fails.
        """
        # --- Global switch ---
        if not self._cfg.enabled or self._llm is None:
            return EnhancedDocument(cleaned=cleaned, enhanced=False)

        enabled_fields = self._enabled_field_names()
        if not enabled_fields:
            # All derived fields disabled — nothing to do
            return EnhancedDocument(cleaned=cleaned, enhanced=False)

        # --- Call LLM ---
        try:
            raw_json = self._call_llm(cleaned.text, enabled_fields)
            parsed = self._parse_response(raw_json, enabled_fields)
        except (LLMCallError, _ParseError) as exc:
            return self._handle_failure(cleaned, str(exc))

        # --- Build EnhancedDocument from parsed fields ---
        doc = EnhancedDocument(
            cleaned=cleaned,
            enhanced=True,
            summary=parsed.get("summary"),
            keywords=parsed.get("keywords"),
            entities=parsed.get("entities"),
            potential_questions=parsed.get("potential_questions"),
            context_padding=parsed.get("context_padding"),
        )
        logger.debug("Enhanced: %s", doc.summary_line())
        return doc

    def enhance_batch(
        self, documents: list[CleanedDocument]
    ) -> list[EnhancedDocument]:
        """
        Enhance a list of CleanedDocuments sequentially.
        Errors on individual documents follow the degradation policy.
        """
        return [self.enhance(doc) for doc in documents]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enabled_field_names(self) -> list[str]:
        """Return the list of field names that are enabled in config."""
        df: DerivedFieldsConfig = self._cfg.derived_fields
        mapping = {
            "summary":             df.summary,
            "keywords":            df.keywords,
            "entities":            df.entities,
            "potential_questions": df.potential_questions,
            "context_padding":     df.context_padding,
        }
        return [name for name, enabled in mapping.items() if enabled]

    def _call_llm(self, text: str, fields: list[str]) -> str:
        """Build prompts and call the LLM. Returns raw response string."""
        # Truncate text to avoid exceeding context window
        truncated = text[:_MAX_TEXT_CHARS]
        if len(text) > _MAX_TEXT_CHARS:
            logger.debug(
                "Text truncated from %d to %d chars for LLM enhancement",
                len(text), _MAX_TEXT_CHARS,
            )

        system_prompt = _build_system_prompt(fields)
        user_prompt = _USER_PROMPT_TEMPLATE.format(text=truncated)

        return self._llm.chat(system_prompt, user_prompt, temperature=0.0)

    def _parse_response(
        self, raw: str, expected_fields: list[str]
    ) -> dict[str, Any]:
        """
        Parse the LLM JSON response.  Raises _ParseError on unrecoverable issues.
        Only returns keys that are in expected_fields; ignores extras.

        Type coercion (defensive second line after prompt-level type contract):
        - str fields (summary, context_padding): if LLM returns a list, join
          the items with a space and log a warning rather than hard-failing.
          This handles models that ignore the "plain string" instruction.
        - list fields (keywords, entities, potential_questions): must be a list;
          hard-fail if a plain string is returned (ambiguous to split).
        """
        # Strip markdown fences if the model ignores the instruction
        cleaned_raw = raw.strip()
        if cleaned_raw.startswith("```"):
            lines = cleaned_raw.splitlines()
            cleaned_raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(cleaned_raw)
        except json.JSONDecodeError as exc:
            raise _ParseError(
                f"LLM response is not valid JSON: {exc}. "
                f"Response snippet: {raw[:200]!r}"
            ) from exc

        if not isinstance(data, dict):
            raise _ParseError(
                f"LLM response is not a JSON object (got {type(data).__name__})"
            )

        # Validate and coerce types for each expected field
        result: dict[str, Any] = {}
        list_fields = {"keywords", "entities", "potential_questions"}
        str_fields  = {"summary", "context_padding"}

        for field_name in expected_fields:
            value = data.get(field_name)
            if value is None:
                raise _ParseError(f"Missing requested field '{field_name}'")

            if field_name in list_fields:
                if not isinstance(value, list):
                    raise _ParseError(
                        f"Field '{field_name}' must be a JSON array, "
                        f"got {type(value).__name__}: {str(value)[:100]!r}"
                    )
                result[field_name] = [str(item) for item in value]

            elif field_name in str_fields:
                if isinstance(value, list):
                    # LLM ignored the "plain string" instruction — coerce to string.
                    # Log at WARNING so the prompt deviation is visible in ops logs.
                    coerced = " ".join(str(item) for item in value).strip()
                    logger.warning(
                        "LLM returned a list for string field '%s' — "
                        "coercing to string (joined %d items). "
                        "Check prompt adherence for this model.",
                        field_name, len(value),
                    )
                    result[field_name] = coerced
                elif not isinstance(value, str):
                    raise _ParseError(
                        f"Field '{field_name}' must be a plain string, "
                        f"got {type(value).__name__}: {str(value)[:100]!r}"
                    )
                else:
                    result[field_name] = value.strip()

        return result

    def _handle_failure(
        self, cleaned: CleanedDocument, error_msg: str
    ) -> EnhancedDocument:
        """Apply degradation policy on LLM failure."""
        policy = self._cfg.degradation_policy

        logger.warning(
            "LLM enhancement failed for doc_id=%s (policy=%s): %s",
            cleaned.doc_id, policy, error_msg,
        )

        if policy == "fail_fast":
            raise EnhancementError(
                f"Enhancement failed for doc_id={cleaned.doc_id}: {error_msg}"
            )

        # rules_only_and_flag: return unenhanced document, never block ingestion
        return EnhancedDocument(
            cleaned=cleaned,
            enhanced=False,
            enhancement_error=error_msg,
        )


# ---------------------------------------------------------------------------
# Internal exception
# ---------------------------------------------------------------------------

class _ParseError(Exception):
    """Internal: LLM response could not be parsed into the expected structure."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EnhancerFactory:
    """
    Builds an Enhancer from AppConfig.

    If enhancement is globally disabled OR enhancement_llm is not configured,
    returns an Enhancer with llm_client=None (always returns enhanced=False).
    """

    @staticmethod
    def from_config(cfg: AppConfig) -> "Enhancer":
        enh_cfg = cfg.enhancement

        if not enh_cfg.enabled:
            return Enhancer(enhancement_cfg=enh_cfg, llm_client=None)

        llm_cfg = cfg.models.enhancement_llm
        if llm_cfg is None:
            logger.warning(
                "enhancement.enabled=true but models.enhancement_llm is not configured. "
                "Running in disabled mode."
            )
            return Enhancer(enhancement_cfg=enh_cfg, llm_client=None)

        llm_client = LLMClient.from_config(llm_cfg)
        return Enhancer(enhancement_cfg=enh_cfg, llm_client=llm_client)
