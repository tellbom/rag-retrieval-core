"""
core/ingestion/cleaner.py

Cleaner: runs the ordered rule pipeline on a raw document and produces
a CleanedDocument (cleaned text + full audit log).

Design rules
------------
- Deterministic: given the same input and config, always produces the same output.
- Non-mutating: the original text is passed in and never stored here.
  The caller (ingestion pipeline) is responsible for persisting originals.
- Auditable: every rule produces a CleaningRecord regardless of whether it
  changed anything.  The complete log can be persisted for compliance review.
- Per-business boilerplate patterns: the caller passes extra regex patterns
  that are prepended to the default set for that business type.
- Soft-dependency resilience: if ftfy or bs4 are missing, the corresponding
  rules run in degraded mode (NOOP with a detail note), not error.

Public API
----------
    cleaner = Cleaner()
    doc = cleaner.clean(
        doc_id="doc-001",
        raw_text="<p>Hello  World</p>",
        business_type="news",
        source_metadata={"title": "...", "url": "..."},
        extra_boilerplate_patterns=[re.compile(r"^Footer.*$", re.MULTILINE)],
    )
    print(doc.text)      # cleaned text
    print(doc.summary()) # one-line audit summary
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from core.ingestion.cleaning_record import CleanedDocument, CleaningRecord, TransformOp
from core.ingestion.rules import (
    DEFAULT_RULE_PIPELINE,
    fix_encoding,
    strip_control_chars,
    normalize_unicode,
    strip_html,
    strip_boilerplate,
    normalize_whitespace,
    fix_repeated_punct,
)

logger = logging.getLogger(__name__)


class Cleaner:
    """
    Applies the rule pipeline to a single raw document.

    The pipeline order is fixed (see DEFAULT_RULE_PIPELINE in rules.py):
        1. fix_encoding
        2. strip_control_chars
        3. normalize_unicode
        4. strip_html
        5. strip_boilerplate   ← extra_patterns injected here
        6. normalize_whitespace
        7. fix_repeated_punct

    Parameters
    ----------
    extra_boilerplate_patterns:
        Additional compiled regex patterns prepended to the default boilerplate
        list for every document cleaned by this instance.
        Pass business-type-specific patterns here.
    """

    def __init__(
        self,
        extra_boilerplate_patterns: list[re.Pattern] | None = None,
    ) -> None:
        self._extra_boilerplate = extra_boilerplate_patterns or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str = "",
        source_metadata: dict | None = None,
    ) -> CleanedDocument:
        """
        Run all rules on `raw_text` and return a CleanedDocument.

        Parameters
        ----------
        doc_id:
            Stable identifier for the source document.
        raw_text:
            The raw input text (may contain HTML, encoding errors, etc.).
        business_type:
            Carried through to CleanedDocument for downstream pipeline stages.
        source_metadata:
            Arbitrary key-value metadata from the source (title, url, dates…).
            Passed through unchanged.

        Returns
        -------
        CleanedDocument
            Immutable: text is the cleaned result, log contains all records.
        """
        original_length = len(raw_text)
        text = raw_text
        records: list[CleaningRecord] = []

        # --- Step 1: encoding fix ---
        text, rec = fix_encoding(text)
        records.append(rec)

        # --- Step 2: control chars ---
        text, rec = strip_control_chars(text)
        records.append(rec)

        # --- Step 3: unicode normalisation ---
        text, rec = normalize_unicode(text)
        records.append(rec)

        # --- Step 4: HTML strip ---
        text, rec = strip_html(text)
        records.append(rec)

        # --- Step 5: boilerplate strip (per-business patterns injected) ---
        text, rec = strip_boilerplate(text, extra_patterns=self._extra_boilerplate)
        records.append(rec)

        # --- Step 6: whitespace normalisation ---
        text, rec = normalize_whitespace(text)
        records.append(rec)

        # --- Step 7: repeated punctuation fix ---
        text, rec = fix_repeated_punct(text)
        records.append(rec)

        doc = CleanedDocument(
            doc_id=doc_id,
            original_length=original_length,
            text=text,
            log=tuple(records),
            business_type=business_type,
            source_metadata=source_metadata or {},
        )

        logger.debug("Cleaned: %s", doc.summary())
        return doc

    def clean_batch(
        self,
        documents: Sequence[tuple[str, str]],
        *,
        business_type: str = "",
    ) -> list[CleanedDocument]:
        """
        Clean multiple (doc_id, raw_text) pairs.
        Convenience method for batch ingestion; calls clean() per document.

        Parameters
        ----------
        documents:
            Sequence of (doc_id, raw_text) tuples.
        business_type:
            Applied to all documents in the batch.

        Returns
        -------
        list[CleanedDocument]
            Same order as input.
        """
        return [
            self.clean(doc_id, raw_text, business_type=business_type)
            for doc_id, raw_text in documents
        ]
