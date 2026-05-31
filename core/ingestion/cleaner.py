"""
core/ingestion/cleaner.py

Cleaner: runs the ordered rule pipeline on a raw document and produces
a CleanedDocument (cleaned text + full audit log).

Design rules
------------
- Generic: no business-type logic.  Extra patterns come from a CleaningProfile
  passed at construction time, never hard-coded by business type.
- Deterministic: given the same input and config, always produces the same output.
- Non-mutating: the original text is passed in and never stored here.
- Auditable: every rule produces a CleaningRecord regardless of outcome.
- Soft-dependency resilience: ftfy / bs4 absence degrades gracefully.

Public API
----------
    # Plain (no extra patterns, defaults on):
    cleaner = Cleaner()

    # With a profile loaded from config:
    profile = CleaningProfile.from_dict(config_data)
    cleaner = Cleaner(profile=profile)

    doc = cleaner.clean(doc_id="doc-001", raw_text="<p>Hello</p>",
                        business_type="news")
    print(doc.text)
    print(doc.summary())
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from core.ingestion.cleaning_profile import CleaningProfile
from core.ingestion.cleaning_record import CleanedDocument, CleaningRecord, TransformOp
from core.ingestion.rules import (
    fix_encoding,
    fix_repeated_punct,
    normalize_unicode,
    normalize_whitespace,
    strip_boilerplate,
    strip_control_chars,
    strip_html,
)

logger = logging.getLogger(__name__)


class Cleaner:
    """
    Applies the rule pipeline to a single raw document.

    Pipeline order (fixed):
        1. fix_encoding
        2. strip_control_chars
        3. normalize_unicode
        4. strip_html
        5. strip_boilerplate   ← profile patterns + optional default patterns
        6. normalize_whitespace
        7. fix_repeated_punct

    Parameters
    ----------
    profile:
        A CleaningProfile that carries externally-configured extra boilerplate
        patterns and the `disable_default_boilerplate` flag.
        Defaults to an empty profile (generic cleaning only, defaults on).
    """

    def __init__(self, profile: CleaningProfile | None = None) -> None:
        self._profile: CleaningProfile = profile or CleaningProfile.empty()

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
            Carried through to CleanedDocument for downstream stages.
        source_metadata:
            Arbitrary metadata passed through unchanged.
        """
        original_length = len(raw_text)
        text = raw_text
        records: list[CleaningRecord] = []

        text, rec = fix_encoding(text)
        records.append(rec)

        text, rec = strip_control_chars(text)
        records.append(rec)

        text, rec = normalize_unicode(text)
        records.append(rec)

        text, rec = strip_html(text)
        records.append(rec)

        text, rec = strip_boilerplate(
            text,
            extra_patterns=self._profile.extra_boilerplate_patterns,
            use_defaults=not self._profile.disable_default_boilerplate,
        )
        records.append(rec)

        text, rec = normalize_whitespace(text)
        records.append(rec)

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
        Calls clean() per document in input order.
        """
        return [
            self.clean(doc_id, raw_text, business_type=business_type)
            for doc_id, raw_text in documents
        ]
