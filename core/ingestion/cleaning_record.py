"""
core/ingestion/cleaning_record.py

Data transfer objects for the rule-based cleaning pipeline.

CleaningRecord
--------------
Captures what one rule transform did to one document.  The complete list of
CleaningRecords for a document forms its audit trail — policy/regulation docs
require the ability to prove that the indexed text faithfully represents the
source, with every change logged.

CleanedDocument
---------------
The output of the full cleaning pipeline: cleaned text + ordered log of all
transforms that were applied.  Immutable after construction.

Design rules
------------
- Transforms are logged even when they produce no change (op=NOOP) so the
  audit trail is complete and reproducible.
- `chars_removed` is always >= 0.  Negative values indicate expansion (rare,
  e.g. unicode normalisation that expands a ligature) and are allowed.
- The original text is NOT stored here — it lives in the original-text store
  (filesystem/object store).  This DTO only captures the delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class TransformOp(str, Enum):
    """Identifies which rule was applied.  String enum for easy serialisation."""
    ENCODING_FIX       = "encoding_fix"
    WHITESPACE_NORM    = "whitespace_norm"
    HTML_STRIP         = "html_strip"
    BOILERPLATE_STRIP  = "boilerplate_strip"
    UNICODE_NORM       = "unicode_norm"
    CONTROL_CHAR_STRIP = "control_char_strip"
    REPEATED_PUNCT_FIX = "repeated_punct_fix"
    NOOP               = "noop"


@dataclass(frozen=True)
class CleaningRecord:
    """
    Log entry for one rule transform applied to one document.

    Attributes
    ----------
    op:
        Which rule was applied.
    chars_before:
        Character count of the text before this rule.
    chars_after:
        Character count of the text after this rule.
    detail:
        Optional human-readable note (e.g. which boilerplate pattern matched,
        or how many HTML tags were stripped).  Kept short — this is a log
        entry, not a diff.
    """
    op: TransformOp
    chars_before: int
    chars_after: int
    detail: str = ""

    @property
    def chars_removed(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def changed(self) -> bool:
        return self.chars_before != self.chars_after


@dataclass(frozen=True)
class CleanedDocument:
    """
    Output of the full cleaning pipeline for one document.

    Attributes
    ----------
    doc_id:
        Stable identifier for the source document.
    original_length:
        Character count of the raw input text (before any rule).
    text:
        The cleaned canonical text.  This becomes the authoritative
        `text` field stored in ES and Qdrant.
    log:
        Ordered list of CleaningRecords, one per rule applied.
        Always present, even if every rule was a NOOP.
    business_type:
        Carried through for downstream pipeline stages.
    source_metadata:
        Any additional metadata from the raw document (title, source url,
        timestamps etc.) passed through unchanged.
    """
    doc_id: str
    original_length: int
    text: str
    log: tuple[CleaningRecord, ...]
    business_type: str = ""
    source_metadata: dict = field(default_factory=dict)

    @property
    def total_chars_removed(self) -> int:
        return self.original_length - len(self.text)

    @property
    def any_change(self) -> bool:
        return any(r.changed for r in self.log)

    def summary(self) -> str:
        """One-line summary for logging."""
        ops = [r.op.value for r in self.log if r.changed]
        return (
            f"doc_id={self.doc_id} "
            f"original={self.original_length} "
            f"cleaned={len(self.text)} "
            f"removed={self.total_chars_removed} "
            f"ops=[{', '.join(ops) or 'none'}]"
        )
