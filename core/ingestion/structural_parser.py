"""
core/ingestion/structural_parser.py

Parses cleaned text into a tree of StructuralNode objects.

Each StructuralNode represents one structural unit (heading, clause,
paragraph, list block, table, step).  The tree preserves parent-child
relationships that become the `parent_id` / `hierarchy_level` fields on Chunk.

Design rules
------------
- Pure function: parse(text) → list[StructuralNode].  No side effects.
- Deterministic: same text always produces same structure.
- Only the structural levels listed in ChunkingConfig.structural.levels
  are used; others fall back to paragraph.
- "Structure First": heading detection takes highest priority; within a
  heading section, clause and paragraph boundaries are detected.
- Table detection: contiguous lines where ≥2 contain a pipe character (|)
  are treated as a table block.
- List detection: contiguous lines starting with a bullet/number prefix.
- Step detection: lines matching Chinese/English numbered step patterns
  (for workflow documents).

Limitations (acceptable for Phase 1)
--------------------------------------
- No cross-reference following (footnotes, appendices).
- No PDF/DOCX native structure; input is already plain text from the cleaner.
- Complex nested tables are treated as one flat table block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class NodeType(str, Enum):
    HEADING    = "heading"
    CLAUSE     = "clause"
    PARAGRAPH  = "paragraph"
    LIST_BLOCK = "list_item"   # matches StructuralLevel enum value
    TABLE      = "table"
    STEP       = "step"
    ROOT       = "root"


@dataclass
class StructuralNode:
    """
    One structural unit in the parsed document tree.

    Attributes
    ----------
    node_type   : what kind of structure this is
    text        : the raw text of this node (without children)
    level       : heading depth (1–6 for headings, 0 for root, 1 for others)
    children    : child nodes (sub-sections, paragraphs within a section)
    heading     : the heading text that introduces this node (for context)
    """
    node_type: NodeType
    text: str
    level: int = 1
    children: list["StructuralNode"] = field(default_factory=list)
    heading: str = ""   # nearest ancestor heading text


# ---------------------------------------------------------------------------
# Heading detection patterns
# ---------------------------------------------------------------------------

# Markdown-style headings: ## Title
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Chinese chapter/section numbering: 第一章, 第1节, 一、, （一）, 1.2.3
_ZH_CHAPTER_RE  = re.compile(r"^第[一二三四五六七八九十百千\d]+[章节条款篇部分]\s*.+", re.MULTILINE)
_ZH_SECTION_RE  = re.compile(r"^[一二三四五六七八九十]{1,2}[、．.]\s*.+", re.MULTILINE)
_ZH_SUBSECT_RE  = re.compile(r"^（[一二三四五六七八九十\d]+）\s*.+", re.MULTILINE)
_NUM_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[^\d].+", re.MULTILINE)  # "1.2 Title"

# Table detection: line contains ≥1 pipe not at start of line
_TABLE_LINE_RE  = re.compile(r"(?<!\|)\|(?!\|)")  # at least one | not doubled

# List item detection
_BULLET_RE = re.compile(r"^[\s]*[•\-\*\uff65\u30fb◆◇○●▪▸►]\s+", re.MULTILINE)
_NUMBERED_LIST_RE = re.compile(r"^[\s]*\d+[.)）]\s+", re.MULTILINE)
_ZH_LIST_RE = re.compile(r"^[\s]*[①②③④⑤⑥⑦⑧⑨⑩]\s*", re.MULTILINE)

# Step detection (workflow documents)
_STEP_RE = re.compile(
    r"^(?:步骤|Step|操作|Action)\s*\d+[：:.]\s*",
    re.MULTILINE | re.IGNORECASE,
)

# Clause detection (policy/legal): "第N条", "Article N", "Section N"
_CLAUSE_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千\d]+条|Article\s+\d+|Section\s+[\d.]+)\s*[：:。\s]",
    re.MULTILINE | re.IGNORECASE,
)


def _detect_heading_level(line: str) -> tuple[int, str] | None:
    """
    Try to detect if `line` is a heading.
    Returns (level, heading_text) or None.
    level 1 = top (chapter), 2 = section, 3 = subsection, 4+ = deeper.
    """
    m = _MD_HEADING_RE.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()

    if _ZH_CHAPTER_RE.match(line):
        return 1, line.strip()
    if _CLAUSE_RE.match(line):
        return 2, line.strip()
    if _ZH_SECTION_RE.match(line):
        return 2, line.strip()
    if _ZH_SUBSECT_RE.match(line):
        return 3, line.strip()
    m = _NUM_HEADING_RE.match(line)
    if m:
        dots = line.split()[0].count(".")
        return dots + 1, line.strip()

    return None


def _is_table_block(lines: list[str]) -> bool:
    """True when ≥2 lines in the block contain a pipe character."""
    return sum(1 for l in lines if _TABLE_LINE_RE.search(l)) >= 2


def _is_list_block(lines: list[str]) -> bool:
    """True when ≥2 lines look like list items."""
    count = sum(
        1 for l in lines
        if _BULLET_RE.match(l) or _NUMBERED_LIST_RE.match(l) or _ZH_LIST_RE.match(l)
    )
    return count >= 2


def _is_step_block(lines: list[str]) -> bool:
    return any(_STEP_RE.match(l) for l in lines)


class StructuralParser:
    """
    Parses cleaned text into a flat list of StructuralNode objects.

    The list is ordered by document position (top-to-bottom).  Each node
    knows its level; the StructuralChunker builds the parent-child tree
    from level information.

    Parameters
    ----------
    enabled_levels:
        Which structural levels to detect, from ChunkingConfig.structural.levels.
        Any level not in this list is folded into PARAGRAPH.
    """

    def __init__(self, enabled_levels: list[str] | None = None) -> None:
        self._enabled = set(enabled_levels or [
            "heading", "clause", "paragraph", "list_item", "table", "step"
        ])

    def parse(self, text: str) -> list[StructuralNode]:
        """
        Parse `text` into an ordered list of StructuralNodes.
        Returns at least one node (the whole text as a PARAGRAPH if nothing
        structural is detected).
        """
        if not text.strip():
            return [StructuralNode(node_type=NodeType.PARAGRAPH, text="", level=1)]

        lines = text.splitlines()
        nodes: list[StructuralNode] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # --- Blank lines: skip ---
            if not line.strip():
                i += 1
                continue

            # --- Heading detection ---
            if "heading" in self._enabled or "clause" in self._enabled:
                heading_info = _detect_heading_level(line)
                if heading_info:
                    level, heading_text = heading_info
                    ntype = NodeType.HEADING if level <= 2 else NodeType.CLAUSE
                    if ntype.value not in self._enabled:
                        ntype = NodeType.PARAGRAPH
                    nodes.append(StructuralNode(
                        node_type=ntype,
                        text=line.strip(),
                        level=level,
                        heading=heading_text,
                    ))
                    i += 1
                    continue

            # --- Step detection ---
            if "step" in self._enabled and _STEP_RE.match(line):
                # Collect this step and continuation lines
                block_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip() and not _STEP_RE.match(lines[j]):
                    block_lines.append(lines[j])
                    j += 1
                nodes.append(StructuralNode(
                    node_type=NodeType.STEP,
                    text="\n".join(block_lines),
                    level=2,
                ))
                i = j
                continue

            # --- Table detection: collect contiguous lines ---
            if "table" in self._enabled:
                block_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    block_lines.append(lines[j])
                    j += 1
                if _is_table_block(block_lines):
                    nodes.append(StructuralNode(
                        node_type=NodeType.TABLE,
                        text="\n".join(block_lines),
                        level=2,
                    ))
                    i = j
                    continue

            # --- List block detection ---
            if "list_item" in self._enabled and (
                _BULLET_RE.match(line) or _NUMBERED_LIST_RE.match(line) or _ZH_LIST_RE.match(line)
            ):
                block_lines = [line]
                j = i + 1
                while j < len(lines) and lines[j].strip() and (
                    _BULLET_RE.match(lines[j])
                    or _NUMBERED_LIST_RE.match(lines[j])
                    or _ZH_LIST_RE.match(lines[j])
                    or (lines[j].startswith("  ") and not _detect_heading_level(lines[j]))
                ):
                    block_lines.append(lines[j])
                    j += 1
                if _is_list_block(block_lines):
                    nodes.append(StructuralNode(
                        node_type=NodeType.LIST_BLOCK,
                        text="\n".join(block_lines),
                        level=2,
                    ))
                    i = j
                    continue

            # --- Default: paragraph (collect until blank line or heading) ---
            block_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    break
                if _detect_heading_level(next_line):
                    break
                if "step" in self._enabled and _STEP_RE.match(next_line):
                    break
                block_lines.append(next_line)
                j += 1
            nodes.append(StructuralNode(
                node_type=NodeType.PARAGRAPH,
                text="\n".join(block_lines),
                level=2,
            ))
            i = j

        return nodes if nodes else [
            StructuralNode(node_type=NodeType.PARAGRAPH, text=text.strip(), level=1)
        ]
