"""
Section builder.

Walks the reading-order block list, derives a hierarchy from ``heading_level``,
and emits Section dicts plus Block->Section IN_SECTION edges. Pure-Python, no
models, no LLM, no Neo4j dependency. Designed to be called early in the
pipeline, before the LLM/embedding servers start.

Returns:
    sections             — list[Section dict] (one per heading + optional synthetic root)
    in_section_edges     — list[(block_id, section_id)]
    parent_child_pairs   — list[{"parent_id", "child_id"}] for HAS_SUBSECTION
    top_level_sections   — list[Section dict] (parent_id is None and level <= 1)
"""

from __future__ import annotations

import re
from typing import Any

import config


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 40) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] if s else "section"


def _build_path(stack: list[dict], max_depth: int) -> str:
    """Join ancestor titles (skipping the synthetic root) into a path string."""
    titles = [s["title"] for s in stack if s["level"] >= 1]
    if len(titles) > max_depth:
        titles = titles[-max_depth:]
    return " > ".join(titles)


def build_sections(
    blocks: list[dict[str, Any]],
    doc_id: str,
) -> tuple[list[dict], list[tuple[str, str]], list[dict], list[dict]]:
    """
    Build the section hierarchy for one document.

    Args:
        blocks: reading-order-sorted block list from ``loader.load_document``.
        doc_id: document identifier (used to scope section IDs and tag nodes).

    Returns:
        sections           — list of Section dicts (see plan for fields)
        in_section_edges   — list[(block_id, section_id)] covering every block
        parent_child_pairs — list[{"parent_id", "child_id"}]
        top_level_sections — list of Section dicts where parent_id is None
    """
    has_any_heading = any(b["type"] == "heading" for b in blocks)

    # Synthetic root — created when (a) there are no headings and the user
    # opted in to a synthetic root, or (b) there are headings (always present
    # so blocks before the first heading have somewhere to go).
    use_synthetic_root = has_any_heading or config.SECTION_EMIT_SYNTHETIC_ROOT_IF_NO_HEADINGS

    sections: list[dict] = []
    in_section_edges: list[tuple[str, str]] = []
    parent_child_pairs: list[dict] = []

    if not blocks:
        return sections, in_section_edges, parent_child_pairs, []

    stack: list[dict] = []

    if use_synthetic_root:
        first_page = blocks[0]["page_number"]
        root = {
            "id":               f"sec_{doc_id[:12]}_root",
            "doc_id":           doc_id,
            "title":            config.SECTION_FALLBACK_TITLE,
            "level":            0,
            "level_inferred":   False,
            "level_gap":        0,
            "path":             "",
            "page_start":       first_page,
            "page_end":         first_page,
            "block_start_id":   blocks[0]["block_id"],
            "block_end_id":     blocks[0]["block_id"],
            "block_count":      0,
            "parent_id":        None,
            "heading_block_id": None,
        }
        sections.append(root)
        stack.append(root)

    last_known_level: int | None = None

    for blk in blocks:
        if blk["type"] == "heading":
            raw_level = blk.get("heading_level")
            if raw_level is None or raw_level < 1:
                lvl = last_known_level if last_known_level is not None else 1
                level_inferred = True
            else:
                lvl = int(raw_level)
                last_known_level = lvl
                level_inferred = False

            # Pop until top.level < lvl (root at level 0 never pops)
            while stack and stack[-1]["level"] >= lvl:
                stack.pop()

            parent = stack[-1] if stack else None
            level_gap = (lvl - parent["level"]) if parent is not None else 0

            sec = {
                "id":               f"sec_{doc_id[:12]}_{blk['block_id']}",
                "doc_id":           doc_id,
                "title":            (blk["text"] or "").strip() or "(untitled)",
                "level":            lvl,
                "level_inferred":   level_inferred,
                "level_gap":        level_gap,
                "path":             "",  # filled after push
                "page_start":       blk["page_number"],
                "page_end":         blk["page_number"],
                "block_start_id":   blk["block_id"],
                "block_end_id":     blk["block_id"],
                "block_count":      0,
                "parent_id":        parent["id"] if parent else None,
                "heading_block_id": blk["block_id"],
            }
            stack.append(sec)
            sec["path"] = _build_path(stack, config.SECTION_PATH_MAX_DEPTH)
            sections.append(sec)
            if parent is not None:
                parent_child_pairs.append({"parent_id": parent["id"], "child_id": sec["id"]})

        # Attach the block to the current deepest section (a heading attaches
        # to its own freshly-pushed section).
        if stack:
            current = stack[-1]
            in_section_edges.append((blk["block_id"], current["id"]))
            current["block_count"] += 1
            current["block_end_id"] = blk["block_id"]
            if blk["page_number"] > current["page_end"]:
                current["page_end"] = blk["page_number"]
            # Propagate page_end up the ancestor chain so each section
            # spans from its heading to its last contained block.
            for ancestor in stack[:-1]:
                if blk["page_number"] > ancestor["page_end"]:
                    ancestor["page_end"] = blk["page_number"]
                ancestor["block_end_id"] = blk["block_id"]
                ancestor["block_count"] += 1

    top_level_sections = [s for s in sections if s["parent_id"] is None]
    return sections, in_section_edges, parent_child_pairs, top_level_sections
