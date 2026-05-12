"""
Document loader: reads document.json produced by the Agentic PDF Parser
and flattens all blocks into a normalised list with enriched metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


BLOCK_TYPES_OF_INTEREST = {
    "paragraph", "heading", "table", "figure",
    "formula", "caption", "list_item", "footnote", "page_header", "page_footer",
}


def extract_table_text(table_data: dict) -> str:
    """Flatten table cells into readable pipe-delimited text for embedding / LLM context."""
    if not table_data:
        return ""
    rows: dict[int, list[str]] = {}
    for cell in table_data.get("cells", []):
        r = cell["row"]
        rows.setdefault(r, []).append(cell["text"].strip())
    lines = [" | ".join(row) for row in rows.values() if any(c.strip() for c in row)]
    return "\n".join(lines)


def make_display_text(block: dict) -> str:
    """Return the best human-readable representation of a block."""
    btype = block["type"]
    if btype == "table" and block.get("table"):
        return extract_table_text(block["table"])
    if btype == "figure" and block.get("figure"):
        return f"[Figure: {block['figure']['asset_path']}]"
    if btype == "formula" and block.get("formula"):
        return block["formula"].get("latex", "")
    return block.get("text") or ""


def load_document(doc_json_path: Path) -> tuple[dict, list[dict[str, Any]], dict[str, dict]]:
    """
    Parse a document.json file from the Agentic PDF Parser.

    Returns:
        raw_doc    — the full parsed JSON dict
        blocks     — flattened list of block dicts with enriched metadata,
                     sorted by (page_index, reading_order)
        block_by_id — {block_id: block} lookup dict
    """
    with doc_json_path.open(encoding="utf-8") as fh:
        raw_doc = json.load(fh)

    blocks: list[dict[str, Any]] = []

    for page in raw_doc["pages"]:
        page_idx = page["index"]
        page_num = page["number"]

        for blk in page["blocks"]:
            if blk["type"] not in BLOCK_TYPES_OF_INTEREST:
                continue

            display_text = make_display_text(blk)
            if not display_text.strip():
                continue

            bbox = blk["provenance"]["bbox"] if blk.get("provenance") else {}

            entry: dict[str, Any] = {
                # identity
                "block_id":      blk["id"],
                "type":          blk["type"],
                # location
                "page_index":    page_idx,
                "page_number":   page_num,
                "reading_order": blk["reading_order"],
                "bbox_x0":       bbox.get("x0"),
                "bbox_y0":       bbox.get("y0"),
                "bbox_x1":       bbox.get("x1"),
                "bbox_y1":       bbox.get("y1"),
                # content
                "text":          display_text,
                "raw_text":      blk.get("text") or "",
                # table metadata
                "has_table":     blk["type"] == "table",
                "table_rows":    blk["table"]["rows"]  if blk["type"] == "table" and blk.get("table") else None,
                "table_cols":    blk["table"]["cols"]  if blk["type"] == "table" and blk.get("table") else None,
                "table_html":    blk["table"]["html"]  if blk["type"] == "table" and blk.get("table") else None,
                # figure / formula metadata
                "figure_path":   blk["figure"]["asset_path"] if blk["type"] == "figure" and blk.get("figure") else None,
                "formula_latex": blk["formula"]["latex"]     if blk["type"] == "formula" and blk.get("formula") else None,
                # heading metadata
                "heading_level": blk.get("level"),
            }
            blocks.append(entry)

    blocks.sort(key=lambda b: (b["page_index"], b["reading_order"]))
    block_by_id = {b["block_id"]: b for b in blocks}
    return raw_doc, blocks, block_by_id


def print_block_summary(blocks: list[dict]) -> None:
    """Print a type-count breakdown of the loaded blocks."""
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b["type"]] = counts.get(b["type"], 0) + 1
    print(f"Total blocks: {len(blocks)}")
    for t, c in sorted(counts.items()):
        print(f"  {t:<15} : {c}")
