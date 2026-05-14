"""
Relationship extraction pipeline.

Sections 4, 6, and 7 of the notebook translated into standalone functions.

Deterministic / structural edges (no models required):
    compute_precedes_edges()
    compute_describes_edges()
    compute_introduces_edges()
    compute_context_edges()

Section-anchored IN_SECTION edges (Block -> Section) are produced by
``kg/sections.py``; ``compute_in_section_edges()`` was removed because its old
Block -> Heading-Block semantics are superseded.

Regex pass (no models required):
    build_ref_index()
    extract_explicit_refs()
    compute_refers_to_regex()

LLM passes (require llm_chat() to be callable — LLM server must be running):
    compute_refers_to_llm()
    compute_table_pair_rels()
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Any

from tqdm import tqdm

import config
from kg.models import cosine_sim, llm_chat


# ── Regex cross-reference pattern ────────────────────────────────────────────
# "Section" is intentionally omitted — section numbers don't map to block IDs
# and produced false positives in earlier testing.

_REF_PATTERN = re.compile(
    r'\b(?P<kind>Table|Figure|Fig\.|Equation|Algorithm)\s*(?P<num>[0-9]+(?:\.[0-9]+)?)\b',
    re.IGNORECASE,
)

_KIND_TO_BLOCK_TYPE = {"Table": "table", "Figure": "figure", "Fig": "figure"}


def extract_explicit_refs(text: str) -> list[dict]:
    """Return list of {kind, num} dicts found in text."""
    refs = []
    for m in _REF_PATTERN.finditer(text):
        kind = m.group("kind").rstrip(".").capitalize()
        refs.append({"kind": kind, "num": m.group("num")})
    return refs


def build_ref_index(blocks: list[dict]) -> dict[tuple, str]:
    """
    Build a mapping: ("Table", "1") → block_id, ("Figure", "2") → block_id, …

    Strategy:
      1. Scan captions for patterns like "Table 1:" and resolve to the spatially
         nearest table/figure block on the same page (not the caption itself).
      2. Fallback: assign sequential numbers to tables/figures without a caption.
    """
    index: dict[tuple, str] = {}

    tf_by_page: dict[int, list[dict]] = {}
    for b in blocks:
        if b["type"] in ("table", "figure"):
            tf_by_page.setdefault(b["page_index"], []).append(b)

    def _nearest_tf_on_page(cap: dict, target_type: str) -> str | None:
        candidates = [
            b for b in tf_by_page.get(cap["page_index"], [])
            if b["type"] == target_type
            and cap["bbox_y0"] is not None
            and b["bbox_y0"] is not None
        ]
        if not candidates:
            return None
        ay0 = cap["bbox_y0"] or 0
        ay1 = cap["bbox_y1"] or 0

        def gap(b: dict) -> float:
            by0, by1 = b["bbox_y0"], b["bbox_y1"]
            if by1 <= ay0:
                return ay0 - by1
            if by0 >= ay1:
                return by0 - ay1
            return 0.0

        candidates.sort(key=gap)
        return candidates[0]["block_id"]

    for cap in (b for b in blocks if b["type"] == "caption"):
        for ref in extract_explicit_refs(cap["text"]):
            target_type = _KIND_TO_BLOCK_TYPE.get(ref["kind"])
            if not target_type:
                continue
            key = (ref["kind"], ref["num"])
            if key not in index:
                resolved = _nearest_tf_on_page(cap, target_type)
                if resolved:
                    index[key] = resolved

    tbl_counter = 1
    for b in blocks:
        if b["type"] == "table":
            key = ("Table", str(tbl_counter))
            if key not in index:
                index[key] = b["block_id"]
            tbl_counter += 1

    fig_counter = 1
    for b in blocks:
        if b["type"] == "figure":
            key = ("Figure", str(fig_counter))
            if key not in index:
                index[key] = b["block_id"]
            fig_counter += 1

    return index


# ── 4B: PRECEDES ─────────────────────────────────────────────────────────────

def compute_precedes_edges(sorted_blocks: list[dict]) -> list[tuple[str, str]]:
    """Consecutive block pairs in global reading order."""
    return [
        (sorted_blocks[i]["block_id"], sorted_blocks[i + 1]["block_id"])
        for i in range(len(sorted_blocks) - 1)
    ]


# ── 4C: DESCRIBES ─────────────────────────────────────────────────────────────

def _vert_gap(anchor: dict, candidate: dict) -> float:
    ay0 = anchor["bbox_y0"] or 0
    ay1 = anchor["bbox_y1"] or 0
    by0 = candidate["bbox_y0"]
    by1 = candidate["bbox_y1"]
    if by1 <= ay0:
        return ay0 - by1
    if by0 >= ay1:
        return by0 - ay1
    return 0.0


def find_nearest_block(
    anchor: dict,
    candidates: list[dict],
    max_y_gap: float = 50.0,
) -> str | None:
    """
    Return block_id of the candidate block vertically nearest to anchor on the
    same page, within max_y_gap PDF points.
    """
    same_page = [
        b for b in candidates
        if b["page_index"] == anchor["page_index"]
        and b["block_id"] != anchor["block_id"]
        and b["bbox_y0"] is not None
    ]
    if not same_page:
        return None
    same_page.sort(key=lambda b: _vert_gap(anchor, b))
    best = same_page[0]
    if _vert_gap(anchor, best) <= max_y_gap:
        return best["block_id"]
    return None


def compute_describes_edges(blocks: list[dict]) -> list[tuple[str, str]]:
    """caption → nearest table/figure on the same page."""
    tf_blocks      = [b for b in blocks if b["type"] in ("table", "figure")]
    caption_blocks = [b for b in blocks if b["type"] == "caption"]
    edges: list[tuple[str, str]] = []
    for cap in caption_blocks:
        target = find_nearest_block(cap, tf_blocks)
        if target:
            edges.append((cap["block_id"], target))
    return edges


# ── 4D: INTRODUCES ────────────────────────────────────────────────────────────

def compute_introduces_edges(sorted_blocks: list[dict]) -> list[tuple[str, str]]:
    """
    heading → immediately following block + any table/figure in scope until
    the next heading of the same or higher level.
    """
    edges: list[tuple[str, str]] = []
    for i, blk in enumerate(sorted_blocks):
        if blk["type"] != "heading":
            continue
        level = blk["heading_level"] or 1
        for j in range(i + 1, len(sorted_blocks)):
            nxt = sorted_blocks[j]
            if nxt["type"] == "heading" and (nxt["heading_level"] or 1) <= level:
                break
            if j == i + 1 or nxt["type"] in ("table", "figure"):
                edges.append((blk["block_id"], nxt["block_id"]))
    return edges


# ── 4E: IN_HEADING_SCOPE (optional legacy edge) ───────────────────────────────

def compute_in_heading_scope_edges(sorted_blocks: list[dict]) -> list[tuple[str, str]]:
    """
    Optional legacy edge: each non-heading block linked to the deepest active
    heading Block (NOT a Section). Off by default — see
    ``config.ENABLE_IN_HEADING_SCOPE``. Provided for backwards-compatible
    queries; the canonical hierarchy edge is now Block -> Section.
    """
    edges: list[tuple[str, str]] = []
    active: dict[int, dict] = {}

    for blk in sorted_blocks:
        if blk["type"] == "heading":
            lvl = blk["heading_level"] or 1
            active[lvl] = blk
            for l in list(active):
                if l > lvl:
                    del active[l]
        elif active:
            deepest = max(active)
            edges.append((blk["block_id"], active[deepest]["block_id"]))

    return edges


# ── 4F: CONTEXT_BEFORE / CONTEXT_AFTER ───────────────────────────────────────

def compute_context_edges(
    sorted_blocks: list[dict],
    window: int = config.CONTEXT_WINDOW,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Returns (context_before_edges, context_after_edges).
    context_before: (block_id, table_id) for N blocks immediately before each table.
    context_after:  (table_id, block_id) for N blocks immediately after each table.
    """
    before_edges: list[tuple[str, str]] = []
    after_edges:  list[tuple[str, str]] = []
    n = len(sorted_blocks)

    for pos, blk in enumerate(sorted_blocks):
        if blk["type"] != "table":
            continue
        tbl_id = blk["block_id"]
        for k in range(1, window + 1):
            if pos - k >= 0:
                before_edges.append((sorted_blocks[pos - k]["block_id"], tbl_id))
            if pos + k < n:
                after_edges.append((tbl_id, sorted_blocks[pos + k]["block_id"]))

    return before_edges, after_edges


# ── 4G: REFERS_TO first pass (regex) ─────────────────────────────────────────

def compute_refers_to_regex(
    blocks: list[dict],
    ref_index: dict[tuple, str],
) -> tuple[list[dict], set[tuple[str, str]]]:
    """
    Scan all blocks for explicit cross-references matching the regex pattern.

    Returns:
        refers_to_edges — list of {src, tgt, methods, mention} dicts
        regex_pairs     — set of (src, tgt) tuples for deduplication in LLM pass
    """
    refers_to_edges: list[dict] = []
    seen: dict[tuple[str, str], dict] = {}

    for blk in blocks:
        if not blk["raw_text"]:
            continue
        for ref in extract_explicit_refs(blk["raw_text"]):
            key = (ref["kind"], ref["num"])
            if key not in ref_index:
                continue
            tgt_id = ref_index[key]
            if tgt_id == blk["block_id"]:
                continue
            pair = (blk["block_id"], tgt_id)
            if pair not in seen:
                mention = f"{ref['kind']} {ref['num']}"
                # Evidence: short context window around the first mention occurrence.
                idx = blk["raw_text"].lower().find(mention.lower())
                if idx >= 0:
                    ev_start = max(0, idx - 20)
                    ev_end = min(len(blk["raw_text"]), idx + len(mention) + 20)
                    evidence = blk["raw_text"][ev_start:ev_end].replace("\n", " ").strip()
                else:
                    evidence = ""
                edge = {
                    "src":                blk["block_id"],
                    "tgt":                tgt_id,
                    "methods":            ["regex"],
                    "mention":            mention,
                    "confidence":         config.DEFAULT_REFERS_TO_CONFIDENCE_REGEX,
                    "evidence":           evidence,
                    "created_by_stage":   "refers_to_regex",
                }
                seen[pair] = edge
                refers_to_edges.append(edge)

    return refers_to_edges, set(seen.keys())


# ── Section 6: REFERS_TO second pass (LLM) ───────────────────────────────────

_SYSTEM_DISCUSSES = """\
You are a scholarly document analysis assistant.
Given a TABLE and a list of TEXT BLOCKS (paragraphs/captions), decide which blocks
discuss, interpret, or summarise the table's content.
Respond ONLY with a JSON array of block IDs that discuss the table.
Return an empty array [] if none qualify.
Example response: ["p0002_b0003", "p0003_b0001"]"""


def _get_nearby_blocks(
    target: dict,
    all_blocks: list[dict],
    page_window: int = 1,
    types: frozenset = frozenset({"paragraph", "caption", "list_item"}),
) -> list[dict]:
    p  = target["page_index"]
    ro = target["reading_order"]
    candidates = [
        b for b in all_blocks
        if b["type"] in types
        and abs(b["page_index"] - p) <= page_window
        and b["block_id"] != target["block_id"]
    ]
    candidates.sort(key=lambda b: abs(b["reading_order"] - ro))
    return candidates


def _llm_find_discussing_blocks(
    table_block: dict,
    candidate_blocks: list[dict],
    max_candidates: int = 6,
) -> list[str]:
    if not candidate_blocks:
        return []
    table_snippet = table_block["text"][:600]
    cand_lines = [
        f'- ID: "{b["block_id"]}" | {b["type"].upper()}: {b["text"][:200].replace(chr(10), " ")}'
        for b in candidate_blocks[:max_candidates]
    ]
    user_prompt = (
        f"TABLE ID: {table_block['block_id']}\n"
        f"TABLE CONTENT (truncated):\n{table_snippet}\n\n"
        f"CANDIDATE BLOCKS:\n" + "\n".join(cand_lines) +
        "\n\nWhich block IDs discuss this table?"
    )
    raw = llm_chat(_SYSTEM_DISCUSSES, user_prompt, max_tokens=200)
    try:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            ids = json.loads(match.group())
            valid_ids = {b["block_id"] for b in candidate_blocks}
            return [i for i in ids if i in valid_ids]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def compute_refers_to_llm(
    blocks: list[dict],
    refers_to_edges: list[dict],
    regex_pairs: set[tuple[str, str]],
    parallel: bool = False,
) -> list[dict]:
    """
    LLM second pass: for each table, ask the LLM which nearby paragraphs
    discuss it. Merges results into (and mutates) refers_to_edges in-place.

    When parallel=True, uses a ThreadPoolExecutor so the LLM server can
    batch decode requests concurrently (requires server started with --parallel N).

    Returns the updated refers_to_edges list.
    """
    refers_to_index: dict[tuple[str, str], dict] = {
        (e["src"], e["tgt"]): e for e in refers_to_edges
    }
    table_blocks = [b for b in blocks if b["type"] == "table"]
    print(f"Running LLM REFERS_TO second pass on {len(table_blocks)} tables …")

    def _process_table(tbl: dict) -> tuple[str, list[str]]:
        all_candidates = _get_nearby_blocks(tbl, blocks, page_window=1)
        candidates = [b for b in all_candidates if (b["block_id"], tbl["block_id"]) not in regex_pairs]
        return tbl["block_id"], _llm_find_discussing_blocks(tbl, candidates)

    if parallel:
        with ThreadPoolExecutor(max_workers=config.LLM_PARALLEL_SLOTS) as pool:
            futures = {pool.submit(_process_table, tbl): tbl for tbl in table_blocks}
            results: list[tuple[str, list[str]]] = []
            for future in tqdm(as_completed(futures), total=len(table_blocks), desc="LLM refers-to"):
                results.append(future.result())
    else:
        results = []
        for tbl in tqdm(table_blocks, desc="LLM refers-to"):
            results.append(_process_table(tbl))
            time.sleep(0.05)

    # Merge results serially — safe whether serial or parallel
    for tbl_id, discussing in results:
        for bid in discussing:
            pair = (bid, tbl_id)
            if pair in refers_to_index:
                edge = refers_to_index[pair]
                if "llm" not in edge["methods"]:
                    edge["methods"].append("llm")
                # When both regex and LLM agree, bump confidence.
                if "regex" in edge["methods"] and "llm" in edge["methods"]:
                    edge["confidence"] = config.DEFAULT_REFERS_TO_CONFIDENCE_BOTH
                    edge["created_by_stage"] = "refers_to_regex+llm"
            else:
                new_edge = {
                    "src":              bid,
                    "tgt":              tbl_id,
                    "methods":          ["llm"],
                    "mention":          None,
                    "confidence":       config.DEFAULT_REFERS_TO_CONFIDENCE_LLM,
                    "evidence":         "",
                    "created_by_stage": "refers_to_llm",
                }
                refers_to_index[pair] = new_edge
                refers_to_edges.append(new_edge)

    return refers_to_edges


# ── Section 7: Table-pair LLM labelling ──────────────────────────────────────

_SYSTEM_TABLE_REL = """\
You are a scholarly document analysis assistant.
TABLE A appears EARLIER in the document than TABLE B.
Describe their relationship using ONE of these labels:
  COMPARES    – both tables report results for the same task or metric
  ABLATES     – one table defines baselines/components that the other ablates or varies
  SUPPLEMENTS – one table provides a supplementary or detailed breakdown of the other
  CONTRASTS   – tables cover distinct tasks or domains being contrasted
  UNRELATED   – no meaningful relationship

Express the relationship FROM TABLE A's perspective (A → B).
If the natural direction is B → A (e.g. B ablates something first defined in A),
set "direction" to "B_to_A". Otherwise set "direction" to "A_to_B".

Respond ONLY with JSON:
{"relationship": "<LABEL>", "reason": "<one sentence>", "direction": "A_to_B" or "B_to_A"}"""


def _llm_label_table_pair(tbl_a: dict, tbl_b: dict) -> dict:
    snippet_a = tbl_a["text"][:400].replace("\n", " | ")
    snippet_b = tbl_b["text"][:400].replace("\n", " | ")
    user_prompt = (
        f"TABLE A (ID: {tbl_a['block_id']}, Page {tbl_a['page_number']}):\n{snippet_a}\n\n"
        f"TABLE B (ID: {tbl_b['block_id']}, Page {tbl_b['page_number']}):\n{snippet_b}\n\n"
        "What is the relationship between TABLE A and TABLE B?"
    )
    raw = llm_chat(_SYSTEM_TABLE_REL, user_prompt, max_tokens=200)
    try:
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, TypeError):
        pass
    return {"relationship": "UNRELATED", "reason": "parse error", "direction": "A_to_B"}


def compute_table_pair_rels(
    blocks: list[dict],
    embeddings: dict[str, Any],
    parallel: bool = False,
) -> list[dict]:
    """
    LLM-label relationships between every candidate table pair.

    Pre-filter (AND):
      1. |page_a - page_b| <= TABLE_PAIR_PAGE_WINDOW
      2. cosine_sim(embed_a, embed_b) >= TABLE_PAIR_SEM_FLOOR

    When parallel=True, uses a ThreadPoolExecutor so the LLM server can
    batch decode requests concurrently (requires server started with --parallel N).

    Returns list of {src, tgt, relationship, reason} dicts (UNRELATED excluded).
    """
    table_blocks = [b for b in blocks if b["type"] == "table"]
    all_pairs    = list(combinations(table_blocks, 2))

    def _is_candidate(a: dict, b: dict) -> bool:
        if abs(a["page_number"] - b["page_number"]) > config.TABLE_PAIR_PAGE_WINDOW:
            return False
        if a["block_id"] in embeddings and b["block_id"] in embeddings:
            return cosine_sim(embeddings[a["block_id"]], embeddings[b["block_id"]]) >= config.TABLE_PAIR_SEM_FLOOR
        return False

    candidates = [(a, b) for a, b in all_pairs if _is_candidate(a, b)]
    n_skipped  = len(all_pairs) - len(candidates)
    print(
        f"{len(all_pairs)} total pairs → {len(candidates)} candidates "
        f"({n_skipped} skipped by page+similarity filter)"
    )

    def _process_pair(tbl_a: dict, tbl_b: dict) -> dict | None:
        result = _llm_label_table_pair(tbl_a, tbl_b)
        rel    = result.get("relationship", "UNRELATED")
        if rel == "UNRELATED":
            return None
        if result.get("direction", "A_to_B") == "B_to_A":
            src, tgt = tbl_b["block_id"], tbl_a["block_id"]
        else:
            src, tgt = tbl_a["block_id"], tbl_b["block_id"]
        return {
            "src":              src,
            "tgt":              tgt,
            "relationship":     rel,
            "reason":           result.get("reason", ""),
            "methods":          ["llm"],
            "model":            config.LLM_MODEL_NAME,
            "scope":            "table",
            "confidence":       config.DEFAULT_TABLE_PAIR_CONFIDENCE,
            "created_by_stage": "table_pair_llm",
        }

    if parallel:
        with ThreadPoolExecutor(max_workers=config.LLM_PARALLEL_SLOTS) as pool:
            futures = {pool.submit(_process_pair, a, b): (a, b) for a, b in candidates}
            raw_results: list[dict | None] = []
            for future in tqdm(as_completed(futures), total=len(candidates), desc="LLM table-pairs"):
                raw_results.append(future.result())
    else:
        raw_results = []
        for tbl_a, tbl_b in tqdm(candidates, desc="LLM table-pairs"):
            raw_results.append(_process_pair(tbl_a, tbl_b))
            time.sleep(0.05)

    table_pair_rels = [r for r in raw_results if r is not None]
    return table_pair_rels
