"""
Global block-to-block semantic similarity.

Computes top-K cosine-similarity edges between arbitrary content blocks using
the already-cached L2-normalised embeddings. Canonicalises direction
(src < tgt) and deduplicates against any table-anchored edges that were
emitted earlier in the pipeline.

Public entrypoint:
    compute_global_block_similarity(
        blocks, embeddings, table_edges, mentions_by_block=None
    ) -> (edges, skip_info)

``edges`` is a list of dicts ready for the Neo4j writer.
``skip_info`` carries the metadata that bubbles up into kg_summary.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from tqdm import tqdm

import config


SimEdge = dict


def compute_global_block_similarity(
    blocks: list[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    table_edges: list[tuple[str, str, float]],
    mentions_by_block: dict[str, int] | None = None,
) -> tuple[list[SimEdge], dict]:
    """
    Build SEMANTICALLY_SIMILAR edges across content block types.

    Args:
        blocks:            block dicts (already sorted by reading order).
        embeddings:        {block_id -> np.ndarray} (L2-normalised float32).
        table_edges:       output of compute_semantic_edges() — used for dedup.
                           Each tuple is (src, tgt, score). Direction is whatever
                           the table-anchored writer produced.
        mentions_by_block: optional {block_id -> mention_count}; blocks with at
                           least one mention bypass the short-text filter.

    Returns:
        edges:     list of edge dicts (one per canonicalised pair).
        skip_info: {"skipped": bool, "reason": str | None, "filtered_count": int}.
    """
    if not config.ENABLE_GLOBAL_BLOCK_SIM:
        return [], {"skipped": True, "reason": "ENABLE_GLOBAL_BLOCK_SIM is False", "filtered_count": 0}

    allowed_types = set(config.BLOCK_SIM_ALLOWED_TYPES)
    short_threshold = config.BLOCK_SIM_SKIP_SHORT_TEXT_CHARS
    mentions_by_block = mentions_by_block or {}

    eligible: list[dict] = []
    for b in blocks:
        if b["type"] not in allowed_types:
            continue
        if b["block_id"] not in embeddings:
            continue
        text_len = len(b.get("text") or "")
        has_entity = mentions_by_block.get(b["block_id"], 0) > 0
        if text_len < short_threshold and not has_entity:
            continue
        eligible.append(b)

    filtered_count = len(eligible)
    if filtered_count > config.BLOCK_SIM_MAX_BLOCKS:
        reason = (
            f"filtered block count ({filtered_count}) exceeded "
            f"BLOCK_SIM_MAX_BLOCKS ({config.BLOCK_SIM_MAX_BLOCKS})"
        )
        print(f"  [warn] Global similarity skipped: {reason}")
        return [], {"skipped": True, "reason": reason, "filtered_count": filtered_count}

    if filtered_count < 2:
        return [], {"skipped": False, "reason": None, "filtered_count": filtered_count}

    block_ids = [b["block_id"] for b in eligible]
    page_indices = np.array([b["page_index"] for b in eligible], dtype=np.int32)

    # Stack into float32 matrix. Vectors are already L2-normalised, so cosine
    # similarity == dot product.
    vectors = [np.asarray(embeddings[bid], dtype=np.float32) for bid in block_ids]
    M = np.stack(vectors, axis=0)
    sims = M @ M.T  # shape (N, N), float32

    n = sims.shape[0]
    np.fill_diagonal(sims, -1.0)

    if config.BLOCK_SIM_SKIP_SAME_PAGE:
        same_page = page_indices[:, None] == page_indices[None, :]
        sims[same_page] = -1.0

    k = min(config.BLOCK_SIM_TOP_K, n - 1)
    min_score = config.BLOCK_SIM_MIN_SCORE

    # Build dedup set against the already-emitted table-anchored edges.
    table_pair_set: set[frozenset] = {frozenset({src, tgt}) for src, tgt, _ in table_edges}

    # Per pair: keep best (max) score and lowest rank seen across both rows.
    pair_best: dict[tuple[str, str], dict] = {}
    for i in tqdm(range(n), desc="Block similarity"):
        row = sims[i]
        if k <= 0:
            continue
        # argpartition gives the k highest indices, unsorted; sort the slice.
        top_idx = np.argpartition(-row, k - 1)[:k]
        top_sorted = top_idx[np.argsort(-row[top_idx])]
        for rank, j in enumerate(top_sorted, start=1):
            score = float(row[j])
            if score < min_score:
                break
            src_id = block_ids[i]
            tgt_id = block_ids[int(j)]
            if src_id == tgt_id:
                continue
            pair_key = (src_id, tgt_id) if src_id < tgt_id else (tgt_id, src_id)
            if frozenset(pair_key) in table_pair_set:
                continue
            existing = pair_best.get(pair_key)
            if existing is None or score > existing["score"]:
                pair_best[pair_key] = {
                    "src":  pair_key[0],
                    "tgt":  pair_key[1],
                    "score": score,
                    "rank":  rank,
                }
            else:
                existing["rank"] = min(existing["rank"], rank)

    edges: list[SimEdge] = []
    for rec in pair_best.values():
        edges.append({
            "src":   rec["src"],
            "tgt":   rec["tgt"],
            "score": round(rec["score"], 4),
            "rank":  rec["rank"],
            "scope": "global",
            "confidence": round(rec["score"], 4),
            "stage": "global_similarity",
        })

    return edges, {"skipped": False, "reason": None, "filtered_count": filtered_count}
