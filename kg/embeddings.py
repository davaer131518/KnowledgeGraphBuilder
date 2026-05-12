"""
Embedding generation and semantic-similarity edge computation.

Embeddings are cached to a pickle file alongside document.json so that
subsequent runs skip re-embedding all blocks.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import config
from kg.models import cosine_sim, embed_text

# Block type pairs eligible for SEMANTICALLY_SIMILAR edges.
# The table is always the target; the other block is the source.
_PAIR_TYPES: set[tuple[str, str]] = {
    ("formula",   "table"),
    ("table",     "table"),
    ("paragraph", "table"),
    ("figure",    "table"),
    ("caption",   "table"),
}


def compute_or_load_embeddings(
    blocks: list[dict],
    doc_sha: str,
    cache_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Return a {block_id → L2-normalised embedding} dict.

    If a cache file for this document SHA already exists in cache_dir it is
    loaded directly.  Otherwise all blocks are embedded and the result is
    persisted to a .pkl file for future runs.

    Args:
        blocks:    The full list of block dicts (from loader.load_document).
        doc_sha:   The document's source_sha256 (first 12 chars used as key).
        cache_dir: Directory where the cache file is written (usually doc json dir).
    """
    cache_path = cache_dir / f"embeddings_{doc_sha[:12]}.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as fh:
            embeddings: dict[str, np.ndarray] = pickle.load(fh)
        print(f"Loaded {len(embeddings)} embeddings from cache ({cache_path.name}).")
        return embeddings

    print(f"Generating embeddings for {len(blocks)} blocks …")
    embeddings = {}
    for blk in tqdm(blocks, desc="Embedding"):
        embed_input = f"[{blk['type'].upper()}] {blk['text']}"
        embeddings[blk["block_id"]] = embed_text(embed_input)

    with cache_path.open("wb") as fh:
        pickle.dump(embeddings, fh)
    print(f"Embedded {len(embeddings)} blocks. Cache saved to {cache_path.name}.")
    return embeddings


def compute_semantic_edges(
    blocks: list[dict],
    embeddings: dict[str, Any],
    top_k: int = config.SEM_SIM_TOP_K,
    min_score: float = config.SEM_SIM_MIN_SCORE,
) -> list[tuple[str, str, float]]:
    """
    For each table block, compute cosine similarity against all eligible block
    type pairs and keep the top_k results above min_score.

    Returns list of (other_block_id, table_id, score) tuples — the non-table
    block is always the *source* in the final SEMANTICALLY_SIMILAR edge.
    """
    block_ids      = [b["block_id"] for b in blocks]
    block_type_map = {b["block_id"]: b["type"] for b in blocks}

    # Accumulate raw (score, other_id) lists keyed by table block_id
    table_sims: dict[str, list[tuple[float, str]]] = {
        b["block_id"]: [] for b in blocks if b["type"] == "table"
    }

    for i, id_a in enumerate(tqdm(block_ids, desc="Similarity")):
        type_a = block_type_map[id_a]
        for j in range(i + 1, len(block_ids)):
            id_b   = block_ids[j]
            type_b = block_type_map[id_b]
            pair   = tuple(sorted([type_a, type_b]))
            if pair not in _PAIR_TYPES:
                continue
            if id_a not in embeddings or id_b not in embeddings:
                continue
            score = cosine_sim(embeddings[id_a], embeddings[id_b])
            if type_a == "table":
                table_sims[id_a].append((score, id_b))
            else:
                table_sims[id_b].append((score, id_a))

    # Select top-K per table above the soft floor
    semantic_edges: list[tuple[str, str, float]] = []
    for tbl_id, sims in table_sims.items():
        for score, other_id in sorted(sims, reverse=True)[:top_k]:
            if score >= min_score:
                semantic_edges.append((other_id, tbl_id, round(score, 4)))

    return semantic_edges
