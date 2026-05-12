"""
Neo4j graph writer.

Translates the extracted document structure, embeddings, and relationships
into Cypher queries and writes them to a running Neo4j instance.
Also exports a kg_summary.json report to the same directory as document.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import config


# ── Low-level query runner ────────────────────────────────────────────────────

def make_runner(driver):
    """Return a run_query() function bound to the given driver."""
    def run_query(query: str, params: dict | None = None) -> list[dict]:
        with driver.session() as session:
            return session.run(query, params or {}).data()
    return run_query


# ── Schema setup ──────────────────────────────────────────────────────────────

def setup_schema(run_query) -> None:
    schema_queries = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Page)     REQUIRE p.page_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Block)    REQUIRE b.block_id IS UNIQUE",
        "CREATE INDEX IF NOT EXISTS FOR (b:Block) ON (b.type)",
    ]
    for q in schema_queries:
        run_query(q)
    print("Constraints & indexes created.")

    try:
        run_query("""
            CREATE VECTOR INDEX block_embedding_index IF NOT EXISTS
            FOR (b:Block) ON (b.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}}
        """)
        print("Vector index created (or already exists).")
    except Exception as e:
        print(f"Vector index skipped (Neo4j < 5.11): {e}")


# ── Node writes ───────────────────────────────────────────────────────────────

def write_document_node(run_query, raw_doc: dict) -> str:
    doc_meta = raw_doc["document"]
    doc_id   = doc_meta["source_sha256"]
    run_query(
        """
        MERGE (d:Document {doc_id: $doc_id})
        SET d.filename  = $filename,
            d.num_pages = $num_pages,
            d.backend   = $backend,
            d.sha256    = $sha256
        """,
        {
            "doc_id":    doc_id,
            "filename":  doc_meta["source_filename"],
            "num_pages": doc_meta["num_pages"],
            "backend":   raw_doc["backend"]["name"],
            "sha256":    doc_id,
        },
    )
    print("Document node created.")
    return doc_id


def write_page_nodes(run_query, raw_doc: dict, doc_id: str) -> None:
    page_params = [
        {
            "page_id":     f"{doc_id}_p{pg['index']}",
            "page_number": pg["number"],
            "page_index":  pg["index"],
            "doc_id":      doc_id,
        }
        for pg in raw_doc["pages"]
    ]
    run_query(
        """
        UNWIND $pages AS p
        MERGE (pg:Page {page_id: p.page_id})
        SET pg.page_number = p.page_number,
            pg.page_index  = p.page_index
        WITH pg, p
        MATCH (d:Document {doc_id: p.doc_id})
        MERGE (pg)-[:PART_OF]->(d)
        """,
        {"pages": page_params},
    )
    print(f"Created {len(page_params)} page nodes.")


def write_block_nodes(
    run_query,
    blocks: list[dict],
    embeddings: dict[str, np.ndarray],
    doc_id: str,
) -> None:
    block_query = """
    UNWIND $blocks AS b
    MERGE (blk:Block {block_id: b.block_id})
    SET blk.type          = b.type,
        blk.text          = b.text,
        blk.page_number   = b.page_number,
        blk.page_index    = b.page_index,
        blk.reading_order = b.reading_order,
        blk.bbox_x0       = b.bbox_x0,
        blk.bbox_y0       = b.bbox_y0,
        blk.bbox_x1       = b.bbox_x1,
        blk.bbox_y1       = b.bbox_y1,
        blk.table_rows    = b.table_rows,
        blk.table_cols    = b.table_cols,
        blk.table_html    = b.table_html,
        blk.figure_path   = b.figure_path,
        blk.formula_latex = b.formula_latex,
        blk.heading_level = b.heading_level,
        blk.embedding     = b.embedding
    WITH blk, b
    MATCH (pg:Page {page_id: b.page_id})
    MERGE (blk)-[:ON_PAGE]->(pg)
    """
    block_params = []
    for b in blocks:
        vec = embeddings.get(b["block_id"])
        block_params.append({
            **b,
            "page_id":   f"{doc_id}_p{b['page_index']}",
            "embedding": vec.tolist() if vec is not None else None,
        })

    batch = config.BLOCK_WRITE_BATCH
    for start in tqdm(range(0, len(block_params), batch), desc="Block nodes"):
        run_query(block_query, {"blocks": block_params[start: start + batch]})
    print(f"Created {len(block_params)} block nodes.")


def write_secondary_labels(run_query, blocks: list[dict]) -> None:
    label_params = [
        {"block_id": b["block_id"], "label": b["type"].capitalize()}
        for b in blocks
    ]
    try:
        run_query(
            """
            UNWIND $items AS item
            MATCH (b:Block {block_id: item.block_id})
            CALL apoc.create.addLabels(b, [item.label]) YIELD node
            RETURN count(node)
            """,
            {"items": label_params},
        )
        print("Secondary labels applied via APOC.")
    except Exception as e:
        print(f"APOC not available — skipping secondary labels ({e}).")


# ── Edge write helpers ────────────────────────────────────────────────────────

def _write_simple_edges(
    run_query,
    edges: list[tuple[str, str]],
    rel_type: str,
) -> None:
    """Write (src, tgt) edge list using MERGE."""
    query = f"""
    UNWIND $edges AS e
    MATCH (a:Block {{block_id: e.src}})
    MATCH (b:Block {{block_id: e.tgt}})
    MERGE (a)-[:{rel_type}]->(b)
    """
    params = [{"src": s, "tgt": t} for s, t in edges]
    batch  = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"{rel_type}: {len(params)} edges written.")


# ── Relationship writes ───────────────────────────────────────────────────────

def write_precedes(run_query, edges: list[tuple[str, str]]) -> None:
    _write_simple_edges(run_query, edges, "PRECEDES")


def write_describes(run_query, edges: list[tuple[str, str]]) -> None:
    _write_simple_edges(run_query, edges, "DESCRIBES")


def write_introduces(run_query, edges: list[tuple[str, str]]) -> None:
    _write_simple_edges(run_query, edges, "INTRODUCES")


def write_in_section(run_query, edges: list[tuple[str, str]]) -> None:
    _write_simple_edges(run_query, edges, "IN_SECTION")


def write_context_edges(
    run_query,
    before_edges: list[tuple[str, str]],
    after_edges:  list[tuple[str, str]],
) -> None:
    _write_simple_edges(run_query, before_edges, "CONTEXT_BEFORE")
    _write_simple_edges(run_query, after_edges,  "CONTEXT_AFTER")


def write_refers_to(run_query, edges: list[dict]) -> None:
    query = """
    UNWIND $edges AS e
    MATCH (a:Block {block_id: e.src})
    MATCH (b:Block {block_id: e.tgt})
    MERGE (a)-[r:REFERS_TO]->(b)
    SET r.methods = e.methods,
        r.mention = e.mention
    """
    params = [
        {"src": e["src"], "tgt": e["tgt"], "methods": e["methods"], "mention": e["mention"]}
        for e in edges
    ]
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"REFERS_TO: {len(params)} edges written.")


def write_semantically_similar(
    run_query,
    edges: list[tuple[str, str, float]],
) -> None:
    query = """
    UNWIND $edges AS e
    MATCH (a:Block {block_id: e.src})
    MATCH (b:Block {block_id: e.tgt})
    MERGE (a)-[r:SEMANTICALLY_SIMILAR]->(b)
    SET r.score = e.score
    """
    params = [{"src": s, "tgt": t, "score": sc} for s, t, sc in edges]
    batch  = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"SEMANTICALLY_SIMILAR: {len(params)} edges written.")


def write_table_pair_rels(run_query, rels: list[dict]) -> None:
    apoc_query = """
    UNWIND $rels AS r
    MATCH (a:Block {block_id: r.src})
    MATCH (b:Block {block_id: r.tgt})
    CALL apoc.create.relationship(a, r.relationship, {reason: r.reason}, b)
    YIELD rel
    RETURN count(rel)
    """
    fallback_query = """
    UNWIND $rels AS r
    MATCH (a:Block {block_id: r.src})
    MATCH (b:Block {block_id: r.tgt})
    MERGE (a)-[rel:TABLE_RELATES_TO {label: r.relationship}]->(b)
    SET rel.reason = r.reason
    """
    try:
        run_query(apoc_query, {"rels": rels})
        print(f"Table-pair relationships written via APOC dynamic rel-types: {len(rels)}")
    except Exception:
        run_query(fallback_query, {"rels": rels})
        print(f"Table-pair relationships written via TABLE_RELATES_TO (fallback): {len(rels)}")


# ── Summary export ────────────────────────────────────────────────────────────

def export_summary(
    run_query,
    raw_doc: dict,
    blocks: list[dict],
    refers_to_edges: list[dict],
    table_pair_rels: list[dict],
    output_dir: Path,
    backend: str,
    run_variant: str,
) -> Path:
    """
    Query Neo4j for final counts and write kg_summary.json to output_dir.
    Returns the path to the written file.
    """
    doc_meta   = raw_doc["document"]
    type_counts: dict[str, int] = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1

    rel_rows = run_query("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)
    edge_counts = {row["rel_type"]: row["count"] for row in rel_rows}

    method_counts: dict[str, int] = {}
    for e in refers_to_edges:
        k = "+".join(sorted(e["methods"]))
        method_counts[k] = method_counts.get(k, 0) + 1

    summary = {
        "document":            doc_meta["source_filename"],
        "backend":             backend,
        "run_variant":         run_variant,
        "total_blocks":        len(blocks),
        "block_types":         type_counts,
        "edges":               edge_counts,
        "refers_to_breakdown": method_counts,
        "table_pair_relations": table_pair_rels,
    }

    out_path = output_dir / "kg_summary.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return out_path


# ── Main orchestrator ─────────────────────────────────────────────────────────

def build_graph(
    driver,
    raw_doc: dict,
    blocks: list[dict],
    embeddings: dict[str, Any],
    sorted_blocks: list[dict],
    precedes_edges: list[tuple[str, str]],
    describes_edges: list[tuple[str, str]],
    introduces_edges: list[tuple[str, str]],
    in_section_edges: list[tuple[str, str]],
    context_before_edges: list[tuple[str, str]],
    context_after_edges: list[tuple[str, str]],
    refers_to_edges: list[dict],
    semantic_edges: list[tuple[str, str, float]],
    table_pair_rels: list[dict],
    output_dir: Path,
    backend: str,
    run_variant: str,
) -> Path:
    """
    Build the complete Neo4j knowledge graph from extracted data.

    Clears the existing graph, creates schema, writes all nodes and
    relationships in the same order as the notebook (Sections 8A–8J),
    and exports kg_summary.json.

    Returns the path to the summary JSON file.
    """
    run_query = make_runner(driver)

    print("\n── Clearing existing graph …")
    run_query("MATCH (n) DETACH DELETE n")
    print("Graph cleared.")

    print("\n── Setting up schema …")
    setup_schema(run_query)

    print("\n── Writing nodes …")
    doc_id = write_document_node(run_query, raw_doc)
    write_page_nodes(run_query, raw_doc, doc_id)
    write_block_nodes(run_query, blocks, embeddings, doc_id)
    write_secondary_labels(run_query, blocks)

    print("\n── Writing structural edges …")
    write_precedes(run_query, precedes_edges)
    write_describes(run_query, describes_edges)
    write_introduces(run_query, introduces_edges)
    write_in_section(run_query, in_section_edges)
    write_context_edges(run_query, context_before_edges, context_after_edges)

    print("\n── Writing semantic edges …")
    write_refers_to(run_query, refers_to_edges)
    write_semantically_similar(run_query, semantic_edges)

    print("\n── Writing LLM-labelled table-pair edges …")
    write_table_pair_rels(run_query, table_pair_rels)

    print("\n── Exporting summary …")
    summary_path = export_summary(
        run_query, raw_doc, blocks, refers_to_edges,
        table_pair_rels, output_dir, backend, run_variant,
    )
    print(f"kg_summary.json written to {summary_path}")

    return summary_path
