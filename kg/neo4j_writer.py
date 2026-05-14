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
        # New: Section + Entity constraints/indexes (idempotent).
        "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        "CREATE INDEX section_doc_id IF NOT EXISTS FOR (s:Section) ON (s.doc_id)",
        "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
        "CREATE INDEX entity_lookup IF NOT EXISTS FOR (e:Entity) ON (e.doc_id, e.normalized_name, e.type)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.normalized_name)",
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
            pg.page_index  = p.page_index,
            pg.doc_id      = p.doc_id
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
    """Block -> Section IN_SECTION edges. (NOTE: Section, not heading Block.)"""
    query = """
    UNWIND $edges AS e
    MATCH (b:Block {block_id: e.block_id})
    MATCH (sec:Section {section_id: e.section_id})
    MERGE (b)-[:IN_SECTION]->(sec)
    """
    params = [{"block_id": s, "section_id": t} for s, t in edges]
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"IN_SECTION: {len(params)} edges written.")


def write_in_heading_scope(run_query, edges: list[tuple[str, str]]) -> None:
    """Optional legacy Block -> Block edge (under a separate rel name)."""
    query = """
    UNWIND $edges AS e
    MATCH (b:Block {block_id: e.src})
    MATCH (h:Block {block_id: e.tgt})
    MERGE (b)-[:IN_HEADING_SCOPE]->(h)
    """
    params = [{"src": s, "tgt": t} for s, t in edges]
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"IN_HEADING_SCOPE: {len(params)} edges written.")


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
    SET r.methods          = e.methods,
        r.mention          = e.mention,
        r.confidence       = e.confidence,
        r.scope            = "reference",
        r.evidence         = e.evidence,
        r.created_by_stage = e.created_by_stage
    """
    params = [
        {
            "src":              e["src"],
            "tgt":              e["tgt"],
            "methods":          e["methods"],
            "mention":          e.get("mention"),
            "confidence":       e.get("confidence", config.DEFAULT_REFERS_TO_CONFIDENCE_REGEX),
            "evidence":         e.get("evidence", ""),
            "created_by_stage": e.get("created_by_stage", "refers_to_regex"),
        }
        for e in edges
    ]
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(query, {"edges": params[start: start + batch]})
    print(f"REFERS_TO: {len(params)} edges written.")


def write_semantically_similar(
    run_query,
    edges: list[dict],
) -> None:
    """Unified writer for both table-scope and global-scope semantic edges."""
    query = """
    UNWIND $edges AS e
    MATCH (a:Block {block_id: e.src})
    MATCH (b:Block {block_id: e.tgt})
    MERGE (a)-[r:SEMANTICALLY_SIMILAR]->(b)
    SET r.score            = e.score,
        r.rank             = e.rank,
        r.methods          = ["embedding"],
        r.model            = $model,
        r.scope            = e.scope,
        r.confidence       = e.confidence,
        r.created_by_stage = e.stage
    """
    params = [
        {
            "src":        e["src"],
            "tgt":        e["tgt"],
            "score":      e["score"],
            "rank":       e.get("rank", 0),
            "scope":      e.get("scope", "global"),
            "confidence": e.get("confidence", e["score"]),
            "stage":      e.get("stage", "table_similarity"),
        }
        for e in edges
    ]
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(params), batch):
        run_query(
            query,
            {"edges": params[start: start + batch], "model": config.EMBED_MODEL_NAME},
        )
    print(f"SEMANTICALLY_SIMILAR: {len(params)} edges written.")


def write_table_pair_rels(run_query, rels: list[dict]) -> None:
    apoc_query = """
    UNWIND $rels AS r
    MATCH (a:Block {block_id: r.src})
    MATCH (b:Block {block_id: r.tgt})
    CALL apoc.create.relationship(a, r.relationship, {
        reason:           r.reason,
        methods:          r.methods,
        model:            r.model,
        scope:            r.scope,
        confidence:       r.confidence,
        created_by_stage: r.created_by_stage
    }, b)
    YIELD rel
    RETURN count(rel)
    """
    fallback_query = """
    UNWIND $rels AS r
    MATCH (a:Block {block_id: r.src})
    MATCH (b:Block {block_id: r.tgt})
    MERGE (a)-[rel:TABLE_RELATES_TO {label: r.relationship}]->(b)
    SET rel.reason           = r.reason,
        rel.methods          = r.methods,
        rel.model            = r.model,
        rel.scope            = r.scope,
        rel.confidence       = r.confidence,
        rel.created_by_stage = r.created_by_stage
    """
    try:
        run_query(apoc_query, {"rels": rels})
        print(f"Table-pair relationships written via APOC dynamic rel-types: {len(rels)}")
    except Exception:
        run_query(fallback_query, {"rels": rels})
        print(f"Table-pair relationships written via TABLE_RELATES_TO (fallback): {len(rels)}")


# ── Section writes ────────────────────────────────────────────────────────────

def write_sections(
    run_query,
    sections: list[dict],
    parent_child_pairs: list[dict],
    top_level_sections: list[dict],
    doc_id: str,
) -> None:
    """Write Section nodes + HAS_SECTION / HAS_SUBSECTION / STARTS_ON_PAGE edges."""
    if not sections:
        print("Section nodes: nothing to write.")
        return

    section_query = """
    UNWIND $sections AS s
    MERGE (sec:Section {section_id: s.id})
    SET sec.doc_id           = s.doc_id,
        sec.title            = s.title,
        sec.level            = s.level,
        sec.level_inferred   = s.level_inferred,
        sec.level_gap        = s.level_gap,
        sec.path             = s.path,
        sec.page_start       = s.page_start,
        sec.page_end         = s.page_end,
        sec.block_start_id   = s.block_start_id,
        sec.block_end_id     = s.block_end_id,
        sec.block_count      = s.block_count,
        sec.heading_block_id = s.heading_block_id
    """
    run_query(section_query, {"sections": sections})
    print(f"Section nodes: {len(sections)} written.")

    if top_level_sections:
        run_query(
            """
            UNWIND $secs AS s
            MATCH (d:Document {doc_id: $doc_id})
            MATCH (sec:Section {section_id: s.id})
            MERGE (d)-[:HAS_SECTION]->(sec)
            """,
            {"secs": top_level_sections, "doc_id": doc_id},
        )
        print(f"HAS_SECTION: {len(top_level_sections)} edges written.")

    if parent_child_pairs:
        run_query(
            """
            UNWIND $pairs AS p
            MATCH (parent:Section {section_id: p.parent_id})
            MATCH (child:Section {section_id: p.child_id})
            MERGE (parent)-[:HAS_SUBSECTION]->(child)
            """,
            {"pairs": parent_child_pairs},
        )
        print(f"HAS_SUBSECTION: {len(parent_child_pairs)} edges written.")

    # STARTS_ON_PAGE — best-effort (skips if matching Page node doesn't exist).
    run_query(
        """
        UNWIND $sections AS s
        MATCH (sec:Section {section_id: s.id})
        MATCH (pg:Page {doc_id: $doc_id, page_number: s.page_start})
        MERGE (sec)-[:STARTS_ON_PAGE]->(pg)
        """,
        {"sections": sections, "doc_id": doc_id},
    )
    print("STARTS_ON_PAGE: written.")


# ── Entity writes ─────────────────────────────────────────────────────────────

def write_entities(run_query, entities: list[dict]) -> None:
    if not entities:
        print("Entity nodes: nothing to write.")
        return
    query = """
    UNWIND $entities AS e
    MERGE (ent:Entity {doc_id: e.doc_id, type: e.type, normalized_name: e.normalized_name})
    SET ent.entity_id           = e.id,
        ent.canonical_name      = e.canonical_name,
        ent.confidence          = e.confidence,
        ent.methods             = e.methods,
        ent.aliases             = e.aliases,
        ent.ambiguous           = e.ambiguous,
        ent.doc_frequency_ratio = e.doc_frequency_ratio
    """
    batch = config.EDGE_WRITE_BATCH
    for start in tqdm(range(0, len(entities), batch), desc="Entity nodes"):
        run_query(query, {"entities": entities[start: start + batch]})
    print(f"Entity nodes: {len(entities)} written.")


def write_mentions(run_query, mention_edges: list[dict]) -> None:
    if not mention_edges:
        print("MENTIONS: nothing to write.")
        return
    query = """
    UNWIND $edges AS m
    MATCH (b:Block {block_id: m.src})
    MATCH (e:Entity {entity_id: m.ent_id})
    MERGE (b)-[r:MENTIONS]->(e)
    SET r.count            = m.count,
        r.spans_flat       = m.spans_flat,
        r.evidence         = m.evidence,
        r.methods          = m.methods,
        r.confidence       = m.confidence,
        r.created_by_stage = "entities"
    """
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(mention_edges), batch):
        run_query(query, {"edges": mention_edges[start: start + batch]})
    print(f"MENTIONS: {len(mention_edges)} edges written.")


def write_shares_entity_with(run_query, pairs: list[dict]) -> None:
    if not pairs:
        return
    query = """
    UNWIND $pairs AS p
    MATCH (a:Block {block_id: p.src})
    MATCH (b:Block {block_id: p.tgt})
    MERGE (a)-[r:SHARES_ENTITY_WITH {entity_id: p.entity_id}]->(b)
    SET r.entity_name      = p.entity_name,
        r.methods          = ["entity_overlap"],
        r.scope            = "global",
        r.confidence       = p.confidence,
        r.created_by_stage = "shares_entity_with"
    """
    batch = config.EDGE_WRITE_BATCH
    for start in range(0, len(pairs), batch):
        run_query(query, {"pairs": pairs[start: start + batch]})
    print(f"SHARES_ENTITY_WITH: {len(pairs)} edges written.")


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
    sections: list[dict] | None = None,
    entities: list[dict] | None = None,
    mention_edges: list[dict] | None = None,
    table_sem_edges: list[dict] | None = None,
    global_sem_edges: list[dict] | None = None,
    global_sim_skip: dict | None = None,
    shares_entity_pairs: list[dict] | None = None,
    spacy_enabled: bool = False,
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

    summary: dict[str, Any] = {
        "document":            doc_meta["source_filename"],
        "backend":             backend,
        "run_variant":         run_variant,
        "total_blocks":        len(blocks),
        "block_types":         type_counts,
        "edges":               edge_counts,
        "refers_to_breakdown": method_counts,
        "table_pair_relations": table_pair_rels,
    }

    # New summary fields
    if sections is not None:
        max_depth = max((s["level"] for s in sections), default=0)
        summary["section_count"] = len(sections)
        summary["section_max_depth"] = int(max_depth)

    if entities is not None:
        ent_by_type: dict[str, int] = {}
        high_freq = 0
        alias_count = 0
        for e in entities:
            ent_by_type[e["type"]] = ent_by_type.get(e["type"], 0) + 1
            if "filtered:high_doc_freq" in e["methods"]:
                high_freq += 1
            alias_count += len(e.get("aliases", []))
        summary["entity_count"] = len(entities)
        summary["entity_count_by_type"] = ent_by_type
        summary["entity_high_freq_filtered_count"] = high_freq
        summary["alias_count"] = alias_count

    if mention_edges is not None:
        summary["mention_edge_count"] = len(mention_edges)

    if table_sem_edges is not None or global_sem_edges is not None:
        ts = len(table_sem_edges) if table_sem_edges is not None else 0
        gs = len(global_sem_edges) if global_sem_edges is not None else 0
        summary["block_semantic_similarity"] = {
            "table_scope": ts,
            "global_scope": gs,
            "total": ts + gs,
        }

    if global_sim_skip is not None:
        summary["global_similarity_skipped"] = bool(global_sim_skip.get("skipped", False))
        if global_sim_skip.get("reason"):
            summary["global_similarity_skip_reason"] = global_sim_skip["reason"]

    if shares_entity_pairs is not None:
        summary["shares_entity_with_edge_count"] = len(shares_entity_pairs)

    summary["spacy_enabled"] = bool(spacy_enabled)

    summary["feature_flags"] = {
        "ENABLE_SECTIONS":            config.ENABLE_SECTIONS,
        "ENABLE_ENTITIES":            config.ENABLE_ENTITIES,
        "ENABLE_GLOBAL_BLOCK_SIM":    config.ENABLE_GLOBAL_BLOCK_SIM,
        "ENABLE_IN_HEADING_SCOPE":    config.ENABLE_IN_HEADING_SCOPE,
        "CREATE_SHARES_ENTITY_WITH":  config.CREATE_SHARES_ENTITY_WITH,
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
    semantic_edges: list[dict],
    table_pair_rels: list[dict],
    output_dir: Path,
    backend: str,
    run_variant: str,
    sections: list[dict] | None = None,
    parent_child_pairs: list[dict] | None = None,
    top_level_sections: list[dict] | None = None,
    in_heading_scope_edges: list[tuple[str, str]] | None = None,
    entities: list[dict] | None = None,
    mention_edges: list[dict] | None = None,
    table_sem_edges: list[dict] | None = None,
    global_sem_edges: list[dict] | None = None,
    global_sim_skip: dict | None = None,
    shares_entity_pairs: list[dict] | None = None,
    spacy_enabled: bool = False,
) -> Path:
    """
    Build the complete Neo4j knowledge graph from extracted data.

    Clears the existing graph, creates schema, writes all nodes and
    relationships, and exports kg_summary.json.
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

    if sections:
        print("\n── Writing Section nodes …")
        write_sections(
            run_query,
            sections,
            parent_child_pairs or [],
            top_level_sections or [],
            doc_id,
        )

    if entities:
        print("\n── Writing Entity nodes …")
        write_entities(run_query, entities)
        write_mentions(run_query, mention_edges or [])

    print("\n── Writing structural edges …")
    write_precedes(run_query, precedes_edges)
    write_describes(run_query, describes_edges)
    write_introduces(run_query, introduces_edges)
    write_in_section(run_query, in_section_edges)
    if config.ENABLE_IN_HEADING_SCOPE and in_heading_scope_edges:
        write_in_heading_scope(run_query, in_heading_scope_edges)
    write_context_edges(run_query, context_before_edges, context_after_edges)

    print("\n── Writing semantic edges …")
    write_refers_to(run_query, refers_to_edges)
    write_semantically_similar(run_query, semantic_edges)

    print("\n── Writing LLM-labelled table-pair edges …")
    write_table_pair_rels(run_query, table_pair_rels)

    if shares_entity_pairs:
        print("\n── Writing SHARES_ENTITY_WITH edges …")
        write_shares_entity_with(run_query, shares_entity_pairs)

    print("\n── Exporting summary …")
    summary_path = export_summary(
        run_query, raw_doc, blocks, refers_to_edges,
        table_pair_rels, output_dir, backend, run_variant,
        sections=sections,
        entities=entities,
        mention_edges=mention_edges,
        table_sem_edges=table_sem_edges,
        global_sem_edges=global_sem_edges,
        global_sim_skip=global_sim_skip,
        shares_entity_pairs=shares_entity_pairs,
        spacy_enabled=spacy_enabled,
    )
    print(f"kg_summary.json written to {summary_path}")

    return summary_path
