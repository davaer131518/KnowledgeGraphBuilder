"""
Knowledge Graph Builder — main entry point.

Converts a document.json from the Agentic PDF Parser into a Neo4j knowledge
graph that encodes how every table in the document connects to surrounding
content.

Usage:
    python build_kg.py apple_10k
    python build_kg.py apple_10k --variant cpu
    python build_kg.py apple_10k --parallel

Reads from:
    <REPO_ROOT>/smoke_tests/<DOCUMENT>/paddle_vl_<VARIANT>/document.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neo4j import GraphDatabase

import config
from kg import (
    loader,
    servers,
    relationships,
    embeddings as emb_module,
    neo4j_writer,
    sections as sections_module,
    entities as entities_module,
    similarity as similarity_module,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a Neo4j knowledge graph from a parsed PDF document.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "document",
        help="Subdirectory name inside smoke_tests/ (e.g. apple_10k).",
    )
    p.add_argument(
        "--variant", "-v",
        default="gpu",
        help="Run variant (gpu or cpu).",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help=(
            "Enable parallel LLM inference via continuous batching "
            f"(--parallel {config.LLM_PARALLEL_SLOTS} slots). "
            "Requires ~5.5 GB VRAM. Reduces LLM pass time by ~40-50%%."
        ),
    )
    return p.parse_args()


# ── Terminal summary ──────────────────────────────────────────────────────────

def _separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_terminal_summary(
    driver,
    blocks: list[dict],
    refers_to_edges: list[dict],
    table_pair_rels: list[dict],
    summary_path: Path,
    sections: list[dict] | None = None,
    entities: list[dict] | None = None,
    global_sim_skip: dict | None = None,
    spacy_enabled: bool = False,
) -> None:
    """Print a rich summary to the terminal after the graph is built."""
    run_query = neo4j_writer.make_runner(driver)

    _separator("═")
    print("  KNOWLEDGE GRAPH BUILD COMPLETE")
    _separator("═")

    # Block counts by type
    print("\nBlock node counts:")
    type_rows = run_query("""
        MATCH (b:Block)
        RETURN b.type AS type, count(b) AS count
        ORDER BY count DESC
    """)
    for row in type_rows:
        print(f"  {row['type']:<15} : {row['count']}")

    # Relationship type distribution
    print("\nRelationship counts:")
    rel_rows = run_query("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)
    for row in rel_rows:
        print(f"  {row['rel_type']:<30} {row['count']}")

    # Sections summary
    if sections:
        print(f"\nSection nodes: {len(sections)} (max depth {max(s['level'] for s in sections)})")

    # Entities summary
    if entities:
        print(f"\nEntity nodes: {len(entities)}  (spaCy active: {spacy_enabled})")
        by_type: dict[str, int] = {}
        for e in entities:
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]):
            print(f"  {t:<20} : {c}")

    # SEMANTICALLY_SIMILAR by scope
    scope_rows = run_query("""
        MATCH ()-[r:SEMANTICALLY_SIMILAR]->()
        RETURN coalesce(r.scope, 'unknown') AS scope, count(r) AS count
        ORDER BY scope
    """)
    if scope_rows:
        print("\nSEMANTICALLY_SIMILAR by scope:")
        for row in scope_rows:
            print(f"  {row['scope']:<10} : {row['count']}")
    if global_sim_skip and global_sim_skip.get("skipped"):
        print(f"  [global similarity skipped: {global_sim_skip.get('reason')}]")

    # REFERS_TO method breakdown
    method_counts: dict[str, int] = {}
    for e in refers_to_edges:
        k = "+".join(sorted(e["methods"]))
        method_counts[k] = method_counts.get(k, 0) + 1
    if method_counts:
        print("\nREFERS_TO breakdown by detection method:")
        for k, v in sorted(method_counts.items()):
            print(f"  method={k}: {v}")

    # Tables ranked by incoming connections
    print("\nTables ranked by incoming connections:")
    print(f"  {'Table ID':<25} {'Page':>4} {'Regex':>6} {'LLM':>5} {'SemSim':>7} {'Cap':>4}  Section")
    print("  " + "-" * 68)
    table_rows = run_query("""
        MATCH (b:Block {type: 'table'})
        OPTIONAL MATCH (x)-[rx:REFERS_TO]->(b) WHERE 'regex' IN rx.methods
        OPTIONAL MATCH (y)-[ry:REFERS_TO]->(b) WHERE 'llm'   IN ry.methods
        OPTIONAL MATCH (z)-[:SEMANTICALLY_SIMILAR]-(b)
        OPTIONAL MATCH (c)-[:DESCRIBES]->(b)
        OPTIONAL MATCH (b)-[:IN_SECTION]->(h)
        RETURN b.block_id AS table_id,
               b.page_number AS page,
               count(DISTINCT x) AS regex_refs,
               count(DISTINCT y) AS llm_refs,
               count(DISTINCT z) AS sem_sim,
               count(DISTINCT c) AS captions,
               coalesce(h.title, h.text) AS section
        ORDER BY regex_refs + llm_refs DESC
    """)
    for row in table_rows:
        section_snippet = (row.get("section") or "—")[:28]
        print(
            f"  {row['table_id']:<25}"
            f"{row['page']:>4}"
            f"{row['regex_refs']:>6}"
            f"{row['llm_refs']:>5}"
            f"{row['sem_sim']:>7}"
            f"{row['captions']:>4}"
            f"  {section_snippet}"
        )

    # Non-UNRELATED table-pair relationships
    if table_pair_rels:
        print(f"\nNon-UNRELATED table-pair relationships ({len(table_pair_rels)}):")
        for r in table_pair_rels:
            print(f"  {r['src']} --[{r['relationship']}]--> {r['tgt']}")
            reason = r.get("reason", "")
            if reason:
                print(f"    {reason[:100]}")

    _separator()
    print(f"Summary JSON : {summary_path}")
    _separator("═")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    backend     = "paddle_vl"
    run_variant = args.variant

    doc_json = (
        config.SMOKE_TESTS_DIR
        / args.document
        / f"{backend}_{run_variant}"
        / "document.json"
    )

    if not doc_json.exists():
        print(f"ERROR: document.json not found at:\n  {doc_json}", file=sys.stderr)
        sys.exit(1)

    print(f"Document : {doc_json}")
    print(f"Backend  : {backend}  Variant: {run_variant}\n")

    embed_proc = None
    llm_proc   = None
    driver     = None

    try:
        # ── 1. Load document ──────────────────────────────────────────────────
        print("── Loading document …")
        raw_doc, blocks, block_by_id = loader.load_document(doc_json)
        loader.print_block_summary(blocks)
        print(f"\nPages   : {raw_doc['document']['num_pages']}")
        print(f"Backend : {raw_doc['backend']['name']} {raw_doc['backend']['version']}")
        print(f"Source  : {raw_doc['document']['source_filename']}\n")

        doc_sha = raw_doc["document"]["source_sha256"]

        # ── 2. Sections (CPU-only) ────────────────────────────────────────────
        sections = parent_child_pairs = top_level_sections = None
        in_section_edges: list[tuple[str, str]] = []
        if config.ENABLE_SECTIONS:
            print("── Building sections …")
            sections, in_section_edges, parent_child_pairs, top_level_sections = (
                sections_module.build_sections(blocks, doc_sha)
            )
            print(f"Sections        : {len(sections)} "
                  f"(top-level={len(top_level_sections)}, subsection-edges={len(parent_child_pairs)})")
            print(f"IN_SECTION      : {len(in_section_edges)}")

        # ── 3. Structural & regex edges (CPU-only) ────────────────────────────
        print("\n── Computing structural relationships …")
        ref_index = relationships.build_ref_index(blocks)
        print(f"Reference index entries: {len(ref_index)}")

        precedes_edges  = relationships.compute_precedes_edges(blocks)
        describes_edges = relationships.compute_describes_edges(blocks)
        introduces_edges = relationships.compute_introduces_edges(blocks)
        context_before_edges, context_after_edges = relationships.compute_context_edges(blocks)

        in_heading_scope_edges: list[tuple[str, str]] = []
        if config.ENABLE_IN_HEADING_SCOPE:
            in_heading_scope_edges = relationships.compute_in_heading_scope_edges(blocks)
            print(f"IN_HEADING_SCOPE: {len(in_heading_scope_edges)}")

        print(f"PRECEDES        : {len(precedes_edges)}")
        print(f"DESCRIBES       : {len(describes_edges)}")
        print(f"INTRODUCES      : {len(introduces_edges)}")
        print(f"CONTEXT_BEFORE  : {len(context_before_edges)}")
        print(f"CONTEXT_AFTER   : {len(context_after_edges)}")

        refers_to_edges, regex_pairs = relationships.compute_refers_to_regex(blocks, ref_index)
        print(f"REFERS_TO (regex): {len(refers_to_edges)}")

        # ── 4. Entity extraction (CPU-only) ───────────────────────────────────
        entities: list[dict] = []
        mention_edges: list[dict] = []
        if config.ENABLE_ENTITIES:
            print("\n── Extracting entities …")
            entities, mention_edges = entities_module.extract_entities(blocks, doc_sha)
            print(f"Entities        : {len(entities)}")
            print(f"MENTIONS        : {len(mention_edges)}")
        spacy_enabled = entities_module.is_spacy_active() if config.ENABLE_ENTITIES else False

        # ── 5–7. Embeddings (embed server up briefly) ─────────────────────────
        print("\n── Generating embeddings …")
        embed_proc = servers.start_embed_server()
        embeddings = emb_module.compute_or_load_embeddings(blocks, doc_sha, doc_json.parent)
        print("Stopping embed server to free VRAM …")
        servers.stop_server(embed_proc)
        embed_proc = None
        print("Embed server stopped.")

        # ── 8. Table-anchored semantic similarity (CPU-only) ──────────────────
        print("\n── Computing table-anchored semantic similarity …")
        table_sem_raw = emb_module.compute_semantic_edges(blocks, embeddings)
        table_sem_edges = emb_module.table_edges_to_dicts(table_sem_raw)
        print(f"SEMANTICALLY_SIMILAR (table scope): {len(table_sem_edges)}")

        # ── 9. Global block-to-block similarity (CPU-only) ────────────────────
        global_sem_edges: list[dict] = []
        global_sim_skip = {"skipped": True, "reason": "ENABLE_GLOBAL_BLOCK_SIM=False", "filtered_count": 0}
        if config.ENABLE_GLOBAL_BLOCK_SIM:
            print("\n── Computing global block-to-block similarity …")
            mentions_by_block: dict[str, int] = {}
            for m in mention_edges:
                mentions_by_block[m["src"]] = mentions_by_block.get(m["src"], 0) + 1
            global_sem_edges, global_sim_skip = similarity_module.compute_global_block_similarity(
                blocks, embeddings, table_sem_raw, mentions_by_block,
            )
            print(f"SEMANTICALLY_SIMILAR (global scope): {len(global_sem_edges)}")
            if global_sim_skip.get("skipped"):
                print(f"  [skipped] {global_sim_skip.get('reason')}")

        all_semantic_edges = table_sem_edges + global_sem_edges

        # ── 10–13. LLM passes (LLM server up briefly) ─────────────────────────
        print("\n── Starting LLM server …")
        llm_proc = servers.start_llm_server(parallel=args.parallel)

        print("\n── LLM REFERS_TO pass …")
        refers_to_edges = relationships.compute_refers_to_llm(
            blocks, refers_to_edges, regex_pairs, parallel=args.parallel
        )
        method_counts: dict[str, int] = {}
        for e in refers_to_edges:
            k = "+".join(sorted(e["methods"]))
            method_counts[k] = method_counts.get(k, 0) + 1
        print(f"REFERS_TO total: {len(refers_to_edges)}")
        for k, v in sorted(method_counts.items()):
            print(f"  method={k}: {v}")

        print("\n── LLM table-pair labelling …")
        table_pair_rels = relationships.compute_table_pair_rels(blocks, embeddings, parallel=args.parallel)
        print(f"Non-UNRELATED table-pair relationships: {len(table_pair_rels)}")

        print("\nStopping LLM server …")
        servers.stop_server(llm_proc)
        llm_proc = None
        print("LLM server stopped.")

        # ── 14. (Optional) SHARES_ENTITY_WITH (CPU-only) ──────────────────────
        shares_entity_pairs: list[dict] = []
        if config.CREATE_SHARES_ENTITY_WITH and entities:
            print("\n── Computing SHARES_ENTITY_WITH edges …")
            shares_entity_pairs = entities_module.compute_shares_entity_with(entities, mention_edges)
            print(f"SHARES_ENTITY_WITH: {len(shares_entity_pairs)}")

        # ── 15–16. Neo4j graph build + summary ────────────────────────────────
        print("\n── Connecting to Neo4j …")
        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
        print("Neo4j connected.")

        summary_path = neo4j_writer.build_graph(
            driver      = driver,
            raw_doc     = raw_doc,
            blocks      = blocks,
            embeddings  = embeddings,
            sorted_blocks         = blocks,   # already sorted by loader
            precedes_edges        = precedes_edges,
            describes_edges       = describes_edges,
            introduces_edges      = introduces_edges,
            in_section_edges      = in_section_edges,
            context_before_edges  = context_before_edges,
            context_after_edges   = context_after_edges,
            refers_to_edges       = refers_to_edges,
            semantic_edges        = all_semantic_edges,
            table_pair_rels       = table_pair_rels,
            output_dir            = doc_json.parent,
            backend               = backend,
            run_variant           = run_variant,
            sections              = sections,
            parent_child_pairs    = parent_child_pairs,
            top_level_sections    = top_level_sections,
            in_heading_scope_edges = in_heading_scope_edges,
            entities              = entities,
            mention_edges         = mention_edges,
            table_sem_edges       = table_sem_edges,
            global_sem_edges      = global_sem_edges,
            global_sim_skip       = global_sim_skip,
            shares_entity_pairs   = shares_entity_pairs,
            spacy_enabled         = spacy_enabled,
        )

        # ── Terminal summary ──────────────────────────────────────────────────
        print_terminal_summary(
            driver, blocks, refers_to_edges, table_pair_rels, summary_path,
            sections=sections,
            entities=entities,
            global_sim_skip=global_sim_skip,
            spacy_enabled=spacy_enabled,
        )

    finally:
        # Always attempt clean shutdown of servers and Neo4j driver
        if embed_proc is not None:
            servers.stop_server(embed_proc)
            print("Embed server stopped.")
        if llm_proc is not None:
            servers.stop_server(llm_proc)
            print("LLM server stopped.")
        if driver is not None:
            driver.close()
            print("Neo4j driver closed.")


if __name__ == "__main__":
    main()
