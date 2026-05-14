# Knowledge Graph Builder

Converts a parsed PDF (`document.json` from the Agentic PDF Parser) into a **Neo4j knowledge graph** with first-class section hierarchy, entity mentions, and block-to-block semantic similarity — using a combination of structural rules, regex, embeddings, lightweight NER, and a local LLM.

---

## How it works

The pipeline runs complementary signal layers to build the graph:

1. **Sections (deterministic)** — derives a document hierarchy from heading levels; produces `Section` nodes with `path`, `level`, `page_start`/`page_end`, and Block→Section `IN_SECTION` edges. No LLM, no models.
2. **Structural** — reading order, layout proximity, heading scope (no models required)
3. **Regex** — explicit cross-references (`"see Table 2"`, `"Figure 1"`, etc.) scanned document-wide
4. **Entities (rule-based + optional spaCy)** — extracts `Entity` nodes (ORG, PERSON, DATE, MONEY, PERCENT, NUMBER, ACRONYM, TABLE_REF, FIGURE_REF, SECTION_REF, LAW_OR_REGULATION, TERM, LOCATION) with regex + light heuristics. If `spacy` and `en_core_web_sm` are installed, spaCy NER augments PERSON/ORG/LOCATION recall; otherwise pure rules. Emits `MENTIONS` edges with `methods`, `confidence`, `spans_flat`, `evidence`.
5. **Embeddings** — bge-m3 via llama-server, cached on disk; computed once and reused across all similarity stages.
6. **Table-anchored semantic similarity** — top-K cosine neighbours per table (`scope: "table"`).
7. **Global block-to-block semantic similarity** — top-K cosine edges across paragraph/caption/figure/formula/table/heading blocks (`scope: "global"`). Canonicalised so `src < tgt` and deduped against table-scope edges via a `frozenset` of pair IDs. Skipped (with a recorded reason) if eligible-block count exceeds `BLOCK_SIM_MAX_BLOCKS`.
8. **LLM pass 1** — for each table, asks the LLM which nearby `{paragraph, caption, list_item}` blocks (within ±1 page) discuss it (Qwen3.5-4B).
9. **LLM pass 2** — for each candidate table pair (pre-filtered by page distance ≤ 10 AND cosine similarity ≥ 0.65), labels the relationship as `SUPPLEMENTS`, `CONTRASTS`, `COMPARES`, `ABLATES`, or `UNRELATED`.
10. **(Optional) `SHARES_ENTITY_WITH`** — Block↔Block edges through shared high-value entities. Off by default; tightly capped on per-entity and global pair limits.

**Pipeline order is server-aware:** all CPU-only work (sections, structural edges, regex, entity extraction) runs first. The embed server is started only for embedding and shut down before the LLM server starts. The LLM server is killed before Neo4j writes, so both GPU servers are down for everything that follows.

### Universal provenance

Every non-structural edge carries a `methods: list[str]` array, plus where applicable `confidence`, `scope`, `evidence`, `model`, and `created_by_stage`. This makes evidence rankable in downstream GraphRAG queries.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| Neo4j 5.x | Running locally on `localhost:7687`; APOC plugin recommended |
| `llama-server.exe` | CUDA-compiled binary at `C:\llama-cpp\llama-server.exe` |
| `Qwen3.5-4B-Q8_0.gguf` | At `C:\llama-cpp\models\` |
| `bge-m3-Q8_0.gguf` | At `C:\llama-cpp\models\` |
| ~6 GB VRAM | bge-m3 (~1.3 GB) + Qwen3.5-4B (~4.5 GB) run simultaneously during embedding; embed server is killed before the LLM passes to free VRAM |

---

## Setup

**1. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**1b. (Optional) Install spaCy for richer NER**

spaCy is **not required**. When present, it adds higher-recall PERSON/ORG/LOCATION/DATE/MONEY/PERCENT/LAW detection that gets merged with the rule-based extractors. When absent, the pipeline runs with rule-based entities only — no errors, no missing nodes.

```bash
pip install "spacy>=3.7,<4.0"
python -m spacy download en_core_web_sm
```

At startup, `kg/entities.py` prints whether spaCy is active. The summary JSON also records `spacy_enabled: true|false`.

**2. Configure credentials**

Copy `.env.example` to `.env` and fill in your Neo4j password:

```bash
copy .env.example .env
```

```dotenv
NEO4J_URI=neo4j://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

**3. (Optional) Adjust tuning knobs**

All thresholds and model paths live in `config.py`. The defaults match the values used during development on Apple's 10-K filing.

---

## Usage

```bash
python build_kg.py <document> [--variant gpu|cpu] [--parallel]
```

`<document>` is the subfolder name inside the Agentic PDF Parser's `smoke_tests/` directory. The script always uses the `paddle_vl` backend and defaults to the `gpu` variant:

```
C:\Users\David Martirosyan\Documents\Projects\Agentic PDF Parser\smoke_tests\<document>\paddle_vl_<variant>\document.json
```

**Examples:**

```bash
python build_kg.py apple_10k                  # serial (default)
python build_kg.py apple_10k --variant cpu
python build_kg.py apple_10k --parallel       # parallel LLM inference
```

### `--parallel` flag

Enables continuous batching on the LLM server (`--parallel 3` slots). Both LLM passes (REFERS_TO and table-pair labelling) are submitted concurrently via a `ThreadPoolExecutor`, allowing the server to batch decode requests from multiple threads in a single GPU forward pass.

| Mode | VRAM | Expected LLM pass time |
|---|---|---|
| Serial (default) | ~4.5 GB | ~9 min (600 pairs) |
| `--parallel` | ~5.5 GB | ~4–5 min (~45–50% reduction) |

The extra ~1 GB covers 3 additional KV-cache slots at `n_ctx=4096`. On a 12 GB card this leaves comfortable headroom.

---

## Output

**Neo4j graph** — the database is cleared and rebuilt on every run. The graph contains:

### Nodes

| Label | Description |
|---|---|
| `Document` | One node per PDF |
| `Page` | One node per page (carries `doc_id`, `page_number`) |
| `Block` | One node per content block with a 1024-dim embedding vector |
| `Section` | One node per heading-derived section. Properties: `section_id`, `doc_id`, `title`, `level`, `level_inferred`, `level_gap`, `path`, `page_start`, `page_end`, `block_count`, `heading_block_id` |
| `Entity` | One node per `(doc_id, type, normalized_name)` triple. Properties: `entity_id`, `canonical_name`, `type`, `confidence`, `methods`, `aliases`, `doc_frequency_ratio`, `ambiguous` |

Entities are **document-scoped by default** — the MERGE key is `(doc_id, type, normalized_name)`, so the same string across two documents produces two separate Entity nodes. A corpus-level `SAME_AS` layer can be added later without changing the schema.

If the APOC plugin is installed, each `Block` also receives a **secondary label** matching its type (`:Table`, `:Paragraph`, `:Heading`, `:Figure`, `:Caption`, `:Formula`, etc.).

### Relationships

| Relationship | Method | Description |
|---|---|---|
| `PART_OF` | Structural | Page → Document |
| `ON_PAGE` | Structural | Block → Page |
| `PRECEDES` | Structural | Reading-order chain between all consecutive blocks |
| `DESCRIBES` | Structural | Caption → nearest table/figure on the same page (spatial proximity) |
| `INTRODUCES` | Structural | Heading → immediately following block + any table/figure in its scope until the next heading of the same or higher level |
| `IN_SECTION` | Structural | Every block → its containing `Section` node |
| `IN_HEADING_SCOPE` | Structural (optional) | Legacy Block→Heading-Block edge, off by default (`ENABLE_IN_HEADING_SCOPE`). `IN_SECTION` is the canonical schema. |
| `HAS_SECTION` | Structural | Document → top-level Section |
| `HAS_SUBSECTION` | Structural | Section → child Section |
| `STARTS_ON_PAGE` | Structural | Section → Page (where the section's heading begins) |
| `CONTEXT_BEFORE` | Structural | N blocks immediately before each table → table |
| `CONTEXT_AFTER` | Structural | Table → N blocks immediately after it |
| `REFERS_TO` | Regex + LLM | A block discusses or references a table. `methods` records detection source (`["regex"]`, `["llm"]`, or `["regex","llm"]`); additional `confidence`, `scope="reference"`, `evidence`, `created_by_stage`. |
| `SEMANTICALLY_SIMILAR` | Embedding | Cosine-similarity edges. `scope` is `"table"` (top-K per table) or `"global"` (top-K across content blocks, canonicalised src<tgt, deduped against table scope). Properties: `score`, `rank`, `methods=["embedding"]`, `model="bge-m3"`, `confidence`, `created_by_stage`. Query without an arrow: `MATCH (a)-[r:SEMANTICALLY_SIMILAR]-(b)`. |
| `MENTIONS` | Rule + spaCy | Block → Entity. Properties: `count`, `spans_flat` (flattened `[s1,e1,s2,e2,…]`), `evidence`, `methods`, `confidence`, `created_by_stage`. |
| `SHARES_ENTITY_WITH` | Entity-overlap (optional) | Block ↔ Block via a shared high-value entity. Off by default (`CREATE_SHARES_ENTITY_WITH=False`); tightly capped on per-entity, type, and global pair limits. Prefer the traversal `MENTIONS<-Entity->MENTIONS` for most queries. |
| `SUPPLEMENTS` / `CONTRASTS` / `COMPARES` / `ABLATES` | LLM | Typed relationship between related table pairs. Requires APOC; falls back to `TABLE_RELATES_TO {label, reason, methods, …}` if APOC is missing. `UNRELATED` pairs are not written. |

**`kg_summary.json`** — written alongside `document.json`. New fields: `section_count`, `section_max_depth`, `entity_count`, `entity_count_by_type`, `entity_high_freq_filtered_count`, `mention_edge_count`, `alias_count`, `block_semantic_similarity` (split by scope), `global_similarity_skipped` (+ optional `global_similarity_skip_reason`), `shares_entity_with_edge_count`, `spacy_enabled`, `feature_flags`.

**Terminal summary** — printed at the end of the run:
- Block counts by type
- Relationship type counts
- `REFERS_TO` breakdown (`regex` / `llm` / `llm+regex`)
- All tables ranked by number of incoming connections
- All non-UNRELATED table-pair relationships with one-sentence reasons

---

## Project structure

```
KnowledgeGraphBuilder/
├── build_kg.py           ← entry point
├── config.py             ← all tuning knobs and paths
├── .env                  ← Neo4j credentials (git-ignored)
├── .env.example          ← credentials template
├── requirements.txt
└── kg/
    ├── loader.py         ← parse document.json into a block list
    ├── servers.py        ← start/stop llama-server processes
    ├── models.py         ← llm_chat(), embed_text(), cosine_sim()
    ├── sections.py       ← deterministic Section hierarchy from heading levels
    ├── relationships.py  ← structural + LLM edge extraction (REFERS_TO, table-pair)
    ├── entities.py       ← rule-based + optional spaCy Entity / MENTIONS extraction
    ├── embeddings.py     ← embedding cache + table-anchored similarity
    ├── similarity.py     ← global block-to-block similarity (vectorised, dedup-aware)
    └── neo4j_writer.py   ← all Cypher writes + kg_summary.json export
```

---

## Configuration reference

### Table-anchored layer (legacy)
| Setting | Default | Description |
|---|---|---|
| `SEM_SIM_TOP_K` | `5` | Max similar blocks per table for table-scope `SEMANTICALLY_SIMILAR` |
| `SEM_SIM_MIN_SCORE` | `0.50` | Minimum cosine score for a table-scope edge |
| `TABLE_PAIR_PAGE_WINDOW` | `10` | Table pairs further apart than this (pages) are skipped (AND'd with `TABLE_PAIR_SEM_FLOOR`) |
| `TABLE_PAIR_SEM_FLOOR` | `0.65` | Table pairs with cosine below this are skipped |
| `CONTEXT_WINDOW` | `3` | N blocks before/after each table for `CONTEXT_*` edges |
| `LLM_N_CTX` / `EMBED_N_CTX` | `4096` / `8192` | Server context windows |
| `EMBED_MAX_CHARS` | `6000` | Max chars per block sent to embed server (halved on 500s) |
| `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | `1024` / `0.0` | LLM determinism |
| `LLM_PARALLEL_SLOTS` | `3` | KV-cache slots when `--parallel` is used (~400 MB VRAM each) |

### Sections
| Setting | Default | Description |
|---|---|---|
| `ENABLE_SECTIONS` | `True` | Build Section nodes + IN_SECTION edges |
| `SECTION_EMIT_SYNTHETIC_ROOT_IF_NO_HEADINGS` | `True` | Emit a single root Section when the doc has no headings |
| `SECTION_PATH_MAX_DEPTH` | `6` | Truncate `Section.path` ancestry chain |
| `ENABLE_IN_HEADING_SCOPE` | `False` | Also emit legacy `(:Block)-[:IN_HEADING_SCOPE]->(:Block)` |

### Entities
| Setting | Default | Description |
|---|---|---|
| `ENABLE_ENTITIES` | `True` | Run entity extraction + MENTIONS writes |
| `ENTITY_USE_SPACY` | `True` | Use spaCy NER if installed; rule-based fallback otherwise |
| `ENTITY_MIN_TERM_LEN` | `3` | Drop entities shorter than this |
| `ENTITY_MIN_CONFIDENCE` | `0.5` | Drop candidates below this (ACRONYMs exempt) |
| `ENTITY_MAX_ENTITIES_PER_BLOCK` | `50` | Per-block cap |
| `ENTITY_MAX_SPANS_PER_MENTION` | `5` | Capped span count per MENTIONS edge |
| `ENTITY_MAX_DOCUMENT_FREQUENCY_RATIO` | `0.25` | TERMs above this ratio are demoted to confidence 0.2 and tagged `filtered:high_doc_freq` |

### Global block similarity
| Setting | Default | Description |
|---|---|---|
| `ENABLE_GLOBAL_BLOCK_SIM` | `True` | Compute global-scope semantic edges |
| `BLOCK_SIM_TOP_K` | `5` | Top-K per block |
| `BLOCK_SIM_MIN_SCORE` | `0.70` | Minimum cosine score |
| `BLOCK_SIM_ALLOWED_TYPES` | paragraph/list_item/caption/figure/formula/table/heading | Eligible source types |
| `BLOCK_SIM_SKIP_SHORT_TEXT_CHARS` | `40` | Min text length unless the block has at least one entity mention |
| `BLOCK_SIM_MAX_BLOCKS` | `8000` | Warn-and-skip threshold; surfaces in `kg_summary.json` |

### Entity-mediated edges
| Setting | Default | Description |
|---|---|---|
| `CREATE_SHARES_ENTITY_WITH` | `False` | Off by default; prefer the entity-mediated traversal |
| `SHARED_ENTITY_MAX_BLOCKS_PER_ENTITY` | `8` | Per-entity block cap |
| `SHARED_ENTITY_MIN_ENTITY_CONFIDENCE` | `0.75` | Minimum entity confidence |
| `SHARED_ENTITY_ALLOWED_TYPES` | ORG, PERSON, LAW_OR_REGULATION, TERM, ACRONYM | Types eligible |
| `SHARED_ENTITY_MAX_PAIRS` | `5000` | Global pair cap |

---

## Embedding cache

Embeddings are cached to `embeddings_<sha256[:12]>.pkl` in the same directory as `document.json`. On subsequent runs the cache is loaded automatically, skipping re-embedding. Delete the `.pkl` file to force a full re-embed.

---

## What's intentionally lightweight

This is a general-purpose, document-agnostic graph. To keep it that way, the implementation deliberately avoids broad inference cost and domain assumptions. The following are **intentionally not implemented**:

- **Full neural coreference resolution.** Entity mentions are linked by string identity, not coreferent reasoning.
- **Claim / atomic-fact extraction.** No fact-level decomposition of paragraphs.
- **Broad LLM-based relation classification.** Only the existing table-pair LLM labelling and the table-anchored `REFERS_TO` LLM pass run. No paragraph-paragraph LLM passes, no all-pairs prompts.
- **All-pairs block similarity.** Global similarity is top-K per block, canonicalised, deduped against table-scope, and bailed out above `BLOCK_SIM_MAX_BLOCKS`.
- **All-pairs `SHARES_ENTITY_WITH`.** Off by default; use the `(Block)-[:MENTIONS]->(Entity)<-[:MENTIONS]-(Block)` traversal instead.
- **Cell-level table expansion.** Tables remain a single `Block` with `extract_table_text()` content.
- **Cross-document entity merging.** Entities are document-scoped (MERGEd on `(doc_id, type, normalized_name)`). A corpus `SAME_AS` layer can be added later without schema migration.
- **Domain-specific entity ontologies.** No financial/legal/scientific schemas; entity types are general.

If any of these become needed, they can be added behind config flags without disturbing the rest of the pipeline.
