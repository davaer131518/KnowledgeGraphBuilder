# Knowledge Graph Builder

Converts a parsed PDF (`document.json` from the Agentic PDF Parser) into a **Neo4j knowledge graph** that encodes how every table in the document connects to surrounding content — other tables, paragraphs, headings, captions, figures, and formulas — using a combination of structural rules, regex, embeddings, and a local LLM.

---

## How it works

The pipeline runs five complementary signal layers to build the graph:

1. **Structural** — reading order, layout proximity, heading scope (no models required)
2. **Regex** — explicit cross-references (`"see Table 2"`, `"Figure 1"`, etc.) scanned document-wide
3. **Embeddings** — top-K cosine similarity between tables and eligible block types (bge-m3 via llama-server), computed globally across the whole document
4. **LLM pass 1** — for each table, asks the LLM which nearby `{paragraph, caption, list_item}` blocks (within ±1 page) discuss it (Qwen3.5-4B)
5. **LLM pass 2** — for each candidate table pair (pre-filtered by page distance ≤ 10 AND cosine similarity ≥ 0.65), labels the relationship as `SUPPLEMENTS`, `CONTRASTS`, `COMPARES`, `ABLATES`, or `UNRELATED`

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
| `Page` | One node per page |
| `Block` | One node per content block (paragraph, table, heading, figure, caption, formula, …) with a 1024-dim embedding vector |

If the APOC plugin is installed, each `Block` also receives a **secondary label** matching its type (`:Table`, `:Paragraph`, `:Heading`, `:Figure`, `:Caption`, `:Formula`, etc.), enabling type-specific Cypher patterns like `MATCH (t:Table)`.

### Relationships

| Relationship | Method | Description |
|---|---|---|
| `PART_OF` | Structural | Page → Document |
| `ON_PAGE` | Structural | Block → Page |
| `PRECEDES` | Structural | Reading-order chain between all consecutive blocks |
| `DESCRIBES` | Structural | Caption → nearest table/figure on the same page (spatial proximity) |
| `INTRODUCES` | Structural | Heading → immediately following block + any table/figure in its scope until the next heading of the same or higher level |
| `IN_SECTION` | Structural | Every non-heading block → its deepest active parent heading |
| `CONTEXT_BEFORE` | Structural | N blocks immediately before each table → table |
| `CONTEXT_AFTER` | Structural | Table → N blocks immediately after it |
| `REFERS_TO` | Regex + LLM | A block discusses or references a table. `methods` property records detection source (`["regex"]`, `["llm"]`, or `["llm","regex"]`). Regex pass covers all block types document-wide; LLM pass is limited to `{paragraph, caption, list_item}` sources within ±1 page of the target table. |
| `SEMANTICALLY_SIMILAR` | Embedding | Top-K cosine neighbours **per table** (source block is always a non-table type: paragraph, figure, caption, or formula, or another table); `score` property. Computed globally across the entire document. |
| `SUPPLEMENTS` / `CONTRASTS` / `COMPARES` / `ABLATES` | LLM | Typed relationship between related table pairs. Requires APOC; falls back to a generic `TABLE_RELATES_TO {label, reason}` edge if APOC is unavailable. `UNRELATED` pairs are not written to the graph. |

**`kg_summary.json`** — written alongside `document.json` with block counts, edge counts, `REFERS_TO` breakdown by detection method, and the full list of table-pair relationships.

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
    ├── relationships.py  ← all edge extraction functions
    ├── embeddings.py     ← embedding cache + semantic edge computation
    └── neo4j_writer.py   ← all Cypher writes + kg_summary.json export
```

---

## Configuration reference

| Setting | Default | Description |
|---|---|---|
| `SEM_SIM_TOP_K` | `5` | Max similar blocks per table for `SEMANTICALLY_SIMILAR` |
| `SEM_SIM_MIN_SCORE` | `0.50` | Minimum cosine score to include an edge |
| `TABLE_PAIR_PAGE_WINDOW` | `10` | Table pairs further apart than this (pages) are skipped — applied together with `TABLE_PAIR_SEM_FLOOR` (both must pass) |
| `TABLE_PAIR_SEM_FLOOR` | `0.65` | Table pairs with cosine similarity below this are skipped — applied together with `TABLE_PAIR_PAGE_WINDOW` (both must pass) |
| `CONTEXT_WINDOW` | `3` | Number of blocks before/after each table for `CONTEXT_*` edges |
| `LLM_N_CTX` | `4096` | LLM context window |
| `EMBED_N_CTX` | `8192` | Embed model context window |
| `EMBED_MAX_CHARS` | `6000` | Max chars sent to embed server per block (halved on 500 errors) |
| `LLM_MAX_TOKENS` | `1024` | Max tokens per LLM completion |
| `LLM_TEMPERATURE` | `0.0` | Deterministic LLM outputs |
| `LLM_PARALLEL_SLOTS` | `3` | KV-cache slots when `--parallel` is used (~400 MB VRAM each) |

---

## Embedding cache

Embeddings are cached to `embeddings_<sha256[:12]>.pkl` in the same directory as `document.json`. On subsequent runs the cache is loaded automatically, skipping re-embedding. Delete the `.pkl` file to force a full re-embed.
