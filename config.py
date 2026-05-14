"""
Central configuration for the Knowledge Graph Builder.

Non-sensitive tuning knobs live here directly.
Neo4j credentials are loaded from a .env file via python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")

# ── Repo / data path helpers ──────────────────────────────────────────────────

def _repo_root() -> Path:
    """Walk up from cwd until a directory containing pyproject.toml is found."""
    here = Path(__file__).parent.resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return here


REPO_ROOT = _repo_root()

# ── Parser output paths ───────────────────────────────────────────────────────
# smoke_tests/ lives inside the Agentic PDF Parser project, which is a sibling
# directory to this one — not a parent, so _repo_root() won't find it.

SMOKE_TESTS_DIR = Path(r"C:\Users\David Martirosyan\Documents\Projects\Agentic PDF Parser\smoke_tests")

# ── llama.cpp paths ───────────────────────────────────────────────────────────

LLAMA_MODELS_DIR  = Path(r"C:\llama-cpp\models")
LLAMA_CPP_DIR     = LLAMA_MODELS_DIR.parent       # contains llama-server.exe

LLM_MODEL_PATH    = LLAMA_MODELS_DIR / "Qwen3.5-4B-Q8_0.gguf"
EMBED_MODEL_PATH  = LLAMA_MODELS_DIR / "bge-m3-Q8_0.gguf"

EMBED_SERVER_PORT = 8091
LLM_SERVER_PORT   = 8092

# ── Neo4j credentials (from .env) ────────────────────────────────────────────

def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing. "
            f"Set it in the .env file at {Path(__file__).parent / '.env'}"
        )
    return val


NEO4J_URI      = _require_env("NEO4J_URI")
NEO4J_USER     = _require_env("NEO4J_USER")
NEO4J_PASSWORD = _require_env("NEO4J_PASSWORD")

# ── Tuning knobs ──────────────────────────────────────────────────────────────

SEM_SIM_TOP_K           = 5      # top-K similar blocks per table
SEM_SIM_MIN_SCORE       = 0.50   # soft floor — pairs below this are discarded even if in top-K
TABLE_PAIR_PAGE_WINDOW  = 10     # table-pair LLM: always check tables within this many pages
TABLE_PAIR_SEM_FLOOR    = 0.65   # table-pair LLM: check distant pairs only if cosine >= this
CONTEXT_WINDOW          = 3      # reading-order blocks before/after each table for CONTEXT edges
LLM_N_CTX               = 4096   # context window passed to llama-server with -c
EMBED_N_CTX             = 8192   # bge-m3 max context (dense HTML/math can be large)
EMBED_MAX_CHARS         = 6000   # first-attempt truncation; embed_text halves on 500 if needed
LLM_MAX_TOKENS          = 1024
LLM_TEMPERATURE         = 0.0    # deterministic for relation extraction

# ── Batch sizes ───────────────────────────────────────────────────────────────

BLOCK_WRITE_BATCH  = 50   # Neo4j UNWIND batch size for block nodes
EDGE_WRITE_BATCH   = 200  # Neo4j UNWIND batch size for relationship writes

# ── Parallelisation ───────────────────────────────────────────────────────────

LLM_PARALLEL_SLOTS = 3   # KV-cache slots when --parallel is used; each adds ~400 MB VRAM at n_ctx=4096

# ── Sections ──────────────────────────────────────────────────────────────────

ENABLE_SECTIONS                            = True
SECTION_FALLBACK_TITLE                     = "<document root>"
SECTION_PATH_MAX_DEPTH                     = 6      # truncate path to avoid huge strings
SECTION_EMIT_SYNTHETIC_ROOT_IF_NO_HEADINGS = True   # synthesise a root section for heading-less docs
ENABLE_IN_HEADING_SCOPE                    = False  # optional legacy (:Block)-[:IN_HEADING_SCOPE]->(:Block)

# ── Entities ──────────────────────────────────────────────────────────────────

ENABLE_ENTITIES                       = True
ENTITY_MIN_TERM_LEN                   = 3
ENTITY_MIN_CONFIDENCE                 = 0.5
ENTITY_MAX_ENTITIES_PER_BLOCK         = 50
ENTITY_MAX_SPANS_PER_MENTION          = 5
ENTITY_EVIDENCE_MAX_CHARS             = 120
ENTITY_MAX_DOCUMENT_FREQUENCY_RATIO   = 0.25   # TERMs above this ratio get demoted to confidence 0.2
ENTITY_USE_SPACY                      = True   # opt-in; falls back gracefully if spaCy is not installed
ENTITY_SPACY_MODEL                    = "en_core_web_sm"
ENTITY_SPACY_MIN_CONFIDENCE           = 0.6
ENTITY_SPACY_TEXT_MAX_CHARS           = 10000  # truncate per-block text before NER

# Common English stopwords + generic doc-vocabulary words filtered from rule-based TERM/ORG extraction.
ENTITY_STOPWORD_BLOCKLIST: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "while", "with", "without",
    "of", "in", "on", "at", "to", "from", "for", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "he", "she", "his", "her", "i", "me", "my", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "may", "might",
    "must", "shall", "can", "not", "no", "yes", "any", "all", "some", "such",
    "than", "then", "so", "also", "more", "most", "less", "least", "very",
    "much", "many", "few", "each", "every", "either", "neither", "both",
    "other", "another", "same", "different", "new", "old",
})

# Generic document-vocabulary terms that should never become TERM entities.
ENTITY_GENERIC_TERMS: frozenset[str] = frozenset({
    "company", "table", "figure", "section", "chapter", "paragraph",
    "document", "report", "page", "appendix", "introduction", "overview",
    "summary", "conclusion", "abstract", "background", "discussion",
    "results", "method", "methods", "methodology", "analysis", "data",
    "information", "services", "service", "business", "operations",
    "management", "directors", "officers", "employees", "shareholders",
    "fiscal", "year", "years", "quarter", "period", "date", "note", "notes",
    "item", "items", "exhibit", "schedule", "form",
})

# ── Global block-to-block semantic similarity ─────────────────────────────────

ENABLE_GLOBAL_BLOCK_SIM         = True
BLOCK_SIM_TOP_K                 = 5
BLOCK_SIM_MIN_SCORE             = 0.70
BLOCK_SIM_ALLOWED_TYPES         = (
    "paragraph", "list_item", "caption", "figure", "formula", "table", "heading",
)
BLOCK_SIM_SKIP_SHORT_TEXT_CHARS = 40
BLOCK_SIM_MAX_BLOCKS            = 8000   # warn-and-skip threshold; no chunked fallback
BLOCK_SIM_SKIP_SAME_PAGE        = False  # set True to discard trivially-near pairs

# ── Entity-mediated edges (optional, off by default) ──────────────────────────

CREATE_SHARES_ENTITY_WITH            = False
SHARED_ENTITY_MAX_BLOCKS_PER_ENTITY  = 8
SHARED_ENTITY_MIN_ENTITY_CONFIDENCE  = 0.75
SHARED_ENTITY_ALLOWED_TYPES          = ("ORG", "PERSON", "LAW_OR_REGULATION", "TERM", "ACRONYM")
SHARED_ENTITY_MAX_PAIRS              = 5000   # global cap

# ── Provenance defaults ───────────────────────────────────────────────────────

DEFAULT_REFERS_TO_CONFIDENCE_REGEX = 0.9
DEFAULT_REFERS_TO_CONFIDENCE_LLM   = 0.85
DEFAULT_REFERS_TO_CONFIDENCE_BOTH  = 0.95
DEFAULT_TABLE_PAIR_CONFIDENCE      = 0.8

LLM_MODEL_NAME   = "qwen3.5-4b"
EMBED_MODEL_NAME = "bge-m3"
