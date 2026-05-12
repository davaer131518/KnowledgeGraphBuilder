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
