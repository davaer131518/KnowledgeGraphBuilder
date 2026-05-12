"""
HTTP client helpers for the llama-server embed and chat endpoints.
"""

from __future__ import annotations

import re

import numpy as np
import requests

import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def llm_chat(system: str, user: str, max_tokens: int = config.LLM_MAX_TOKENS) -> str:
    """
    POST to the LLM server /v1/chat/completions endpoint.

    Strips Qwen3 <think>…</think> reasoning blocks so callers always
    receive only the final answer text.
    Thinking mode is disabled via chat_template_kwargs to save token budget.
    """
    resp = requests.post(
        f"http://127.0.0.1:{config.LLM_SERVER_PORT}/v1/chat/completions",
        json={
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens":  max_tokens,
            "temperature": config.LLM_TEMPERATURE,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _THINK_RE.sub("", raw).strip()


def embed_text(text: str) -> np.ndarray:
    """
    POST to the embed server /v1/embeddings endpoint; return L2-normalised vector.

    If the server returns 500 (context overflow on dense/HTML text), halves the
    input length and retries up to 4 times before raising.
    """
    max_chars = config.EMBED_MAX_CHARS
    last_err: Exception | None = None

    for attempt in range(5):
        truncated = text[:max_chars]
        resp = requests.post(
            f"http://127.0.0.1:{config.EMBED_SERVER_PORT}/v1/embeddings",
            json={"input": truncated, "encoding_format": "float"},
            timeout=60,
        )
        if resp.ok:
            vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        last_err = RuntimeError(
            f"Embed server {resp.status_code} (attempt {attempt + 1}, "
            f"len={len(truncated)}): {resp.text[:400]}"
        )
        if resp.status_code != 500 or max_chars <= 100:
            break
        max_chars //= 2

    raise last_err  # type: ignore[misc]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two L2-normalised vectors (simple dot product)."""
    return float(np.dot(a, b))
