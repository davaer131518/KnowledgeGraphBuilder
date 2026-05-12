"""
llama-server lifecycle helpers.

Each model (embed + LLM) runs as a separate llama-server.exe background process
on its own port. Using the CUDA-compiled binary from the PDF parser backend means
GPU offload works out of the box without rebuilding llama-cpp-python from source.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import config


def start_llama_server(
    model_path: Path,
    port: int,
    extra_args: tuple[str, ...] = (),
    health_timeout: int = 120,
) -> subprocess.Popen:
    """
    Start llama-server.exe as a background process and block until its
    /health endpoint returns HTTP 200.

    Args:
        model_path:     Path to the GGUF model file.
        port:           Port to bind the server on.
        extra_args:     Additional CLI flags (e.g. --embedding --pooling mean).
        health_timeout: Seconds to wait before raising TimeoutError.

    Returns:
        The running Popen handle (keep it to kill the server later).
    """
    server_bin = config.LLAMA_CPP_DIR / "llama-server.exe"
    cmd = [
        str(server_bin),
        "-m",     str(model_path),
        "--port", str(port),
        "--host", "127.0.0.1",
        "-ngl",   "-1",           # offload all layers to GPU
        *extra_args,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + health_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}) "
                f"for {model_path.name}"
            )
        try:
            with urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    return proc
        except (URLError, OSError):
            pass
        time.sleep(1.0)

    proc.kill()
    raise TimeoutError(
        f"llama-server did not become ready within {health_timeout}s "
        f"(port {port}, model {model_path.name})"
    )


def stop_server(proc: subprocess.Popen | None) -> None:
    """Kill a llama-server process if it is still running."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def start_embed_server() -> subprocess.Popen:
    print(f"Starting embedding server ({config.EMBED_MODEL_PATH.name}) on port {config.EMBED_SERVER_PORT} …")
    proc = start_llama_server(
        config.EMBED_MODEL_PATH,
        config.EMBED_SERVER_PORT,
        extra_args=("--embedding", "--pooling", "mean", "-c", str(config.EMBED_N_CTX)),
    )
    print(f"Embedding server ready on port {config.EMBED_SERVER_PORT}.")
    return proc


def start_llm_server(parallel: bool = False) -> subprocess.Popen:
    extra = ("-c", str(config.LLM_N_CTX))
    if parallel:
        extra += ("--parallel", str(config.LLM_PARALLEL_SLOTS))
    print(f"Starting LLM server ({config.LLM_MODEL_PATH.name}) on port {config.LLM_SERVER_PORT} …")
    proc = start_llama_server(
        config.LLM_MODEL_PATH,
        config.LLM_SERVER_PORT,
        extra_args=extra,
    )
    slots_msg = f" ({config.LLM_PARALLEL_SLOTS} parallel slots)" if parallel else ""
    print(f"LLM server ready on port {config.LLM_SERVER_PORT}{slots_msg}.")
    return proc
