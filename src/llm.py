"""Minimal Ollama chat client. No mock fallback: if Ollama is unreachable we raise."""
from __future__ import annotations

import time

import requests

from .config import load_config


class OllamaUnavailable(RuntimeError):
    pass


def _base() -> str:
    return load_config()["llm"]["base_url"].rstrip("/")


def check_ollama(model: str | None = None) -> None:
    """Raise OllamaUnavailable with a clear message if the server or model is missing."""
    try:
        r = requests.get(f"{_base()}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise OllamaUnavailable(
            f"Ollama is not reachable at {_base()} ({type(e).__name__}). "
            "Start it (e.g. open the Ollama app or run `ollama serve`) and try again.") from e
    if model:
        names = {m["name"] for m in r.json().get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            raise OllamaUnavailable(f"Model '{model}' is not pulled. Run: ollama pull {model}")


def num_ctx_for(prompt_tokens_est: int, num_predict: int) -> int:
    """Context window sized to the prompt, bucketed so Ollama rarely reloads the model."""
    need = int(prompt_tokens_est * 1.2) + num_predict + 256
    for b in (4096, 8192, 16384, 32768):
        if need <= b:
            return b
    return 32768


def chat(messages: list[dict], model: str, num_ctx: int, num_predict: int | None = None,
         json_format: bool = False) -> dict:
    cfg = load_config()["llm"]
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": cfg["temperature"],
            "seed": cfg["seed"],
            "num_ctx": num_ctx,
            "num_predict": num_predict or cfg["num_predict"],
        },
    }
    if json_format:
        body["format"] = "json"
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{_base()}/api/chat", json=body, timeout=cfg["request_timeout_s"])
    except requests.ConnectionError as e:
        raise OllamaUnavailable(f"Ollama is not reachable at {_base()}. Start Ollama and retry.") from e
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:300]}")
    d = r.json()
    return {
        "content": d["message"]["content"],
        "prompt_tokens": d.get("prompt_eval_count", 0),
        "completion_tokens": d.get("eval_count", 0),
        "prompt_eval_s": d.get("prompt_eval_duration", 0) / 1e9,
        "eval_s": d.get("eval_duration", 0) / 1e9,
        "load_s": d.get("load_duration", 0) / 1e9,
        "wall_s": time.perf_counter() - t0,
        "num_ctx": num_ctx,
        "model": model,
    }
