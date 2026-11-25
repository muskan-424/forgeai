"""LLM access: Ollama backend, deterministic Mock backend, disk cache, usage accounting."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import requests

from . import config


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached_seconds: float = 0.0   # model time that cache hits replayed (what those calls originally cost)

    def add(self, other: "Usage") -> None:
        for f in ("calls", "cache_hits", "prompt_tokens", "completion_tokens", "seconds", "cached_seconds"):
            setattr(self, f, getattr(self, f) + getattr(other, f))


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """The model server cannot be reached or the model is missing. Retrying or 'repairing' is pointless,
    so callers must stop and surface this instead of treating it as a bad model answer."""


class LLM:
    """chat() -> text, json() -> parsed object. Results cached on disk by request hash."""

    def __init__(self, model: str | None = None, backend: str | None = None,
                 mock_fn: Optional[Callable[[list[dict], Optional[dict]], str]] = None,
                 use_cache: bool = True):
        self.model = model or config.AGENT_MODEL
        self.backend = backend or config.BACKEND
        self.mock_fn = mock_fn
        self.use_cache = use_cache and self.backend == "ollama"
        self.usage = Usage()
        if self.use_cache:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ----- core -----
    def chat(self, messages: list[dict], temperature: float = 0.0,
             schema: dict | None = None, max_tokens: int = 700) -> str:
        key = hashlib.sha256(json.dumps(
            [self.model, messages, temperature, schema, max_tokens], sort_keys=True).encode()).hexdigest()
        cpath = config.CACHE_DIR / f"{key}.json"
        if self.use_cache and cpath.exists():
            self.usage.cache_hits += 1
            entry = json.loads(cpath.read_text(encoding="utf-8"))
            self.usage.cached_seconds += entry.get("seconds", 0.0)
            return entry["text"]

        t0 = time.time()
        if self.backend == "mock":
            if not self.mock_fn:
                raise LLMError("mock backend needs mock_fn")
            text = self.mock_fn(messages, schema)
            pt = ct = 0
        else:
            text, pt, ct = self._ollama_chat(messages, temperature, schema, max_tokens)
        dt = time.time() - t0
        self.usage.calls += 1
        self.usage.prompt_tokens += pt
        self.usage.completion_tokens += ct
        self.usage.seconds += dt
        if self.use_cache:
            cpath.write_text(json.dumps({"text": text, "seconds": round(dt, 2)}), encoding="utf-8")
        return text

    def _ollama_chat(self, messages, temperature, schema, max_tokens):
        body = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 7},
        }
        if schema is not None:
            body["format"] = schema
        try:
            r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=config.LLM_TIMEOUT_S)
            if r.status_code == 404:
                raise LLMUnavailable(f"Ollama has no model '{self.model}' (HTTP 404). Check `ollama list`; if it is "
                                     f"empty, Ollama is probably not using your OLLAMA_MODELS folder - restart it "
                                     f"from a window where OLLAMA_MODELS is set.")
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMUnavailable(f"Ollama call failed ({e}). Is `ollama serve` running and "
                                 f"`ollama pull {self.model}` done?") from e
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)

    # ----- structured output with one repair retry -----
    def json(self, messages: list[dict], schema: dict | None = None,
             temperature: float = 0.0, max_tokens: int = 700, validate: Callable | None = None):
        """Return parsed JSON. `validate(obj)` may raise to trigger a repair retry."""
        last_err = ""
        msgs = list(messages)
        for attempt in range(2):
            text = self.chat(msgs, temperature=temperature if attempt == 0 else 0.0,
                             schema=schema, max_tokens=max_tokens)
            try:
                obj = extract_json(text)
                if validate:
                    validate(obj)
                return obj
            except Exception as e:  # noqa: BLE001 - any parse/validation failure triggers repair
                last_err = str(e)
                msgs = list(messages) + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was not valid ({last_err[:200]}). "
                                                 "Reply again with ONLY valid JSON matching the required shape."},
                ]
        raise LLMError(f"No valid JSON after repair: {last_err}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "mock":
            return [_hash_embed(t) for t in texts]
        out = []
        for t in texts:
            key = hashlib.sha256(("emb|" + config.EMBED_MODEL + "|" + t).encode()).hexdigest()
            cpath = config.CACHE_DIR / f"{key}.json"
            if self.use_cache and cpath.exists():
                out.append(json.loads(cpath.read_text(encoding="utf-8"))["v"])
                continue
            try:
                r = requests.post(f"{config.OLLAMA_URL}/api/embeddings",
                                  json={"model": config.EMBED_MODEL, "prompt": t},
                                  timeout=config.LLM_TIMEOUT_S)
                r.raise_for_status()
            except requests.RequestException as e:
                raise LLMUnavailable(f"Embedding call failed ({e}); `ollama pull {config.EMBED_MODEL}`") from e
            v = r.json()["embedding"]
            if self.use_cache:
                cpath.write_text(json.dumps({"v": v}), encoding="utf-8")
            out.append(v)
        return out


def extract_json(text: str):
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = t.find(open_c), t.rfind(close_c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON object found in model output")


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-words hashing embedding for the mock backend."""
    v = [0.0] * dim
    for w in re.findall(r"\w+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return v
