"""Central configuration. Override with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BACKEND = os.environ.get("FORGE_BACKEND", "ollama")  # "ollama" | "mock"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
AGENT_MODEL = os.environ.get("FORGE_AGENT_MODEL", "qwen2.5:3b")
JUDGE_MODEL = os.environ.get("FORGE_JUDGE_MODEL", "llama3.2:3b")  # different family on purpose
EMBED_MODEL = os.environ.get("FORGE_EMBED_MODEL", "nomic-embed-text")
CACHE_DIR = Path(os.environ.get("FORGE_CACHE_DIR", ROOT / ".llm_cache"))
LLM_TIMEOUT_S = int(os.environ.get("FORGE_LLM_TIMEOUT", "300"))
PROMPTS_DIR = ROOT / "prompts"
