"""Loads the meta-prompts from prompts/*.txt. Placeholders look like <<name>> so JSON braces are safe."""
from __future__ import annotations

from . import config


def render(name: str, **vars: str) -> str:
    text = (config.PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
    for k, v in vars.items():
        text = text.replace(f"<<{k}>>", str(v))
    return text
