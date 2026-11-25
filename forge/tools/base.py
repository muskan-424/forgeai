from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Manifest:
    name: str
    description: str                    # shown to the agent model
    args: dict                          # {arg_name: "description"}
    when_needed: str                    # shown to the generator: when is this tool justified?
    keywords: list[str] = field(default_factory=list)  # brief/requirement words that hint at need
    requires_file_type: str = ""        # e.g. "csv": gate = such a document must exist
    strong_keywords: list[str] = field(default_factory=list)  # explicit mentions: enough evidence on their own
    provides_facts: bool = True         # False for pure helpers (calculator, date): they compute, they cannot
                                        # make a topic answerable, so they never count as grounding


class Tool:
    manifest: Manifest

    def __init__(self, config: dict | None = None, docs_dir: str = "."):
        self.config = config or {}
        self.docs_dir = docs_dir

    def run(self, args: dict) -> str:  # pragma: no cover - interface
        raise NotImplementedError
