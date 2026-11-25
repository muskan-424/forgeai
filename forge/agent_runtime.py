"""Runs any AgentDefinition: retrieve -> (tool loop) -> contract-validated answer."""
from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from . import config
from .ingest import build_chunks
from .llm import LLM, LLMError
from .retrieval import Retriever
from .schemas import AgentAnswer, AgentDefinition, AgentTrace, Chunk, Citation
from .tools import build_tool

MAX_TOOL_STEPS = 2

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "object", "properties": {
            "doc": {"type": "string"}, "quote": {"type": "string"}}, "required": ["doc", "quote"]}},
        "refused": {"type": "boolean"},
        "refusal_reason": {"type": "string"},
    },
    "required": ["answer", "citations", "refused", "refusal_reason"],
}

TOOL_SCHEMA = {
    "type": "object",
    "properties": {**ANSWER_SCHEMA["properties"],
                   "tool": {"type": "string"}, "args": {"type": "object"}},
    "required": ["answer", "citations", "refused", "refusal_reason"],
}


from .textmatch import norm, quote_verified  # noqa: E402  (shared with eval generator and judge)


def load_definition(path: str | Path) -> AgentDefinition:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    d = AgentDefinition(**data)
    # resolve docs_dir relative to the definition file
    if not Path(d.docs_dir).is_absolute():
        d.docs_dir = str((p.parent / d.docs_dir).resolve())
    return d


def save_definition(d: AgentDefinition, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(d.model_dump(), sort_keys=False, allow_unicode=True, width=100),
                          encoding="utf-8")


def render_system_prompt(d: AgentDefinition, tool_specs: str) -> str:
    parts = [d.system_prompt.strip()]
    if d.rules:
        parts.append("RULES:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(d.rules)))
    c = d.output_contract
    contract = ("OUTPUT FORMAT: reply with ONE JSON object: "
                '{"answer": str, "citations": [{"doc": str, "quote": str}], "refused": bool, "refusal_reason": str}. '
                f"Keep the answer under {c.max_answer_words} words. ")
    if c.require_citations:
        contract += ("Every factual answer must include at least one citation whose 'quote' is copied VERBATIM "
                     "from the provided context and whose 'doc' is that passage's document. ")
    contract += (f"If the context does not contain the answer or the request is out of scope, set refused=true, "
                 f"answer to: \"{d.refusal_message}\", citations to [], and give a short refusal_reason.")
    parts.append(contract)
    if tool_specs:
        parts.append("TOOLS (optional): to call a tool, reply with JSON {\"tool\": name, \"args\": {...}, "
                     "\"answer\": \"\", \"citations\": [], \"refused\": false, \"refusal_reason\": \"\"}. "
                     "You will then receive the result and must give the final answer.\n" + tool_specs)
    return "\n\n".join(parts)


class Agent:
    def __init__(self, definition: AgentDefinition, llm: LLM | None = None, embed_llm: LLM | None = None):
        self.d = definition
        self.llm = llm or LLM(model=definition.model or config.AGENT_MODEL)
        rc = definition.retrieval
        self.chunks = build_chunks(definition.docs_dir, rc.chunk_size, rc.chunk_overlap)
        self.retriever = Retriever(self.chunks, rc, embed_llm or self.llm)
        self.tools = {t.name: build_tool(t.name, t.config, definition.docs_dir) for t in definition.tools}

    def _tool_specs(self) -> str:
        lines = []
        for t in self.tools.values():
            m = t.manifest
            lines.append(f"- {m.name}: {m.description} Args: {m.args}")
        return "\n".join(lines)

    @staticmethod
    def _context_block(chunks: list[Chunk]) -> str:
        if not chunks:
            return "(no relevant passages found)"
        return "\n\n".join(f"[{c.doc}]\n{c.text}" for c in chunks)

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AgentTrace:
        t0 = time.time()
        trace = AgentTrace(question=question)
        hits = self.retriever.search(question)
        trace.retrieved = [c for c, _ in hits]

        if not trace.retrieved and not self.tools:
            trace.answer = AgentAnswer(answer=self.d.refusal_message, refused=True,
                                       refusal_reason="no relevant passages retrieved")
            trace.latency_s = time.time() - t0
            return trace

        system = render_system_prompt(self.d, self._tool_specs())
        messages = [{"role": "system", "content": system}]
        for q, a in (history or [])[-3:]:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content":
                         f"CONTEXT:\n{self._context_block(trace.retrieved)}\n\nQUESTION: {question}"})
        schema = TOOL_SCHEMA if self.tools else ANSWER_SCHEMA
        tool_texts: list[str] = []

        try:
            for step in range(MAX_TOOL_STEPS + 1):
                obj = self.llm.json(messages, schema=schema, temperature=self.d.temperature)
                trace.raw_output = str(obj)
                tool = obj.get("tool") if isinstance(obj, dict) else None
                if tool and tool in self.tools and step < MAX_TOOL_STEPS:
                    res = self.tools[tool].run(obj.get("args") or {})
                    trace.tool_calls.append({"tool": tool, "args": obj.get("args") or {}, "result": res})
                    tool_texts.append(res)
                    messages.append({"role": "assistant", "content": str(obj)})
                    messages.append({"role": "user", "content":
                                     f"TOOL RESULT ({tool}): {res}\nNow give the final answer JSON (no more tool calls)."})
                    continue
                trace.answer = AgentAnswer(
                    answer=str(obj.get("answer", "")),
                    citations=[Citation(doc=str(c.get("doc", "")), quote=str(c.get("quote", "")))
                               for c in (obj.get("citations") or []) if isinstance(c, dict)],
                    refused=bool(obj.get("refused", False)),
                    refusal_reason=str(obj.get("refusal_reason", "")),
                )
                break
        except LLMError as e:
            trace.error = str(e)
            trace.contract_ok = False

        self._validate(trace, tool_texts)
        trace.latency_s = time.time() - t0
        return trace

    def _validate(self, trace: AgentTrace, tool_texts: list[str]) -> None:
        a, c = trace.answer, self.d.output_contract
        corpus = [(ch.doc, norm(ch.text)) for ch in trace.retrieved] + [("__tool__", norm(t)) for t in tool_texts]
        verified = [quote_verified(cit.quote, [text for _, text in corpus]) for cit in a.citations]
        trace.citations_verified = verified
        if a.refused:
            if not a.answer:
                a.answer = self.d.refusal_message
            return
        problems = []
        if not trace.error and not a.answer.strip():
            problems.append("empty answer")
        if c.require_citations and not tool_texts and not a.citations:
            problems.append("missing citations")
        if len(a.answer.split()) > c.max_answer_words * 1.5:
            problems.append("too long")
        if problems:
            trace.contract_ok = False
            trace.error = trace.error or "; ".join(problems)
