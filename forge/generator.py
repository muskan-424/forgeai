"""Agent generator: requirements + documents -> a complete declarative AgentDefinition.

- Tools: attached only if (1) the tool's precondition holds (e.g. a CSV exists), (2) the brief/requirements
  contain evidence of the need, AND (3) a model necessity check agrees the documents alone are not enough.
  Every decision, attached or rejected, is recorded with its reason.
- Retrieval parameters: derived from document statistics, not guessed by the model.
- System prompt: the model writes only the persona paragraph; the rules are deterministic, individually
  patchable strings (the optimizer edits these, not a monolithic prompt)."""
from __future__ import annotations

import os
import re
import statistics
from pathlib import Path

from .ingest import build_chunks, inventory
from .intake import SLOT_QUESTIONS
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .schemas import AgentDefinition, RetrievalConfig, Requirements, ToolRef
from .textmatch import content_tokens
from .tools import manifests

TOOL_CHECK_SCHEMA = {"type": "object", "properties": {"needed": {"type": "boolean"}, "reason": {"type": "string"}},
                     "required": ["needed", "reason"]}
PROMPT_SCHEMA = {"type": "object", "properties": {"system_prompt": {"type": "string"}}, "required": ["system_prompt"]}

BASE_RULES = [
    "Answer only from the provided CONTEXT or tool results. Never use outside knowledge.",
    "Cite the supporting sentence verbatim; never invent or paraphrase a citation quote.",
    "Copy numbers, names and conditions exactly as written in the context. Never guess or compute a figure that is not there.",
    "If the context does not contain the answer, refuse: set refused=true and use only the standard refusal message.",
    "A refusal must be clean: no facts, no citations and no partial answer inside a refusal.",
    "If a question is ambiguous, answer the most likely meaning using the documents and state that assumption briefly.",
]


# ------------------------------------------------------------------ tools
def _evidence(manifest, text: str, inv: list[dict]) -> tuple[str, bool]:
    """(evidence, strong): words in the brief/requirements showing the need for this tool; strong = explicit mention."""
    low = text.lower()
    for k in manifest.strong_keywords:
        if re.search(rf"\b{re.escape(k)}\b", low):
            return f"the brief explicitly mentions '{k}'", True
    for k in manifest.keywords:
        if re.search(rf"\b{re.escape(k)}\b", low):
            return f"requirements mention '{k}'", False
    if manifest.requires_file_type == "csv":     # table columns named in the requirements also count
        req_tokens = content_tokens(text)
        for d in inv:
            if d["type"] == "csv":
                hit = [c for c in d.get("columns", []) if content_tokens(c) & req_tokens]
                if hit:
                    return f"requirements refer to table column(s) {hit} of {d['doc']}", False
    return "", False


def select_tools(req: Requirements, inv: list[dict], llm: LLM) -> tuple[list[ToolRef], list[dict]]:
    text = " ".join([req.brief, req.scope] + req.must_answer)
    attached, decisions = [], []
    for m in manifests():
        d = {"tool": m.name, "attached": False, "reason": ""}
        csv_docs = [x for x in inv if x["type"] == "csv"]
        if m.requires_file_type and not any(x["type"] == m.requires_file_type for x in inv):
            d["reason"] = f"precondition failed: no .{m.requires_file_type} document provided"
        else:
            ev, strong = _evidence(m, text, inv)
            if not ev:
                d["reason"] = "no evidence in the brief that this tool is needed"
            else:
                try:
                    if strong:        # an explicit mention is not second-guessed by a small model's yes/no
                        obj = {"needed": True, "reason": "explicitly requested in the brief"}
                    else:
                        obj = llm.json([{"role": "user", "content": render(
                        "generator_tool_check", brief=req.brief, topics="; ".join(req.must_answer), tool=m.name,
                        tool_description=m.description, when_needed=m.when_needed)}],
                        schema=TOOL_CHECK_SCHEMA, temperature=0.0, max_tokens=120)
                except LLMUnavailable:
                    raise
                except LLMError:
                    obj = {"needed": False, "reason": "necessity check failed"}
                if obj.get("needed"):
                    cfg = {}
                    if m.name == "csv_lookup":
                        best = max(csv_docs, key=lambda x: x.get("rows", 0))
                        cfg = {"csv_path": best["doc"]}
                        if len(csv_docs) > 1:
                            ev += f" (several CSVs; using the largest: {best['doc']})"
                    attached.append(ToolRef(name=m.name, justification=f"{ev}; {obj.get('reason', '')}".strip("; "),
                                            config=cfg))
                    d.update(attached=True, reason=f"{ev}; model: {obj.get('reason', '')}")
                else:
                    d["reason"] = f"evidence found ({ev}) but model says documents suffice: {obj.get('reason', '')}"
        decisions.append(d)
    return attached, decisions


# ------------------------------------------------------------------ retrieval parameters
def retrieval_params(docs_dir: str, inv: list[dict]) -> RetrievalConfig:
    prose = [d for d in inv if d["type"] != "csv"]
    prose_words = sum(d["words"] for d in prose)
    total_words = sum(d["words"] for d in inv) or 1
    avg_par = statistics.mean([d.get("avg_paragraph_chars", 600) for d in prose]) if prose else 600
    chunk_size = int(min(max(avg_par * 1.6, 500), 1200) // 50 * 50)
    overlap = int(chunk_size * 0.12)
    n_chunks = len(build_chunks(docs_dir, chunk_size, overlap))
    top_k = 3 if n_chunks <= 6 else 4 if n_chunks <= 60 else 5
    prose_share = prose_words / total_words
    # Embeddings cost a query embedding per question plus an index build (about a minute for 18 chunks on this CPU).
    # BM25 with stemming and heading text is enough for small corpora; the optimizer can still turn them on later.
    use_emb = prose_share >= 0.5 and 60 < n_chunks <= 300
    return RetrievalConfig(chunk_size=chunk_size, chunk_overlap=overlap, top_k=top_k,
                           bm25_weight=0.5 if use_emb else 1.0, use_embeddings=use_emb)


# ------------------------------------------------------------------ prompt + rules
def build_rules(req: Requirements, tools: list[ToolRef]) -> list[str]:
    rules = list(BASE_RULES)
    if any(t.name == "calculator" for t in tools):
        rules.append("For any arithmetic (totals, differences, percentages) use the calculator tool; do not compute in your head.")
    if any(t.name == "csv_lookup" for t in tools):
        rules.append("For exact items, codes or prices in the table, call csv_lookup and quote its row; do not rely on memory.")
    if any(t.name == "current_date" for t in tools):
        rules.append("When an answer depends on today's date, call current_date first.")
    if any(t.name == "open_meteo" for t in tools):
        rules.append("Use open_meteo only for live weather questions; never for policy or catalogue questions.")
    generic = SLOT_QUESTIONS["must_refuse"][1]          # already covered by the base refusal rule
    refuse = [t for t in req.must_refuse if "not covered by the documents" not in t and t != generic]
    for t in refuse:
        rules.append(f"Refuse questions about: {t}.")
    gaps = [e["topic"] for e in req.groundability if e["status"] == "unsupported"]
    if gaps:
        rules.append("The documents do not cover: " + "; ".join(gaps) + ". If asked about these, refuse.")
    return rules


def refusal_topics(req: Requirements) -> list[str]:
    """What the runtime refusal gate checks against: the brief's forbidden topics and the unsupported gaps
    (without the generic 'anything not covered' default, which is not a topic)."""
    generic = SLOT_QUESTIONS["must_refuse"][1]
    out = []
    for t in req.must_refuse:
        t = t.replace(" (not covered by the documents)", "").strip()
        if t and t != generic and t not in out:
            out.append(t)
    return out


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    stop = {"i", "need", "an", "a", "the", "that", "for", "to", "of", "and", "assistant", "answers", "answer"}
    return "-".join([w for w in words if w not in stop][:4] + ["agent"])


def write_persona(req: Requirements, llm: LLM) -> tuple[str, str]:
    """Returns (persona paragraph, how it was produced: 'model' or 'fallback')."""
    try:
        obj = llm.json([{"role": "user", "content": render(
            "generator_system_prompt", scope=req.scope, audience=req.audience, tone=req.tone,
            topics="; ".join(req.must_answer))}], schema=PROMPT_SCHEMA, temperature=0.2, max_tokens=250)
        text = str(obj.get("system_prompt", "")).strip()
        if 40 <= len(text) <= 1200 and re.match(r"(?i)you\b", text) and not text.lstrip().startswith("{"):
            return text, "model"
    except LLMUnavailable:
        raise
    except LLMError:
        pass
    return (f"You are an assistant for {req.audience}. You help with: {req.scope} "
            f"Your tone is {req.tone}."), "fallback"


def generate_definition(req: Requirements, docs_dir: str, llm: LLM, out_dir: str | None = None
                        ) -> tuple[AgentDefinition, dict]:
    inv = inventory(docs_dir)
    tools, tool_decisions = select_tools(req, inv, llm)
    retrieval = retrieval_params(docs_dir, inv)
    persona, persona_src = write_persona(req, llm)
    rel_docs = os.path.relpath(docs_dir, out_dir) if out_dir else docs_dir
    d = AgentDefinition(
        name=_slug(req.brief), description=req.scope, system_prompt=persona,
        rules=build_rules(req, tools), docs_dir=rel_docs.replace("\\", "/"), retrieval=retrieval, tools=tools,
        refusal_message="I can't help with that; it is outside what my documents cover.",
        refusal_topics=refusal_topics(req))
    notes = {"tool_decisions": tool_decisions, "persona_source": persona_src,
             "retrieval_basis": {"documents": len(inv), "chunks": len(build_chunks(docs_dir, retrieval.chunk_size, retrieval.chunk_overlap))}}
    return d, notes
