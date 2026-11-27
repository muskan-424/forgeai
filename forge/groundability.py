"""Groundability guard: can the documents (or the tools) actually support what the brief asks for?

A topic is 'supported' only if BOTH checks agree:
  - lexical:  the topic's own words appear in the best-matching passages (deterministic floor), and
  - model:    an entailment check says 'yes' AND quotes a sentence that really occurs in those passages.
An unsupported topic is never silently dropped: it is flagged, turned into a must-refuse rule, and, if EVERY
topic is unsupported, generation is blocked."""
from __future__ import annotations

import math
import re

from .ingest import build_chunks, inventory
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .retrieval import Retriever, tokenize
from .schemas import RetrievalConfig, Requirements
from .textmatch import split_sentences
from .tools import manifests

CHECK_SCHEMA = {"type": "object", "properties": {
    "coverage": {"type": "string", "enum": ["yes", "partial", "no"]}}, "required": ["coverage"]}
EVIDENCE_SENTENCES = 2

LEXICAL_SUPPORTED = 0.5   # at least half the topic's content words must occur in the retrieved passages
LEXICAL_STRONG = 0.75     # with a verified evidence quote, this much overlap outweighs a cautious 'partial'
LEXICAL_FLOOR = 0.25      # below this the documents cannot be supporting the topic, whatever the model says


class GroundabilityError(RuntimeError):
    pass


def tool_support(topic: str, inv: list[dict]) -> str | None:
    """A tool that can serve the topic: keyword evidence in the topic AND its precondition (e.g. a CSV exists)."""
    for m in manifests():
        if m.requires_file_type and not any(d["type"] == m.requires_file_type for d in inv):
            continue
        # Helpers (calculator, date) cannot make a topic answerable by themselves, so a loose word like 'price' does
        # not count; but a topic that explicitly names their job ('how many days remain until') IS what they serve
        # (rehearsal: the date capability was flagged as a document gap and became a refusal rule).
        words = m.keywords if m.provides_facts else m.strong_keywords
        if any(re.search(rf"\b{re.escape(k)}\b", topic.lower()) for k in words):
            return m.name
    return None


def weighted_coverage(topic: str, texts: list[str], retriever: Retriever) -> float:
    """Share of the topic's words found in the passages, weighted by rarity (BM25 idf over the corpus).
    Distinctive words ('company', 'car') count far more than generic ones ('policy'); words that occur nowhere
    in the corpus get the maximum weight, so a topic built on absent words cannot look covered."""
    toks = set(tokenize(topic))
    if not toks:
        return 0.0
    src: set[str] = set()
    for t in texts:
        src |= set(tokenize(t))
    bm = retriever.bm25
    unseen = math.log(1 + (bm.N + 0.5) / 0.5)
    w = {t: bm.idf.get(t, unseen) for t in toks}
    return sum(w[t] for t in toks if t in src) / sum(w.values())


def best_sentences(topic: str, hits, retriever: Retriever, k: int = EVIDENCE_SENTENCES) -> list[tuple[str, str]]:
    """The k sentences (doc, sentence) of the retrieved passages that best cover the topic's distinctive words.
    Chosen deterministically, so the evidence is real by construction; the model never writes a quote."""
    scored = []
    for chunk, _ in hits:
        for sent in split_sentences(chunk.text):
            # the sentence's OWN words count fully; the section heading only breaks ties (otherwise a document
            # title that matches the topic makes an irrelevant opening sentence win)
            own = weighted_coverage(topic, [sent], retriever)
            scored.append((own + 0.25 * weighted_coverage(topic, [sent + " " + chunk.heading], retriever), chunk.doc, sent))
    scored.sort(key=lambda x: -x[0])
    return [(doc, sent) for sc, doc, sent in scored[:k] if sc > 0]


def check_topic(topic: str, retriever: Retriever, llm: LLM) -> dict:
    hits = retriever.search(topic, k=3)
    if not hits:
        return {"topic": topic, "status": "unsupported", "lexical": 0.0, "model": None, "evidence": "", "doc": "",
                "note": "no passage shares any words with the topic"}
    lex = round(weighted_coverage(topic, [c.text + " " + c.heading for c, _ in hits], retriever), 2)
    evidence = best_sentences(topic, hits, retriever)
    if not evidence:
        return {"topic": topic, "status": "unsupported", "lexical": lex, "model": None, "evidence": "", "doc": "",
                "note": "no sentence matches the topic's distinctive words"}
    doc, quote = evidence[0]
    if weighted_coverage(topic, [quote], retriever) >= 0.99:      # every distinctive word of the topic is in ONE sentence
        return {"topic": topic, "status": "supported", "lexical": lex, "model": None, "evidence": quote, "doc": doc,
                "note": "every word of the topic occurs in one sentence of the documents"}
    listing = "\n".join(f"[{d}] {s}" for d, s in evidence)
    try:
        obj = llm.json([{"role": "user", "content": render("groundability_check", sentences=listing, topic=topic)}],
                       schema=CHECK_SCHEMA, temperature=0.0, max_tokens=40)
    except LLMUnavailable:
        raise
    except LLMError:
        obj = {"coverage": "no"}
    model = obj.get("coverage", "no")

    note = ""
    if model == "yes" and lex >= LEXICAL_SUPPORTED:
        status = "supported"
    elif model == "partial" and lex >= LEXICAL_STRONG:
        status = "supported"
        note = f"model cautious ('partial'); topic words fully present (lexical {lex}) in the quoted sentence"
    elif lex < LEXICAL_FLOOR or (model == "no" and lex < LEXICAL_SUPPORTED):
        status = "unsupported"
        note = f"model said '{model}', lexical coverage {lex}"
    else:
        status = "partial"
        note = f"model said '{model}', lexical coverage {lex}"
    return {"topic": topic, "status": status, "lexical": lex, "model": model, "evidence": quote, "doc": doc, "note": note}


def check_groundability(req: Requirements, docs_dir: str, llm: LLM) -> dict:
    """Annotates `req` in place and returns the report (entries, gaps, warnings, blocking)."""
    inv = inventory(docs_dir)
    retriever = Retriever(build_chunks(docs_dir), RetrievalConfig(top_k=3), None)
    entries, warnings = [], []
    for topic in req.must_answer:
        e = check_topic(topic, retriever, llm)
        if e["status"] in ("unsupported", "partial"):
            tool = tool_support(topic, inv)
            if tool:
                e.update(status="supported_by_tool", note=f"documents weak; tool '{tool}' can serve this", tool=tool)
        entries.append(e)
    gaps = [e["topic"] for e in entries if e["status"] == "unsupported"]
    partial = [e["topic"] for e in entries if e["status"] == "partial"]
    for t in gaps:
        warnings.append(f"UNSUPPORTED: the documents do not cover '{t}'. The agent will refuse it instead of guessing.")
        req.must_refuse.append(f"{t} (not covered by the documents)")
    for t in partial:
        warnings.append(f"PARTIAL: the documents cover '{t}' only in part; expect refusals or gaps there.")
    blocking = bool(entries) and all(e["status"] == "unsupported" for e in entries)
    req.groundability = entries
    return {"entries": entries, "gaps": gaps, "partial": partial, "warnings": warnings, "blocking": blocking}
