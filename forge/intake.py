"""Intake interviewer: brief + document folder -> structured requirements, with at most 5 clarifying questions.

Anti-hallucination: whatever the model claims the brief says (audience, tone, refusals, scope, topics) is
checked against the brief text itself; unsupported claims are treated as 'not stated' and asked about instead.
Unattended mode (auto) answers every question with a recorded ASSUMPTION so the run never blocks."""
from __future__ import annotations

import re
from typing import Callable

from .ingest import build_chunks, inventory, inventory_text
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .schemas import Requirements
from .retrieval import _STOP, stem, tokenize
from .textmatch import content_tokens, coverage

MAX_QUESTIONS = 5

EXTRACT_SCHEMA = {"type": "object", "properties": {
    "scope": {"type": "string"}, "audience": {"type": "string"}, "tone": {"type": "string"},
    "topics": {"type": "array", "items": {"type": "string"}},
    "must_refuse": {"type": "array", "items": {"type": "string"}}},
    "required": ["scope", "audience", "tone", "topics", "must_refuse"]}

# A refusal is only believable if the brief contains words that forbid something. Without this, a 3B model
# files topics the brief wants ANSWERED (seen in a real run: 'Leave policy') under must_refuse.
_REFUSAL_CUE = re.compile(
    r"\b(must not|mustn't|should not|shouldn't|do not|don't|never|refuse|decline|cannot|can't|hand off|escalate|"
    r"avoid|off[- ]limits|not (?:to )?(?:answer|give|provide|advise|discuss|offer))\b", re.I)
QUESTIONS_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
                    "required": ["questions"]}

# slot -> (question to ask, default used in unattended mode)
SLOT_QUESTIONS = {
    "must_refuse": ("Are there topics the assistant must refuse or hand off to a human, even if the documents mention them?",
                    "anything not covered by the provided documents"),
    "tone": ("What tone should the assistant use (for example formal, friendly, concise)?",
             "professional, concise and neutral"),
    "audience": ("Who will be asking the assistant questions?", "the people described in the brief"),
    "scope": ("What should be in scope, and what should be out of scope?", ""),
    "must_answer": ("Which questions must the assistant be able to answer reliably?", ""),
}
ASK_ORDER = ["must_refuse", "tone", "audience", "must_answer"]   # scope is never asked: the brief is the scope


# ---- backstop: a 3B model's topic list is unreliable, so read the brief's own sentences as well -----------------
_REQUEST_CUE = re.compile(r"\b(tell|answer|answers|explain|explains|help|helps|provide|provides|give|gives|advise|calculate|"
                          r"compute|look up|find|show|guide|summari[sz]e|check|list|inform|cover|handle|support|assist)\b", re.I)
_META = {"assistant", "answer", "question", "help", "need", "want", "tone", "friendly", "formal", "clear", "concise",
         "polite", "keep", "also", "should", "would", "must", "please", "provide", "explain", "tell", "give", "look",
         "find", "show", "document", "based", "system", "tool", "new", "build", "able", "like", "make", "use",
         "calculate", "compute", "include", "including", "includes", "lookup"}
_GENERIC = {"rules", "rule", "terms", "details", "information", "things", "process", "procedure", "procedures", "policy", "policies", "matters", "issues", "topics", "guidance", "support", "requirements", "conditions", "questions"}
_FUNC = {"about", "regarding", "from", "into", "over", "under", "between", "through", "during", "before", "after",
         "just", "only", "very", "any", "some", "each", "every", "other", "such", "than", "then", "there", "these",
         "those", "where", "when", "while", "whether", "who", "whom", "whose", "will", "shall", "may", "might", "has",
         "have", "had", "having", "been", "being", "get", "gets", "got", "their", "them", "they", "its", "our", "your"}
_BRIDGE = {"in", "of", "the", "for", "on", "to", "at", "a", "an", "with", "by"}   # may sit INSIDE a topic; 'and'/'or' may not
_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+|[,;:()]")
MAX_BACKSTOP_TOPICS = 6


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def refusal_phrases(brief: str) -> list[str]:
    """What the brief forbids, taken from the text after the forbidding words ('must not give legal advice about
    contracts' -> 'legal advice about contracts')."""
    out = []
    for sent in _sentences(brief):
        m = _REFUSAL_CUE.search(sent)
        if not m:
            continue
        tail = sent[m.end():].strip(" .,:;")
        tail = re.sub(r"^(?:to\s+)?(?:give|provide|answer|discuss|offer|advise on|advise|make|do|comment on|talk about|"
                      r"share|help with)\s+(?:any\s+|about\s+)?", "", tail, flags=re.I).strip()
        if 1 <= len(tail.split()) <= 10:
            out.append(tail)
    return out


def corpus_vocab(docs_dir: str) -> set[str]:
    return {t for c in build_chunks(docs_dir) for t in tokenize(c.text + " " + c.heading)}


def backstop_topics(brief: str, req: Requirements, vocab: set[str]) -> list[tuple[str, str]]:
    """(topic, source sentence) for requests the model's topic list does not account for. Topics are verbatim runs of
    the brief's own words. A run is added only if (a) it contains a word NO document uses (a likely gap, e.g.
    'pension'), or (b) most of its sentence is not covered by anything extracted."""
    known = " ".join([req.scope, req.audience, req.tone] + req.must_answer + req.must_refuse)
    covered = {stem(w.lower().removesuffix("'s")) for w in _TOKEN.findall(known)}
    covered |= {stem(m) for m in _META}
    out: list[tuple[str, str]] = []
    for sent in _sentences(brief):
        if not _REQUEST_CUE.search(sent) or _REFUSAL_CUE.search(sent):
            continue
        toks = [w.removesuffix("'s") for w in _TOKEN.findall(sent)]
        content = [w for w in toks if (w.isalpha() or "-" in w) and w.lower() not in _STOP | _FUNC and len(w) > 2]
        flags = [(w.lower() not in _STOP | _FUNC and len(w) > 2 and stem(w.lower()) not in covered and (w.isalpha() or "-" in w))
                 for w in toks]
        if not any(flags):
            continue
        content_stems = [stem(w.lower()) for w in content if stem(w.lower()) not in {stem(m) for m in _META}]
        frac_uncovered = (sum(1 for s_ in content_stems if s_ not in covered) / len(content_stems)) if content_stems else 0
        runs, cur, gap = [], [], []
        for w, f in zip(toks, flags):
            if f:
                cur += gap + [w]
                gap = []
            elif cur and w.lower() in _BRIDGE and len(gap) < 2:
                gap.append(w)
            else:
                if cur:
                    runs.append(cur)
                cur, gap = [], []
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) == 1 and run[0].lower() in _GENERIC:
                continue                                # 'rules', 'terms'...: too vague to be a topic by themselves
            unseen = any(stem(w.lower()) not in vocab for w in run if w.lower() not in _STOP)
            if unseen or (frac_uncovered >= 0.5 and len(run) >= 2):
                out.append((" ".join(run[:6]), sent))
    seen, dedup = [], []
    for t, s_ in out:
        if not any(coverage(t, x) >= 0.8 for x in seen):
            seen.append(t)
            dedup.append((t, s_))
    return dedup[:MAX_BACKSTOP_TOPICS]


def _stated(value: str, brief: str, min_cov: float | None = None) -> bool:
    """Is this value traceable to the brief? (guards against the model inventing 'stated' requirements)
    Short phrases must match fully: 'Remote work' must not pass just because the brief ends a sentence with 'work'."""
    if not value.strip():
        return False
    if min_cov is None:
        n = len(content_tokens(value))
        min_cov = 1.0 if n <= 2 else 0.67 if n <= 4 else 0.5
    return coverage(value, brief) >= min_cov


def _split_list(text: str) -> list[str]:
    return [p.strip(" .") for p in re.split(r"[;\n,]|\band\b", text) if len(p.strip(" .")) > 2]


def extract_draft(brief: str, inv_text: str, headings: list[str], llm: LLM,
                  titles: frozenset = frozenset(), vocab: set[str] | None = None) -> tuple[Requirements, list[str]]:
    """Returns (draft requirements, list of slots the brief genuinely leaves open)."""
    req = Requirements(brief=brief)
    try:
        obj = llm.json([{"role": "user", "content": render("intake_extract", brief=brief, inventory=inv_text)}],
                       schema=EXTRACT_SCHEMA, temperature=0.0, max_tokens=350)
    except LLMUnavailable:
        raise
    except LLMError:
        obj = {}
    for slot in ("scope", "audience", "tone"):
        v = str(obj.get(slot, "")).strip()
        if _stated(v, brief):
            setattr(req, slot, v)
    topics: list[str] = []
    for t in obj.get("topics", []):
        t = str(t).strip()
        if t and _stated(t, brief) and t.lower() not in {x.lower() for x in topics}:
            topics.append(t)                                   # every topic must come from the BRIEF's own words
    req.must_answer = topics[:10]
    if _REFUSAL_CUE.search(brief):                             # no forbidding words in the brief -> nothing to refuse
        for t in obj.get("must_refuse", []):
            t = str(t).strip()
            if t and _stated(t, brief) and not any(coverage(t, a) >= 0.8 for a in req.must_answer):
                req.must_refuse.append(t)
        req.must_refuse = req.must_refuse[:6]
    # deterministic backstop 1: refusals stated in the brief's own words
    for phrase in refusal_phrases(brief):
        if not any(coverage(phrase, x) >= 0.8 for x in req.must_refuse) and not any(coverage(phrase, a) >= 0.8 for a in req.must_answer):
            req.must_refuse.append(phrase)
            req.assumptions.append(f"Refusal '{phrase}' taken verbatim from the brief (the model's list had missed it).")
    # deterministic backstop 2: requests the model's topic list does not account for
    if vocab is not None:
        for topic, sent in backstop_topics(brief, req, vocab):
            if len(req.must_answer) < 10:
                req.must_answer.append(topic)
                req.assumptions.append(f"Topic '{topic}' taken verbatim from the brief sentence \"{sent[:90]}\" "
                                       "(the model's topic list had missed it).")
    missing = [s for s in ASK_ORDER if not getattr(req, s)]
    return req, missing


def interview(brief: str, docs_dir: str, llm: LLM, ask: Callable[[str], str] | None = None,
              max_questions: int = MAX_QUESTIONS, progress=print) -> Requirements:
    """ask=None means unattended: every open slot gets a recorded default assumption."""
    inv = inventory(docs_dir)
    headings = [h for d in inv for h in d.get("headings", [])] + [c for d in inv for c in d.get("columns", [])]
    default_topics = _default_topics(inv)
    titles = frozenset(d['headings'][0].lower() for d in inv if len(d.get('headings', [])) > 1)   # a title is not a topic
    req, missing = extract_draft(brief, inventory_text(inv), headings, llm, titles, corpus_vocab(docs_dir))

    questions: list[tuple[str, str]] = [(s, SLOT_QUESTIONS[s][0]) for s in missing][:max_questions]
    if ask is not None and len(questions) < max_questions:
        try:
            extra = llm.json([{"role": "user", "content": render(
                "intake_questions", brief=brief, inventory=inventory_text(inv), n=str(max_questions - len(questions)))}],
                schema=QUESTIONS_SCHEMA, temperature=0.2, max_tokens=200).get("questions", [])
        except LLMUnavailable:
            raise
        except LLMError:
            extra = []
        questions += [("extra", str(q).strip()) for q in extra if str(q).strip()][:max_questions - len(questions)]

    for slot, q in questions:
        answer = (ask(q) if ask else "").strip()
        req.clarifications.append({"q": q, "a": answer if answer else "(no answer; default assumed)", "slot": slot})
        if slot == "extra":
            if answer:
                req.assumptions.append(f"Clarification: {q} -> {answer}")
            continue
        if answer and answer.lower() not in {"skip", "n/a", "none", "-"}:
            if slot in ("must_answer", "must_refuse"):
                setattr(req, slot, _split_list(answer))
            else:
                setattr(req, slot, answer)
        else:
            default = SLOT_QUESTIONS[slot][1]
            _apply_default(req, slot, default, default_topics)

    # slots we did not get to ask about (more than max_questions open) also get defaults
    for slot in ASK_ORDER + ["scope"]:
        if not getattr(req, slot):
            _apply_default(req, slot, SLOT_QUESTIONS[slot][1], default_topics)
    assert len(req.clarifications) <= max_questions
    return req


def _default_topics(inv: list[dict]) -> list[str]:
    """Topics to assume when the brief names none: each document's section names (not its title); CSV -> its name."""
    topics: list[str] = []
    for d in inv:
        if d["type"] == "csv":
            topics.append(re.sub(r"[_\-]+", " ", d["doc"].rsplit(".", 1)[0]).strip())
            continue
        hs = d.get("headings", [])
        topics += [h for h in (hs[1:] if len(hs) > 1 else hs) if 3 <= len(h) <= 40]
    return topics[:8]


def _apply_default(req: Requirements, slot: str, default: str, topics: list[str]) -> None:
    if slot == "must_answer":
        req.must_answer = topics or [req.scope or req.brief[:60]]
        req.assumptions.append(f"must_answer not specified; assumed the documents' own sections: {req.must_answer}")
    elif slot == "scope":
        req.scope = req.brief.strip().split(".")[0][:200]
        req.assumptions.append(f"scope not specified; taken from the brief: '{req.scope}'")
    elif slot == "must_refuse":
        req.must_refuse = [default]
        req.assumptions.append(f"must_refuse not specified; assumed: {default}")
    else:
        setattr(req, slot, default)
        req.assumptions.append(f"{slot} not specified; assumed: {default}")


def requirements_markdown(req: Requirements) -> str:
    def bullets(xs):
        return "\n".join(f"- {x}" for x in xs) or "- (none)"
    g = "\n".join(f"- **{e['topic']}**: {e['status']}" + (f" ({e.get('note')})" if e.get("note") else "")
                  for e in req.groundability) or "- (not checked)"
    qa = "\n".join(f"- Q: {c['q']}\n  A: {c['a']}" for c in req.clarifications) or "- (none asked)"
    return (f"# Requirements\n\n**Brief:** {req.brief}\n\n## Scope\n{req.scope}\n\n## Audience\n{req.audience}\n\n"
            f"## Tone\n{req.tone}\n\n## Must answer\n{bullets(req.must_answer)}\n\n## Must refuse\n{bullets(req.must_refuse)}\n\n"
            f"## Groundability (can the documents support the brief?)\n{g}\n\n## Clarifying questions\n{qa}\n\n"
            f"## Assumptions\n{bullets(req.assumptions)}\n")
