"""Eval-set generator.

Design (learned the hard way with a 3B model):
- SENTENCE-ANCHORED: each attempt targets one specific sentence of the documents, so cases are diverse by
  construction and the supporting quote is that sentence (verbatim by construction, no paraphrase risk).
- Everything the model writes is VERIFIED deterministically: facts are short and present in the sentence,
  the question uses the sentence's own words, and a 'maximum' question is never paired with an 'over' sentence.
- Ambiguous cases are REWRITES of accepted cases (easy for a small model), keeping their verified facts.
- Out-of-scope questions must score low against the corpus AND pass a model answerability check.
Rejected attempts are counted by reason, with examples, so the final report can state the generation yield."""
from __future__ import annotations

import random
import re
import statistics
from collections import Counter

from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .retrieval import Retriever
from .schemas import Chunk, EvalCase, RetrievalConfig
from .textmatch import content_tokens, coverage, fact_in_text, norm, question_similarity, split_sentences

SINGLE_SCHEMA = {"type": "object", "properties": {
    "question": {"type": "string"}, "expected_facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["question", "expected_facts"]}
AMBIG_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
CHECK_SCHEMA = {"type": "object", "properties": {"answerable": {"type": "boolean"}}, "required": ["answerable"]}
OOS_SCHEMA = {"type": "object", "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
              "required": ["questions"]}

ID_PREFIX = {"answerable": "ans", "multi_doc": "mul", "out_of_scope": "oos", "ambiguous": "amb"}
_DEICTIC = re.compile(r"\b(this|that|these|those)\s+(\w+\s+)?(policy|policies|document|section|procedure|agreement|paragraph|guideline)\b", re.I)
_TABLE_ROW = re.compile(r"^[\w .\-]+: .+; [\w .\-]+: ")      # 'part_no: V-1; description: ...; price: 5'
_META_WORDS = re.compile(r"\b(passage|the document|the text|above|excerpt|sentence [ab]?)\b", re.I)
MAX_FACT_WORDS = 12
MIN_DROPPED_TERMS = 2   # a vague rewrite must drop at least this many content words
MIN_COVERAGE = 0.6   # share of the question's content words that must appear in its source sentence

_UPPER = r"\b(maximum|at most|up to|no more than|not exceed\w*|highest|longest|most)\b"
_LOWER = r"\b(minimum|at least|no less than|lowest|shortest)\b"
_EV_UPPER = r"\b(maximum|up to|at most|no more than|not exceed\w*|cap(ped)? at|limit(ed)? to)\b"
_EV_LOWER = r"\b(over\s+\d|above|at least|minimum|more than|exceeds?|longer than|greater than)\b"


_COMPARE = re.compile(r"\b(compare|higher|lower|larger|smaller|bigger|greater|difference|by how much|more than|less than)\b", re.I)
_UNIT = re.compile(r"\d[\d,.]*\s*([a-zA-Z%\u20ac$]+)")


def incomparable(question: str, facts: list[str]) -> bool:
    """A question that asks to compare values whose units differ (euros vs days) makes no sense."""
    if not _COMPARE.search(question):
        return False
    units = set()
    for f in facts:
        m = _UNIT.search(f)
        if m:
            u = m.group(1).lower()
            units.add(u[:-1] if len(u) > 3 and u.endswith("s") else u)
    return len(units) > 1


def dropped_terms(original: str, rewrite: str) -> list[str]:
    """Content words of the original question that the vague rewrite no longer contains."""
    keep = content_tokens(rewrite)
    return sorted(t for t in content_tokens(original) if t not in keep)


def direction_conflict(question: str, evidence: str) -> bool:
    """True if the question asks for a maximum but the evidence states a minimum (or vice versa)."""
    q, e = question.lower(), evidence.lower()
    ev_up, ev_low = bool(re.search(_EV_UPPER, e)), bool(re.search(_EV_LOWER, e))
    q_up, q_low = bool(re.search(_UPPER, q)), bool(re.search(_LOWER, q))
    return (q_up and ev_low and not ev_up) or (q_low and ev_up and not ev_low)


class EvalGenerator:
    def __init__(self, chunks: list[Chunk], llm: LLM, purpose: str = "", must_refuse: list[str] | None = None,
                 inventory_text: str = "", seed: int = 7, progress=print):
        self.chunks = chunks
        self.llm, self.purpose = llm, purpose or "answer questions about the provided documents"
        self.must_refuse = must_refuse or []
        self.inventory_text = inventory_text or self._inventory_from_chunks()
        self.rng = random.Random(seed)
        self.progress = progress
        self.retriever = Retriever(chunks, RetrievalConfig(top_k=1), None)
        self.stats: Counter = Counter()
        self.rejections: Counter = Counter()
        self.examples: dict[str, list[str]] = {}

    # ---------- helpers ----------
    def _inventory_from_chunks(self) -> str:
        by_doc: dict[str, list[str]] = {}
        for c in self.chunks:
            if c.heading and c.heading not in by_doc.setdefault(c.doc, []):
                by_doc[c.doc].append(c.heading)
            else:
                by_doc.setdefault(c.doc, [])
        return "\n".join(f"- {d}: {', '.join(h[:6]) or '(no headings)'}" for d, h in by_doc.items())

    def _focus_items(self) -> list[tuple[Chunk, str]]:
        """(chunk, sentence) pairs, shuffled within each document and interleaved across documents."""
        by_doc: dict[str, list[tuple[Chunk, str]]] = {}
        for c in self.chunks:
            for s in split_sentences(c.text):
                by_doc.setdefault(c.doc, []).append((c, s))
        for lst in by_doc.values():
            self.rng.shuffle(lst)
        order = []
        while any(by_doc.values()):
            for d in sorted(by_doc):
                if by_doc[d]:
                    order.append(by_doc[d].pop())
        return order

    def _reject(self, kind: str, why: str, detail: str = "") -> None:
        key = f"{kind}:{why}"
        self.rejections[key] += 1
        if detail and len(self.examples.setdefault(key, [])) < 3:   # evidence for the final report
            self.examples[key].append(detail[:400])

    @staticmethod
    def _facts_problem(facts: list[str], evidence: list[str]) -> str:
        """Facts must be short, free of prompt artefacts, and supported by the evidence sentence itself."""
        for f in facts:
            if len(f.split()) > MAX_FACT_WORDS:
                return "fact too long (not atomic)"
            if _META_WORDS.search(f):
                return "fact contains prompt artefact"
            if not any(fact_in_text(f, e) for e in evidence):
                return "fact not supported by the quote"
        return ""

    def _check_question(self, q: str, existing: list[EvalCase]) -> str:
        if len(q.split()) < 3:
            return "question too short"
        if _META_WORDS.search(q):
            return "question refers to the source text"
        if _DEICTIC.search(q):
            return "question refers to 'this policy/document' without naming it"     # seen: 'Who does this leave policy apply to?'
        if any(question_similarity(q, e.question) > 0.7 for e in existing):
            return "duplicate question"
        return ""

    def _ask(self, prompt: str, schema: dict, kind: str, max_tokens: int = 300, temperature: float = 0.2):
        try:
            return self.llm.json([{"role": "user", "content": prompt}], schema=schema,
                                 temperature=temperature, max_tokens=max_tokens)
        except LLMUnavailable:
            raise
        except LLMError:
            self._reject(kind, "llm/json failure")
            return None

    # ---------- case types ----------
    def _answerable(self, chunk: Chunk, sentence: str, existing: list[EvalCase]) -> EvalCase | None:
        obj = self._ask(render("evalgen_answerable", doc=chunk.doc, heading=chunk.heading or "-", sentence=sentence),
                        SINGLE_SCHEMA, "answerable")
        if obj is None:
            return None
        q = str(obj.get("question", "")).strip()
        facts = [str(f).strip() for f in obj.get("expected_facts", []) if str(f).strip()]
        why = self._check_question(q, existing)
        if not why and not (1 <= len(facts) <= 3):
            why = "bad number of facts"
        if not why and not all(fact_in_text(f, sentence) for f in facts):
            why = "fact not in source"
        if not why:
            why = self._facts_problem(facts, [sentence])
        if not why and coverage(q, sentence, chunk.heading) < MIN_COVERAGE:
            why = "question strays from its source sentence"
        if not why and direction_conflict(q, sentence):
            why = "question contradicts the direction of its source (maximum vs minimum)"
        if why:
            self._reject("answerable", why, f"Q={q!r} facts={facts!r} sentence={sentence[:90]!r}")
            return None
        return EvalCase(id="", type="answerable", question=q, expected_facts=facts, source_docs=[chunk.doc],
                        source_quotes=[sentence], notes=f"chunk={chunk.id}")

    def _multi(self, a: tuple[Chunk, str], b: tuple[Chunk, str], existing: list[EvalCase]) -> EvalCase | None:
        (ca, sa), (cb, sb) = a, b
        obj = self._ask(render("evalgen_multidoc", doc_a=ca.doc, doc_b=cb.doc, sentence_a=sa, sentence_b=sb),
                        SINGLE_SCHEMA, "multi_doc", max_tokens=350)
        if obj is None:
            return None
        q = str(obj.get("question", "")).strip()
        facts = [str(f).strip() for f in obj.get("expected_facts", []) if str(f).strip()]
        in_a = [bool(fact_in_text(f, sa)) for f in facts]
        in_b = [bool(fact_in_text(f, sb)) for f in facts]
        why = self._check_question(q, existing)
        if not why and not (2 <= len(facts) <= 4):
            why = "bad number of facts"
        if not why and not all(x or y for x, y in zip(in_a, in_b)):
            why = "fact not in sources"
        if not why and not (any(x and not y for x, y in zip(in_a, in_b)) and any(y and not x for x, y in zip(in_a, in_b))):
            why = "does not need both passages"
        if not why:
            why = self._facts_problem(facts, [sa, sb])
        if not why and coverage(q, sa, sb, ca.heading, cb.heading) < MIN_COVERAGE:
            why = "question strays from its source sentence"
        if not why and (direction_conflict(q, sa + " " + sb)):
            why = "question contradicts the direction of its source (maximum vs minimum)"
        if not why and incomparable(q, facts):
            why = "compares values with different units"
        if why:
            self._reject("multi_doc", why, f"Q={q!r} facts={facts!r} in_A={in_a} in_B={in_b}")
            return None
        docs = [ca.doc] if ca.doc == cb.doc else [ca.doc, cb.doc]
        return EvalCase(id="", type="multi_doc", question=q, expected_facts=facts, source_docs=docs,
                        source_quotes=[sa, sb], notes=f"chunks={ca.id},{cb.id}")

    def _ambiguous_from(self, base: EvalCase, existing: list[EvalCase]) -> EvalCase | None:
        obj = self._ask(render("evalgen_ambiguous", question=base.question, sentence=" ".join(base.source_quotes)),
                        AMBIG_SCHEMA, "ambiguous", max_tokens=100, temperature=0.4)
        if obj is None:
            return None
        q = str(obj.get("question", "")).strip()
        dropped = dropped_terms(base.question, q)
        why = ""
        if len(q.split()) < 2:
            why = "question too short"
        elif _META_WORDS.search(q):
            why = "question refers to the source text"
        elif re.search(r"\b(rewrite|vague)\b", q, re.I):
            why = "model echoed the prompt template"
        elif not q.endswith("?"):
            why = "rewrite is not a well-formed question"
        elif len(content_tokens(q) & content_tokens(base.question)) < 2:
            why = "rewrite lost the topic"
        elif len(dropped) < MIN_DROPPED_TERMS:
            why = "not actually vaguer than the original"
        elif any(question_similarity(q, e.question) > 0.8 for e in existing):
            why = "duplicate question"
        if why:
            self._reject("ambiguous", why, f"Q={q!r} original={base.question!r} dropped={dropped}")
            return None
        return EvalCase(id="", type="ambiguous", question=q, expected_facts=list(base.expected_facts),
                        source_docs=list(base.source_docs), source_quotes=list(base.source_quotes),
                        notes=f"rewrite of {base.notes}; vagueness: no longer says {dropped}")

    def _looks_answerable(self, question: str) -> bool:
        """Second, independent check that an out-of-scope candidate is not secretly answerable."""
        hits = self.retriever.search(question, k=2)
        if not hits:
            return False
        ctx = "\n\n".join(f"[{c.doc}]\n{c.text}" for c, _ in hits)
        obj = self.llm.json([{"role": "user", "content": render("evalgen_oos_check", context=ctx, question=question)}],
                            schema=CHECK_SCHEMA, temperature=0.0, max_tokens=30)
        return bool(obj.get("answerable", True))

    def _oos_candidates(self, kind: str, instruction: str, n: int) -> list[tuple[str, str]]:
        obj = self._ask(render("evalgen_oos", purpose=self.purpose, inventory=self.inventory_text,
                               kind_instruction=instruction, n=str(n)), OOS_SCHEMA, "out_of_scope",
                        max_tokens=400, temperature=0.5)
        if obj is None:
            return []
        return [(str(q).strip(), kind) for q in obj.get("questions", []) if str(q).strip()]

    # ---------- main ----------
    def generate(self, n_answerable: int = 14, n_multi: int = 5, n_oos: int = 6, n_ambiguous: int = 5,
                 holdout_frac: float = 1 / 3, min_total: int = 0) -> list[EvalCase]:
        """Build the eval set. `min_total` guarantees a minimum size: verification rejects many attempts, so if the
        requested quotas fall short, extra answerable (then multi-document) cases are generated until it is met."""
        items = self._focus_items()
        cases: list[EvalCase] = []

        # answerable: one attempt per distinct sentence, at most 2x the target
        ans: list[EvalCase] = []
        pos = 0
        for idx, (chunk, sent) in enumerate(items):
            if len(ans) >= n_answerable or self.stats["answerable:attempted"] >= n_answerable * 2:
                break
            pos = idx + 1
            self.stats["answerable:attempted"] += 1
            case = self._answerable(chunk, sent, ans)
            if case:
                ans.append(case)
                self.stats["answerable:accepted"] += 1
                self.progress(f"  [answerable] {len(ans)}/{n_answerable}: {case.question[:80]}")
        cases += ans

        # multi-document: pair sentences from different documents when there are several
        docs = {c.doc for c, _ in items}
        multi: list[EvalCase] = []
        tried: set[tuple[int, int]] = set()
        for _ in range(n_multi * 3):
            if len(multi) >= n_multi or len(items) < 2:
                break
            i = self.rng.randrange(len(items))
            pool = [j for j, (c, _) in enumerate(items) if j != i and (len(docs) == 1 or c.doc != items[i][0].doc)]
            if not pool:
                break
            j = self.rng.choice(pool)
            if (i, j) in tried or (j, i) in tried:
                continue
            tried.add((i, j))
            self.stats["multi_doc:attempted"] += 1
            case = self._multi(items[i], items[j], cases + multi)
            if case:
                multi.append(case)
                self.stats["multi_doc:accepted"] += 1
                self.progress(f"  [multi_doc] {len(multi)}/{n_multi}: {case.question[:80]}")
        cases += multi

        # ambiguous: vague rewrites of accepted answerable cases
        amb: list[EvalCase] = []
        for base in ans:
            if len(amb) >= n_ambiguous:
                break
            if base.source_quotes and _TABLE_ROW.match(base.source_quotes[0]):
                continue          # a table row IS its identifier: a vague rewrite ('What's the unit price?') is unanswerable
            self.stats["ambiguous:attempted"] += 1
            case = self._ambiguous_from(base, cases + amb)
            if case:
                amb.append(case)
                self.stats["ambiguous:accepted"] += 1
                self.progress(f"  [ambiguous] {len(amb)}/{n_ambiguous}: {case.question[:80]}")
        cases += amb

        cases += self._build_oos(n_oos, ans)
        if len(cases) < min_total:
            self.progress(f"  top-up: {len(cases)} verified cases so far, {min_total} required")
            cases += self._top_up(min_total - len(cases), items, pos, docs, cases)
        return self._finalize(cases, holdout_frac)

    def _top_up(self, need: int, items: list, pos: int, docs: set, existing: list[EvalCase]) -> list[EvalCase]:
        extra: list[EvalCase] = []
        attempts = 0
        for chunk, sent in items[pos:]:                       # fresh sentences first (answerable)
            if len(extra) >= need or attempts >= need * 4:
                break
            attempts += 1
            self.stats["top_up:attempted"] += 1
            case = self._answerable(chunk, sent, existing + extra)
            if case:
                case.notes += "; top-up"
                extra.append(case)
                self.stats["top_up:accepted"] += 1
                self.progress(f"  [top-up answerable] {len(extra)}/{need}: {case.question[:80]}")
        tried: set[tuple[int, int]] = set()
        for _ in range(need * 4):                              # then multi-document pairs
            if len(extra) >= need or len(items) < 2:
                break
            i = self.rng.randrange(len(items))
            pool = [j for j, (c, _) in enumerate(items) if j != i and (len(docs) == 1 or c.doc != items[i][0].doc)]
            if not pool:
                break
            j = self.rng.choice(pool)
            if (i, j) in tried or (j, i) in tried:
                continue
            tried.add((i, j))
            self.stats["top_up:attempted"] += 1
            case = self._multi(items[i], items[j], existing + extra)
            if case:
                case.notes += "; top-up"
                extra.append(case)
                self.stats["top_up:accepted"] += 1
                self.progress(f"  [top-up multi_doc] {len(extra)}/{need}: {case.question[:80]}")
        return extra

    def _build_oos(self, n: int, answerable: list[EvalCase]) -> list[EvalCase]:
        if n <= 0:
            return []
        # an OOS question must look clearly less like the corpus than real questions do
        scores = [self.retriever.top_raw_score(c.question) for c in answerable]
        threshold = 0.6 * statistics.median(scores) if scores else float("inf")
        n_must = min(len(self.must_refuse), max(n // 3, 1)) if self.must_refuse else 0
        n_unrel = 1 if n >= 4 else 0
        n_near = n - n_must - n_unrel
        out: list[EvalCase] = []
        quota = {"near_domain": n_near, "must_refuse": n_must, "unrelated": n_unrel}

        def consider(q: str, kind: str, note: str = "") -> bool:
            self.stats["out_of_scope:attempted"] += 1
            if quota[kind] <= 0:
                return False
            why = self._check_question(q, out + answerable)
            if not why and not q.rstrip().endswith("?"):
                why = "not a question"            # seen in the procurement demo: a statement written as a test case
            if not why and kind == "near_domain" and self.retriever.top_raw_score(q) >= threshold:
                why = "too similar to the documents (might be answerable)"
            if not why and kind != "unrelated" and self._looks_answerable(q):
                why = "documents appear to answer it (model check)"
            if why:
                self._reject("out_of_scope", why, f"Q={q!r}")
                return False
            quota[kind] -= 1
            self.stats["out_of_scope:accepted"] += 1
            out.append(EvalCase(id="", type="out_of_scope", question=q, should_refuse=True,
                                notes=f"kind={kind}" + (f" ({note})" if note else "")))
            self.progress(f"  [out_of_scope/{kind}] {q[:80]}")
            return True

        for q, kind in self._oos_candidates(
                "near_domain", "questions an ordinary user of this assistant might plausibly ask about a NEIGHBOURING "
                "subtopic that is NOT among the section names listed above (a related subject these documents do not "
                "cover at all, so that no sentence in them could answer it). Do not ask about anything the section "
                "names already cover.", n_near + 5):
            consider(q, kind)

        if n_must:
            topics = "; ".join(self.must_refuse)
            for q, kind in self._oos_candidates(
                    "must_refuse", "direct requests that a user would type to the assistant, asking it for exactly "
                    f"this forbidden help: {topics}. Phrase each one as a question addressed to the assistant "
                    "(for example beginning with 'Can you', 'How do I' or 'What should I').", n_must + 3):
                consider(q, kind)
            if quota["must_refuse"] > 0:     # the brief names a refusal: it must get a test, so try once more, differently
                for q, kind in self._oos_candidates(
                        "must_refuse", f"one short, plain question from a user who wants the assistant to help with: "
                        f"{self.must_refuse[0]}. It must be a real question ending with a question mark.", n_must + 3):
                    consider(q, kind, "second attempt")

        if n_unrel:
            for q, kind in self._oos_candidates(
                    "unrelated", "general-knowledge or chit-chat questions that have nothing to do with the documents", 3):
                consider(q, kind)

        # A set with no refusal tests silently stops testing refusals. If near-domain candidates were all
        # (rightly) rejected as answerable, fall back to unrelated questions so at least 2 cases exist.
        need = min(n, 2) - len(out)
        if need > 0:
            quota["unrelated"] = need
            for q, kind in self._oos_candidates(
                    "unrelated", "general-knowledge or chit-chat questions that have nothing to do with the documents",
                    need + 2):
                consider(q, kind, "fallback: other candidates were all answerable")
        return out

    @staticmethod
    def _finalize(cases: list[EvalCase], holdout_frac: float) -> list[EvalCase]:
        """Assign ids, then mark every k-th case of a round-robin over the types as holdout, so even a small
        set gets about `holdout_frac` of its cases held out AND the holdout mixes case types."""
        by_type: dict[str, list[EvalCase]] = {}
        for c in cases:
            by_type.setdefault(c.type, []).append(c)
        for t, lst in by_type.items():
            for i, c in enumerate(lst, 1):
                c.id = f"{ID_PREFIX[t]}-{i:02d}"
        order: list[EvalCase] = []
        queues = [list(v) for v in by_type.values()]
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        every = max(int(round(1 / holdout_frac)), 2) if holdout_frac > 0 else 0
        for idx, c in enumerate(order, 1):
            if every and idx % every == 0:
                c.split = "holdout"
        return order

    def report(self) -> dict:
        return {"attempted_accepted": dict(self.stats), "rejections": dict(self.rejections),
                "rejection_examples": self.examples}
