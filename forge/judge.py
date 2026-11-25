"""Judge: deterministic checks first, LLM only for facts the deterministic matcher could not confirm.

Bias defences
- position bias:      pointwise scoring (one answer at a time), never A-vs-B.
- verbosity bias:     the LLM only answers "is THIS fact stated?"; length/style are explicitly irrelevant,
                      and deterministic matching decides first.
- self-preference:    the judge model is a different family from the agent model (config.JUDGE_MODEL).
- instability:        two passes with reversed fact order; any disagreement => 'uncertain' => counted as NOT stated.
- trust but verify:   calibrate() scores synthetic gold / wrong / padded / wrongly-refused answers
                      and reports the judge's own accuracy.
"""
from __future__ import annotations


from . import config
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .schemas import AgentAnswer, AgentTrace, Chunk, Citation, EvalCase, JudgeVerdict
from .textmatch import SOFT_REFUSAL, fact_in_text, numbers, quote_verified

FACTS_SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "stated": {"type": "boolean"}},
    "required": ["n", "stated"]}}}, "required": ["results"]}

MAX_REFUSAL_WORDS = 45
AMBIGUOUS_RECALL = 0.5   # ambiguous questions: a good answer to the likely meaning covers at least half the facts


class Judge:
    def __init__(self, llm: LLM | None = None, use_llm: bool = True):
        self.llm = llm or LLM(model=config.JUDGE_MODEL)
        self.use_llm = use_llm

    # ------------------------------------------------------------------ main
    def judge(self, case: EvalCase, trace: AgentTrace) -> JudgeVerdict:
        v = JudgeVerdict(case_id=case.id)
        a = trace.answer

        if trace.error and not a.answer.strip() and not a.refused:
            v.reasons.append(f"agent error: {trace.error[:120]}")
            v.refusal_correct = False
            return v

        verified_idx = [i for i, ok in enumerate(trace.citations_verified) if ok]
        soft = (not a.refused and not verified_idx and bool(SOFT_REFUSAL.search(a.answer)))
        refused = a.refused or soft
        if soft:
            v.reasons.append("soft refusal (refused flag not set)")

        if case.should_refuse:
            leak = refused and self._refusal_leaks(a)
            v.refusal_correct = refused and not leak
            v.correct = v.refusal_correct
            v.grounded = v.refusal_correct  # answering an out-of-scope question is, by definition, ungrounded
            v.passed = v.refusal_correct
            if not refused:
                v.reasons.append("hallucination: answered an out-of-scope question")
            elif leak:
                v.reasons.append("hallucination: refusal flag set but the answer still asserts content")
            return v

        # ----- answerable / multi_doc / ambiguous -----
        v.refusal_correct = not refused
        if refused:
            v.correct = False
            v.grounded = True
            v.reasons.append("wrong refusal: refused an answerable question")
            return v

        self._grade_facts(case, trace, v)
        threshold = AMBIGUOUS_RECALL if case.type == "ambiguous" else 1.0
        recall = (sum(v.fact_hits) / len(v.fact_hits)) if v.fact_hits else 0.0
        v.correct = recall >= threshold and not v.uncertain
        if not v.correct:
            v.reasons.append(f"facts matched {sum(v.fact_hits)}/{len(v.fact_hits)}"
                             + (" (uncertain judge, counted as miss)" if v.uncertain else ""))
        v.grounded = self._grounded(case, trace, v)
        v.passed = v.correct and v.grounded and v.refusal_correct
        return v

    @staticmethod
    def _refusal_leaks(a: AgentAnswer) -> bool:
        """A clean refusal is short and cites nothing. Citations or a long answer mean content leaked through."""
        return bool(a.citations) or len(a.answer.split()) > MAX_REFUSAL_WORDS

    # ------------------------------------------------------------------ facts
    def _grade_facts(self, case: EvalCase, trace: AgentTrace, v: JudgeVerdict) -> None:
        answer = trace.answer.answer
        residual: list[int] = []
        for i, fact in enumerate(case.expected_facts):
            m = fact_in_text(fact, answer)
            v.fact_hits.append(bool(m))
            v.fact_methods.append(m or "miss")
            if not m:
                residual.append(i)
        if residual and self.use_llm and answer.strip():
            res = self._llm_facts(case.question, answer, [case.expected_facts[i] for i in residual])
            v.llm_used = True
            for i, r in zip(residual, res):
                if r is True:
                    v.fact_hits[i], v.fact_methods[i] = True, "llm"
                elif r is None:
                    v.fact_methods[i] = "uncertain"
                    v.uncertain = True

    def _llm_facts(self, question: str, answer: str, facts: list[str]) -> list[bool | None]:
        """Two passes (original and reversed fact order). True only if both say stated; None if they disagree."""
        passes: list[list[bool]] = []
        for order in (list(range(len(facts))), list(reversed(range(len(facts))))):
            listing = "\n".join(f"{k + 1}. {facts[j]}" for k, j in enumerate(order))
            prompt = render("judge_facts", question=question, answer=answer, facts=listing)
            try:
                obj = self.llm.json([{"role": "user", "content": prompt}], schema=FACTS_SCHEMA,
                                    temperature=0.0, max_tokens=200)
                got = {int(r["n"]): bool(r["stated"]) for r in obj.get("results", [])}
                row = [False] * len(facts)
                for k, j in enumerate(order):
                    row[j] = got.get(k + 1, False)
                passes.append(row)
            except LLMUnavailable:
                raise
            except (LLMError, KeyError, ValueError, TypeError):
                return [None] * len(facts)
        a, b = passes
        return [x if x == y else None for x, y in zip(a, b)]

    # ------------------------------------------------------------------ grounding
    def _grounded(self, case: EvalCase, trace: AgentTrace, v: JudgeVerdict) -> bool:
        a = trace.answer
        used_tools = bool(trace.tool_calls)
        ok = True
        if not used_tools:
            if not a.citations:
                v.reasons.append("ungrounded: no citations")
                ok = False
            elif not all(trace.citations_verified):
                v.reasons.append("ungrounded: citation quote not found in retrieved context (fabricated)")
                ok = False
            elif case.source_docs:
                cited = self._cited_docs(trace)
                if not cited & set(case.source_docs):
                    v.reasons.append(f"ungrounded: cited {sorted(cited)} but expected {case.source_docs}")
                    ok = False
        # numbers in the answer must come from the context, tool output or the question itself
        support = " ".join([c.text for c in trace.retrieved] + [str(t.get("result", "")) for t in trace.tool_calls]
                           + [trace.question])
        extra = numbers(a.answer) - numbers(support)
        if extra:
            v.reasons.append(f"ungrounded: numbers not in context: {sorted(extra)}")
            ok = False
        return ok

    @staticmethod
    def _cited_docs(trace: AgentTrace) -> set[str]:
        docs: set[str] = set()
        for cit, ok in zip(trace.answer.citations, trace.citations_verified):
            if not ok:
                continue
            docs |= {c.doc for c in trace.retrieved if quote_verified(cit.quote, [c.text])}
        return docs

    # ------------------------------------------------------------------ calibration
    def calibrate(self, cases: list[EvalCase], per_kind: int = 3, progress=print) -> dict:
        """Score synthetic answers whose correct verdict is known, and report how often the judge is right."""
        answerable = [c for c in cases if not c.should_refuse and c.expected_facts and c.source_quotes]
        oos = [c for c in cases if c.should_refuse]
        probes: list[tuple[str, EvalCase, AgentTrace, bool]] = []  # (kind, case, trace, should_pass)
        for i, c in enumerate(answerable[:per_kind]):
            other = answerable[(i + 1) % len(answerable)]
            probes.append(("gold", c, _trace(c, "; ".join(c.expected_facts) + ".", [c]), True))
            filler = ("This is a common question and there are several considerations to keep in mind. "
                      "Organisations usually publish guidance, and staff are encouraged to read it carefully, "
                      "speak to their manager, and follow the official process. ") * 4
            probes.append(("padded_gold", c, _trace(c, filler + " ".join(c.expected_facts) + ". " + filler, [c]), True))
            if other.expected_facts != c.expected_facts:
                probes.append(("wrong_facts", c, _trace(c, "; ".join(other.expected_facts) + ".", [c, other]), False))
            probes.append(("padded_wrong", c, _trace(c, filler, [c]), False))
            probes.append(("wrong_refusal", c, _trace(c, "", [c], refused=True), False))
        for c in oos[:per_kind]:
            probes.append(("oos_refused", c, _trace(c, "", [], refused=True), True))
            probes.append(("oos_answered", c, _trace(c, "Sure, the answer is 42 and you can find it in section 7.", [], refused=False), False))
            hedged = _trace(c, "You must request this through the HR portal at least 14 days in advance. "
                               "The documents do not mention a specific procedure.", [], refused=True)
            hedged.answer.citations = [Citation(doc="doc", quote="Leave must be requested at least 14 days in advance.")]
            hedged.citations_verified = [True]
            probes.append(("oos_hedged", c, hedged, False))

        by_kind: dict[str, list[bool]] = {}
        false_pass = false_fail = llm_false_accepts = 0
        for kind, c, tr, should in probes:
            got = self.judge(c, tr)
            right = got.passed == should
            by_kind.setdefault(kind, []).append(right)
            if got.passed and not should:
                false_pass += 1
            if not got.passed and should:
                false_fail += 1
            if not should and any(m == "llm" for m in got.fact_methods):
                llm_false_accepts += sum(1 for m in got.fact_methods if m == "llm")
            progress(f"  calibration {kind:<13} expected={'pass' if should else 'fail'} got={'pass' if got.passed else 'fail'}")
        n = len(probes)
        correct = sum(sum(v) for v in by_kind.values())
        return {"probes": n, "accuracy": round(correct / n, 3) if n else None, "false_pass": false_pass,
                "false_fail": false_fail, "llm_false_accepts": llm_false_accepts,
                "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()}}


def _trace(case: EvalCase, answer: str, context_cases: list[EvalCase], refused: bool = False) -> AgentTrace:
    """Synthetic AgentTrace for calibration: retrieved context = the cases' source quotes."""
    chunks = [Chunk(id=f"cal#{i}", doc=(cc.source_docs[0] if cc.source_docs else "doc"), text=" ".join(cc.source_quotes))
              for i, cc in enumerate(context_cases)]
    cites, verified = [], []
    if not refused and case.source_quotes:
        cites = [Citation(doc=case.source_docs[0] if case.source_docs else "doc", quote=case.source_quotes[0])]
        verified = [True]
    return AgentTrace(question=case.question, retrieved=chunks,
                      answer=AgentAnswer(answer=answer, citations=cites, refused=refused,
                                         refusal_reason="calibration" if refused else ""),
                      citations_verified=verified)
