"""Critic: classify every failed case into exactly one root-cause category, by rules (no model call).

  retrieval_miss  the sentence(s) that answer the question never reached the model (or it refused for that reason)
  hallucination   the answer asserts things the context does not support, or content leaked through a refusal /
                  an out-of-scope question was answered
  wrong_refusal   the context DID contain the answer, but the agent refused
  prompt_defect   the context contained the answer, the agent answered grounded, but the answer is wrong/incomplete
  agent_error     the agent run itself failed (timeout, server error); no patch can fix that
"""
from __future__ import annotations

from collections import Counter

from .schemas import CaseResult
from .textmatch import norm

CATEGORIES = ("retrieval_miss", "hallucination", "wrong_refusal", "prompt_defect", "agent_error")
PRIORITY = ("retrieval_miss", "hallucination", "wrong_refusal", "prompt_defect")   # tie-break when counts are equal

EXPLANATION = {
    "retrieval_miss": "the passage that answers the question was not among the retrieved context",
    "hallucination": "the answer asserted content that the context does not support (or answered an out-of-scope question)",
    "wrong_refusal": "the context contained the answer but the agent refused",
    "prompt_defect": "the context contained the answer and the agent did not refuse, but the answer was wrong or incomplete",
    "agent_error": "the agent run failed before producing an answer",
}


def context_has_answer(result: CaseResult) -> bool:
    """Do ALL the expected supporting sentences occur in what the agent was given (retrieved text + tool output)?"""
    c, t = result.case, result.trace
    if not c.source_quotes:
        return False
    ctx = norm(" ".join(ch.text for ch in t.retrieved) + " " + " ".join(str(x.get("result", "")) for x in t.tool_calls))
    return all(norm(q) in ctx for q in c.source_quotes)


def classify(result: CaseResult) -> dict | None:
    """None if the case passed; otherwise {'category', 'subtype', 'detail'}."""
    v, t, c = result.verdict, result.trace, result.case
    if v.passed:
        return None
    reasons = " ".join(v.reasons)
    refused = t.answer.refused or "soft refusal" in reasons
    if t.error and not t.answer.answer.strip() and not refused:
        return {"category": "agent_error", "subtype": "", "detail": t.error[:140]}
    if c.should_refuse:
        return {"category": "hallucination", "subtype": "out_of_scope",
                "detail": "refusal leaked content" if refused else "answered an out-of-scope question"}
    has = context_has_answer(result)
    if refused:
        return ({"category": "wrong_refusal", "subtype": "", "detail": "context had the answer; agent refused"} if has else
                {"category": "retrieval_miss", "subtype": "refused", "detail": "answer not in retrieved context; agent refused"})
    if not has:
        return {"category": "retrieval_miss", "subtype": "answered", "detail": "answer not in retrieved context"}
    if not v.grounded:
        # answer facts right but the citation is not a verbatim quote: a quoting defect, not a content hallucination
        content = "numbers not in context" in reasons or not v.correct
        return {"category": "hallucination", "subtype": "ungrounded" if content else "citation",
                "detail": "; ".join(v.reasons)[:140]}
    return {"category": "prompt_defect", "subtype": "", "detail": "; ".join(v.reasons)[:140]}


def classify_all(results: list[CaseResult]) -> list[dict]:
    out = []
    for r in results:
        k = classify(r)
        if k:
            out.append({"case_id": r.case.id, "type": r.case.type, "question": r.case.question, **k})
    return out


def dominant(failures: list[dict]) -> str | None:
    """The category to fix next: most frequent, ties broken by PRIORITY. agent_error is never 'dominant'."""
    counts = Counter(f["category"] for f in failures if f["category"] != "agent_error")
    if not counts:
        return None
    return max(counts, key=lambda k: (counts[k], -PRIORITY.index(k)))
