"""Optimizer: turn the dominant failure category into ONE targeted, typed patch of the agent definition.

Patches are lists of whitelisted operations (set_param / append_rule); the optimizer never regenerates the
definition. Retrieval failures walk a fixed ladder of retrieval changes. Behavioural failures first get a
proven library rule; if that exact rule is already present (it did not fix the failures), the escalation is a
rule written by the model from the failing examples, and after that a lower temperature."""
from __future__ import annotations

import re

from .critic import EXPLANATION, dominant
from .llm import LLM, LLMError, LLMUnavailable
from .prompts import render
from .schemas import AgentDefinition
from .textmatch import norm, question_similarity

RULE_SCHEMA = {"type": "object", "properties": {"rule": {"type": "string"}}, "required": ["rule"]}

ALLOWED_PARAMS = {"refusal_gate", "temperature", "retrieval.top_k", "retrieval.use_embeddings", "retrieval.bm25_weight",
                  "retrieval.chunk_size", "retrieval.chunk_overlap", "retrieval.min_score"}

LIBRARY = {
    ("hallucination", "out_of_scope"):
        "If the question asks about something the CONTEXT does not state, refuse, even if it sounds related to the "
        "documents' topic; never borrow a rule or figure from a different topic.",
    ("hallucination", "ungrounded"):
        "Every number, name and date in your answer must appear in the CONTEXT or in a tool result; "
        "if you cannot find it there, refuse.",
    ("hallucination", "citation"):
        "Each citation quote must be one or more whole sentences copied exactly, word for word, from the CONTEXT; "
        "never reword, shorten or merge sentences inside a quote.",
    ("wrong_refusal", ""):
        "Refuse only when no sentence in the CONTEXT answers the question; if a sentence answers it, even in "
        "different words, answer using that sentence.",
    ("prompt_defect", ""):
        "Answer the exact question asked, giving the exact figure and unit from the CONTEXT, and answer every "
        "part of a multi-part question.",
}


def _op_sig(op: dict) -> str:
    if op["op"] == "set_param":
        return f"param:{op['path']}={op['value']}"
    return "rule:" + norm(op["text"])[:80]


def patch_signatures(patch: dict) -> list[str]:
    return [_op_sig(o) for o in patch["ops"]]


# ------------------------------------------------------------------ retrieval ladder
def retrieval_steps(d: AgentDefinition, n_chunks: int) -> list[tuple[list[dict], str]]:
    r = d.retrieval
    steps: list[tuple[list[dict], str]] = []
    if r.top_k < 8:
        k = min(r.top_k + 2, 8)
        steps.append(([{"op": "set_param", "path": "retrieval.top_k", "value": k}],
                      f"retrieve more passages (top_k {r.top_k} -> {k})"))
    if not r.use_embeddings and n_chunks > 6:
        steps.append(([{"op": "set_param", "path": "retrieval.use_embeddings", "value": True},
                       {"op": "set_param", "path": "retrieval.bm25_weight", "value": 0.5}],
                      "add embedding similarity to the lexical search (hybrid retrieval)"))
    if r.chunk_size < 1500:
        size = min(int(r.chunk_size * 1.5) // 50 * 50, 1500)
        steps.append(([{"op": "set_param", "path": "retrieval.chunk_size", "value": size},
                       {"op": "set_param", "path": "retrieval.chunk_overlap", "value": int(size * 0.15)}],
                      f"larger chunks so answers are not split (chunk_size {r.chunk_size} -> {size})"))
    if r.top_k < 8:
        steps.append(([{"op": "set_param", "path": "retrieval.top_k", "value": 8}], "retrieve up to 8 passages"))
    return steps


# ------------------------------------------------------------------ behavioural rules
def _tailored_rule(category: str, examples: list[str], rules: list[str], llm: LLM | None) -> str:
    if llm is None:
        return ""
    try:
        obj = llm.json([{"role": "user", "content": render(
            "optimizer_rule", category=category, explanation=EXPLANATION[category],
            examples="\n".join(examples[:3]) or "(none)", rules="\n".join(f"- {r}" for r in rules))}],
            schema=RULE_SCHEMA, temperature=0.2, max_tokens=120)
    except LLMUnavailable:
        raise
    except LLMError:
        return ""
    rule = str(obj.get("rule", "")).strip()
    if not (25 <= len(rule) <= 260) or rule.startswith("{"):
        return ""
    # a 3B model sometimes writes junk (seen in a real run: a rule that scripts a refusal sentence for ANSWERABLE
    # questions). A rule must be one short, general instruction: no quoted script, no refusal wording unless the
    # failure is a hallucination.
    if re.search(r"['\"\u2018\u2019\u201c\u201d]", rule) or len(rule.split()) > 45:
        return ""
    if category != "hallucination" and re.search(r"\b(cannot|can't|unable to) (answer|help)\b", rule, re.I):
        return ""
    if any(question_similarity(rule, r) > 0.6 for r in rules):
        return ""                                   # restates an existing rule: no new information
    return rule


def propose(d: AgentDefinition, failures: list[dict], history: list[dict], n_chunks: int,
            examples: dict[str, str] | None = None, llm: LLM | None = None) -> dict | None:
    """Returns a patch {'category','ops','rationale','addresses','source'} or None if nothing new can be tried.
    `history` = every patch already tried (kept OR reverted), so the same step is never repeated."""
    cat = dominant(failures)
    if cat is None:
        return None
    tried = {s for h in history for s in patch_signatures(h)}
    targets = [f for f in failures if f["category"] == cat]
    addresses = [f["case_id"] for f in targets]

    if cat == "retrieval_miss":
        for ops, why in retrieval_steps(d, n_chunks):
            if not any(_op_sig(o) in tried for o in ops):
                return {"category": cat, "ops": ops, "rationale": why, "addresses": addresses, "source": "ladder"}
        return None

    if cat == "wrong_refusal" and any(f["subtype"] == "gate" for f in targets):
        # the gate blocked legitimate questions: switching it off is the fix
        if d.refusal_gate and "param:refusal_gate=False" not in tried:
            return {"category": cat, "ops": [{"op": "set_param", "path": "refusal_gate", "value": False}],
                    "rationale": "turn the refusal gate off: it blocked answerable questions",
                    "addresses": addresses, "source": "ladder"}
        return None

    existing = {norm(r) for r in d.rules}
    subtypes = sorted({f["subtype"] for f in targets})
    ops, why, source = [], [], "library"
    for st in subtypes:
        lib = LIBRARY.get((cat, st)) or LIBRARY.get((cat, ""))
        if lib is None:
            continue
        if norm(lib) not in existing and f"rule:{norm(lib)[:80]}" not in tried:
            ops.append({"op": "append_rule", "text": lib})
            why.append(f"add proven rule for {cat}/{st or 'general'}")
    if (not ops and cat == "hallucination" and "out_of_scope" in subtypes and d.refusal_topics
            and not d.refusal_gate and "param:refusal_gate=True" not in tried):
        # prompt rules did not make the model refuse: enforce it with a runtime classification before answering
        ops.append({"op": "set_param", "path": "refusal_gate", "value": True})
        why.append("enable the runtime refusal gate (rules alone did not make the agent refuse forbidden topics)")
        source = "ladder"
    if not ops:   # library wording already tried: escalate to a rule written from the failing examples
        ex = [examples[f["case_id"]] for f in targets if examples and f["case_id"] in examples]
        rule = _tailored_rule(cat, ex, d.rules, llm)
        if rule and f"rule:{norm(rule)[:80]}" not in tried:
            ops.append({"op": "append_rule", "text": rule})
            why.append(f"model-written rule from {len(ex)} failing example(s) for {cat}")
            source = "model"
    if not ops and d.temperature > 0.0 and "param:temperature=0.0" not in tried:
        ops.append({"op": "set_param", "path": "temperature", "value": 0.0})
        why.append("make answers deterministic (temperature 0)")
        source = "ladder"
    if not ops:
        return None
    return {"category": cat, "ops": ops, "rationale": "; ".join(why), "addresses": addresses, "source": source}


def apply_patch(d: AgentDefinition, patch: dict) -> AgentDefinition:
    """Return a patched COPY; unknown operations or parameters raise, so a patch can never do more than it says."""
    new = d.model_copy(deep=True)
    for op in patch["ops"]:
        if op["op"] == "append_rule":
            if norm(op["text"]) not in {norm(r) for r in new.rules}:
                new.rules.append(op["text"])
        elif op["op"] == "set_param":
            if op["path"] not in ALLOWED_PARAMS:
                raise ValueError(f"parameter not patchable: {op['path']}")
            target, _, field = op["path"].rpartition(".")
            obj = getattr(new, target) if target else new
            setattr(obj, field, op["value"])
        else:
            raise ValueError(f"unknown patch operation: {op['op']}")
    return AgentDefinition(**new.model_dump())      # re-validate the whole definition


def describe(patch: dict) -> str:
    parts = []
    for o in patch["ops"]:
        parts.append(f"{o['path']} = {o['value']}" if o["op"] == "set_param" else f"+ rule: {o['text']}")
    return "; ".join(parts)
