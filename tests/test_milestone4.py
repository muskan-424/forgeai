import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge.agent_runtime import Agent, load_definition  # noqa: E402
from forge.critic import classify, classify_all, dominant  # noqa: E402
from forge.factory import run_factory  # noqa: E402
from forge.judge import Judge  # noqa: E402
from forge.llm import LLM  # noqa: E402
from forge.loop import run_loop  # noqa: E402
from forge.optimizer import LIBRARY, apply_patch, propose, retrieval_steps  # noqa: E402
from forge.report import build_report  # noqa: E402
from forge.retrieval import tokenize  # noqa: E402
from forge.schemas import (AgentAnswer, AgentDefinition, AgentTrace, CaseResult, Chunk, Citation,  # noqa: E402
                           EvalCase, RetrievalConfig)
from test_milestone2 import make_mock  # noqa: E402
from test_milestone3 import DOCS, HR_BRIEF, fake  # noqa: E402


# ------------------------------------------------------------------ helpers
def mk_case(cid, typ="answerable", split="dev", refuse=False):
    quote = f"The limit for item {cid} is 7 days per year."
    return EvalCase(id=cid, type=typ, question=f"What is the limit for item {cid}?",
                    expected_facts=[] if refuse else ["7 days"], source_docs=[] if refuse else ["d.md"],
                    source_quotes=[] if refuse else [quote], should_refuse=refuse, split=split)


def NO_LLM_JUDGE():
    return Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": []})), use_llm=False)


def result_for(case, *, retrieved=True, refused=False, answer=None, cite=True, leak=False, error=""):
    quote = case.source_quotes[0] if case.source_quotes else "unrelated text"
    chunks = [Chunk(id="c", doc="d.md", text=quote if retrieved else "something unrelated entirely")]
    if refused:
        ans = AgentAnswer(answer="I can't help with that; it is outside what my documents cover.", refused=True)
        if leak:
            ans.citations = [Citation(doc="d.md", quote=quote)]
            ans.answer = "Probably 7 days. The documents do not say more than that, so I cannot help any further here."
        verified = [True] if leak else []
    else:
        ans = AgentAnswer(answer=answer if answer is not None else "The limit is 7 days.",
                          citations=[Citation(doc="d.md", quote=quote)] if cite else [])
        verified = [True] if cite else []
    t = AgentTrace(question=case.question, retrieved=chunks, answer=ans, citations_verified=verified, error=error)
    return CaseResult(case=case, trace=t, verdict=NO_LLM_JUDGE().judge(case, t))


# ------------------------------------------------------------------ critic
def test_critic_classifies_every_failure_category():
    c = mk_case("a")
    oos = mk_case("o", "out_of_scope", refuse=True)
    assert classify(result_for(c))["category"] is None if False else classify(result_for(c)) is None   # a pass
    cases = {
        "retrieval_miss/refused": result_for(c, retrieved=False, refused=True),
        "retrieval_miss/answered": result_for(c, retrieved=False, answer="The limit is 9 days."),
        "wrong_refusal": result_for(c, refused=True),
        "hallucination/ungrounded": result_for(c, answer="The limit is 7 days, plus 14 days of notice.", cite=True),
        "prompt_defect": result_for(c, answer="The limit is a few days."),
        "hallucination/out_of_scope": result_for(oos, answer="Sure, it is 3.", cite=False),
        "hallucination/leak": result_for(oos, refused=True, leak=True),
    }
    got = {k: classify(v) for k, v in cases.items()}
    for key, k in got.items():
        cat, _, sub = key.partition("/")
        assert k["category"] == cat, (key, k)
    assert got["retrieval_miss/refused"]["subtype"] == "refused" and got["retrieval_miss/answered"]["subtype"] == "answered"
    assert got["hallucination/ungrounded"]["subtype"] == "ungrounded"
    err = result_for(c, answer="", error="Ollama call failed")
    err.verdict.passed = False
    assert classify(err)["category"] == "agent_error"


def test_dominant_category_ties_break_by_priority_and_ignore_agent_errors():
    f = lambda cat: {"category": cat, "case_id": "x", "subtype": ""}   # noqa: E731
    assert dominant([f("prompt_defect"), f("prompt_defect"), f("hallucination")]) == "prompt_defect"
    assert dominant([f("prompt_defect"), f("hallucination")]) == "hallucination"        # tie -> priority
    assert dominant([f("hallucination"), f("retrieval_miss")]) == "retrieval_miss"
    assert dominant([f("agent_error")]) is None and dominant([]) is None


# ------------------------------------------------------------------ optimizer
def base_def(**kw):
    d = AgentDefinition(name="t", system_prompt="You are a test agent.", rules=["Answer only from the context."],
                        docs_dir=".", retrieval=RetrievalConfig(top_k=3, chunk_size=500, chunk_overlap=60), temperature=0.1)
    return d.model_copy(update=kw)


def fails(cat, sub="", cid="x"):
    return [{"category": cat, "subtype": sub, "case_id": cid}]


def test_retrieval_ladder_walks_forward_and_never_repeats():
    d, history = base_def(), []
    seen = []
    for _ in range(12):                                          # the ladder is finite: it must run out
        p = propose(d, fails("retrieval_miss"), history, n_chunks=40)
        if p is None:
            break
        seen.append(p["rationale"])
        history.append(p)
        d = apply_patch(d, p)
    assert d.retrieval.top_k == 8 and d.retrieval.use_embeddings and d.retrieval.chunk_size > 500
    assert len(seen) == len(set(seen)) and len(seen) >= 3
    assert propose(d, fails("retrieval_miss"), history, n_chunks=40) is None             # ladder exhausted
    assert d.retrieval.chunk_size == 1500                           # capped
    tiny = [o for ops, _ in retrieval_steps(base_def(), n_chunks=3) for o in ops]
    assert not any(o["path"] == "retrieval.use_embeddings" for o in tiny)                 # nothing to embed in 3 chunks


def test_behavioural_escalation_library_then_model_rule_then_temperature():
    d, history = base_def(), []
    p1 = propose(d, fails("wrong_refusal"), history, 10)
    assert p1["source"] == "library" and p1["ops"][0]["text"] == LIBRARY[("wrong_refusal", "")]
    history.append(p1)
    d = apply_patch(d, p1)
    llm = LLM(backend="mock", mock_fn=lambda m, s: json.dumps(
        {"rule": "When a sentence of the context states the answer, give that answer instead of refusing."}))
    p2 = propose(d, fails("wrong_refusal"), history, 10, {"x": "Q: ... | A: refused"}, llm)
    assert p2["source"] == "model" and "instead of refusing" in p2["ops"][0]["text"]
    history.append(p2)
    d = apply_patch(d, p2)
    p3 = propose(d, fails("wrong_refusal"), history, 10, {}, llm)
    assert p3["ops"] == [{"op": "set_param", "path": "temperature", "value": 0.0}]
    history.append(p3)
    assert propose(apply_patch(d, p3), fails("wrong_refusal"), history, 10, {}, llm) is None


def test_model_written_rule_is_rejected_if_useless_and_hallucination_gets_subtype_rules():
    d = base_def(rules=[LIBRARY[("hallucination", "out_of_scope")]])
    hist = [{"ops": [{"op": "append_rule", "text": LIBRARY[("hallucination", "out_of_scope")]}]}]
    echo = LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"rule": LIBRARY[("hallucination", "out_of_scope")]}))
    p = propose(d, fails("hallucination", "out_of_scope"), hist, 10, {}, echo)              # restates an existing rule
    assert p["ops"][0]["path"] == "temperature"
    both = fails("hallucination", "out_of_scope", "a") + fails("hallucination", "ungrounded", "b")
    p = propose(base_def(), both, [], 10)
    assert {o["text"] for o in p["ops"]} == {LIBRARY[("hallucination", "out_of_scope")], LIBRARY[("hallucination", "ungrounded")]}
    assert set(p["addresses"]) == {"a", "b"}


def test_apply_patch_is_typed_and_returns_a_copy():
    d = base_def()
    new = apply_patch(d, {"ops": [{"op": "set_param", "path": "retrieval.top_k", "value": 6},
                                  {"op": "append_rule", "text": "Be brief."}]})
    assert new.retrieval.top_k == 6 and new.rules[-1] == "Be brief." and d.retrieval.top_k == 3 and len(d.rules) == 1
    with pytest.raises(ValueError):
        apply_patch(d, {"ops": [{"op": "set_param", "path": "system_prompt", "value": "x"}]})
    with pytest.raises(ValueError):
        apply_patch(d, {"ops": [{"op": "delete_everything"}]})
    with pytest.raises(Exception):
        apply_patch(d, {"ops": [{"op": "set_param", "path": "retrieval.top_k", "value": "many"}]})


# ------------------------------------------------------------------ loop with a scripted agent
class FakeAgent:
    """Behaviour depends on the definition, so patches have real, checkable effects."""
    def __init__(self, d, cases, tick, oos_break_rule=None):
        self.d, self.by_q, self.tick, self.oos_break_rule = d, {c.question: c for c in cases}, tick, oos_break_rule

    def _has(self, text):
        return any(text in r for r in self.d.rules)

    def ask(self, q):
        t = self._ask(q)
        t.latency_s = 10.0
        return t

    def _ask(self, q):
        self.tick[0] += 10
        c = self.by_q[q]
        if c.id.startswith("retr"):                                   # needs a bigger top_k to find the answer
            found = self.d.retrieval.top_k >= 5
            return result_for(c, retrieved=found, refused=not found).trace
        if c.id.startswith("wref"):                                   # refuses unless told to answer
            return result_for(c, refused=not self._has("Refuse only when")).trace
        if c.id.startswith("oos"):                                    # answers only if a rule breaks it
            broken = self.oos_break_rule and self._has(self.oos_break_rule)
            return (result_for(c, answer="Sure, 3.", cite=False) if broken else result_for(c, refused=True)).trace
        return result_for(c).trace


def loop_setup(oos_break_rule=None, tick=None):
    tick = tick if tick is not None else [0.0]
    dev = [mk_case("retr1"), mk_case("wref1"), mk_case("oos1", "out_of_scope", refuse=True)]
    hold = [mk_case("retrH", split="holdout"), mk_case("okH", split="holdout")]
    factory = lambda d: FakeAgent(d, dev + hold, tick, oos_break_rule)   # noqa: E731
    return dev, hold, factory, tick


def test_loop_fixes_failures_one_targeted_patch_at_a_time_and_logs_everything(tmp_path):
    dev, hold, factory, tick = loop_setup()
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, None, factory,
                   progress=lambda *_: None, clock=lambda: tick[0])
    log = res["changelog"]
    assert [c["category"] for c in log] == ["retrieval_miss", "wrong_refusal"]                # priority order
    assert [c["decision"] for c in log] == ["kept", "kept"]
    assert log[0]["fixed"] == ["retr1"] and log[0]["broken"] == []
    assert (log[0]["dev_passed_before"], log[0]["dev_passed_after"]) == (1, 2)
    assert (log[1]["dev_passed_before"], log[1]["dev_passed_after"]) == (2, 3)
    assert log[0]["failures_addressed"] == ["retr1"] and "top_k 3 -> 5" in log[0]["what_changed"] or "retrieval.top_k = 5" in log[0]["what_changed"]
    assert res["stop_reason"] == "all dev cases pass"
    assert res["best_definition"].retrieval.top_k == 5 and any("Refuse only when" in r for r in res["best_definition"].rules)
    sc = res["scorecards"]
    assert [s["iteration"] for s in sc] == [0, 1, 2, "final"]
    assert sc[0]["dev"]["overall"]["passed"] == 1 and sc[-1]["dev"]["overall"]["passed"] == 3
    assert sc[0]["holdout"]["overall"]["passed"] == 1 and sc[-1]["holdout"]["overall"]["passed"] == 2   # holdout improved too
    for f in ("changelog.json", "scorecards.json", "final_agent_definition.yaml", "iterations/iter_0/results.json",
              "iterations/iter_2/agent_definition.yaml"):
        assert (tmp_path / f).exists(), f


def test_loop_reverts_a_patch_that_breaks_more_than_it_fixes_and_stops_early(tmp_path):
    dev, hold, factory, tick = loop_setup(oos_break_rule="Refuse only when")                    # the fix breaks oos1
    dev = [c for c in dev if c.id != "retr1"]                                                    # leave only wref1 + oos1
    d0 = base_def()
    res = run_loop(d0, dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, None, factory,
                   progress=lambda *_: None, clock=lambda: tick[0])
    log = res["changelog"]
    assert log[0]["decision"] == "reverted" and log[0]["fixed"] == ["wref1"] and log[0]["broken"] == ["oos1"]
    assert "broke ['oos1']" in log[0]["decision_reason"]
    assert [c["decision"] for c in log] == ["reverted", "reverted"]                             # 2nd: temperature, no effect
    assert log[1]["what_changed"] == "temperature = 0.0"
    assert "early stop" in res["stop_reason"]
    assert res["best_definition"] is d0 and d0.rules == ["Answer only from the context."]       # nothing leaked in
    assert res["scorecards"][-1]["dev"]["overall"]["passed"] == res["scorecards"][0]["dev"]["overall"]["passed"] == 1


def test_loop_respects_the_time_budget_and_the_iteration_cap(tmp_path):
    dev, hold, factory, tick = loop_setup()
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path / "a"), None, 3, 60, factory,
                   progress=lambda *_: None, clock=lambda: tick[0])
    assert "budget" in res["stop_reason"] and res["changelog"] == []                             # 50s used, 33s needed
    dev, hold, factory, tick = loop_setup()
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path / "b"), None, 1, None, factory,
                   progress=lambda *_: None, clock=lambda: tick[0])
    assert len(res["changelog"]) == 1 and "maximum of 1" in res["stop_reason"]


def test_loop_with_nothing_to_fix_changes_nothing(tmp_path):
    ok = [mk_case("ok1"), mk_case("ok2")]
    tick = [0.0]
    res = run_loop(base_def(), ok, [], NO_LLM_JUDGE(), str(tmp_path), None, 3, None,
                   lambda d: FakeAgent(d, ok, tick), progress=lambda *_: None, clock=lambda: tick[0])
    assert res["changelog"] == [] and res["stop_reason"] == "all dev cases pass"


# ------------------------------------------------------------------ report
def test_report_states_numbers_with_intervals_failures_and_cautions(tmp_path):
    dev, hold, factory, tick = loop_setup(oos_break_rule="Refuse only when")
    dev = [c for c in dev if c.id != "retr1"]
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, None, factory,
                   progress=lambda *_: None, clock=lambda: tick[0])
    ctx = {"name": "t", "requirements": {"brief": "b", "scope": "s", "audience": "a", "tone": "t", "must_answer": ["x"],
                                         "must_refuse": ["y"]},
           "groundability": {"entries": [{"topic": "x", "status": "supported", "note": ""}], "warnings": []},
           "notes": {"tool_decisions": [{"tool": "calculator", "attached": False, "reason": "no evidence"}]},
           "eval_report": {"attempted_accepted": {"answerable:attempted": 4, "answerable:accepted": 2},
                           "rejections": {"answerable:duplicate question": 2}},
           "calibration": {"probes": 5, "accuracy": 1.0, "false_pass": 0, "false_fail": 0, "llm_false_accepts": 0,
                           "by_kind": {"gold": "2/2"}},
           "cases": dev + hold, "loop": res, "final_definition": res["best_definition"], "total_seconds": 120.0,
           "usage": {"agent": {"calls": 3, "cache_hits": 0, "seconds": 30.0}}}
    md, js = build_report(ctx)
    for needle in ("# Quality report", "## Verdict", "95% CI", "## Per-iteration scorecard", "## Changelog",
                   "REVERTED", "## What still fails and why", "wref1", "Root cause", "How much to trust",
                   "Known limits", "same model family"):
        assert needle in md, needle
    assert js["residual_failures"] and js["stop_reason"] and js["changelog"][0]["decision"] == "reverted"


def test_report_warns_that_a_perfect_score_on_few_cases_is_weak_evidence(tmp_path):
    ok = [mk_case("ok1"), mk_case("ok2")]
    tick = [0.0]
    res = run_loop(base_def(), ok, [], NO_LLM_JUDGE(), str(tmp_path), None, 3, None,
                   lambda d: FakeAgent(d, ok, tick), progress=lambda *_: None, clock=lambda: tick[0])
    ctx = {"name": "t", "requirements": {"brief": "b", "scope": "s", "audience": "a", "tone": "t", "must_answer": [], "must_refuse": []},
           "groundability": {"entries": [], "warnings": []}, "notes": {"tool_decisions": []}, "eval_report": None,
           "calibration": None, "cases": ok, "loop": res, "final_definition": res["best_definition"], "total_seconds": 1.0,
           "usage": {}}
    md, _ = build_report(ctx)
    assert "weak evidence, not proof" in md and "No patch was needed" in md and "Calibration was skipped" in md


# ------------------------------------------------------------------ the whole factory, end to end, mocked
def router():
    m3, m2 = fake(), make_mock()

    def fn(messages, schema):
        p = messages[0]["content"]
        if p.startswith(("INTAKE", "GROUNDABILITY", "TOOL NECESSITY", "SYSTEM PROMPT WRITER")):
            return m3(messages, schema)
        if p.startswith("PROMPT RULE WRITER"):
            return json.dumps({"rule": "Prefer the sentence that names the exact item the user asked about."})
        return m2(messages, schema)
    return fn


def extractive(messages, schema):
    """A mock agent: answers with the context sentence that best overlaps the question, refuses if none does."""
    user = messages[-1]["content"]
    ctx = user.split("CONTEXT:\n", 1)[1].split("\n\nQUESTION:", 1)[0]
    question = user.split("QUESTION:", 1)[1].strip()
    qt = set(tokenize(question))
    best, score = "", 0
    for line in ctx.splitlines():
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            sc = len(qt & set(tokenize(s)))
            if len(s.split()) >= 4 and sc > score:
                best, score = s, sc
    if score >= 2:
        return json.dumps({"answer": best, "citations": [{"doc": "d", "quote": best}], "refused": False, "refusal_reason": ""})
    return json.dumps({"answer": "I can't help with that; it is outside what my documents cover.", "citations": [],
                       "refused": True, "refusal_reason": "not in context"})


def test_full_factory_end_to_end_with_mocks_writes_every_artifact(tmp_path):
    out = tmp_path / "run"
    res = run_factory(HR_BRIEF, DOCS, str(out), auto=True, n_eval=(3, 1, 2, 1), max_iters=2,
                      llm=LLM(backend="mock", mock_fn=router()), judge=NO_LLM_JUDGE(),
                      agent_factory=lambda d: Agent(d, LLM(backend="mock", mock_fn=extractive)), progress=lambda *_: None)
    for f in ("requirements.md", "groundability.json", "agent_definition.yaml", "generation_notes.json", "evals.json",
              "evals_generation_report.json", "judge_calibration.json", "changelog.json", "scorecards.json",
              "final_agent_definition.yaml", "report.md", "report.json", "iterations/iter_0/results.json"):
        assert (out / f).exists(), f
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## Verdict" in md and "## Changelog" in md and "## What still fails and why" in md
    cases = json.loads((out / "evals.json").read_text(encoding="utf-8"))
    assert {"answerable", "out_of_scope"} <= {c["type"] for c in cases}
    final = load_definition(out / "final_agent_definition.yaml")                              # loads from its own location
    assert Path(final.docs_dir).resolve() == Path(DOCS).resolve() and final.tools == []
    sc = res["loop"]["scorecards"][0]["dev"]["overall"]
    assert sc["n"] > 0 and sc["passed"] >= 1                                                  # the extractive agent is not useless
    assert json.loads((out / "judge_calibration.json").read_text(encoding="utf-8"))["accuracy"] == 1.0


def test_junk_model_written_rules_are_rejected():
    """Real run: the 3B model wrote a rule that scripted a refusal sentence for answerable questions."""
    d = base_def(rules=[LIBRARY[("prompt_defect", "")]])
    hist = [{"ops": [{"op": "append_rule", "text": LIBRARY[("prompt_defect", "")]}]}]
    junk = ["If the answer matches a fact in the context, explicitly state 'The context contains this information, "
            "but does not provide a refusal or correction, so I cannot answer this question'.",
            "When the context contains the figure, you must say that you cannot answer and refuse politely here.",
            "x " * 60]
    for rule in junk:
        llm = LLM(backend="mock", mock_fn=lambda m, s, r=rule: json.dumps({"rule": r}))
        p = propose(d, fails("prompt_defect"), hist, 10, {"x": "Q: .."}, llm)
        assert p["ops"][0]["path"] == "temperature", rule                       # junk rejected -> next step on the ladder
    good = LLM(backend="mock", mock_fn=lambda m, s: json.dumps(
        {"rule": "Prefer the sentence that names the exact item the user asked about, and quote its figure."}))
    assert propose(d, fails("prompt_defect"), hist, 10, {"x": "Q: .."}, good)["source"] == "model"


def test_holdout_is_spread_across_types_even_for_tiny_sets():
    from forge.evalgen import EvalGenerator
    cases = [EvalCase(id="", type=t, question=f"q{i}", should_refuse=(t == "out_of_scope"))
             for i, t in enumerate(["answerable"] * 4 + ["multi_doc"] * 2 + ["ambiguous", "out_of_scope", "out_of_scope"])]
    out = EvalGenerator._finalize(cases, 1 / 3)
    hold = [c for c in out if c.split == "holdout"]
    assert len(out) == 9 and len(hold) == 3                      # a third, not 'whatever each type rounds to'
    assert len({c.type for c in hold}) >= 2                      # and it mixes case types
    assert len({c.id for c in out}) == 9


def test_report_flags_a_small_noisy_gain_and_a_model_written_rule(tmp_path):
    ok = [mk_case("ok1"), mk_case("ok2")]
    tick = [0.0]
    res = run_loop(base_def(), ok, [], NO_LLM_JUDGE(), str(tmp_path), None, 3, None,
                   lambda d: FakeAgent(d, ok, tick), progress=lambda *_: None, clock=lambda: tick[0])
    res["changelog"] = [{"iteration": 2, "category": "prompt_defect", "source": "model", "what_changed": "+ rule: x",
                         "why": "y", "failures_addressed": ["a"], "fixed": ["a"], "broken": [], "dev_passed_before": 5,
                         "dev_passed_after": 6, "dev_total": 6, "decision": "kept", "decision_reason": "net gain on dev cases",
                         "seconds": 1.0}]
    ctx = {"name": "t", "requirements": {"brief": "b", "scope": "s", "audience": "a", "tone": "t", "must_answer": [], "must_refuse": []},
           "groundability": {"entries": [], "warnings": []}, "notes": {"tool_decisions": []}, "eval_report": None,
           "calibration": None, "cases": ok, "loop": res, "final_definition": res["best_definition"], "total_seconds": 1.0, "usage": {}}
    md, _ = build_report(ctx)
    assert "Check iteration 2's kept patch" in md and "within normal run-to-run noise" in md and "written by the 3B model" in md


# ------------------------------------------------------------------ citation handling (found in the HR demo baseline)
def test_quote_verifier_accepts_a_section_label_but_not_rewording():
    """Real demo failures: '6. Unpaid leave: <real sentences>' is cosmetic; 'Employees receive...' for
    'Shore-based employees receive...' is a genuinely reworded quote."""
    from forge.textmatch import quote_verified
    ctx = ["Employees with at least 2 years of service may request unpaid leave of up to 3 months. "
           "Unpaid leave requires approval from the department director.",
           "Shore-based employees receive 25 days of paid annual leave per leave year (1 January to 31 December)."]
    assert quote_verified("Unpaid leave requires approval from the department director.", ctx)                 # verbatim
    assert quote_verified("6. Unpaid leave: Employees with at least 2 years of service may request unpaid leave of "
                          "up to 3 months. Unpaid leave requires approval from the department director.", ctx)  # label + real sentences
    assert not quote_verified("Employees receive 25 days of paid annual leave per leave year.", ctx)           # reworded
    assert not quote_verified("6. Unpaid leave: Employees may request unpaid leave of up to 6 months.", ctx)   # changed content
    assert not quote_verified("", ctx) and not quote_verified("3. Header: ok", ctx)


def test_critic_separates_citation_defects_from_content_hallucination():
    c = mk_case("a")
    def failing(answer, verified):
        t = result_for(c, answer=answer).trace
        t.citations_verified = verified
        return CaseResult(case=c, trace=t, verdict=NO_LLM_JUDGE().judge(c, t))
    cite = classify(failing("The limit is 7 days.", [False]))                    # right answer, citation not verbatim
    assert (cite["category"], cite["subtype"]) == ("hallucination", "citation")
    content = classify(failing("The limit is 7 days, plus 14 days of notice.", [True]))   # invented figure
    assert (content["category"], content["subtype"]) == ("hallucination", "ungrounded")
    wrong = classify(failing("The limit is a few days.", [False]))               # wrong answer AND bad citation
    assert wrong["subtype"] == "ungrounded"


def test_optimizer_answers_a_citation_defect_with_a_quoting_rule_not_a_numbers_rule():
    p = propose(base_def(), fails("hallucination", "citation"), [], 10)
    assert p["ops"][0]["text"] == LIBRARY[("hallucination", "citation")] and "word for word" in p["ops"][0]["text"]
    assert "number" not in p["ops"][0]["text"].lower()


def test_agent_runtime_verifies_label_prefixed_quotes():
    import json as _json
    from forge.agent_runtime import Agent as _Agent
    d = load_definition(Path(__file__).resolve().parent.parent / "examples" / "sample_agent.yaml")
    def answer(quote):
        obj = {"answer": "25 days.", "refused": False, "refusal_reason": "", "citations": [{"doc": "leave_policy.md", "quote": quote}]}
        return _Agent(d, LLM(backend="mock", mock_fn=lambda m, s: _json.dumps(obj))).ask("How many days of annual leave do I get?")
    ok = answer("1. Annual Leave: Shore staff are entitled to 25 days of paid annual leave per calendar year.")
    assert ok.citations_verified == [True]
    bad = answer("Employees are entitled to 25 days of paid annual leave per calendar year.")   # a word changed
    assert bad.citations_verified == [False]


def test_cache_hits_report_the_time_they_replayed(tmp_path, monkeypatch):
    """A cached run must not look free: the replayed model time is accounted for separately."""
    from forge import config
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(LLM, "_ollama_chat", lambda self, m, t, sc, mt: ("hello", 10, 5))
    clock = iter([100.0, 142.5, 200.0, 200.0])
    monkeypatch.setattr("forge.llm.time.time", lambda: next(clock))
    a = LLM(model="x", backend="ollama")
    assert a.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert a.usage.calls == 1 and abs(a.usage.seconds - 42.5) < 1e-6 and a.usage.cache_hits == 0
    b = LLM(model="x", backend="ollama")                                           # second run: identical request
    assert b.chat([{"role": "user", "content": "hi"}]) == "hello"
    assert b.usage.calls == 0 and b.usage.cache_hits == 1 and abs(b.usage.cached_seconds - 42.5) < 1e-6


def test_a_stalled_call_cannot_eat_the_budget_and_agent_errors_are_retried(tmp_path):
    """Real failure: one call took 9,310 s (machine stalled), the 75-minute budget was gone, the loop never ran."""
    dev, hold, factory, tick = loop_setup()
    class Stalling(FakeAgent):
        def ask(self, q):
            t = super().ask(q)
            if "retr1" in q:
                t.latency_s = 9310.0                               # the stall
            return t
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, 1500, lambda d: Stalling(d, dev + hold, tick),
                   progress=lambda *_: None, clock=lambda: tick[0])
    assert len(res["changelog"]) >= 1                              # stall counted as 600 s (not 9,310 s): 1,500 s budget survived


def test_infrastructure_errors_are_retried_once_before_judging():
    from forge.evalrun import run_eval
    c = mk_case("ok1")
    calls = {"n": 0}
    class Flaky:
        def ask(self, q):
            calls["n"] += 1
            if calls["n"] == 1:
                return AgentTrace(question=q, error="Ollama call failed (Read timed out)")
            return result_for(c).trace
    out = run_eval(Flaky(), [c], NO_LLM_JUDGE(), progress=lambda *_: None, sleep=lambda s: None)
    assert calls["n"] == 2 and out[0].verdict.passed


# ------------------------------------------------------------------ infrastructure failures must not be blamed on a patch
def test_agent_errors_are_retried_with_backoff_before_a_case_is_scored():
    from forge.evalrun import run_eval
    c = mk_case("ok1")
    waits, calls = [], {"n": 0}

    class Down:
        def ask(self, q):
            calls["n"] += 1
            if calls["n"] <= 2:
                return AgentTrace(question=q, error="Ollama call failed (connection refused)")
            return result_for(c).trace
    out = run_eval(Down(), [c], NO_LLM_JUDGE(), progress=lambda *_: None, sleep=waits.append)
    assert waits == [15, 45] and calls["n"] == 3 and out[0].verdict.passed              # server came back on the third try
    calls["n"] = -100                                                                   # never recovers
    out = run_eval(Down(), [c], NO_LLM_JUDGE(), progress=lambda *_: None, sleep=lambda s: None)
    assert not out[0].verdict.passed and "agent error" in out[0].verdict.reasons[0]


class ServerDownAfterPatch(FakeAgent):
    """Real incident: Ollama restarted during iteration 3. With top_k >= 5 every call fails like a dead server."""
    def _ask(self, q):
        if self.d.retrieval.top_k >= 5:
            return AgentTrace(question=q, error="Ollama call failed (connection refused)")
        return super()._ask(q)


def test_an_iteration_with_server_errors_is_inconclusive_not_blamed_on_the_patch(tmp_path):
    dev, hold, factory, tick = loop_setup()
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, None,
                   lambda d: ServerDownAfterPatch(d, dev + hold, tick), progress=lambda *_: None,
                   clock=lambda: tick[0], sleep=lambda s: None)
    log = res["changelog"]
    assert log and log[0]["decision"] == "inconclusive" and log[0]["fixed"] == [] and log[0]["broken"] == []
    assert log[0]["infrastructure_errors"] and "INCONCLUSIVE" in log[0]["decision_reason"]
    # the same patch is tried again (never recorded as tried) and the early-stop counter was not advanced
    assert len(log) == 3 and {c["what_changed"] for c in log} == {log[0]["what_changed"]}
    assert "early stop" not in res["stop_reason"] and res["best_definition"].retrieval.top_k == 3
    assert res["scorecards"][-1]["dev"]["overall"]["passed"] == res["scorecards"][0]["dev"]["overall"]["passed"]


def test_a_broken_baseline_stops_the_loop_and_the_report_says_inconclusive(tmp_path):
    dev, hold, factory, tick = loop_setup()

    class AllDown:
        def __init__(self, d): pass
        def ask(self, q): return AgentTrace(question=q, error="Ollama has no model 'x' (HTTP 404)")
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path), None, 3, None, AllDown,
                   progress=lambda *_: None, clock=lambda: 0.0, sleep=lambda s: None)
    assert res["changelog"] == [] and "baseline is not trustworthy" in res["stop_reason"]
    dev, hold, factory, tick = loop_setup()
    res = run_loop(base_def(), dev, hold, NO_LLM_JUDGE(), str(tmp_path / "b"), None, 3, None,
                   lambda d: ServerDownAfterPatch(d, dev + hold, tick), progress=lambda *_: None,
                   clock=lambda: tick[0], sleep=lambda s: None)
    ctx = {"name": "t", "requirements": {"brief": "b", "scope": "s", "audience": "a", "tone": "t", "must_answer": [], "must_refuse": []},
           "groundability": {"entries": [], "warnings": []}, "notes": {"tool_decisions": []}, "eval_report": None,
           "calibration": None, "cases": dev + hold, "loop": res, "final_definition": res["best_definition"],
           "total_seconds": 1.0, "usage": {}}
    md, _ = build_report(ctx)
    assert "INCONCLUSIVE" in md and "says nothing about its patch" in md
