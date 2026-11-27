import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forge.agent_runtime import Agent, load_definition  # noqa: E402
from forge.critic import classify  # noqa: E402
from forge.generator import refusal_topics  # noqa: E402
from forge.llm import LLM  # noqa: E402
from forge.loop import run_loop  # noqa: E402
from forge.optimizer import LIBRARY, apply_patch, propose  # noqa: E402
from forge.report import build_report  # noqa: E402
from forge.schemas import CaseResult, Requirements  # noqa: E402
from test_milestone4 import NO_LLM_JUDGE, base_def, fails, mk_case, result_for  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_agent.yaml"
ANSWER = {"answer": "25 days.", "refused": False, "refusal_reason": "",
          "citations": [{"doc": "leave_policy.md", "quote": "25 days of paid annual leave"}]}


def gated_agent(gate_reply, gate=True, topics=("legal advice about contracts",)):
    """Agent whose model answers normally, and answers the gate prompt with `gate_reply` (a dict or raw text)."""
    calls = {"gate": 0, "main": 0}

    def fn(messages, schema):
        if messages[0]["content"].startswith("REFUSAL GATE"):
            calls["gate"] += 1
            return gate_reply if isinstance(gate_reply, str) else json.dumps(gate_reply)
        calls["main"] += 1
        return json.dumps(ANSWER)
    d = load_definition(SAMPLE).model_copy(update={"refusal_gate": gate, "refusal_topics": list(topics)})
    return Agent(d, LLM(backend="mock", mock_fn=fn)), calls


# ---------------------------------------------------------------- runtime
def test_gate_blocks_a_forbidden_question_without_running_the_main_model():
    agent, calls = gated_agent({"forbidden": True, "topic": "legal advice about contracts"})
    tr = agent.ask("Is clause 4 of our supplier contract legally enforceable?")
    assert tr.answer.refused and tr.answer.answer == agent.d.refusal_message and not tr.answer.citations
    assert tr.gate_blocked == "legal advice about contracts" and "forbidden topic" in tr.answer.refusal_reason
    assert calls == {"gate": 1, "main": 0} and tr.retrieved == [] and tr.error == ""       # a clean, cheap refusal


def test_gate_lets_other_questions_through_to_the_normal_path():
    agent, calls = gated_agent({"forbidden": False, "topic": ""})
    tr = agent.ask("What does the supplier contract say about annual leave days?")        # overlaps 'contracts': the model decides
    assert not tr.answer.refused and tr.answer.answer == "25 days." and tr.gate_blocked == ""
    assert calls == {"gate": 1, "main": 1}


def test_gate_is_off_by_default_and_needs_topics():
    agent, calls = gated_agent({"forbidden": True, "topic": "legal advice about contracts"}, gate=False)
    assert not agent.ask("How many days of annual leave do I get?").answer.refused and calls["gate"] == 0
    agent, calls = gated_agent({"forbidden": True, "topic": "x"}, gate=True, topics=())
    assert not agent.ask("How many days of annual leave do I get?").answer.refused and calls["gate"] == 0
    assert load_definition(SAMPLE).refusal_gate is False            # existing definitions are unaffected


def test_gate_ignores_a_yes_that_names_no_listed_topic_and_fails_open_on_errors():
    agent, calls = gated_agent({"forbidden": True, "topic": "something entirely different"})
    q = "What does the supplier contract say about annual leave days?"
    assert not agent.ask(q).answer.refused and calls["gate"] == 1                      # 'yes' without a real topic is ignored
    agent, calls = gated_agent("this is not json at all")
    tr = agent.ask(q)
    assert not tr.answer.refused and tr.error == "" and calls["main"] == 1            # an error never blocks a question


# ---------------------------------------------------------------- generator, critic, report
def test_refusal_topics_are_only_what_the_requester_forbade():
    req = Requirements(brief="b", must_refuse=["anything not covered by the provided documents", "legal advice about contracts",
                                               "pension contributions (not covered by the documents)"])
    assert refusal_topics(req) == ["legal advice about contracts"]          # gaps stay prompt rules, not gate topics
    assert refusal_topics(Requirements(brief="b", must_refuse=["anything not covered by the provided documents"])) == []


def test_critic_blames_the_gate_when_it_blocks_an_answerable_question():
    c = mk_case("a")
    t = result_for(c, refused=True).trace
    t.gate_blocked = "legal advice about contracts"
    cr = CaseResult(case=c, trace=t, verdict=NO_LLM_JUDGE().judge(c, t))
    k = classify(cr)
    assert (k["category"], k["subtype"]) == ("wrong_refusal", "gate") and "refusal gate blocked" in k["detail"]


# ---------------------------------------------------------------- optimizer ladder
def test_optimizer_enables_the_gate_after_the_proven_rule_fails_and_before_model_written_rules():
    d, history = base_def(refusal_topics=["legal advice about contracts"]), []
    f = fails("hallucination", "out_of_scope")
    p1 = propose(d, f, history, 10)
    assert p1["ops"][0]["text"] == LIBRARY[("hallucination", "out_of_scope")]
    history.append(p1)
    d = apply_patch(d, p1)
    llm = LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"rule": "Prefer refusing when unsure about the topic."}))
    p2 = propose(d, f, history, 10, {"x": "Q: .."}, llm)
    assert p2["ops"] == [{"op": "set_param", "path": "refusal_gate", "value": True}]         # gate comes BEFORE the model-written rule
    history.append(p2)
    d = apply_patch(d, p2)
    assert d.refusal_gate is True
    p3 = propose(d, f, history, 10, {"x": "Q: .."}, llm)
    assert p3["source"] == "model"                                                           # gate already on: escalate as before


def test_optimizer_never_proposes_the_gate_without_topics_and_switches_it_off_when_it_blocks_good_questions():
    d, history = base_def(), []                                                              # no refusal_topics
    p1 = propose(d, fails("hallucination", "out_of_scope"), history, 10)
    history.append(p1)
    p2 = propose(apply_patch(d, p1), fails("hallucination", "out_of_scope"), history, 10)
    assert p2 is None or p2["ops"][0].get("path") != "refusal_gate"
    on = base_def(refusal_gate=True, refusal_topics=["legal advice about contracts"])
    off = propose(on, fails("wrong_refusal", "gate"), [], 10)
    assert off["ops"] == [{"op": "set_param", "path": "refusal_gate", "value": False}]
    assert apply_patch(on, off).refusal_gate is False
    assert propose(base_def(refusal_gate=False), fails("wrong_refusal", "gate"), [], 10) is None


# ---------------------------------------------------------------- the loop, end to end with a scripted agent
class GateFake:
    """oos* cases are answered (a hallucination) unless the gate is on; ok* cases are blocked by a gate with false positives."""
    def __init__(self, d, cases, false_positive):
        self.d, self.by_q, self.fp = d, {c.question: c for c in cases}, false_positive

    def ask(self, q):
        c = self.by_q[q]
        if c.id.startswith("oos"):
            if self.d.refusal_gate:
                t = result_for(c, refused=True).trace
                t.gate_blocked = "legal advice about contracts"
            else:
                t = result_for(c, answer="Sure, 3.", cite=False).trace
        elif self.d.refusal_gate and self.fp:
            t = result_for(c, refused=True).trace
            t.gate_blocked = "legal advice about contracts"
        else:
            t = result_for(c).trace
        t.latency_s = 10.0
        return t


def run_gate_loop(tmp_path, false_positive):
    dev = [mk_case("oos1", "out_of_scope", refuse=True), mk_case("ok1"), mk_case("ok2")]
    d = base_def(refusal_topics=["legal advice about contracts"])
    res = run_loop(d, dev, [], NO_LLM_JUDGE(), str(tmp_path), None, 3, None, lambda x: GateFake(x, dev, false_positive),
                   progress=lambda *_: None, clock=lambda: 0.0)
    return d, dev, res


def test_loop_turns_the_gate_on_when_rules_fail_and_keeps_it_when_it_helps(tmp_path):
    d, dev, res = run_gate_loop(tmp_path, false_positive=False)
    log = res["changelog"]
    assert [c["decision"] for c in log] == ["reverted", "kept"]
    assert "+ rule:" in log[0]["what_changed"] and log[1]["what_changed"] == "refusal_gate = True"
    assert log[1]["fixed"] == ["oos1"] and log[1]["broken"] == []
    assert res["best_definition"].refusal_gate is True and res["stop_reason"] == "all dev cases pass"
    assert res["scorecards"][-1]["dev"]["overall"]["passed"] == 3


def test_loop_reverts_the_gate_when_it_blocks_legitimate_questions(tmp_path):
    d, dev, res = run_gate_loop(tmp_path, false_positive=True)
    gate_patch = [c for c in res["changelog"] if c["what_changed"] == "refusal_gate = True"][0]
    assert gate_patch["decision"] == "reverted" and set(gate_patch["broken"]) == {"ok1", "ok2"} and gate_patch["fixed"] == ["oos1"]
    assert res["best_definition"].refusal_gate is False              # the net-loss patch never reaches the final definition


def test_report_states_whether_the_gate_is_on(tmp_path):
    d, dev, res = run_gate_loop(tmp_path, false_positive=False)
    ctx = {"name": "t", "requirements": {"brief": "b", "scope": "s", "audience": "a", "tone": "t", "must_answer": [], "must_refuse": []},
           "groundability": {"entries": [], "warnings": []}, "notes": {"tool_decisions": []}, "eval_report": None,
           "calibration": None, "cases": dev, "loop": res, "final_definition": res["best_definition"], "total_seconds": 1.0, "usage": {}}
    md, _ = build_report(ctx)
    assert "**Refusal gate:** ON (forbidden topics checked before answering: legal advice about contracts)" in md
    assert "refusal_gate = True" in md


def test_gate_never_calls_the_model_for_questions_that_share_no_word_with_a_forbidden_topic():
    """Measured on the real model: unconstrained, it blocked quotations/payment/warranty questions as 'contract-related'."""
    agent, calls = gated_agent({"forbidden": True, "topic": "legal advice about contracts"})
    for q in ("How many days of annual leave do I get?",                          # (the sample corpus is HR text)
              "What is the meal allowance for domestic travel?",
              "Do I need a medical certificate for sick leave?"):
        tr = agent.ask(q)
        assert not tr.answer.refused and tr.gate_blocked == ""
    assert calls["gate"] == 0 and calls["main"] == 3                                   # no gate cost, no false blocks
    tr = agent.ask("What should I do if a supplier breaches the contract?")            # 'contract' overlaps -> model asked
    assert tr.answer.refused and calls["gate"] == 1


# ---------------------------------------------------------------- default: enforce refusals the requester stated
def test_gate_starts_on_only_when_the_brief_explicitly_forbids_something(tmp_path):
    """Real 20-case procurement run: the contract-breach case landed in the holdout, so the optimizer never saw it, the
    gate stayed off and the agent gave legal advice. A stated refusal must not depend on the dev/holdout split."""
    from forge.generator import explicit_refusals, generate_definition
    from test_milestone3 import DOCS, L
    stated = Requirements(brief="b", scope="s", audience="a", tone="t", must_answer=["annual leave"],
                          must_refuse=["legal advice about contracts", "pension (not covered by the documents)"])
    assert explicit_refusals(stated) == ["legal advice about contracts"]
    d, notes = generate_definition(stated, DOCS, L(), out_dir=str(tmp_path))
    assert d.refusal_gate is True and "explicitly forbids legal advice about contracts" in notes["refusal_gate"]
    assert d.refusal_topics == ["legal advice about contracts"]
    gaps_only = Requirements(brief="b", scope="s", audience="a", tone="t", must_answer=["annual leave"],
                             must_refuse=["anything not covered by the provided documents",
                                          "pension contributions (not covered by the documents)"])
    d2, notes2 = generate_definition(gaps_only, DOCS, L(), out_dir=str(tmp_path))
    assert d2.refusal_gate is False and d2.refusal_topics == []                           # gaps alone: no gate topics
    assert notes2["refusal_gate"].startswith("off")


# ---------------------------------------------------------------- deterministic tool pre-fetch (rehearsal finding)
def test_date_and_weather_questions_get_tool_results_before_the_model_answers(monkeypatch):
    """Rehearsal: the 3B model never called the attached tools; it refused the weather and invented a day count."""
    from datetime import date, timedelta
    from forge.schemas import ToolRef
    from forge.tools.open_meteo import OpenMeteo
    monkeypatch.setattr(OpenMeteo, "run", lambda self, args: f"{args['location']}, NL: 7.5 C, wind 22 km/h")
    seen = []

    def fn(messages, schema):
        seen.append(messages[-1]["content"])
        return json.dumps(ANSWER)
    d = load_definition(SAMPLE).model_copy(update={"tools": [ToolRef(name="current_date"), ToolRef(name="open_meteo")]})
    agent = Agent(d, LLM(backend="mock", mock_fn=fn))
    target = (date.today() + timedelta(days=108)).isoformat()
    tr = agent.ask(f"Our annual leave certificate expires on {target}. How many days remain until it expires?")
    assert f"108 days from today until {target}" in seen[-1] and "TOOL RESULTS" in seen[-1]
    assert [t["tool"] for t in tr.tool_calls] == ["current_date", "current_date"] and all(t["prefetched"] for t in tr.tool_calls)
    tr = agent.ask("What is the current weather at the terminal in Rotterdam?")
    assert tr.tool_calls[0] == {"tool": "open_meteo", "args": {"location": "Rotterdam"},
                                "result": "Rotterdam, NL: 7.5 C, wind 22 km/h", "prefetched": True}
    assert "Rotterdam, NL: 7.5 C" in seen[-1]
    tr = agent.ask("How many days of annual leave do I get?")                       # no trigger: nothing fetched
    assert tr.tool_calls == [] and "TOOL RESULTS" not in seen[-1]
    plain = Agent(load_definition(SAMPLE), LLM(backend="mock", mock_fn=fn))         # no tools attached: never fetches
    assert plain.ask("What is the weather in Rotterdam today?").tool_calls == []
