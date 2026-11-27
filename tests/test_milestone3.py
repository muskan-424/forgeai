import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.agent_runtime import load_definition, save_definition  # noqa: E402
from forge.factory import run_stage1  # noqa: E402
from forge.generator import build_rules, retrieval_params, select_tools  # noqa: E402
from forge.groundability import GroundabilityError, check_groundability, tool_support  # noqa: E402
from forge.ingest import inventory  # noqa: E402
from forge.intake import MAX_QUESTIONS, interview  # noqa: E402
from forge.llm import LLM  # noqa: E402
from forge.schemas import Requirements  # noqa: E402

DOCS = str(Path(__file__).resolve().parent.parent / "examples" / "sample_docs")
HR_BRIEF = "I need an assistant that answers HR policy questions for shore staff. Keep a formal tone."


def fake(extract=None, verdicts=None, tools_needed=(), questions=None, persona=None):
    """Mock model that routes by the prompt's header line."""
    extract = extract if extract is not None else {
        "scope": "Answers HR policy questions", "audience": "shore staff", "tone": "formal",
        "topics": ["annual leave", "sick leave", "travel expenses"], "must_refuse": []}
    verdicts = verdicts or {}

    def fn(messages, schema):
        p = messages[0]["content"]
        if p.startswith("INTAKE EXTRACT"):
            return json.dumps(extract)
        if p.startswith("INTAKE QUESTIONS"):
            return json.dumps({"questions": questions or ["Do these policies apply to contractors?"]})
        if p.startswith("GROUNDABILITY CHECK"):
            topic = re.search(r"TOPIC: (.*)", p).group(1).strip()
            # a misbehaving model may also try to smuggle in a quote; the guard must ignore it
            return json.dumps({"coverage": verdicts.get(topic, "yes"), "evidence_quote": "FABRICATED QUOTE"})
        if p.startswith("TOOL NECESSITY CHECK"):
            tool = re.search(r"CANDIDATE TOOL: (\w+)", p).group(1)
            return json.dumps({"needed": tool in tools_needed, "reason": "needed" if tool in tools_needed else "docs suffice"})
        if p.startswith("SYSTEM PROMPT WRITER"):
            return json.dumps({"system_prompt": persona or "You are a formal assistant that helps shore staff with HR policy questions."})
        raise AssertionError("unexpected prompt: " + p[:60])
    return fn


def L(**kw):
    return LLM(backend="mock", mock_fn=fake(**kw))


# ---------------------------------------------------------------- intake
def test_intake_auto_asks_only_open_slots_and_records_assumptions():
    req = interview(HR_BRIEF, DOCS, L(), ask=None)
    asked = [c["slot"] for c in req.clarifications]
    assert "tone" not in asked and "audience" not in asked           # the brief already said these
    assert "must_refuse" in asked                                     # nothing was said about refusals
    assert len(req.clarifications) <= MAX_QUESTIONS
    assert req.tone == "formal" and req.audience == "shore staff"
    assert req.must_refuse == ["anything not covered by the provided documents"]
    assert any("must_refuse not specified" in a for a in req.assumptions)


def test_intake_does_not_trust_invented_requirements():
    invented = {"scope": "Answers HR policy questions", "audience": "shore staff", "tone": "playful and witty pirate",
                "topics": ["cryptocurrency mining"], "must_refuse": ["quantum physics"]}
    req = interview(HR_BRIEF, DOCS, L(extract=invented), ask=None)
    assert req.tone != "playful and witty pirate"                     # not in the brief -> treated as unstated
    assert "quantum physics" not in req.must_refuse
    assert "cryptocurrency mining" not in req.must_answer
    assert "tone" in [c["slot"] for c in req.clarifications]
    assert req.must_answer                                            # fell back to the documents' own sections


def test_intake_interactive_uses_answers_and_never_exceeds_five_questions():
    seen = []

    def ask(q):
        seen.append(q)
        return "refund policy; legal advice" if "refuse" in q else "ok"
    empty = {"scope": "", "audience": "", "tone": "", "topics": [], "must_refuse": []}
    req = interview("Build me a helper.", DOCS, L(extract=empty), ask=ask)
    assert len(seen) <= MAX_QUESTIONS and len(req.clarifications) <= MAX_QUESTIONS
    assert req.must_refuse == ["refund policy", "legal advice"]
    assert req.audience == "ok"
    # everything still unanswered gets a default, so downstream stages never see an empty slot
    assert req.must_answer and req.scope and req.tone and req.audience


# ---------------------------------------------------------------- groundability
def test_groundability_supported_unsupported_and_evidence_is_deterministic():
    req = Requirements(brief=HR_BRIEF, must_answer=["annual leave", "pension scheme", "sick leave certificate"])
    rep = check_groundability(req, DOCS, L(verdicts={"sick leave certificate": "no"}))
    st = {e["topic"]: e for e in rep["entries"]}
    corpus = " ".join(p.read_text(encoding="utf-8") for p in Path(DOCS).glob("*.md"))
    assert st["annual leave"]["status"] == "supported"
    assert "25 days" in st["annual leave"]["evidence"] and st["annual leave"]["evidence"] in corpus   # a real sentence
    assert all(e["evidence"] != "FABRICATED QUOTE" for e in rep["entries"])                          # model quotes ignored
    assert st["pension scheme"]["status"] == "unsupported"            # no lexical overlap at all, model never asked
    assert st["sick leave certificate"]["status"] == "partial"        # words are present but the model says 'no'
    assert rep["gaps"] == ["pension scheme"] and not rep["blocking"]
    assert "pension scheme (not covered by the documents)" in req.must_refuse   # never silently dropped
    assert any("UNSUPPORTED" in w for w in rep["warnings"])


def test_groundability_cautious_partial_with_full_lexical_coverage_counts_as_supported():
    req = Requirements(brief="x", must_answer=["Daily Allowances"])
    rep = check_groundability(req, DOCS, L(verdicts={"Daily Allowances": "partial"}))
    e = rep["entries"][0]
    assert e["status"] == "supported" and "cautious" in e["note"] and "meal allowance" in e["evidence"]


def test_groundability_lexical_floor_overrules_a_confident_model():
    req = Requirements(brief="x", must_answer=["annual leave quantum chromodynamics gluon confinement"])
    rep = check_groundability(req, DOCS, L())
    assert rep["entries"][0]["status"] == "unsupported" or rep["entries"][0]["status"] == "partial"
    assert rep["entries"][0]["status"] != "supported"


def test_groundability_tool_can_support_a_topic_and_blocks_only_when_nothing_is_supported():
    req = Requirements(brief="x", must_answer=["current weather forecast at sea"])
    rep = check_groundability(req, DOCS, L())
    assert rep["entries"][0]["status"] == "supported_by_tool" and rep["entries"][0]["tool"] == "open_meteo"
    assert not rep["blocking"]
    req = Requirements(brief="x", must_answer=["pension scheme", "company car policy"])
    rep = check_groundability(req, DOCS, L())
    assert rep["blocking"] and len(rep["gaps"]) == 2
    assert tool_support("supplier price list", inventory(DOCS)) is None   # csv tool needs a CSV; calculator cannot ground facts
    assert tool_support("total cost budget forecast", inventory(DOCS)) is None


# ---------------------------------------------------------------- generator
def csv_docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "parts.csv").write_text("part_no,description,unit_price\nV-100,Gate valve,120\nP-200,Pump seal,45\n", encoding="utf-8")
    (d / "suppliers.md").write_text("# Suppliers\n\nAll orders above 1000 euros need two quotations from approved suppliers.\n", encoding="utf-8")
    return str(d)


def test_hr_brief_attaches_no_tools_and_explains_why():
    req = Requirements(brief=HR_BRIEF, scope="HR policy questions", must_answer=["annual leave", "sick leave"])
    tools, decisions = select_tools(req, inventory(DOCS), L(tools_needed=("csv_lookup", "open_meteo", "calculator")))
    assert tools == []                                                  # an HR agent with a weather tool is a defect
    reasons = {d["tool"]: d["reason"] for d in decisions}
    assert "precondition failed" in reasons["csv_lookup"]
    assert "no evidence" in reasons["open_meteo"]
    assert all(not d["attached"] for d in decisions)


def test_procurement_brief_attaches_csv_lookup_with_config_and_nothing_else(tmp_path):
    docs = csv_docs(tmp_path)
    req = Requirements(brief="Assistant for ship spares procurement: look up part prices in the catalogue",
                       scope="spares procurement", must_answer=["part prices", "supplier rules"])
    tools, decisions = select_tools(req, inventory(docs), L(tools_needed=("csv_lookup",)))
    assert [t.name for t in tools] == ["csv_lookup"]
    assert tools[0].config == {"csv_path": "parts.csv"} and tools[0].justification
    assert not any(d["attached"] for d in decisions if d["tool"] in ("open_meteo", "current_date"))


def test_model_necessity_check_can_veto_a_tool_with_keyword_evidence(tmp_path):
    docs = csv_docs(tmp_path)
    req = Requirements(brief="Answer questions about stock levels of spare items", scope="stock levels",
                       must_answer=["item availability"])
    tools, decisions = select_tools(req, inventory(docs), L(tools_needed=()))
    assert tools == []
    assert any("model says documents suffice" in d["reason"] for d in decisions if d["tool"] == "csv_lookup")


def test_weather_tool_only_with_explicit_need():
    req = Requirements(brief="Port agent that also reports current weather and wind at the berth",
                       scope="port operations", must_answer=["berth weather"])
    tools, _ = select_tools(req, inventory(DOCS), L(tools_needed=("open_meteo",)))
    assert [t.name for t in tools] == ["open_meteo"]


def test_retrieval_params_follow_document_statistics(tmp_path):
    p = retrieval_params(DOCS, inventory(DOCS))
    assert p.top_k == 3 and 500 <= p.chunk_size <= 1200 and p.chunk_overlap < p.chunk_size
    docs = csv_docs(tmp_path)
    c = retrieval_params(docs, inventory(docs))
    assert c.bm25_weight >= 0.5
    big = tmp_path / "big"
    big.mkdir()
    (big / "a.md").write_text("\n\n".join(f"Section {i}\n\n" + ("Policy sentence about topic %d. " % i) * 30 for i in range(40)), encoding="utf-8")
    b = retrieval_params(str(big), inventory(str(big)))
    assert b.top_k >= 4


def test_rules_encode_clean_refusals_gaps_and_tool_usage():
    req = Requirements(brief="b", must_refuse=["legal advice", "pension scheme (not covered by the documents)"],
                       groundability=[{"topic": "pension scheme", "status": "unsupported"}])
    from forge.schemas import ToolRef
    rules = build_rules(req, [ToolRef(name="calculator", justification="x")])
    joined = "\n".join(rules)
    assert "A refusal must be clean" in joined                          # the lesson from the first baseline run
    assert "Refuse questions about: legal advice." in joined
    assert "The documents do not cover: pension scheme" in joined
    assert "use the calculator tool" in joined


# ---------------------------------------------------------------- end to end
def test_stage1_writes_all_artifacts_and_definition_loads(tmp_path):
    out = tmp_path / "run"
    res = run_stage1(HR_BRIEF, DOCS, str(out), auto=True, llm=L(), progress=lambda *_: None)
    for f in ("requirements.json", "requirements.md", "groundability.json", "agent_definition.yaml",
              "generation_notes.json", "llm_usage.json"):
        assert (out / f).exists(), f
    d = load_definition(out / "agent_definition.yaml")                 # docs_dir resolves relative to the definition
    assert Path(d.docs_dir).resolve() == Path(DOCS).resolve()
    assert d.tools == [] and d.rules and d.system_prompt.startswith("You are")
    assert "Groundability" in (out / "requirements.md").read_text(encoding="utf-8")


def test_stage1_blocks_when_documents_cannot_support_the_brief(tmp_path):
    out = tmp_path / "run"
    extract = {"scope": "Answers pension questions", "audience": "shore staff", "tone": "formal",
               "topics": ["pension scheme", "company car"], "must_refuse": []}
    with pytest.raises(GroundabilityError):
        run_stage1("pension scheme and company car questions for shore staff, formal tone", DOCS, str(out),
                   auto=True, llm=L(extract=extract), progress=lambda *_: None)
    assert (out / "groundability.json").exists() and not (out / "agent_definition.yaml").exists()
    res = run_stage1("pension scheme and company car questions for shore staff, formal tone", DOCS, str(out),
                     auto=True, force=True, llm=L(extract=extract), progress=lambda *_: None)
    assert (out / "agent_definition.yaml").exists()


def test_bad_persona_falls_back_to_a_template(tmp_path):
    res = run_stage1(HR_BRIEF, DOCS, str(tmp_path / "r"), auto=True, llm=L(persona="{not a prompt}"),
                     progress=lambda *_: None)
    assert res["notes"]["persona_source"] == "fallback"
    assert res["definition"].system_prompt.startswith("You are an assistant for")


# ---------------------------------------------------------------- regressions found in the demo pre-flight
def test_topics_come_from_the_brief_including_ones_the_documents_cannot_support():
    """Real pre-flight: the model copied document headings, and the brief's own 'pension' topic never reached the guard."""
    brief = ("Answer HR policy questions for shore staff: leave and sick pay, travel and expenses. "
             "It should also tell staff how their pension contributions work.")
    extract = {"scope": "HR policy questions for shore staff", "audience": "shore staff", "tone": "friendly",
               "topics": ["leave and sick pay", "travel and expenses", "pension contributions", "Remote work"],
               "must_refuse": ["Leave policy"]}
    req = interview(brief, DOCS, L(extract=extract), ask=None)
    assert "pension contributions" in req.must_answer and "leave and sick pay" in req.must_answer
    assert "Remote work" not in req.must_answer                      # a document section the brief never mentioned
    rep = check_groundability(req, DOCS, L())
    st = {e["topic"]: e["status"] for e in rep["entries"]}
    assert st["pension contributions"] == "unsupported" and "pension contributions" in rep["gaps"]   # the guard sees the gap


def test_must_refuse_needs_a_forbidding_cue_in_the_brief():
    """Real pre-flight: 'Leave policy' and 'Travel and expenses policy' were filed under must_refuse for a brief that
    asks the assistant to ANSWER them."""
    wants = {"scope": "", "audience": "", "tone": "", "topics": ["leave policy"],
             "must_refuse": ["Leave policy", "Travel and expenses policy"]}
    req = interview("Answer questions about the leave policy and the travel and expenses policy for staff.", DOCS,
                    L(extract=wants), ask=None)
    assert "Leave policy" not in req.must_refuse and "Travel and expenses policy" not in req.must_refuse
    assert req.must_refuse == ["anything not covered by the provided documents"]                  # default, recorded
    forbids = {"scope": "", "audience": "", "tone": "", "topics": ["leave policy"],
               "must_refuse": ["legal advice about contracts", "leave policy"]}
    req = interview("Answer questions about the leave policy. It must not give legal advice about contracts.", DOCS,
                    L(extract=forbids), ask=None)
    assert req.must_refuse == ["legal advice about contracts"]       # cue present; restating a topic is still rejected


class _NoModel:
    """An LLM that fails the test if the generator consults it."""
    usage = None

    def json(self, *a, **k):
        raise AssertionError("the model must not be consulted for an explicit mention")


def test_explicit_mentions_attach_tools_without_asking_the_model(tmp_path):
    docs = csv_docs(tmp_path)
    brief = ("Procurement assistant: look up spare parts in the catalogue and calculate order totals "
             "including the supplier's volume discount.")
    req = Requirements(brief=brief, scope="spares procurement", must_answer=["part prices", "order totals"])
    tools, decisions = select_tools(req, inventory(docs), _NoModel())
    assert [t.name for t in tools] == ["csv_lookup", "calculator"]
    assert tools[0].config == {"csv_path": "parts.csv"} and "explicitly mentions" in tools[0].justification
    assert not any(d["attached"] for d in decisions if d["tool"] in ("open_meteo", "current_date"))


def test_explicit_mention_cannot_override_a_missing_precondition():
    req = Requirements(brief="Answer questions from the catalogue and look up prices", scope="x", must_answer=["prices"])
    tools, decisions = select_tools(req, inventory(DOCS), _NoModel())          # DOCS has no CSV
    assert tools == [] and "precondition failed" in {d["tool"]: d["reason"] for d in decisions}["csv_lookup"]


HR_DEMO_BRIEF = ("I need an assistant that answers HR policy questions for shore staff at a shipping company: leave and "
                 "sick pay, travel and expenses, and working hours, notice periods and the disciplinary process. It should "
                 "also tell staff how their pension contributions work. Keep a clear, friendly tone.")
HR_DEMO_EXTRACT = {"scope": "Answers HR policy questions for shore staff", "audience": "shore staff at a shipping company",
                   "tone": "friendly", "must_refuse": [],
                   "topics": ["leave and sick pay", "travel and expenses", "working hours", "notice periods",
                              "disciplinary process"]}          # what the 3B model returned in the real pre-flight: no pension


def test_backstop_recovers_the_pension_gap_the_model_missed_and_invents_nothing_else():
    req = interview(HR_DEMO_BRIEF, DOCS, L(extract=HR_DEMO_EXTRACT), ask=None)
    assert req.must_answer[-1] == "pension contributions" and len(req.must_answer) == 6   # exactly one topic added
    assert any("pension contributions" in a and "taken verbatim" in a for a in req.assumptions)
    rep = check_groundability(req, DOCS, L())
    assert "pension contributions" in rep["gaps"]                                          # the guard now sees it
    assert "pension contributions (not covered by the documents)" in req.must_refuse


def test_backstop_takes_refusals_from_the_brief_and_ignores_context_phrases():
    pr = ("An assistant for the procurement team. It should look up spare parts in the catalogue and calculate order totals. "
          "It must not give legal advice about contracts.")
    ex = {"scope": "", "audience": "the procurement team", "tone": "", "topics": [], "must_refuse": []}
    req = interview(pr, DOCS, L(extract=ex), ask=None)
    assert req.must_refuse == ["legal advice about contracts"]
    assert any("order totals" in t for t in req.must_answer)
    ctx = interview("Answer questions about leave for staff at a shipping company.", DOCS,
                    L(extract={"scope": "", "audience": "staff at a shipping company", "tone": "", "topics": ["leave"],
                               "must_refuse": []}), ask=None)
    assert ctx.must_answer == ["leave"]                              # 'shipping company', 'about' are context, not topics
    from forge.intake import backstop_topics, corpus_vocab
    r2 = Requirements(brief="x", audience="the team", must_answer=[])
    topics = [t for t, _ in backstop_topics("It should explain the approval rules and the terms.", r2, corpus_vocab(DOCS))]
    assert "rules" not in topics and "terms" not in topics           # generic single words are not gaps


class _Boom:
    def json(self, *a, **k):
        raise AssertionError("the model must not be consulted when one sentence contains every topic word")


def test_guard_skips_the_model_when_one_sentence_has_every_topic_word_and_picks_real_evidence(tmp_path):
    from forge.groundability import check_topic
    from forge.ingest import build_chunks
    from forge.retrieval import Retriever
    from forge.schemas import RetrievalConfig
    d = tmp_path / "docs"
    d.mkdir()
    (d / "policy.md").write_text(
        "# Travel and Expenses Policy\n\nEffective 1 January 2025. Applies to all employees.\n\n"
        "## Claims\n\nExpense claims must be submitted within 30 days of the trip together with receipts.\n", encoding="utf-8")
    r = Retriever(build_chunks(str(d)), RetrievalConfig(top_k=3), None)
    e = check_topic("expense claims", r, _Boom())
    assert e["status"] == "supported" and e["model"] is None and "30 days" in e["evidence"]   # not the title-matching intro
    e = check_topic("travel expenses", r, L(verdicts={"travel expenses": "yes"}))
    assert "Effective" not in e["evidence"]                                                    # real sentence, not the opener


# ---------------------------------------------------------------- regressions from the rehearsal on an unseen brief
def test_rehearsal_false_gaps_do_not_happen_again(tmp_path):
    from forge.retrieval import stem
    assert stem("validity") == stem("valid") and stem("security") == stem("secure")
    d = tmp_path / "docs"
    d.mkdir()
    (d / "certs.txt").write_text("SHIP CERTIFICATES\n\nThe Safety Management Certificate is valid for 5 years. "
                                 "A certificate is treated as expired from the day after its expiry date.\n", encoding="utf-8")
    req = Requirements(brief="x", must_answer=["Certificate Validity", "many days remain until", "current weather"])
    rep = check_groundability(req, str(d), L(verdicts={"Certificate Validity": "yes"}))
    st = {e["topic"]: e["status"] for e in rep["entries"]}
    assert st["Certificate Validity"] == "supported"                       # validity ~ valid
    assert st["many days remain until"] == "supported_by_tool"            # the date tool's job, not a document gap
    assert st["current weather"] == "supported_by_tool"
    assert rep["gaps"] == []                                               # so no harmful refusal rules
    assert tool_support("supplier price list", inventory(DOCS)) is None    # a loose 'price' still grounds nothing


def test_request_verbs_never_become_topics():
    from forge.intake import backstop_topics, corpus_vocab
    r = Requirements(brief="x", audience="ship agents", must_answer=["terminal regulations"])
    brief = "It should tell the agent the terminal regulations and report the current weather at the port."
    topics = [t for t, _ in backstop_topics(brief, r, corpus_vocab(DOCS))]
    assert "report" not in topics and not any(t.split()[0] == "report" for t in topics)
