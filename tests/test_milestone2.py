import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.evalgen import (EvalGenerator, coverage, direction_conflict, dropped_terms,  # noqa: E402
                           incomparable, split_sentences)
from forge.evalrun import scorecard, wilson  # noqa: E402
from forge.ingest import build_chunks  # noqa: E402
from forge.judge import Judge  # noqa: E402
from forge.llm import LLM  # noqa: E402
from forge.schemas import (AgentAnswer, AgentTrace, CaseResult, Chunk, Citation, EvalCase,  # noqa: E402
                           JudgeVerdict)
from forge.textmatch import fact_in_text, numbers, question_similarity  # noqa: E402

DOCS = Path(__file__).resolve().parent.parent / "examples" / "sample_docs"


# ---------------- textmatch ----------------
def test_fact_matching_exact_fuzzy_and_number_strictness():
    text = "Shore staff are entitled to 25 days of paid annual leave per calendar year."
    assert fact_in_text("25 days of paid annual leave", text) == "exact"
    assert fact_in_text("25 paid annual leave days", text) == "fuzzy"
    assert fact_in_text("30 days of paid annual leave", text) == ""   # wrong number must never match
    assert fact_in_text("medical certificate", text) == ""
    assert numbers("1,200.50 and 5") == {"1200.5", "5"}
    assert question_similarity("how many sick days", "how many sick days") == 1.0


# ---------------- judge (deterministic path) ----------------
def case_leave():
    return EvalCase(id="ans-01", type="answerable", question="How many days of annual leave?",
                    expected_facts=["25 days"], source_docs=["leave_policy.md"],
                    source_quotes=["Shore staff are entitled to 25 days of paid annual leave per calendar year."])


def trace(answer, cites=None, verified=None, refused=False, ctx="Shore staff are entitled to 25 days of paid annual leave per calendar year.",
          doc="leave_policy.md", q="How many days of annual leave?"):
    cites = cites if cites is not None else [Citation(doc=doc, quote=ctx)]
    verified = verified if verified is not None else [True] * len(cites)
    return AgentTrace(question=q, retrieved=[Chunk(id="c", doc=doc, text=ctx)],
                      answer=AgentAnswer(answer=answer, citations=cites, refused=refused),
                      citations_verified=verified)


def J():
    return Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": []})), use_llm=False)


def test_judge_passes_correct_grounded_answer():
    v = J().judge(case_leave(), trace("You get 25 days of paid leave per year."))
    assert v.passed and v.correct and v.grounded


def test_judge_fails_wrong_number_and_fabricated_citation_and_wrong_doc():
    assert not J().judge(case_leave(), trace("You get 30 days.")).correct
    v = J().judge(case_leave(), trace("You get 25 days.", verified=[False]))
    assert v.correct and not v.grounded and not v.passed
    v = J().judge(case_leave(), trace("You get 25 days.", doc="expenses_policy.md"))
    assert not v.grounded


def test_judge_flags_numbers_not_in_context():
    v = J().judge(case_leave(), trace("You get 25 days, and 12 bank holidays."))
    assert not v.grounded and any("numbers not in context" in r for r in v.reasons)


def test_judge_refusal_logic():
    c = case_leave()
    v = J().judge(c, trace("", refused=True, cites=[]))
    assert not v.passed and any("wrong refusal" in r for r in v.reasons)
    oos = EvalCase(id="oos-01", type="out_of_scope", question="Capital of France?", should_refuse=True)
    assert J().judge(oos, trace("", refused=True, cites=[])).passed
    v = J().judge(oos, trace("Paris.", cites=[], verified=[]))
    assert not v.passed and any("hallucination" in r for r in v.reasons)
    soft = J().judge(oos, trace("I can't help with that; it is outside what my documents cover.", cites=[], verified=[]))
    assert soft.passed and any("soft refusal" in r for r in soft.reasons)


def test_judge_agent_error_is_a_failure():
    t = AgentTrace(question="q", error="Ollama call failed")
    assert not J().judge(case_leave(), t).passed


# ---------------- judge (LLM path, mocked) ----------------
def test_llm_judge_upgrades_paraphrase_only_when_both_passes_agree():
    case = EvalCase(id="a", type="answerable", question="Notice period?", expected_facts=["two weeks written notice"],
                    source_docs=["leave_policy.md"], source_quotes=["x" * 20])
    t = trace("A fortnight of formal notice is needed.", ctx="x" * 20, cites=[Citation(doc="leave_policy.md", quote="x" * 20)])
    agree = Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": [{"n": 1, "stated": True}]})))
    v = agree.judge(case, t)
    assert v.fact_methods == ["llm"] and v.correct and v.llm_used

    calls = {"n": 0}

    def flip(m, s):
        calls["n"] += 1
        return json.dumps({"results": [{"n": 1, "stated": calls["n"] == 1}]})
    v = Judge(LLM(backend="mock", mock_fn=flip)).judge(case, t)
    assert v.uncertain and not v.correct and v.fact_methods == ["uncertain"]


def test_calibration_with_a_correct_mock_judge_is_perfect():
    cases = [
        case_leave(),
        EvalCase(id="ans-02", type="answerable", question="Meal allowance abroad?", expected_facts=["65 euros per day"],
                 source_docs=["expenses_policy.md"], source_quotes=["International travel carries a meal allowance of 65 euros per day."]),
        EvalCase(id="oos-01", type="out_of_scope", question="Capital of France?", should_refuse=True),
    ]
    judge = Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": [{"n": 1, "stated": False}]})))
    cal = judge.calibrate(cases, per_kind=2, progress=lambda *_: None)
    assert cal["accuracy"] == 1.0 and cal["false_pass"] == 0 and cal["false_fail"] == 0
    assert {"gold", "padded_gold", "wrong_facts", "padded_wrong", "wrong_refusal", "oos_refused", "oos_answered"} <= set(cal["by_kind"])


def test_calibration_catches_a_sycophantic_judge():
    cases = [case_leave(), EvalCase(id="ans-02", type="answerable", question="Meal allowance abroad?",
                                    expected_facts=["65 euros per day"], source_docs=["expenses_policy.md"],
                                    source_quotes=["International travel carries a meal allowance of 65 euros per day."])]
    yes = Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": [{"n": 1, "stated": True}] })))
    cal = yes.calibrate(cases, per_kind=2, progress=lambda *_: None)
    assert cal["false_pass"] > 0 and cal["llm_false_accepts"] > 0 and cal["accuracy"] < 1.0


# ---------------- scorecard ----------------
def test_wilson_interval_and_scorecard():
    lo, hi = wilson(7, 10)
    assert 0.39 < lo < 0.40 and 0.89 < hi < 0.90
    assert wilson(0, 0) == (0.0, 0.0)
    c = case_leave()
    ok = CaseResult(case=c, trace=trace("25 days"), verdict=JudgeVerdict(case_id=c.id, passed=True))
    bad = CaseResult(case=c.model_copy(update={"id": "x", "split": "holdout"}), trace=trace("30"),
                     verdict=JudgeVerdict(case_id="x", passed=False, reasons=["facts matched 0/1"]))
    s = scorecard([ok, bad])
    assert s["overall"]["passed"] == 1 and s["overall"]["n"] == 2
    assert s["by_split"]["holdout"]["rate"] == 0.0 and len(s["failures"]) == 1


# ---------------- eval generator (mocked LLM) ----------------
def _words(text, n):
    return re.findall(r"[A-Za-z]{5,}", text)[:n]


def make_mock(bad_question=None, secretly_answerable="", vague=True):
    """Mock that understands the sentence-anchored prompts."""
    def fn(messages, schema):
        p = messages[0]["content"]
        if "ANSWERABILITY CHECK" in p:
            q = p.split("QUESTION:")[1].splitlines()[0].strip().lower()
            return json.dumps({"answerable": bool(secretly_answerable) and secretly_answerable.lower() in q})
        if "OUT-OF-SCOPE" in p:
            return json.dumps({"questions": ["What is the company's pension contribution rate?",
                                             "How do I apply for a visa for my spouse?",
                                             "What is the capital of France?",
                                             "Who won the football world cup in 2018?",
                                             "How many days of annual leave do I get?"]})
        if "AMBIGUOUS" in p:
            orig = p.split("ORIGINAL QUESTION:")[1].splitlines()[0].strip()
            kept = " ".join(re.findall(r"[A-Za-z]+", orig)[:3])
            return json.dumps({"question": f"{kept}?" if vague else orig})
        facts = lambda sent: re.findall(r"\d+ [a-z]+(?: per day)?", sent)[:1] or [" ".join(sent.split()[:2])]
        if "SENTENCE A:" in p:
            sa = p.split("SENTENCE A:\n")[1].split("\n\n")[0].strip()
            sb = p.split("SENTENCE B:\n")[1].split("\n\n")[0].strip()
            q = bad_question or f"Compare {' '.join(_words(sa, 2))} with {' '.join(_words(sb, 2))}"
            return json.dumps({"question": q, "expected_facts": facts(sa) + facts(sb)})
        sent = p.split("SENTENCE:\n")[1].split("\n\n")[0].strip()
        q = bad_question or f"Explain {' '.join(_words(sent, 4))}"
        return json.dumps({"question": q, "expected_facts": facts(sent)})
    return fn


def test_split_sentences_and_coverage_and_direction_guards():
    text = "Staff get 25 days of leave. Short one.\nUnused leave may be carried over, up to a maximum of 5 days."
    assert split_sentences(text) == ["Staff get 25 days of leave.",
                                     "Unused leave may be carried over, up to a maximum of 5 days."]
    sentence = "Flights over 6 hours may be booked in premium economy with line manager approval."
    assert coverage("Which flights may be booked in premium economy?", sentence) == 1.0
    assert coverage("What is the maximum travel time for booking premium economy flights?", sentence) < 0.6
    # the real bug found on the 3B model: 'maximum' question against an 'over 6 hours' sentence
    assert direction_conflict("What is the maximum travel time for premium economy?", sentence)
    assert not direction_conflict("From how many hours can premium economy be booked?", sentence)
    up = "Unused annual leave may be carried over, up to a maximum of 5 days, into the following year."
    assert not direction_conflict("What is the maximum number of days of leave carried over?", up)
    assert direction_conflict("What is the minimum number of days carried over?", up)


def test_evalgen_builds_anchored_verified_cases_and_splits():
    chunks = build_chunks(DOCS, 300, 40)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=make_mock()), progress=lambda *_: None)
    cases = gen.generate(n_answerable=4, n_multi=2, n_oos=4, n_ambiguous=2)
    assert {"answerable", "multi_doc", "out_of_scope", "ambiguous"} <= {c.type for c in cases}
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))
    alltext = {ch.doc: " ".join(c.text for c in chunks if c.doc == ch.doc) for ch in chunks}
    for c in cases:
        if c.type in ("answerable", "ambiguous"):
            assert c.source_quotes[0] in alltext[c.source_docs[0]]                # quote is verbatim by construction
            assert all(fact_in_text(f, c.source_quotes[0]) for f in c.expected_facts)
        if c.type == "multi_doc":
            assert len(c.source_docs) == 2 and len(c.source_quotes) == 2
        if c.type == "out_of_scope":
            assert c.should_refuse
    qs = [c.question for c in cases]
    assert len(qs) == len(set(qs))                                                # sentence anchoring -> no duplicates
    assert not any("annual leave do i get" in c.question.lower() for c in cases if c.type == "out_of_scope")
    assert any(c.split == "holdout" for c in cases) and any(c.split == "dev" for c in cases)


def test_evalgen_rejects_questions_that_stray_and_reports_examples():
    chunks = build_chunks(DOCS, 300, 40)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=make_mock(bad_question="What colour is the company logo today?")),
                        progress=lambda *_: None)
    cases = gen.generate(n_answerable=3, n_multi=1, n_oos=0, n_ambiguous=2)
    assert cases == []                                                            # nothing survives; ambiguous needs answerable
    rep = gen.report()
    assert rep["rejections"]["answerable:question strays from its source sentence"] > 0
    assert rep["rejection_examples"]["answerable:question strays from its source sentence"]


def test_evalgen_rejects_a_rewrite_that_is_not_vague():
    chunks = build_chunks(DOCS, 300, 40)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=make_mock(vague=False)), progress=lambda *_: None)
    cases = gen.generate(n_answerable=2, n_multi=0, n_oos=0, n_ambiguous=2)
    assert [c for c in cases if c.type == "ambiguous"] == []
    assert any(k.startswith("ambiguous:") for k in gen.report()["rejections"])


def test_oos_model_check_is_used_only_when_documents_overlap():
    chunks = build_chunks(DOCS, 300, 40)
    llm = LLM(backend="mock", mock_fn=make_mock(secretly_answerable="claims above"))
    gen = EvalGenerator(chunks, llm, progress=lambda *_: None)
    assert gen._looks_answerable("What happens to claims above 500 euros?") is True    # model says answerable -> reject
    assert gen._looks_answerable("What happens to claims for travel insurance?") is False
    calls = llm.usage.calls
    assert gen._looks_answerable("What is the pension contribution rate?") is False     # zero overlap: no model call needed
    assert llm.usage.calls == calls


def test_facts_problem_rules():
    f = EvalGenerator._facts_problem
    quote = "International travel carries a meal allowance of 65 euros per day."
    assert f(["65 euros per day"], [quote]) == ""
    assert f(["Passage A: 65 euros per day"], [quote]) == "fact contains prompt artefact"
    assert f(["a very long fact " * 5], [quote]) == "fact too long (not atomic)"
    assert f(["90 euros per day"], [quote]) == "fact not supported by the quote"


def test_incomparable_units_and_measured_vagueness():
    # the real nonsense question the 3B model produced
    assert incomparable("Compare the meal allowance with the paid annual leave. Which is higher?",
                        ["40 euros per day", "25 days"])
    assert not incomparable("How many days of sick leave, and what is the notice period?", ["10 days", "7 days"])
    assert not incomparable("Compare the two limits, which is higher?", ["40 euros per day", "65 euros per day"])
    orig = "How many days of paid annual leave do Shore staff get per year?"
    assert {"annual", "paid"} <= set(dropped_terms(orig, "how many days of leave do shore staff get?"))
    assert dropped_terms("What is the meal allowance for domestic travel for staff?",
                         "what's the meal allowance for domestic travel?") == ["staff"]   # too mild: only 1 dropped


def test_ambiguous_rewrites_reject_template_echo_and_malformed_questions():
    chunks = build_chunks(DOCS, 300, 40)
    for bad in ("the vague rewrite", "how many days of leave do shore staff get it", "What?"):
        base_mock = make_mock()

        def fn(m, s, b=bad):
            return json.dumps({"question": b}) if "AMBIGUOUS" in m[0]["content"] else base_mock(m, s)
        gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=fn), progress=lambda *_: None)
        cases = gen.generate(n_answerable=2, n_multi=0, n_oos=0, n_ambiguous=2)
        assert [c for c in cases if c.type == "ambiguous"] == [], bad
        assert any(k.startswith("ambiguous:") for k in gen.report()["rejections"]), bad


def test_hedged_refusal_with_content_fails():
    """Real bug from the first baseline run: refused=True plus an invented procedure with a citation passed."""
    oos = EvalCase(id="oos-02", type="out_of_scope", question="How do I extend my leave?", should_refuse=True)
    hedged = trace("they must request the extension through the HR portal at least 14 days in advance. "
                   "There is no specific procedure mentioned.", refused=True)       # keeps the leave-policy citation
    v = J().judge(oos, hedged)
    assert not v.passed and any("still asserts content" in r for r in v.reasons)
    long_no_cite = trace("word " * 60, cites=[], verified=[], refused=True)
    assert not J().judge(oos, long_no_cite).passed
    clean = trace("I can't help with that; it is outside what my documents cover.", cites=[], verified=[], refused=True)
    assert J().judge(oos, clean).passed


def test_calibration_includes_hedged_refusal_probe():
    cases = [case_leave(), EvalCase(id="oos-01", type="out_of_scope", question="Capital of France?", should_refuse=True)]
    judge = Judge(LLM(backend="mock", mock_fn=lambda m, s: json.dumps({"results": [{"n": 1, "stated": False}]})))
    cal = judge.calibrate(cases, per_kind=1, progress=lambda *_: None)
    assert cal["by_kind"]["oos_hedged"] == "1/1" and cal["accuracy"] == 1.0


def test_split_sentences_skips_headings_but_keeps_table_rows():
    chunk = ("Travel and Expenses Policy\nBooking Travel\nAll business travel must be booked at least 7 days ahead.\n"
             "part_no: V-100; description: Gate valve; unit_price: 120")
    assert split_sentences(chunk) == ["All business travel must be booked at least 7 days ahead.",
                                      "part_no: V-100; description: Gate valve; unit_price: 120"]


def test_oos_falls_back_to_unrelated_questions_when_every_near_domain_candidate_is_answerable():
    chunks = build_chunks(DOCS, 300, 40)
    base_mock = make_mock()

    def fn(messages, schema):
        p = messages[0]["content"]
        if "OUT-OF-SCOPE" in p:
            if "general-knowledge" in p:
                return json.dumps({"questions": ["What is the capital of France?", "Who won the 2018 football world cup?"]})
            return json.dumps({"questions": ["What is the meal allowance for domestic travel?",
                                              "How many days of paid annual leave do staff get?",
                                              "How many days of paid sick leave can an employee take?"]})
        return base_mock(messages, schema)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=fn), progress=lambda *_: None)
    cases = gen.generate(n_answerable=3, n_multi=0, n_oos=3, n_ambiguous=0)
    oos = [c for c in cases if c.type == "out_of_scope"]
    assert len(oos) >= 2 and all("fallback" in c.notes for c in oos)
    assert gen.report()["rejections"]["out_of_scope:too similar to the documents (might be answerable)"] >= 3


def test_out_of_scope_cases_must_be_questions():
    chunks = build_chunks(DOCS, 300, 40)
    base_mock = make_mock()

    def fn(messages, schema):
        p = messages[0]["content"]
        if "OUT-OF-SCOPE" in p:
            return json.dumps({"questions": ["procurement_procedure.pdf does not provide legal advice on contracts.",
                                             "What is the capital of France?", "Who won the 2018 football world cup?"]})
        return base_mock(messages, schema)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=fn), progress=lambda *_: None)
    cases = gen.generate(n_answerable=2, n_multi=0, n_oos=3, n_ambiguous=0)
    oos = [c.question for c in cases if c.type == "out_of_scope"]
    assert oos and all(q.endswith("?") for q in oos)
    assert gen.report()["rejections"].get("out_of_scope:not a question", 0) >= 1


def test_a_refusal_named_in_the_brief_always_gets_a_test_case():
    """Real procurement demo: the only candidate for 'legal advice about contracts' was a statement, which was
    (rightly) rejected, and the brief's explicit refusal ended up with no test at all."""
    chunks = build_chunks(DOCS, 300, 40)
    base_mock = make_mock()
    seen = {"must": 0}

    def fn(messages, schema):
        p = messages[0]["content"]
        if "OUT-OF-SCOPE" in p and "forbidden help" in p:               # first attempt: the model writes a statement
            seen["must"] += 1
            return json.dumps({"questions": ["The procedure does not provide legal advice on contracts."]})
        if "OUT-OF-SCOPE" in p and "one short, plain question" in p:    # second, differently worded attempt works
            seen["must"] += 1
            return json.dumps({"questions": ["Can you tell me if this supplier contract clause is legally enforceable?"]})
        return base_mock(messages, schema)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=fn), must_refuse=["legal advice about contracts"],
                        progress=lambda *_: None)
    cases = gen.generate(n_answerable=3, n_multi=0, n_oos=3, n_ambiguous=0)
    must = [c for c in cases if c.type == "out_of_scope" and "must_refuse" in c.notes]
    assert seen["must"] == 2 and len(must) == 1 and "legally enforceable" in must[0].question
    assert "second attempt" in must[0].notes
    assert gen.report()["rejections"].get("out_of_scope:not a question", 0) >= 1


# ------------------------------------------------------------------ eval size: minimum guarantee and weak-test filters
def test_min_cases_tops_up_when_verification_rejects_too_many():
    """Real demos ended at 11-13 cases because the quotas were met only partly; a minimum must be guaranteed."""
    chunks = build_chunks(DOCS, 300, 40)
    gen = EvalGenerator(chunks, LLM(backend="mock", mock_fn=make_mock()), progress=lambda *_: None)
    few = gen.generate(n_answerable=2, n_multi=0, n_oos=0, n_ambiguous=0)
    assert len(few) == 2
    gen2 = EvalGenerator(chunks, LLM(backend="mock", mock_fn=make_mock()), progress=lambda *_: None)
    cases = gen2.generate(n_answerable=2, n_multi=0, n_oos=0, n_ambiguous=0, min_total=7)
    assert len(cases) >= 7 and any("top-up" in c.notes for c in cases)
    assert gen2.report()["attempted_accepted"]["top_up:accepted"] >= 5
    assert len({c.question for c in cases}) == len(cases)                     # still no duplicates
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


def test_deictic_questions_and_table_row_rewrites_are_rejected(tmp_path):
    gen = EvalGenerator(build_chunks(DOCS, 300, 40), LLM(backend="mock", mock_fn=make_mock()), progress=lambda *_: None)
    assert "without naming" in gen._check_question("Who does this leave policy apply to?", [])
    assert "without naming" in gen._check_question("What does that document say about notice?", [])
    assert gen._check_question("Who does the leave policy apply to?", []) == ""
    d = tmp_path / "docs"
    d.mkdir()
    rows = [f"V-{100 + i},Gate valve number {i} bronze,{50 + i}" for i in range(12)]
    (d / "parts.csv").write_text("\n".join(["part_no,description,unit_price"] + rows) + "\n", encoding="utf-8")
    g2 = EvalGenerator(build_chunks(str(d)), LLM(backend="mock", mock_fn=make_mock()), progress=lambda *_: None)
    cases = g2.generate(n_answerable=4, n_multi=0, n_oos=0, n_ambiguous=3)
    assert [c for c in cases if c.type == "answerable"]                          # rows still make answerable cases
    assert [c for c in cases if c.type == "ambiguous"] == []                     # but are never rewritten as vague
    assert g2.report()["attempted_accepted"].get("ambiguous:attempted", 0) == 0


def test_keep_awake_requests_and_releases_the_wakelock():
    from forge.power import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, keep_awake
    calls = []
    with keep_awake(calls.append) as on:
        assert on and calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert calls[-1] == ES_CONTINUOUS                                            # released afterwards
    calls.clear()
    try:
        with keep_awake(calls.append):
            raise RuntimeError("run failed")
    except RuntimeError:
        pass
    assert calls[-1] == ES_CONTINUOUS                                            # even when the run crashes
