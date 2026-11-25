import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.agent_runtime import Agent, load_definition, norm  # noqa: E402
from forge.ingest import build_chunks, chunk_csv, chunk_text, inventory  # noqa: E402
from forge.llm import LLM, extract_json  # noqa: E402
from forge.retrieval import Retriever  # noqa: E402
from forge.schemas import RetrievalConfig, ToolRef  # noqa: E402
from forge.tools import REGISTRY, build_tool  # noqa: E402
from forge.tools.calculator import safe_eval  # noqa: E402

EX = Path(__file__).resolve().parent.parent / "examples"


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_calculator_safe():
    assert safe_eval("12.5 * 8 + 3") == 103
    assert safe_eval("2^3") == 8
    assert safe_eval("1,200 + 5") == 1205
    for bad in ("__import__('os').system('x')", "2**1000", "open('f')"):
        with pytest.raises(Exception):
            safe_eval(bad)


def test_chunking_keeps_headings_and_covers_text():
    text = (EX / "sample_docs" / "leave_policy.md").read_text(encoding="utf-8")
    chunks = chunk_text("leave_policy.md", text, 300, 50)
    assert len(chunks) >= 2
    assert any(c.heading == "Sick Leave" for c in chunks)
    joined = " ".join(c.text for c in chunks)
    assert "25 days" in joined and "medical certificate" in joined


def test_csv_chunks_are_self_describing():
    ch = chunk_csv("p.csv", "part,price\nA1,10\nB2,20\n", rows_per_chunk=1)
    assert len(ch) == 2 and "part: B2; price: 20" in ch[1].text


def test_inventory_lists_docs():
    inv = {d["doc"]: d for d in inventory(EX / "sample_docs")}
    assert {"leave_policy.md", "expenses_policy.md"} <= set(inv)
    assert "Annual Leave" in inv["leave_policy.md"]["headings"]


def test_bm25_finds_relevant_chunk():
    chunks = build_chunks(EX / "sample_docs", 300, 50)
    r = Retriever(chunks, RetrievalConfig(top_k=2), None)
    top = r.search("how many days of sick leave")[0][0]
    assert top.heading == "Sick Leave"


def test_registry_has_four_tools_and_csv_lookup(tmp_path):
    assert set(REGISTRY) == {"csv_lookup", "calculator", "open_meteo", "current_date"}
    (tmp_path / "p.csv").write_text("part,price\nValve,10\nPump,99\n", encoding="utf-8")
    t = build_tool("csv_lookup", {"csv_path": "p.csv"}, str(tmp_path))
    assert "Pump" in t.run({"column": "part", "value": "pump"})
    assert "unknown column" in t.run({"column": "zzz", "value": "x"})


def _mock(answer_obj):
    return lambda messages, schema: json.dumps(answer_obj)


def test_agent_verifies_citations_and_flags_fabricated_quote():
    d = load_definition(EX / "sample_agent.yaml")
    good = {"answer": "25 days.", "refused": False, "refusal_reason": "",
            "citations": [{"doc": "leave_policy.md", "quote": "25 days of paid annual leave"}]}
    agent = Agent(d, LLM(backend="mock", mock_fn=_mock(good)))
    tr = agent.ask("How many days of annual leave do I get?")
    assert tr.citations_verified == [True] and tr.contract_ok

    fake = dict(good, citations=[{"doc": "leave_policy.md", "quote": "30 days of paid leave"}])
    agent = Agent(d, LLM(backend="mock", mock_fn=_mock(fake)))
    assert agent.ask("How many days of annual leave do I get?").citations_verified == [False]


def test_agent_flags_missing_citations_and_accepts_refusal():
    d = load_definition(EX / "sample_agent.yaml")
    nocite = {"answer": "25 days.", "refused": False, "refusal_reason": "", "citations": []}
    tr = Agent(d, LLM(backend="mock", mock_fn=_mock(nocite))).ask("annual leave days?")
    assert not tr.contract_ok and "missing citations" in tr.error
    ref = {"answer": "", "refused": True, "refusal_reason": "out of scope", "citations": []}
    tr = Agent(d, LLM(backend="mock", mock_fn=_mock(ref))).ask("annual leave days?")
    assert tr.answer.refused and tr.contract_ok


def test_agent_tool_loop():
    d = load_definition(EX / "sample_agent.yaml")
    d.tools = [ToolRef(name="calculator", justification="test")]
    calls = iter([
        {"tool": "calculator", "args": {"expression": "25 - 5"}, "answer": "", "citations": [],
         "refused": False, "refusal_reason": ""},
        {"answer": "20 days remain.", "citations": [], "refused": False, "refusal_reason": ""},
    ])
    agent = Agent(d, LLM(backend="mock", mock_fn=lambda m, s: json.dumps(next(calls))))
    tr = agent.ask("I have 25 days and used 5, how many remain?")
    assert tr.tool_calls[0]["result"] == "20" and tr.contract_ok


def test_agent_refuses_without_context_and_tools():
    d = load_definition(EX / "sample_agent.yaml")
    tr = Agent(d, LLM(backend="mock", mock_fn=_mock({}))).ask("zzzz qqqq")
    assert tr.answer.refused
