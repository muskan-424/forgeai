"""Milestone-2 CLI.

  python eval_cli.py gen   --docs examples/sample_docs --out runs/sample/evals.json [--purpose "..."] [--n 14 5 6 5]
  python eval_cli.py run   --definition examples/sample_agent.yaml --evals runs/sample/evals.json
                           [--out runs/sample/scorecard.json] [--calibrate]
"""
import argparse
import json
import sys
from pathlib import Path

from forge import config
from forge.agent_runtime import Agent, load_definition
from forge.evalgen import EvalGenerator
from forge.evalrun import run_eval, scorecard
from forge.ingest import build_chunks
from forge.judge import Judge
from forge.llm import LLM
from forge.schemas import AgentTrace, CaseResult, EvalCase


def cmd_gen(a) -> int:
    chunks = build_chunks(a.docs)
    llm = LLM(model=config.AGENT_MODEL)
    gen = EvalGenerator(chunks, llm, purpose=a.purpose, must_refuse=a.must_refuse)
    n_ans, n_mul, n_oos, n_amb = a.n
    cases = gen.generate(n_ans, n_mul, n_oos, n_amb, min_total=a.min_cases)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([c.model_dump() for c in cases], indent=2), encoding="utf-8")
    (out.parent / "evals_generation_report.json").write_text(json.dumps(gen.report(), indent=2), encoding="utf-8")
    print(f"\n{len(cases)} cases -> {out}")
    print(json.dumps(gen.report(), indent=2))
    print(f"LLM calls: {llm.usage.calls} (cache hits {llm.usage.cache_hits}), {llm.usage.seconds:.0f}s")
    return 0


def cmd_run(a) -> int:
    cases = [EvalCase(**c) for c in json.loads(Path(a.evals).read_text(encoding="utf-8"))]
    agent = Agent(load_definition(a.definition))
    judge = Judge()
    out = {}
    if a.calibrate:
        print("Calibrating judge...")
        out["judge_calibration"] = judge.calibrate(cases)
        print(json.dumps(out["judge_calibration"], indent=2))
    print(f"Running {len(cases)} cases...")
    results = run_eval(agent, cases, judge)
    out["scorecard"] = scorecard(results)
    out["results"] = [r.model_dump() for r in results]
    print(json.dumps({k: v for k, v in out["scorecard"].items() if k != "failures"}, indent=2))
    for f in out["scorecard"]["failures"]:
        print(f"FAIL {f['id']}: {f['question'][:70]} -> {'; '.join(f['reasons'])[:150]}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


def cmd_rejudge(a) -> int:
    """Re-score the saved agent traces with the CURRENT judge. No agent calls, so a judge fix is cheap to apply."""
    saved = json.loads(Path(a.scorecard).read_text(encoding="utf-8"))
    judge = Judge(use_llm=not a.no_llm)
    results = []
    for r in saved["results"]:
        case, trace = EvalCase(**r["case"]), AgentTrace(**r["trace"])
        results.append(CaseResult(case=case, trace=trace, verdict=judge.judge(case, trace)))
    old = {r["case"]["id"]: r["verdict"]["passed"] for r in saved["results"]}
    for r in results:
        if old[r.case.id] != r.verdict.passed:
            print(f"CHANGED {r.case.id}: {'PASS' if old[r.case.id] else 'FAIL'} -> {'PASS' if r.verdict.passed else 'FAIL'}"
                  f"  ({'; '.join(r.verdict.reasons)[:120]})")
    saved["scorecard"] = scorecard(results)
    saved["results"] = [r.model_dump() for r in results]
    saved["rejudged_with_current_judge"] = True
    if a.calibrate:
        saved["judge_calibration"] = judge.calibrate([r.case for r in results])
    Path(a.out or a.scorecard).write_text(json.dumps(saved, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in saved["scorecard"].items() if k != "failures"}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--docs", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--purpose", default="")
    g.add_argument("--must-refuse", nargs="*", default=[])
    g.add_argument("--min-cases", type=int, default=20)
    g.add_argument("--n", nargs=4, type=int, default=[14, 5, 6, 5], metavar=("ANS", "MULTI", "OOS", "AMB"))
    r = sub.add_parser("run")
    r.add_argument("--definition", required=True)
    r.add_argument("--evals", required=True)
    r.add_argument("--out")
    r.add_argument("--calibrate", action="store_true")
    j = sub.add_parser("rejudge")
    j.add_argument("--scorecard", required=True)
    j.add_argument("--out")
    j.add_argument("--no-llm", action="store_true")
    j.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    return {"gen": cmd_gen, "run": cmd_run, "rejudge": cmd_rejudge}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
