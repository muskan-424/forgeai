"""Factory pipeline: intake -> groundability guard -> generator -> eval generation -> judge calibration
-> critic/optimizer loop -> honest report. Every stage writes its artifacts into the run folder."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from . import config
from .agent_runtime import Agent, save_definition
from .evalgen import EvalGenerator
from .generator import generate_definition
from .groundability import GroundabilityError, check_groundability
from .ingest import build_chunks
from .intake import SLOT_QUESTIONS, interview, requirements_markdown
from .judge import Judge
from .llm import LLM
from .loop import run_loop
from .report import build_report
from .schemas import EvalCase


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_stage1(brief: str, docs_dir: str, out_dir: str, auto: bool = True,
               ask: Callable[[str], str] | None = None, force: bool = False,
               llm: LLM | None = None, progress=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    llm = llm or LLM(model=config.AGENT_MODEL)

    progress("[1/6] intake: reading the brief and the documents")
    req = interview(brief, docs_dir, llm, ask=None if auto else ask)
    progress(f"      {len(req.clarifications)} clarifying question(s), {len(req.assumptions)} assumption(s)")

    progress("[2/6] groundability guard: can the documents support the brief?")
    report = check_groundability(req, docs_dir, llm)
    for w in report["warnings"]:
        progress(f"      ! {w}")
    _dump(out / "requirements.json", req.model_dump())
    (out / "requirements.md").write_text(requirements_markdown(req), encoding="utf-8")
    _dump(out / "groundability.json", report)
    if report["blocking"] and not force:
        _dump(out / "llm_usage.json", llm.usage.__dict__)
        raise GroundabilityError(
            "Blocked: none of the topics the brief asks for are supported by the documents. "
            f"See {out / 'groundability.json'}. Fix the brief or the documents, or re-run with force=True.")

    progress("[3/6] generator: tools, retrieval parameters, system prompt")
    definition, notes = generate_definition(req, docs_dir, llm, out_dir=str(out))
    save_definition(definition, out / "agent_definition.yaml")
    _dump(out / "generation_notes.json", notes)
    _dump(out / "llm_usage.json", llm.usage.__dict__)
    progress(f"      tools attached: {[t.name for t in definition.tools] or 'none'}")
    return {"requirements": req, "groundability": report, "definition": definition, "notes": notes,
            "usage": llm.usage, "out_dir": str(out)}


def _refusal_topics(req) -> list[str]:
    generic = SLOT_QUESTIONS["must_refuse"][1]
    return [t.replace(" (not covered by the documents)", "") for t in req.must_refuse if t != generic]


def run_factory(brief: str, docs_dir: str, out_dir: str, auto: bool = True, ask: Callable[[str], str] | None = None,
                force: bool = False, n_eval: tuple[int, int, int, int] = (8, 3, 4, 3), min_cases: int = 20, max_iters: int = 3,
                budget_minutes: float | None = None, calibrate: bool = True, reuse_evals: bool = False,
                llm: LLM | None = None, judge: Judge | None = None, agent_factory: Callable | None = None,
                progress=print) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    llm = llm or LLM(model=config.AGENT_MODEL)
    s1 = run_stage1(brief, docs_dir, out_dir, auto=auto, ask=ask, force=force, llm=llm, progress=progress)
    req, definition = s1["requirements"], s1["definition"]
    definition.docs_dir = str((out / definition.docs_dir).resolve())          # absolute in memory

    # ---------------- eval set ----------------
    evals_path = out / "evals.json"
    eval_report = None
    if reuse_evals and evals_path.exists():
        progress("[4/6] eval set: reusing evals.json")
        cases = [EvalCase(**c) for c in json.loads(evals_path.read_text(encoding="utf-8"))]
        rp = out / "evals_generation_report.json"
        eval_report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    else:
        progress("[4/6] eval generator: building a verified test set from the documents")
        gen = EvalGenerator(build_chunks(definition.docs_dir), llm, purpose=req.scope, must_refuse=_refusal_topics(req),
                            progress=progress)
        cases = gen.generate(*n_eval, min_total=min_cases)
        eval_report = gen.report()
        _dump(evals_path, [c.model_dump() for c in cases])
        _dump(out / "evals_generation_report.json", eval_report)
    dev = [c for c in cases if c.split == "dev"]
    hold = [c for c in cases if c.split == "holdout"]
    progress(f"      {len(cases)} cases ({len(dev)} dev / {len(hold)} holdout)")

    # ---------------- judge + calibration ----------------
    judge = judge or Judge()
    calibration = None
    if calibrate:
        progress("[5/6] judge calibration: scoring synthetic answers with known verdicts")
        calibration = judge.calibrate(cases, per_kind=2, progress=lambda *_: None)
        _dump(out / "judge_calibration.json", calibration)
        progress(f"      judge accuracy {calibration['accuracy']:.0%}, false passes {calibration['false_pass']}")

    # ---------------- critic / optimizer loop ----------------
    progress("[6/6] critic-optimizer loop")
    agent_llm = LLM(model=config.AGENT_MODEL)
    factory = agent_factory or (lambda d: Agent(d, agent_llm))
    result = run_loop(definition, dev, hold, judge, str(out), opt_llm=llm, max_iters=max_iters,
                      budget_s=budget_minutes * 60 if budget_minutes else None, agent_factory=factory, progress=progress)

    # ---------------- report ----------------
    total = time.time() - t0
    md, js = build_report({
        "name": result["best_definition"].name, "requirements": req.model_dump(), "groundability": s1["groundability"],
        "notes": s1["notes"], "eval_report": eval_report, "calibration": calibration, "cases": cases, "loop": result,
        "final_definition": result["best_definition"], "total_seconds": total,
        "usage": {"factory (intake, guard, generator, evalgen)": llm.usage.__dict__,
                  "agent": agent_llm.usage.__dict__, "judge": judge.llm.usage.__dict__}})
    (out / "report.md").write_text(md, encoding="utf-8")
    _dump(out / "report.json", js)
    progress(f"Done in {total / 60:.1f} min. Report: {out / 'report.md'}")
    return {"loop": result, "report_md": md, "report": js, "out_dir": str(out), "requirements": req}
