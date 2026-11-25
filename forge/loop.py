"""Critic-optimizer loop: run evals -> classify failures -> ONE targeted patch -> re-run -> keep or revert.

Rules of the loop (all enforced here, not left to the model):
- the optimizer only ever sees DEV cases; HOLDOUT cases are scored at the start and the end, never used to patch;
- a patch is kept only if it fixes more dev cases than it breaks (net gain); otherwise it is reverted, and the
  attempted step is remembered so it is not retried;
- at most `max_iters` iterations (default 3), a wall-clock budget, and early stop when two patches in a row
  fail to pay for themselves;
- every iteration writes its definition, results and patch, and the changelog records what changed, why,
  which failures it addressed, which cases it fixed or broke, and the score before and after."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from .agent_runtime import Agent, save_definition
from .critic import classify_all
from .evalrun import is_infra_error, run_eval, scorecard
from .ingest import build_chunks
from .judge import Judge
from .llm import LLM
from .optimizer import apply_patch, describe, propose
from .schemas import AgentDefinition, CaseResult, EvalCase
from . import config


def default_agent_factory(d: AgentDefinition):
    return Agent(d, LLM(model=d.model or config.AGENT_MODEL))


def save_definition_at(d: AgentDefinition, path: Path) -> None:
    """Write a definition whose docs_dir is relative to the file, so it loads from wherever it is stored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = d.model_copy(deep=True)
    if os.path.isabs(c.docs_dir):
        c.docs_dir = os.path.relpath(c.docs_dir, path.parent).replace("\\", "/")
    save_definition(c, path)


def evaluate(d: AgentDefinition, cases: list[EvalCase], judge: Judge, agent_factory, progress,
             sleep=time.sleep) -> list[CaseResult]:
    if not cases:
        return []
    return run_eval(agent_factory(d), cases, judge, progress=progress, sleep=sleep)


def _n_chunks(d: AgentDefinition) -> int:
    try:
        return len(build_chunks(d.docs_dir, d.retrieval.chunk_size, d.retrieval.chunk_overlap))
    except Exception:  # noqa: BLE001 - corpus size only steers the ladder; never fail the loop over it
        return 0


STALL_CAP_S = 600   # one question never counts for more than this: a sleeping laptop or a hung call must not eat the budget


def compute_seconds(results: list[CaseResult]) -> float:
    return sum(min(r.trace.latency_s, STALL_CAP_S) for r in results)


def _passed(results: list[CaseResult]) -> set[str]:
    return {r.case.id for r in results if r.verdict.passed}


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_iteration(out: Path, n: int, d: AgentDefinition, dev: list[CaseResult], holdout: list[CaseResult],
                    patch: dict | None, decision: str) -> None:
    folder = out / "iterations" / f"iter_{n}"
    save_definition_at(d, folder / "agent_definition.yaml")
    _dump(folder / "results.json", {
        "iteration": n, "decision": decision, "patch": patch,
        "dev": scorecard(dev) if dev else None, "holdout": scorecard(holdout) if holdout else None,
        "dev_results": [r.model_dump() for r in dev], "holdout_results": [r.model_dump() for r in holdout]})


def run_loop(definition: AgentDefinition, dev_cases: list[EvalCase], holdout_cases: list[EvalCase], judge: Judge,
             out_dir: str, opt_llm: LLM | None = None, max_iters: int = 3, budget_s: float | None = None,
             agent_factory: Callable = default_agent_factory, progress=print, clock=time.time, sleep=time.sleep) -> dict:
    out = Path(out_dir)
    t_start = clock()

    # ---------------- iteration 0: baseline (dev AND holdout) ----------------
    progress(f"== baseline: {len(dev_cases)} dev + {len(holdout_cases)} holdout cases")
    dev0 = evaluate(definition, dev_cases, judge, agent_factory, progress, sleep)
    dev_seconds = max(compute_seconds(dev0), 1e-6)             # one dev pass: used to budget later iterations
    hold0 = evaluate(definition, holdout_cases, judge, agent_factory, progress, sleep)
    spent = compute_seconds(dev0) + compute_seconds(hold0)     # compute time used so far (stalls capped)
    _save_iteration(out, 0, definition, dev0, hold0, None, "baseline")

    best, best_dev = definition, dev0
    history: list[dict] = []          # every patch tried, kept or reverted
    changelog: list[dict] = []
    scorecards = [{"iteration": 0, "decision": "baseline", "dev": scorecard(dev0),
                   "holdout": scorecard(hold0) if hold0 else None}]
    consecutive_reverts = 0
    stop_reason = f"reached the maximum of {max_iters} iterations"
    base_infra = [r.case.id for r in dev0 + hold0 if is_infra_error(r.trace)]
    if base_infra:
        max_iters = 0
        stop_reason = (f"the model server failed on {base_infra} during the baseline even after retries; "
                       "the baseline is not trustworthy, so no patch was tried")

    for n in range(1, max_iters + 1):
        failures = classify_all(best_dev)
        if not failures:
            stop_reason = "all dev cases pass"
            break
        if budget_s is not None and spent + dev_seconds * 1.1 > budget_s:
            stop_reason = f"time budget ({budget_s:.0f}s) would be exceeded by another iteration"
            break
        examples = {r.case.id: f"Q: {r.case.question} | A: {r.trace.answer.answer[:160]!r} | why: {'; '.join(r.verdict.reasons)[:120]}"
                    for r in best_dev if not r.verdict.passed}
        patch = propose(best, failures, history, _n_chunks(best), examples, opt_llm)
        if patch is None:
            stop_reason = "no new patch to try for the remaining failures"
            break
        progress(f"== iteration {n}: {patch['category']} x{len(patch['addresses'])} -> {describe(patch)}")
        candidate = apply_patch(best, patch)

        before_ids = _passed(best_dev)
        t1 = clock()
        cand_dev = evaluate(candidate, dev_cases, judge, agent_factory, progress, sleep)
        dev_seconds = max(compute_seconds(cand_dev), 1e-6)
        spent += compute_seconds(cand_dev)
        after_ids = _passed(cand_dev)
        infra = [r.case.id for r in cand_dev if is_infra_error(r.trace)]
        fixed, broken = sorted(after_ids - before_ids), sorted(before_ids - after_ids)
        if infra:      # the server failed: this iteration says nothing about the patch (real incident: Ollama restarted)
            fixed, broken = [], []
        kept = not infra and len(after_ids) > len(before_ids)
        decision = "inconclusive" if infra else ("kept" if kept else "reverted")
        changelog.append({
            "iteration": n, "category": patch["category"], "source": patch["source"],
            "what_changed": describe(patch), "why": patch["rationale"],
            "failures_addressed": patch["addresses"], "fixed": fixed, "broken": broken,
            "dev_passed_before": len(before_ids), "dev_passed_after": len(after_ids), "dev_total": len(dev_cases),
            "decision": decision,
            "infrastructure_errors": infra,
            "decision_reason": (f"INCONCLUSIVE: the model server failed on {infra} even after retries, so these results say "
                                "nothing about the patch; it was reverted and will be tried again" if infra else
                                "net gain on dev cases" if kept else
                                "no net gain: " + (f"broke {broken}" if broken else "fixed nothing") + "; patch reverted"),
            "seconds": round(clock() - t1, 1)})
        _save_iteration(out, n, candidate, cand_dev, [], patch, decision)
        scorecards.append({"iteration": n, "decision": decision, "dev": scorecard(cand_dev), "holdout": None})
        progress(f"   dev {len(before_ids)} -> {len(after_ids)} of {len(dev_cases)}: {decision}"
                 + (f" (fixed {fixed})" if fixed else "") + (f" (broke {broken})" if broken else ""))
        if infra:
            continue           # not recorded as tried and not counted against the early stop: the patch was never judged
        history.append(patch)
        if kept:
            best, best_dev, consecutive_reverts = candidate, cand_dev, 0
        else:
            consecutive_reverts += 1
            if consecutive_reverts >= 2:
                stop_reason = "two consecutive patches did not pay for themselves (early stop)"
                break

    # ---------------- final holdout with the best definition ----------------
    if best is definition:
        hold_final = hold0
    else:
        progress("== final holdout run with the best definition")
        hold_final = evaluate(best, holdout_cases, judge, agent_factory, progress, sleep)
    scorecards.append({"iteration": "final", "decision": "final", "dev": scorecard(best_dev),
                       "holdout": scorecard(hold_final) if hold_final else None})
    save_definition_at(best, out / "final_agent_definition.yaml")
    _dump(out / "changelog.json", changelog)
    _dump(out / "scorecards.json", scorecards)
    return {"best_definition": best, "changelog": changelog, "scorecards": scorecards, "stop_reason": stop_reason,
            "baseline": {"dev": dev0, "holdout": hold0}, "final": {"dev": best_dev, "holdout": hold_final},
            "seconds": round(clock() - t_start, 1)}
