"""Run an agent over eval cases, judge every result, build the scorecard."""
from __future__ import annotations

import math
import time
from collections import defaultdict

from .agent_runtime import Agent
from .judge import Judge
from .schemas import CaseResult, EvalCase


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n. Honest about small samples."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(centre - half, 0), 3), round(min(centre + half, 1), 3))


RETRY_WAITS_S = (15, 45)     # a restarting model server needs seconds, not milliseconds


def is_infra_error(trace) -> bool:
    """The agent produced no answer because the run itself failed (server down, timeout), not because it answered badly."""
    return bool(trace.error) and not trace.answer.answer.strip() and not trace.answer.refused


def run_eval(agent: Agent, cases: list[EvalCase], judge: Judge, progress=print, sleep=time.sleep,
             retry_waits=RETRY_WAITS_S) -> list[CaseResult]:
    results = []
    for i, case in enumerate(cases, 1):
        trace = agent.ask(case.question)
        for wait in retry_waits:
            if not is_infra_error(trace):
                break
            progress(f"  ... agent error on {case.id} ({trace.error[:60]}); waiting {wait}s, then retrying")
            sleep(wait)                                  # a dropped connection or a timeout is not a wrong answer
            trace = agent.ask(case.question)
        verdict = judge.judge(case, trace)
        results.append(CaseResult(case=case, trace=trace, verdict=verdict))
        progress(f"  [{i}/{len(cases)}] {case.id:<8} {'PASS' if verdict.passed else 'FAIL'}  "
                 f"{trace.latency_s:5.1f}s  {case.question[:60]}")
    return results


def _rate(rs: list[CaseResult]) -> dict:
    n = len(rs)
    k = sum(r.verdict.passed for r in rs)
    lo, hi = wilson(k, n)
    return {"passed": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [lo, hi]}


def scorecard(results: list[CaseResult]) -> dict:
    by_split, by_type = defaultdict(list), defaultdict(list)
    for r in results:
        by_split[r.case.split].append(r)
        by_type[r.case.type].append(r)
    non_oos = [r for r in results if not r.case.should_refuse]
    oos = [r for r in results if r.case.should_refuse]
    return {
        "overall": _rate(results),
        "by_split": {k: _rate(v) for k, v in sorted(by_split.items())},
        "by_type": {k: _rate(v) for k, v in sorted(by_type.items())},
        "refusal_accuracy": _rate(oos)["rate"] if oos else None,
        "wrong_refusal_rate": round(sum("wrong refusal" in " ".join(r.verdict.reasons) for r in non_oos) / len(non_oos), 3) if non_oos else None,
        "grounded_rate": round(sum(r.verdict.grounded for r in results) / len(results), 3) if results else None,
        "uncertain_judgements": sum(r.verdict.uncertain for r in results),
        "mean_latency_s": round(sum(r.trace.latency_s for r in results) / len(results), 1) if results else None,
        "failures": [{"id": r.case.id, "type": r.case.type, "split": r.case.split, "question": r.case.question,
                      "reasons": r.verdict.reasons} for r in results if not r.verdict.passed],
    }
