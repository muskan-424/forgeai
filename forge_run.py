"""ForgeAI: one paragraph + a folder of documents in, a tested agent and an honest report out.

  python forge_run.py --brief "I need an assistant that answers HR policy questions for shore staff" \
                      --docs examples/sample_docs --out runs/hr_sample --auto

Without --auto the interviewer asks its clarifying questions (at most 5) on the terminal.
Artifacts in --out: requirements.md, groundability.json, agent_definition.yaml, evals.json,
judge_calibration.json, iterations/, changelog.json, scorecards.json, final_agent_definition.yaml, report.md."""
import argparse
import sys
from pathlib import Path

from forge.factory import run_factory, run_stage1
from forge.groundability import GroundabilityError
from forge.llm import LLMUnavailable
from forge.power import keep_awake


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--brief")
    g.add_argument("--brief-file")
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--auto", action="store_true", help="unattended: answer clarifying questions with recorded assumptions")
    ap.add_argument("--force", action="store_true", help="generate even if no topic is supported by the documents")
    ap.add_argument("--stage1-only", action="store_true", help="stop after generating the agent definition (no evals)")
    ap.add_argument("--n-eval", nargs=4, type=int, default=[8, 3, 4, 3], metavar=("ANS", "MULTI", "OOS", "AMB"),
                    help="number of eval cases per type (default 8 3 4 3)")
    ap.add_argument("--min-cases", type=int, default=20,
                    help="minimum eval-set size (default 20, as the assignment requires); extra cases are generated if verification rejects too many")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--budget-minutes", type=float, default=None, help="stop iterating when another iteration would exceed this")
    ap.add_argument("--no-calibrate", action="store_true", help="skip the judge self-calibration")
    ap.add_argument("--reuse-evals", action="store_true", help="reuse evals.json in --out if present")
    a = ap.parse_args()
    brief = a.brief or Path(a.brief_file).read_text(encoding="utf-8").strip()

    def ask(q: str) -> str:
        return input(f"\n? {q}\n> ")

    try:
        with keep_awake():              # ask Windows not to sleep during a long run (a sleep once stalled a call for 2.6 h)
            if a.stage1_only:
                res = run_stage1(brief, a.docs, a.out, auto=a.auto, ask=ask, force=a.force)
                u = res["usage"]
                print(f"\nDone. LLM calls: {u.calls} (cache hits {u.cache_hits}), {u.seconds:.0f}s")
            else:
                run_factory(brief, a.docs, a.out, auto=a.auto, ask=ask, force=a.force, n_eval=tuple(a.n_eval),
                            min_cases=a.min_cases, max_iters=a.max_iters, budget_minutes=a.budget_minutes,
                            calibrate=not a.no_calibrate, reuse_evals=a.reuse_evals)
    except GroundabilityError as e:
        print(f"\n{e}")
        return 2
    except LLMUnavailable as e:
        print(f"\nModel server problem: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
