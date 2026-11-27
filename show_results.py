"""One-screen summary of the committed results (no model calls):  python show_results.py"""
import json
from pathlib import Path

RUNS = Path(__file__).parent / "runs"


def rate(sc):
    if not sc:
        return "-"
    o = sc["overall"]
    lo, hi = o["ci95"]
    return f"{o['passed']}/{o['n']} ({o['rate']:.0%}, 95% CI {lo:.0%}-{hi:.0%})"


def demo(name, folder):
    d = RUNS / folder
    scs = json.loads((d / "scorecards.json").read_text(encoding="utf-8"))
    log = json.loads((d / "changelog.json").read_text(encoding="utf-8"))
    cal = json.loads((d / "judge_calibration.json").read_text(encoding="utf-8"))
    cases = json.loads((d / "evals.json").read_text(encoding="utf-8"))
    types = {}
    for c in cases:
        types[c["type"]] = types.get(c["type"], 0) + 1
    print(f"\n=== {name} ===")
    print(f"  eval set      : {len(cases)} generated cases {types}")
    print(f"  dev           : {rate(scs[0]['dev'])}  ->  {rate(scs[-1]['dev'])}")
    print(f"  holdout       : {rate(scs[0]['holdout'])}  ->  {rate(scs[-1]['holdout'])}   (never used by the optimizer)")
    print(f"  judge         : {cal['accuracy']:.0%} on {cal['probes']} calibration probes, {cal['false_pass']} false passes")
    for c in log:
        print(f"  iteration {c['iteration']}    : {c['decision'].upper():8} {c['what_changed'][:70]}"
              f"  (dev {c['dev_passed_before']} -> {c['dev_passed_after']})")


print("FORGEAI RESULTS (from the committed run artifacts)")
demo("HR policy assistant (3 PDFs)", "demo_hr_policy")
demo("Ship-spares procurement assistant (CSV + 2 PDFs)", "demo_procurement")

g = json.loads((RUNS / "rehearsal_westerhaven" / "groundability.json").read_text(encoding="utf-8"))
notes = json.loads((RUNS / "rehearsal_westerhaven" / "generation_notes.json").read_text(encoding="utf-8"))
print("\n=== Rehearsal on an unseen brief (.docx + .txt + Markdown table) ===")
for e in g["entries"]:
    print(f"  {e['status']:17} {e['topic']}" + (f"  [{e['tool']}]" if e.get("tool") else ""))
print("  tools attached: " + ", ".join(t["tool"] for t in notes["tool_decisions"] if t["attached"]))
print("  live questions: 6/6 correct after the tool pre-fetch fix (see runs/rehearsal_westerhaven/live_questions_after_fix.txt)")
print("\nEvery kept gain is one case on 14 and the reports flag it as within run-to-run noise.")
