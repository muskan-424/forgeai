"""Honest report: what was built, how well it works WITH uncertainty, what still fails and why, what to distrust."""
from __future__ import annotations

from collections import Counter

from .critic import EXPLANATION, classify_all
from .evalrun import wilson


def fmt(sc: dict | None) -> str:
    if not sc or not sc["n"]:
        return "n/a"
    lo, hi = sc["ci95"]
    return f"{sc['passed']}/{sc['n']} ({sc['rate']:.0%}; 95% CI {lo:.0%}-{hi:.0%})"


def _residual(results, label: str) -> list[dict]:
    out = []
    by_id = {r.case.id: r for r in results}
    for f in classify_all(results):
        r = by_id[f["case_id"]]
        out.append({"set": label, "id": f["case_id"], "type": f["type"], "question": f["question"],
                    "category": f["category"], "why_category": EXPLANATION[f["category"]], "detail": f["detail"],
                    "judge_reasons": r.verdict.reasons, "agent_answer": r.trace.answer.answer[:240],
                    "refused": r.trace.answer.refused})
    return out


def build_report(ctx: dict) -> tuple[str, dict]:
    loop, req, cases = ctx["loop"], ctx["requirements"], ctx["cases"]
    first, last = loop["scorecards"][0], loop["scorecards"][-1]
    d0, d1 = first["dev"]["overall"], last["dev"]["overall"]
    h0 = first["holdout"]["overall"] if first["holdout"] else None
    h1 = last["holdout"]["overall"] if last["holdout"] else None
    residual = _residual(loop["final"]["dev"], "dev") + _residual(loop["final"]["holdout"], "holdout")
    kept = [c for c in loop["changelog"] if c["decision"] == "kept"]
    inconclusive = [c for c in loop["changelog"] if c["decision"] == "inconclusive"]
    n_total = len(cases)

    L: list[str] = []
    add = L.append
    add(f"# Quality report: {ctx['name']}\n")
    add(f"**Brief:** {req['brief']}\n")

    # ---------------- verdict ----------------
    add("## Verdict\n")
    add(f"- **Dev set** (used by the optimizer): {fmt(d0)} at baseline -> {fmt(d1)} after {len(kept)} kept patch(es) "
        f"({len(loop['changelog'])} tried).")
    if h1:
        add(f"- **Holdout set** (never used to patch): {fmt(h0)} at baseline -> {fmt(h1)} final.")
    add(f"- The loop stopped because: {loop['stop_reason']}.")
    for c in inconclusive:
        add(f"- **Iteration {c['iteration']} is INCONCLUSIVE:** the model server failed on {', '.join(c['infrastructure_errors'])} "
            "even after retries, so that iteration says nothing about its patch.")
    dev_n = d1["n"] if d1 else 0
    for c in kept:
        net = c["dev_passed_after"] - c["dev_passed_before"]
        notes = []
        if dev_n < 20 and net <= 2:
            notes.append(f"a gain of {net} case(s) on {dev_n} dev cases is within normal run-to-run noise, so it does not "
                         "prove the patch helped")
        if c["source"] == "model":
            notes.append("the rule was written by the 3B model and should be read by a person before it is trusted")
        if notes:
            add(f"- **Check iteration {c['iteration']}'s kept patch:** " + "; ".join(notes) + ".")
    add(f"- {len(residual)} case(s) still fail; see 'What still fails and why'.")
    tiny = [s for s in (d1, h1) if s and s["n"] < 30]
    add(f"- **How much to trust these numbers:** the eval set has only {n_total} cases, so every rate carries a wide "
        "confidence interval (shown above). "
        + ("A perfect score on so few cases is weak evidence, not proof. " if any(s["rate"] == 1.0 for s in tiny) else "")
        + "Differences of one or two cases are within noise.\n")

    # ---------------- what was built ----------------
    d = ctx["final_definition"]
    add("## What was built\n")
    add(f"- **Scope:** {req['scope']}\n- **Audience:** {req['audience']}\n- **Tone:** {req['tone']}")
    add(f"- **Must answer:** {', '.join(req['must_answer']) or '-'}\n- **Must refuse:** {', '.join(req['must_refuse']) or '-'}")
    add(f"- **Retrieval:** chunk {d.retrieval.chunk_size} chars (overlap {d.retrieval.chunk_overlap}), top-{d.retrieval.top_k}, "
        f"{'hybrid BM25+embeddings' if d.retrieval.use_embeddings else 'BM25 only'}")
    add(f"- **Tools attached:** {', '.join(t.name for t in d.tools) or 'none'}")
    for t in ctx["notes"]["tool_decisions"]:
        add(f"  - `{t['tool']}`: {'ATTACHED' if t['attached'] else 'not attached'}: {t['reason']}")
    add(f"- **Rules in the final prompt:** {len(d.rules)}")
    add(f"- **Refusal gate:** {'ON' if d.refusal_gate else 'off'}" +
        (f" (forbidden topics checked before answering: {', '.join(d.refusal_topics)})" if d.refusal_topics else
         " (no forbidden topics to check)") + "\n")

    # ---------------- groundability ----------------
    g = ctx["groundability"]
    add("## Can the documents support the brief?\n")
    for e in g["entries"]:
        add(f"- **{e['topic']}**: {e['status']}" + (f" ({e['note']})" if e.get("note") else ""))
    for w in g["warnings"]:
        add(f"- WARNING: {w}")
    add("")

    # ---------------- eval set ----------------
    add("## Evaluation set\n")
    types = Counter(c.type for c in cases)
    splits = Counter(c.split for c in cases)
    add(f"- {n_total} generated cases: " + ", ".join(f"{v} {k}" for k, v in types.items()) +
        f"; split {splits.get('dev', 0)} dev / {splits.get('holdout', 0)} holdout.")
    er = ctx.get("eval_report") or {}
    acc = er.get("attempted_accepted", {})
    rej = er.get("rejections", {})
    if acc:
        att = sum(v for k, v in acc.items() if k.endswith(":attempted"))
        ok = sum(v for k, v in acc.items() if k.endswith(":accepted"))
        add(f"- Generation yield: {ok} of {att} attempts passed verification; {sum(rej.values())} were discarded.")
        for k, v in sorted(rej.items(), key=lambda kv: -kv[1])[:5]:
            add(f"  - {v}x {k}")
    add("- The cases were written by the same model family as the agent (and verified mechanically against the "
        "documents), so blind spots shared by that family may be invisible to this evaluation.\n")

    # ---------------- judge ----------------
    cal = ctx.get("calibration")
    add("## How reliable is the judge?\n")
    if cal:
        add(f"- Calibration on {cal['probes']} synthetic answers with known verdicts: accuracy {cal['accuracy']:.0%}, "
            f"false passes {cal['false_pass']}, false fails {cal['false_fail']}, "
            f"facts wrongly accepted by the model judge {cal['llm_false_accepts']}.")
        add("  - by probe: " + ", ".join(f"{k} {v}" for k, v in cal["by_kind"].items()))
    else:
        add("- Calibration was skipped for this run.")
    add("- Defences: deterministic checks first (refusal flag, verbatim citations, numbers present in the context, "
        "fact matching); the model judge only sees facts the matcher could not confirm, one answer at a time, "
        "over two passes with reversed fact order, and disagreement counts as a miss.")
    add("- Calibration probes are synthetic and easy; they show the judge is not trivially fooled, not that it "
        "is right on every real answer.\n")

    # ---------------- scorecard ----------------
    add("## Per-iteration scorecard\n")
    add("| iteration | decision | dev | holdout |\n|---|---|---|---|")
    for s in loop["scorecards"]:
        add(f"| {s['iteration']} | {s['decision']} | {fmt(s['dev']['overall'])} | {fmt(s['holdout']['overall']) if s['holdout'] else '-'} |")
    add("")

    # ---------------- changelog ----------------
    add("## Changelog\n")
    if not loop["changelog"]:
        add("No patch was needed or possible; the generated definition was not changed.\n")
    for c in loop["changelog"]:
        add(f"### Iteration {c['iteration']}: {c['decision'].upper()} ({c['category']})")
        add(f"- **Changed:** {c['what_changed']}\n- **Why:** {c['why']} (source: {c['source']})")
        add(f"- **Failures addressed:** {', '.join(c['failures_addressed'])}")
        add(f"- **Score before -> after (dev):** {c['dev_passed_before']} -> {c['dev_passed_after']} of {c['dev_total']}")
        add(f"- **Fixed:** {', '.join(c['fixed']) or 'none'}; **broke:** {', '.join(c['broken']) or 'none'}")
        add(f"- **Decision:** {c['decision_reason']}\n")

    # ---------------- residual failures ----------------
    add("## What still fails and why\n")
    if not residual:
        add("Every evaluated case passed. With so few cases this does not mean the agent is flawless; "
            "see 'How much to trust these numbers'.\n")
    for r in residual:
        add(f"- **{r['id']}** ({r['set']}, {r['type']}): \"{r['question']}\"")
        add(f"  - Root cause: **{r['category']}**, meaning {r['why_category']}.")
        add(f"  - Judge: {'; '.join(r['judge_reasons']) or r['detail']}")
        add(f"  - Agent said: \"{r['agent_answer']}\"")
    add("")
    cats = Counter(r["category"] for r in residual)
    if cats:
        add("Failure categories remaining: " + ", ".join(f"{k} x{v}" for k, v in cats.items()) + ".\n")

    # ---------------- cost ----------------
    add("## Cost\n")
    replayed = False
    for name, u in ctx["usage"].items():
        extra = f" + {u['cached_seconds']:.0f}s replayed from the cache" if u.get("cached_seconds") else ""
        add(f"- {name}: {u['calls']} fresh model calls, {u['cache_hits']} answered from the cache; "
            f"{u['seconds']:.0f}s of fresh model time{extra}")
        replayed = replayed or u["cache_hits"] > u["calls"]
    if replayed:
        add("- NOTE: this run mostly replayed identical requests from the cache, so the time above understates what a "
            "first, uncached run costs (on this CPU roughly 30-150 s per agent call).")
    add(f"- Total wall-clock: {ctx['total_seconds'] / 60:.1f} min\n")

    add("## Known limits of this run\n")
    add("- Small open models (3B) on CPU: answers and judgements are noisy; one run is one sample.")
    add("- Eval cases are limited to what the documents literally say; reasoning across many documents is under-tested.")
    add("- The optimizer can only change retrieval parameters, prompt rules and temperature, not the documents or the model.")

    js = {"name": ctx["name"], "scorecards": loop["scorecards"], "changelog": loop["changelog"],
          "stop_reason": loop["stop_reason"], "residual_failures": residual, "calibration": cal,
          "eval_composition": {"types": dict(types), "splits": dict(splits)}, "usage": ctx["usage"],
          "total_seconds": ctx["total_seconds"]}
    return "\n".join(L) + "\n", js
