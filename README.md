# ForgeAI: the agent that builds agents

Give ForgeAI a one-paragraph brief and a folder of documents. It interviews the requester, checks that the
documents can actually support the brief, generates an agent definition, writes its own test set, judges the
agent, diagnoses failures, applies targeted fixes, and hands back the agent plus an honest quality report.
Everything runs on small local models (Ollama): `qwen2.5:3b` for the agent and generation, `llama3.2:3b` as the
judge (a different model family on purpose), `nomic-embed-text` for optional embeddings.

```
brief + docs/
  -> intake (<=5 questions)  -> groundability guard  -> generator (prompt, retrieval, tools, output contract)
  -> eval generator (verified)  -> judge calibration
  -> loop (max 3): run -> judge -> critic classifies -> ONE typed patch -> re-run -> keep or revert
  -> report.md (scorecards, changelog, what still fails and why)
```

## Quick start

```powershell
pip install -r requirements.txt            # pydantic, pyyaml, pypdf, requests, pytest, fpdf2
ollama pull qwen2.5:3b ; ollama pull llama3.2:3b ; ollama pull nomic-embed-text
python -m pytest tests -q                  # 84 tests, no model needed (mock backend)

# the factory, unattended
python forge_run.py --brief-file demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out runs/my_run --auto \
       --n-eval 6 3 3 2 --budget-minutes 50

# chat with any produced agent
python chat.py runs/demo_hr_policy/final_agent_definition.yaml
```

Useful flags: `--stage1-only` (just intake, guard, generator), `--no-calibrate`, `--reuse-evals`, `--max-iters`,
`--force` (generate even if nothing is supported), and without `--auto` the interviewer asks its questions on the
terminal. Set `FORGE_CACHE_DIR` to a roomy drive: every model response is cached on disk, so reruns are fast.

## How each requirement is met, and where the evidence is

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Intake interviewer, at most 5 questions, requirements doc | `forge/intake.py`. Open slots only; unattended mode records every default as an assumption. Claims about the brief are checked against the brief's own words; deterministic backstops read the brief's sentences when the model's list misses a topic or a refusal. | `runs/*/requirements.md` |
| 2 | Groundability guard | `forge/groundability.py`. A topic is supported only if rare-word lexical coverage and the model agree, with deterministic evidence sentences (the model never writes quotes). Unsupported topics become refusal rules; all-unsupported blocks generation. | HR demo: **pension contributions flagged unsupported**, refused by the agent, and tested (`runs/demo_hr_policy/groundability.json`) |
| 3 | Agent generator | `forge/generator.py`. Tool gate (precondition + evidence in the brief + necessity check for weak evidence), retrieval parameters from document statistics, persona by the model, rules as separate patchable strings. | `runs/*/agent_definition.yaml`, `generation_notes.json` (every tool decision, attached or rejected, with its reason) |
| 4 | Eval generator, >= 20 cases, generated not hand-written | `forge/evalgen.py`. Sentence-anchored cases with deterministic verification (verbatim sentence, atomic facts in it, question words from it, direction and unit checks, out-of-scope candidates must score low and pass a model answerability check). Prompts in `prompts/`. **Note: the demos use 11-13 cases, not 20+** (see limits). | `runs/*/evals.json`, `evals_generation_report.json` (yield and rejection examples) |
| 5 | Judge with bias defences | `forge/judge.py`. Deterministic checks first; the model only sees unconfirmed facts, pointwise, two passes with reversed order, disagreement counts as a miss; different model family; self-calibration with synthetic probes (gold, padded, wrong, hedged refusal...). | `runs/*/judge_calibration.json` |
| 6 | Critic-optimizer loop, max 3 iterations, targeted fixes | `forge/critic.py` (5 rule-based categories), `forge/optimizer.py` (whitelisted typed patches), `forge/loop.py` (dev-only optimization, revert on no net gain, early stop, compute-time budget). | `runs/*/changelog.json`, `iterations/iter_N/` |
| 7 | Honest reporting | `forge/report.py`: Wilson intervals, per-iteration scorecard, residual failures with root cause, calibration, generation yield, cost, caveats. | `runs/*/report.md` |
| 8 | Runnable product | `chat.py` loads any definition; shows verified citations and tool calls. | tested on both final definitions |
| 9 | Two full demonstrations | HR policy (3 PDFs) and ship-spares procurement (CSV catalogue + 2 PDFs). Synthesis prompts committed. | `demos/*/synthesis_prompts.md`, `runs/demo_hr_policy/`, `runs/demo_procurement/` |

Stretch goals: **budget enforcement** (compute-time budget and early stopping) is done. A **human checkpoint** is only
partial (the interviewer is interactive, but there is no approve/edit step on the requirements document). The prompt
tournament, factory-vs-human baseline, regression mode and web UI are not implemented.

## The two demonstrations, read honestly

| | HR policy | Ship-spares procurement |
|---|---|---|
| Documents | 3 PDFs | 1 CSV (60 parts) + 2 PDFs |
| Tools attached | none (correct) | `csv_lookup`, `calculator` (weather/date rejected, with reasons) |
| Planted trap | brief asks for pension help the documents cannot support | brief forbids legal advice about contracts |
| Dev baseline -> final | 8/9 -> 9/9 | 5/8 -> 5/8 |
| Holdout | 3/4 -> 3/4 | 3/4 -> 3/4 |
| Patches | 1 tried, 1 kept | 2 tried, 0 kept (early stop) |

- **HR:** the pension gap was flagged, refused and tested end to end. The one kept patch (a rule to quote word for word)
  fixed one case, which the report itself calls within run-to-run noise. The remaining failure is a real agent error
  on a badly worded question.
- **Procurement:** the optimizer could not fix its failures and **correctly reverted both patches**. One failure is the
  brief's own refusal being violated (a contract-breach question was answered with a loosely related procedure), one is
  an out-of-scope question answered from general knowledge, and one is a bad test (a vague question that lost the part
  number). Prompt rules are too weak a lever for a 3B model here. The next step would be a runtime refusal gate
  (a cheap classification before answering), which the optimizer could switch on; it roughly doubles latency, so it
  was not added unasked.

Neither demo claims 100%. With 11-13 cases, every rate has a wide confidence interval and one-case differences are noise.

## Limits you should know about

- **Small eval sets.** The brief asks for at least 20 cases; the demos have 11-13 because the agent takes 30-200 s per
  question on this CPU and a run with a 3-iteration loop already takes 35-60 minutes. `--n-eval` raises it.
- **Eval quality is mixed.** Cases are verified mechanically (real sentence, real facts, no direction or unit errors), but a 3B
  model still writes some clumsy or ambiguous questions; the reports show which failures trace back to the test.
- **Same model family wrote the tests and answers them**, so shared blind spots may be invisible. The judge is a
  different family.
- **Cost figures in `runs/demo_hr_policy/report.md` understate the real cost.** That run replayed cached results, and the
  cache did not yet record replayed time (fixed afterwards; the procurement report shows it). A first uncached HR run
  took roughly 45 minutes.
- The HR demo was produced before the out-of-scope prompt for named refusals was revised, so its eval set differs
  slightly from what the current generator would write. Reruns change results (model sampling and prompt edits).
- `runs/_dev_history/` holds earlier development runs, including a procurement run whose loop never executed because
  one call stalled for 2.6 hours (the budget now counts compute time, and agent errors are retried once).

## Defects the real runs exposed (all fixed, each with a regression test)

Running on real documents and a real model found problems that mock tests could not: PDFs breaking sentences at line
wraps; the chunker dropping section headings; retrieval without stemming (false "not covered" verdicts); topics the
brief wanted answered being filed as refusals; a model vetoing explicitly requested tools; a calculator wrongly
"grounding" topics; a judge that passed a hedged refusal; a critic that filed quoting defects as hallucinations; the
optimizer's model writing a nonsense rule; no refusal test generated for a refusal named in the brief; and a stalled
call eating the whole time budget.

## Layout

```
forge/            the factory (intake, groundability, generator, evalgen, judge, critic, optimizer, loop, report, runtime)
forge/tools/      registry: csv_lookup, calculator, open_meteo, current_date (manifests + gating metadata)
prompts/          every meta-prompt the factory uses (committed)
demos/            documents, briefs and synthesis prompts for the two demonstrations
runs/             complete artifacts of both demonstration runs
tests/            84 tests: deterministic logic, mocked pipeline end to end, PDF ingestion
ARCHITECTURE.md   the original design plan; this README describes what was actually built
```
