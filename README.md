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
python -m pytest tests -q                  # 107 tests, no model needed (mock backend)

# the factory, unattended
python forge_run.py --brief-file demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out runs/my_run --auto \
       --n-eval 10 4 4 3 --min-cases 20 --budget-minutes 150

# chat with any produced agent
python chat.py runs/demo_hr_policy/final_agent_definition.yaml
```

Useful flags: `--stage1-only` (just intake, guard, generator), `--min-cases` (default 20), `--no-calibrate`, `--reuse-evals`, `--max-iters`,
`--force` (generate even if nothing is supported), and without `--auto` the interviewer asks its questions on the
terminal. Set `FORGE_CACHE_DIR` to a roomy drive: every model response is cached on disk, so reruns are fast.

## How each requirement is met, and where the evidence is

| # | Requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Intake interviewer, at most 5 questions, requirements doc | `forge/intake.py`. Open slots only; unattended mode records every default as an assumption. Claims about the brief are checked against the brief's own words; deterministic backstops read the brief's sentences when the model's list misses a topic or a refusal. | `runs/*/requirements.md` |
| 2 | Groundability guard | `forge/groundability.py`. A topic is supported only if rare-word lexical coverage and the model agree, with deterministic evidence sentences (the model never writes quotes). Unsupported topics become refusal rules; all-unsupported blocks generation. | HR demo: **pension contributions flagged unsupported**, refused by the agent, and tested (`runs/demo_hr_policy/groundability.json`) |
| 3 | Agent generator | `forge/generator.py`. Tool gate (precondition + evidence in the brief + necessity check for weak evidence), retrieval parameters from document statistics, persona by the model, rules as separate patchable strings. | `runs/*/agent_definition.yaml`, `generation_notes.json` (every tool decision, attached or rejected, with its reason) |
| 4 | Eval generator, >= 20 cases, generated not hand-written | `forge/evalgen.py`. Sentence-anchored cases with deterministic verification (verbatim sentence, atomic facts in it, question words from it, direction and unit checks, out-of-scope candidates must score low and pass a model answerability check). Prompts in `prompts/`. A minimum size is guaranteed (`--min-cases`, default 20): if verification rejects too many attempts, extra answerable and multi-document cases are generated. Table rows are never rewritten as vague questions, and questions that say 'this policy' without naming it are rejected. Both demos use 20 cases. | `runs/*/evals.json`, `evals_generation_report.json` (yield and rejection examples) |
| 5 | Judge with bias defences | `forge/judge.py`. Deterministic checks first; the model only sees unconfirmed facts, pointwise, two passes with reversed order, disagreement counts as a miss; different model family; self-calibration with synthetic probes (gold, padded, wrong, hedged refusal...). | `runs/*/judge_calibration.json` |
| 6 | Critic-optimizer loop, max 3 iterations, targeted fixes | `forge/critic.py` (5 rule-based categories), `forge/optimizer.py` (whitelisted typed patches), `forge/loop.py` (dev-only optimization, revert on no net gain, early stop, compute-time budget, server errors retried with backoff and reported as INCONCLUSIVE instead of blamed on a patch). The optimizer can also switch on a runtime **refusal gate** (below). | `runs/*/changelog.json`, `iterations/iter_N/` |
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
| Eval cases | 20 (14 dev / 6 holdout) | 20 (14 dev / 6 holdout) |
| Dev baseline -> final | 13/14 -> 14/14 | 11/14 -> 13/14 |
| Holdout | 5/6 -> 5/6 | 5/6 -> 5/6 |
| Patches | 1 tried, 1 kept | 3 tried, 2 kept; refusal gate on from the start |

- **HR:** the pension gap was flagged as unsupported, turned into a refusal rule, and tested with a freshly generated
  pension question, which the agent refused. The one kept patch (a rule to quote word for word) fixed one dev case,
  which the report itself calls within run-to-run noise. The remaining failure is a real agent error on a badly worded
  holdout question (it says "25 days" where the policy says pro rata).
- **Procurement:** a retrieval patch (top-k 4 -> 6) fixed a supplier lookup, and two rules (quote word for word; refuse
  what the context does not state) fixed the total-cost-of-ownership question that had failed in every earlier run. A
  third, model-written rule broke that case again and was reverted. Each kept gain is one case on 14, flagged as noise.
  Still failing: a fabricated quote on an otherwise right answer, and a holdout case that blends two reporting deadlines.
- **What the holdout taught us:** in the same run with the gate left to the optimizer
  (`runs/_dev_history/demo_procurement_20cases_gate_off/`), the contract-breach question sat in the holdout, the
  optimizer never saw it, the gate stayed off, and the agent gave legal advice (holdout 4/6). A refusal the brief states
  explicitly is now enforced from the start; with it, that case is refused and the holdout is 5/6, with no dev case
  affected.

**The refusal gate** (`prompts/refusal_gate.txt`, `Agent._gate`): the definition always lists the brief's forbidden topics
and the groundability gaps. The gate starts **on** when the brief explicitly forbids something (otherwise off), and the
optimizer can switch it on or off. When on, a question that shares a content word with a forbidden topic gets a short
classification call (about 8-16 s) before answering; a yes refuses cleanly without the main call. It was tuned against
the real 3B model: unconstrained it blocked 4 of 5 legitimate questions that merely mentioned suppliers or warranties, so
the lexical precondition is essential. Limits: a paraphrase that shares no word with the topic ("can we sue...") slips
past, and borderline questions can be blocked. The optimizer reverts the gate if it costs more cases than it fixes.

Neither demo claims 100% on held-out data. With 20 cases (about 14 dev / 6 holdout), every rate has a wide confidence interval and one-case differences are noise.

## Rehearsal on an unseen brief (preparing for the live review)

The review uses a brief and documents ForgeAI has never seen, so the factory was rehearsed on a new case built to differ
from both demos: a ship-agent assistant for a fictional container terminal (`rehearsal/westerhaven/`), with file types
the demos never used (a Word .docx without detectable headings, a plain .txt, Markdown with a fee table), a brief that
needs the weather and date tools, a planted gap (pilotage fees) and an explicit refusal (medical advice about crew).

**What the first attempt exposed, and what was fixed (generic fixes, each with a regression test):**
- Markdown table rows were dropped as "headings", so the fee table could never be tested or cited. Pipe tables are now
  turned into self-describing rows ("Vessel length: over 300 metres; Fee per metre per day: EUR 3.20").
- Three **false gaps** became refusal rules, so the agent would have refused the brief's own core questions:
  "Certificate Validity" (no stemming link between *validity* and *valid*), "many days remain until" (the date tool's
  job was treated as a document gap), and the junk topic "report" (a verb taken from "...and report the weather").
  Fixed with -ity stemming, letting helper tools support topics that explicitly name their job, and excluding request
  verbs from topics.
- The refusal gate now checks only what the requester explicitly forbade, so a false gap can never become a hard block.

**After the fixes** (`runs/rehearsal_westerhaven/`): 8 topics from the brief; certificate validity and expiry dates
supported by the documents; weather and "days remain until" supported by the attached `open_meteo` and `current_date`
tools; pilotage fees the only gap; CSV and calculator rejected with reasons; refusal gate on for medical advice; the
interactive intake asked 2 questions (tone, and whether the booking document is complete).

**Live question round** (`runs/rehearsal_westerhaven/live_questions*.txt`), after closing other apps (the first attempt,
with 0.5 GB free RAM, timed out after 1,083 s):

| Question | First answer | After the fix |
|---|---|---|
| Is the chief engineer fit for duty after heart surgery? (forbidden) | refused | refused (by the prompt rule: the gate's word check missed this paraphrase) |
| Pilotage fees for a 300 m ship? (the gap) | refused | refused |
| Berth fee per metre per day, 250 m vessel? | EUR 2.60, citation verified | unchanged |
| How long is an adult seafarer's medical certificate valid? | 2 years, citation verified | unchanged |
| Current weather at the terminal in Rotterdam? | **refused; tool never called** | live Open-Meteo data, tool result cited |
| Days until the certificate expires on 2027-03-15? | **"183 days" (invented; tool never called)** | **108 days** (correct), tool result cited |

The most important finding: the factory attached the right tools, but the 3B model almost never chose to call them (the
procurement catalogue answers also came from retrieval, not from the CSV tool). The runtime now **pre-fetches** a tool
when a question clearly needs it (a weather word plus a named place; a date or date wording) and hands the result to the
model to read. The model-initiated tool path remains for everything else. A full timed run on the rehearsal was not done.

**For the live review on this hardware (CPU only, 7 GB RAM):** boot fresh, close everything else (VS Code windows,
browsers, chat apps, OneDrive sync), and check `ollama list`. Run `--stage1-only` first (about 2-10 minutes) to show
requirements, groundability and tool choices, then the full run with a time box, for example
`--min-cases 20 --budget-minutes 45`; the loop stops iterating when the budget is spent and the report says so. A first,
uncached full run takes one to three hours here; a machine with more RAM or a GPU is far faster.

## Limits you should know about

- **Eval sets are still small.** Each demo has 20 cases, the brief's minimum, split about 14 dev / 6 holdout, so the
  intervals stay wide (a 6-case holdout spans tens of percentage points). `--min-cases` raises it, at roughly 1-4 minutes
  per agent call on this CPU.
- **Eval quality is mixed.** Cases are verified mechanically (real sentence, real facts, no direction or unit errors), but a 3B
  model still writes some clumsy or ambiguous questions; the reports show which failures trace back to the test.
- **Same model family wrote the tests and answers them**, so shared blind spots may be invisible. The judge is a
  different family.
- **Reported run times are mostly cache replays.** The demo reports show fresh model time and replayed time separately;
  a first, uncached run of either demo takes one to three hours on this CPU (the slowest single agent answer seen was
  about 8 minutes, for a question that needed several tool calls).
- Reruns change results (model sampling and prompt edits); the committed runs are one sample each. The two demo runs were produced before the rehearsal fixes (stemming, table rows, gate topics); those fixes were verified by tests and by the rehearsal's stages 1-3, not by re-running the demos.
- **Hardware is the practical limit.** With other apps open, the 3B model can time out entirely (seen in the rehearsal). See the live-review checklist above.
- `runs/_dev_history/` holds earlier development runs, including a procurement run whose loop never executed because
  one call stalled for 2.6 hours (the budget now counts compute time), and procurement runs interrupted by a model-server outage (agent errors are now
  retried with backoff, and an iteration with unresolved server errors is reported as INCONCLUSIVE).

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
rehearsal/        an unseen brief + document folder (.docx, .txt, Markdown table) used to rehearse the live review
runs/             complete artifacts of both demonstration runs, plus runs/rehearsal_westerhaven (stages 1-3)
tests/            107 tests: deterministic logic, mocked pipeline end to end, PDF ingestion, refusal gate, loop robustness
ARCHITECTURE.md   the original design plan; this README describes what was actually built
```
