# Quality report: hr-policy-questions-shore-agent

**Brief:** I need an assistant that answers HR policy questions for shore staff at a shipping company: leave and sick pay, travel and expenses, and working hours, notice periods and the disciplinary process. It should also tell staff how their pension contributions work. Keep a clear, friendly tone.

## Verdict

- **Dev set** (used by the optimizer): 8/9 (89%; 95% CI 56%-98%) at baseline -> 9/9 (100%; 95% CI 70%-100%) after 1 kept patch(es) (1 tried).
- **Holdout set** (never used to patch): 3/4 (75%; 95% CI 30%-95%) at baseline -> 3/4 (75%; 95% CI 30%-95%) final.
- The loop stopped because: all dev cases pass.
- **Check iteration 1's kept patch:** a gain of 1 case(s) on 9 dev cases is within normal run-to-run noise, so it does not prove the patch helped.
- 1 case(s) still fail; see 'What still fails and why'.
- **How much to trust these numbers:** the eval set has only 13 cases, so every rate carries a wide confidence interval (shown above). A perfect score on so few cases is weak evidence, not proof. Differences of one or two cases are within noise.

## What was built

- **Scope:** Answers HR policy questions for shore staff at a shipping company regarding leave and sick pay, travel and expenses, working hours, notice periods, and the disciplinary process.
- **Audience:** shore staff at a shipping company
- **Tone:** friendly
- **Must answer:** leave and sick pay, travel and expenses, working hours, notice periods, disciplinary process, pension contributions
- **Must refuse:** anything not covered by the provided documents, pension contributions (not covered by the documents)
- **Retrieval:** chunk 500 chars (overlap 60), top-4, BM25 only
- **Tools attached:** none
  - `csv_lookup`: not attached: precondition failed: no .csv document provided
  - `calculator`: not attached: no evidence in the brief that this tool is needed
  - `open_meteo`: not attached: no evidence in the brief that this tool is needed
  - `current_date`: not attached: no evidence in the brief that this tool is needed
- **Rules in the final prompt:** 8

## Can the documents support the brief?

- **leave and sick pay**: supported (every word of the topic occurs in one sentence of the documents)
- **travel and expenses**: supported
- **working hours**: supported (every word of the topic occurs in one sentence of the documents)
- **notice periods**: supported (every word of the topic occurs in one sentence of the documents)
- **disciplinary process**: partial (model said 'partial', lexical coverage 0.39)
- **pension contributions**: unsupported (no passage shares any words with the topic)
- WARNING: UNSUPPORTED: the documents do not cover 'pension contributions'. The agent will refuse it instead of guessing.
- WARNING: PARTIAL: the documents cover 'disciplinary process' only in part; expect refusals or gaps there.

## Evaluation set

- 13 generated cases: 6 answerable, 3 multi_doc, 2 ambiguous, 2 out_of_scope; split 9 dev / 4 holdout.
- Generation yield: 13 of 28 attempts passed verification; 14 were discarded.
  - 7x out_of_scope:too similar to the documents (might be answerable)
  - 2x answerable:question strays from its source sentence
  - 2x ambiguous:rewrite lost the topic
  - 1x answerable:question contradicts the direction of its source (maximum vs minimum)
  - 1x multi_doc:does not need both passages
- The cases were written by the same model family as the agent (and verified mechanically against the documents), so blind spots shared by that family may be invisible to this evaluation.

## How reliable is the judge?

- Calibration on 16 synthetic answers with known verdicts: accuracy 100%, false passes 0, false fails 0, facts wrongly accepted by the model judge 0.
  - by probe: gold 2/2, padded_gold 2/2, wrong_facts 2/2, padded_wrong 2/2, wrong_refusal 2/2, oos_refused 2/2, oos_answered 2/2, oos_hedged 2/2
- Defences: deterministic checks first (refusal flag, verbatim citations, numbers present in the context, fact matching); the model judge only sees facts the matcher could not confirm, one answer at a time, over two passes with reversed fact order, and disagreement counts as a miss.
- Calibration probes are synthetic and easy; they show the judge is not trivially fooled, not that it is right on every real answer.

## Per-iteration scorecard

| iteration | decision | dev | holdout |
|---|---|---|---|
| 0 | baseline | 8/9 (89%; 95% CI 56%-98%) | 3/4 (75%; 95% CI 30%-95%) |
| 1 | kept | 9/9 (100%; 95% CI 70%-100%) | - |
| final | final | 9/9 (100%; 95% CI 70%-100%) | 3/4 (75%; 95% CI 30%-95%) |

## Changelog

### Iteration 1: KEPT (hallucination)
- **Changed:** + rule: Each citation quote must be one or more whole sentences copied exactly, word for word, from the CONTEXT; never reword, shorten or merge sentences inside a quote.
- **Why:** add proven rule for hallucination/citation (source: library)
- **Failures addressed:** amb-02
- **Score before -> after (dev):** 8 -> 9 of 9
- **Fixed:** amb-02; **broke:** none
- **Decision:** net gain on dev cases

## What still fails and why

- **mul-02** (holdout, multi_doc): "How many months of annual leave do employees receive if they join in the middle of the year, and what is the maximum time frame for the HR director to make a final decision on a grievance?"
  - Root cause: **prompt_defect**, meaning the context contained the answer and the agent did not refuse, but the answer was wrong or incomplete.
  - Judge: facts matched 1/2
  - Agent said: "If an employee joins in the middle of the year, they receive 25 days of paid annual leave per leave year. The maximum time frame for the HR director to make a final decision on a grievance is 20 working days."

Failure categories remaining: prompt_defect x1.

## Cost

- factory (intake, guard, generator, evalgen): 0 model calls (26 cache hits), 0s of model time
- agent: 13 model calls (13 cache hits), 740s of model time
- judge: 1 model calls (11 cache hits), 40s of model time
- Total wall-clock: 13.1 min

## Known limits of this run

- Small open models (3B) on CPU: answers and judgements are noisy; one run is one sample.
- Eval cases are limited to what the documents literally say; reasoning across many documents is under-tested.
- The optimizer can only change retrieval parameters, prompt rules and temperature, not the documents or the model.
