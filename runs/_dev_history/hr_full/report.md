# Quality report: hr-policy-questions-shore-agent

**Brief:** I need an assistant that answers HR policy questions for shore staff. Keep a formal tone.

## Verdict

- **Dev set** (used by the optimizer): 5/6 (83%; 95% CI 44%-97%) at baseline -> 6/6 (100%; 95% CI 61%-100%) after 1 kept patch(es) (2 tried).
- **Holdout set** (never used to patch): 0/1 (0%; 95% CI 0%-79%) at baseline -> 0/1 (0%; 95% CI 0%-79%) final.
- The loop stopped because: all dev cases pass.
- 1 case(s) still fail; see 'What still fails and why'.
- **How much to trust these numbers:** the eval set has only 7 cases, so every rate carries a wide confidence interval (shown above). A perfect score on so few cases is weak evidence, not proof. Differences of one or two cases are within noise.

## What was built

- **Scope:** This assistant covers HR policies related to expenses and leave for shore staff.
- **Audience:** shore staff who need answers to HR policy questions
- **Tone:** formal
- **Must answer:** Booking Travel, Daily Allowances, Claiming Expenses, Annual Leave
- **Must refuse:** anything not covered by the provided documents
- **Retrieval:** chunk 500 chars (overlap 60), top-3, BM25 only
- **Tools attached:** none
  - `csv_lookup`: not attached: precondition failed: no .csv document provided
  - `calculator`: not attached: no evidence in the brief that this tool is needed
  - `open_meteo`: not attached: no evidence in the brief that this tool is needed
  - `current_date`: not attached: no evidence in the brief that this tool is needed
- **Rules in the final prompt:** 7

## Can the documents support the brief?

- **Booking Travel**: supported (model cautious ('partial'); topic words fully present (lexical 1.0) in the quoted sentence)
- **Daily Allowances**: supported
- **Claiming Expenses**: supported
- **Annual Leave**: supported

## Evaluation set

- 7 generated cases: 4 answerable, 2 multi_doc, 1 ambiguous; split 6 dev / 1 holdout.
- Generation yield: 7 of 20 attempts passed verification; 13 were discarded.
  - 6x out_of_scope:too similar to the documents (might be answerable)
  - 2x multi_doc:fact not in sources
  - 1x answerable:question strays from its source sentence
  - 1x multi_doc:compares values with different units
  - 1x ambiguous:rewrite lost the topic
- The cases were written by the same model family as the agent (and verified mechanically against the documents), so blind spots shared by that family may be invisible to this evaluation.

## How reliable is the judge?

- Calibration on 10 synthetic answers with known verdicts: accuracy 100%, false passes 0, false fails 0, facts wrongly accepted by the model judge 0.
  - by probe: gold 2/2, padded_gold 2/2, wrong_facts 2/2, padded_wrong 2/2, wrong_refusal 2/2
- Defences: deterministic checks first (refusal flag, verbatim citations, numbers present in the context, fact matching); the model judge only sees facts the matcher could not confirm, one answer at a time, over two passes with reversed fact order, and disagreement counts as a miss.
- Calibration probes are synthetic and easy; they show the judge is not trivially fooled, not that it is right on every real answer.

## Per-iteration scorecard

| iteration | decision | dev | holdout |
|---|---|---|---|
| 0 | baseline | 5/6 (83%; 95% CI 44%-97%) | 0/1 (0%; 95% CI 0%-79%) |
| 1 | reverted | 5/6 (83%; 95% CI 44%-97%) | - |
| 2 | kept | 6/6 (100%; 95% CI 61%-100%) | - |
| final | final | 6/6 (100%; 95% CI 61%-100%) | 0/1 (0%; 95% CI 0%-79%) |

## Changelog

### Iteration 1: REVERTED (prompt_defect)
- **Changed:** + rule: Answer the exact question asked, giving the exact figure and unit from the CONTEXT, and answer every part of a multi-part question.
- **Why:** add proven rule for prompt_defect/general (source: library)
- **Failures addressed:** amb-01
- **Score before -> after (dev):** 5 -> 5 of 6
- **Fixed:** none; **broke:** none
- **Decision:** no net gain: fixed nothing; patch reverted

### Iteration 2: KEPT (prompt_defect)
- **Changed:** + rule: If the answer matches a fact in the context, explicitly state 'The context contains this information, but does not provide a refusal or correction, so I cannot answer this question based on the given context.'
- **Why:** model-written rule from 1 failing example(s) for prompt_defect (source: model)
- **Failures addressed:** amb-01
- **Score before -> after (dev):** 5 -> 6 of 6
- **Fixed:** amb-01; **broke:** none
- **Decision:** net gain on dev cases

## What still fails and why

- **ans-03** (holdout, answerable): "What is the minimum duration for booking a flight in premium economy?"
  - Root cause: **prompt_defect**, meaning the context contained the answer and the agent did not refuse, but the answer was wrong or incomplete.
  - Judge: facts matched 0/1
  - Agent said: "7 days"

Failure categories remaining: prompt_defect x1.

## Cost

- factory (intake, guard, generator, evalgen): 7 model calls (15 cache hits), 194s of model time
- agent: 20 model calls (0 cache hits), 867s of model time
- judge: 5 model calls (17 cache hits), 127s of model time
- Total wall-clock: 19.9 min

## Known limits of this run

- Small open models (3B) on CPU: answers and judgements are noisy; one run is one sample.
- Eval cases are limited to what the documents literally say; reasoning across many documents is under-tested.
- The optimizer can only change retrieval parameters, prompt rules and temperature, not the documents or the model.
