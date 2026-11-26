# Quality report: procurement-team-ship-management-agent

**Brief:** An assistant for the procurement team of a ship management company. It should look up spare parts in the catalogue (prices, lead times, stock status), calculate order totals including the supplier's volume discount, and explain the approved-supplier terms and the approval and quotation rules. It must not give legal advice about contracts.

## Verdict

- **Dev set** (used by the optimizer): 11/14 (79%; 95% CI 52%-92%) at baseline -> 13/14 (93%; 95% CI 68%-99%) after 2 kept patch(es) (3 tried).
- **Holdout set** (never used to patch): 5/6 (83%; 95% CI 44%-97%) at baseline -> 5/6 (83%; 95% CI 44%-97%) final.
- The loop stopped because: reached the maximum of 3 iterations.
- **Check iteration 1's kept patch:** a gain of 1 case(s) on 14 dev cases is within normal run-to-run noise, so it does not prove the patch helped.
- **Check iteration 2's kept patch:** a gain of 1 case(s) on 14 dev cases is within normal run-to-run noise, so it does not prove the patch helped.
- 2 case(s) still fail; see 'What still fails and why'.
- **How much to trust these numbers:** the eval set has only 20 cases, so every rate carries a wide confidence interval (shown above). Differences of one or two cases are within noise.

## What was built

- **Scope:** Build a document-based assistant for the procurement team of a ship management company.
- **Audience:** Procurement team of a ship management company.
- **Tone:** professional, concise and neutral
- **Must answer:** Quotations, Volume discounts, spare parts in the catalogue, lead times, stock status, order totals, approved-supplier terms
- **Must refuse:** legal advice about contracts
- **Retrieval:** chunk 500 chars (overlap 60), top-6, BM25 only
- **Tools attached:** csv_lookup, calculator
  - `csv_lookup`: ATTACHED: the brief explicitly mentions 'catalogue'; model: explicitly requested in the brief
  - `calculator`: ATTACHED: the brief explicitly mentions 'calculate'; model: explicitly requested in the brief
  - `open_meteo`: not attached: no evidence in the brief that this tool is needed
  - `current_date`: not attached: no evidence in the brief that this tool is needed
- **Rules in the final prompt:** 11
- **Refusal gate:** ON (forbidden topics checked before answering: legal advice about contracts)

## Can the documents support the brief?

- **Quotations**: supported (every word of the topic occurs in one sentence of the documents)
- **Volume discounts**: supported (every word of the topic occurs in one sentence of the documents)
- **spare parts in the catalogue**: supported (model cautious ('partial'); topic words fully present (lexical 1.0) in the quoted sentence)
- **lead times**: supported (every word of the topic occurs in one sentence of the documents)
- **stock status**: supported (every word of the topic occurs in one sentence of the documents)
- **order totals**: supported (every word of the topic occurs in one sentence of the documents)
- **approved-supplier terms**: partial (model said 'partial', lexical coverage 0.35)
- WARNING: PARTIAL: the documents cover 'approved-supplier terms' only in part; expect refusals or gaps there.

## Evaluation set

- 20 generated cases: 10 answerable, 3 multi_doc, 3 ambiguous, 4 out_of_scope; split 14 dev / 6 holdout.
- Generation yield: 20 of 43 attempts passed verification; 13 were discarded.
  - 6x multi_doc:fact not in sources
  - 4x answerable:question strays from its source sentence
  - 2x multi_doc:does not need both passages
  - 1x multi_doc:question refers to the source text
- The cases were written by the same model family as the agent (and verified mechanically against the documents), so blind spots shared by that family may be invisible to this evaluation.

## How reliable is the judge?

- Calibration on 16 synthetic answers with known verdicts: accuracy 100%, false passes 0, false fails 0, facts wrongly accepted by the model judge 0.
  - by probe: gold 2/2, padded_gold 2/2, wrong_facts 2/2, padded_wrong 2/2, wrong_refusal 2/2, oos_refused 2/2, oos_answered 2/2, oos_hedged 2/2
- Defences: deterministic checks first (refusal flag, verbatim citations, numbers present in the context, fact matching); the model judge only sees facts the matcher could not confirm, one answer at a time, over two passes with reversed fact order, and disagreement counts as a miss.
- Calibration probes are synthetic and easy; they show the judge is not trivially fooled, not that it is right on every real answer.

## Per-iteration scorecard

| iteration | decision | dev | holdout |
|---|---|---|---|
| 0 | baseline | 11/14 (79%; 95% CI 52%-92%) | 5/6 (83%; 95% CI 44%-97%) |
| 1 | kept | 12/14 (86%; 95% CI 60%-96%) | - |
| 2 | kept | 13/14 (93%; 95% CI 68%-99%) | - |
| 3 | reverted | 12/14 (86%; 95% CI 60%-96%) | - |
| final | final | 13/14 (93%; 95% CI 68%-99%) | 5/6 (83%; 95% CI 44%-97%) |

## Changelog

### Iteration 1: KEPT (retrieval_miss)
- **Changed:** retrieval.top_k = 6
- **Why:** retrieve more passages (top_k 4 -> 6) (source: ladder)
- **Failures addressed:** amb-02, ans-09
- **Score before -> after (dev):** 11 -> 12 of 14
- **Fixed:** ans-09; **broke:** none
- **Decision:** net gain on dev cases

### Iteration 2: KEPT (hallucination)
- **Changed:** + rule: Each citation quote must be one or more whole sentences copied exactly, word for word, from the CONTEXT; never reword, shorten or merge sentences inside a quote.; + rule: If the question asks about something the CONTEXT does not state, refuse, even if it sounds related to the documents' topic; never borrow a rule or figure from a different topic.
- **Why:** add proven rule for hallucination/citation; add proven rule for hallucination/out_of_scope (source: library)
- **Failures addressed:** amb-02, oos-02
- **Score before -> after (dev):** 12 -> 13 of 14
- **Fixed:** oos-02; **broke:** none
- **Decision:** net gain on dev cases

### Iteration 3: REVERTED (hallucination)
- **Changed:** + rule: When answering, ensure all information is directly supported by the context and avoid making assertions not grounded in the provided text or tables.
- **Why:** model-written rule from 1 failing example(s) for hallucination (source: model)
- **Failures addressed:** amb-02
- **Score before -> after (dev):** 13 -> 12 of 14
- **Fixed:** none; **broke:** oos-02
- **Decision:** no net gain: broke ['oos-02']; patch reverted

## What still fails and why

- **amb-02** (dev, ambiguous): "What's Orion Safety Ltd's services?"
  - Root cause: **hallucination**, meaning the answer asserted content that the context does not support (or answered an out-of-scope question).
  - Judge: ungrounded: citation quote not found in retrieved context (fabricated)
  - Agent said: "Orion Safety Ltd provides services for lifesaving and fire-fighting equipment."
- **amb-01** (holdout, ambiguous): "What's the reporting time for purchases?"
  - Root cause: **prompt_defect**, meaning the context contained the answer and the agent did not refuse, but the answer was wrong or incomplete.
  - Judge: facts matched 0/1
  - Agent said: "The reporting time for purchases is 24 hours for safety-critical items or when the vessel risks going off-hire, and 48 hours for purchases that are approved retrospectively."

Failure categories remaining: hallucination x1, prompt_defect x1.

## Cost

- factory (intake, guard, generator, evalgen): 0 fresh model calls, 40 answered from the cache; 0s of fresh model time + 469s replayed from the cache
- agent: 0 fresh model calls, 69 answered from the cache; 0s of fresh model time + 7344s replayed from the cache
- judge: 0 fresh model calls, 38 answered from the cache; 0s of fresh model time + 774s replayed from the cache
- NOTE: this run mostly replayed identical requests from the cache, so the time above understates what a first, uncached run costs (on this CPU roughly 30-150 s per agent call).
- Total wall-clock: 0.0 min

## Known limits of this run

- Small open models (3B) on CPU: answers and judgements are noisy; one run is one sample.
- Eval cases are limited to what the documents literally say; reasoning across many documents is under-tested.
- The optimizer can only change retrieval parameters, prompt rules and temperature, not the documents or the model.
