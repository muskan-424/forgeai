# Quality report: procurement-team-ship-management-agent

**Brief:** An assistant for the procurement team of a ship management company. It should look up spare parts in the catalogue (prices, lead times, stock status), calculate order totals including the supplier's volume discount, and explain the approved-supplier terms and the approval and quotation rules. It must not give legal advice about contracts.

## Verdict

- **Dev set** (used by the optimizer): 5/8 (62%; 95% CI 31%-86%) at baseline -> 6/8 (75%; 95% CI 41%-93%) after 1 kept patch(es) (3 tried).
- **Holdout set** (never used to patch): 3/4 (75%; 95% CI 30%-95%) at baseline -> 3/4 (75%; 95% CI 30%-95%) final.
- The loop stopped because: reached the maximum of 3 iterations.
- **Check iteration 2's kept patch:** a gain of 1 case(s) on 8 dev cases is within normal run-to-run noise, so it does not prove the patch helped.
- 3 case(s) still fail; see 'What still fails and why'.
- **How much to trust these numbers:** the eval set has only 12 cases, so every rate carries a wide confidence interval (shown above). Differences of one or two cases are within noise.

## What was built

- **Scope:** Build a document-based assistant for the procurement team of a ship management company.
- **Audience:** Procurement team of a ship management company.
- **Tone:** professional, concise and neutral
- **Must answer:** Quotations, Volume discounts, spare parts in the catalogue, lead times, stock status, order totals, approved-supplier terms
- **Must refuse:** legal advice about contracts
- **Retrieval:** chunk 500 chars (overlap 60), top-4, BM25 only
- **Tools attached:** csv_lookup, calculator
  - `csv_lookup`: ATTACHED: the brief explicitly mentions 'catalogue'; model: explicitly requested in the brief
  - `calculator`: ATTACHED: the brief explicitly mentions 'calculate'; model: explicitly requested in the brief
  - `open_meteo`: not attached: no evidence in the brief that this tool is needed
  - `current_date`: not attached: no evidence in the brief that this tool is needed
- **Rules in the final prompt:** 9
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

- 12 generated cases: 5 answerable, 2 multi_doc, 2 ambiguous, 3 out_of_scope; split 8 dev / 4 holdout.
- Generation yield: 12 of 22 attempts passed verification; 2 were discarded.
  - 2x answerable:question strays from its source sentence
- The cases were written by the same model family as the agent (and verified mechanically against the documents), so blind spots shared by that family may be invisible to this evaluation.

## How reliable is the judge?

- Calibration on 16 synthetic answers with known verdicts: accuracy 100%, false passes 0, false fails 0, facts wrongly accepted by the model judge 0.
  - by probe: gold 2/2, padded_gold 2/2, wrong_facts 2/2, padded_wrong 2/2, wrong_refusal 2/2, oos_refused 2/2, oos_answered 2/2, oos_hedged 2/2
- Defences: deterministic checks first (refusal flag, verbatim citations, numbers present in the context, fact matching); the model judge only sees facts the matcher could not confirm, one answer at a time, over two passes with reversed fact order, and disagreement counts as a miss.
- Calibration probes are synthetic and easy; they show the judge is not trivially fooled, not that it is right on every real answer.

## Per-iteration scorecard

| iteration | decision | dev | holdout |
|---|---|---|---|
| 0 | baseline | 5/8 (62%; 95% CI 31%-86%) | 3/4 (75%; 95% CI 30%-95%) |
| 1 | reverted | 5/8 (62%; 95% CI 31%-86%) | - |
| 2 | kept | 6/8 (75%; 95% CI 41%-93%) | - |
| 3 | reverted | 4/8 (50%; 95% CI 22%-78%) | - |
| final | final | 6/8 (75%; 95% CI 41%-93%) | 3/4 (75%; 95% CI 30%-95%) |

## Changelog

### Iteration 1: REVERTED (hallucination)
- **Changed:** + rule: If the question asks about something the CONTEXT does not state, refuse, even if it sounds related to the documents' topic; never borrow a rule or figure from a different topic.
- **Why:** add proven rule for hallucination/out_of_scope (source: library)
- **Failures addressed:** oos-02, oos-03
- **Score before -> after (dev):** 5 -> 5 of 8
- **Fixed:** none; **broke:** none
- **Decision:** no net gain: fixed nothing; patch reverted

### Iteration 2: KEPT (hallucination)
- **Changed:** refusal_gate = True
- **Why:** enable the runtime refusal gate (rules alone did not make the agent refuse forbidden topics) (source: ladder)
- **Failures addressed:** oos-02, oos-03
- **Score before -> after (dev):** 5 -> 6 of 8
- **Fixed:** oos-03; **broke:** none
- **Decision:** net gain on dev cases

### Iteration 3: REVERTED (retrieval_miss)
- **Changed:** retrieval.top_k = 6
- **Why:** retrieve more passages (top_k 4 -> 6) (source: ladder)
- **Failures addressed:** amb-02
- **Score before -> after (dev):** 6 -> 4 of 8
- **Fixed:** none; **broke:** ans-02, ans-04
- **Decision:** no net gain: broke ['ans-02', 'ans-04']; patch reverted

## What still fails and why

- **amb-02** (dev, ambiguous): "What's the unit price?"
  - Root cause: **retrieval_miss**, meaning the passage that answers the question was not among the retrieved context.
  - Judge: facts matched 0/1
  - Agent said: "The unit prices are provided in the 'unit_price_eur' column of the spares_catalogue.csv file. For example, the Fire extinguisher 9 kg, foam (part_no: OS-3736) has a unit price of 103.04 EUR."
- **oos-02** (dev, out_of_scope): "How do I calculate the total cost of ownership for spare parts?"
  - Root cause: **hallucination**, meaning the answer asserted content that the context does not support (or answered an out-of-scope question).
  - Judge: hallucination: answered an out-of-scope question
  - Agent said: "To calculate the total cost of ownership for spare parts, you would need to consider the initial purchase cost, any volume discounts, freight costs, and the cost of maintenance or replacement over the warranty period. However, the provided "
- **amb-01** (holdout, ambiguous): "What's the reporting time for purchases?"
  - Root cause: **hallucination**, meaning the answer asserted content that the context does not support (or answered an out-of-scope question).
  - Judge: ungrounded: cited [] but expected ['procurement_procedure.pdf']
  - Agent said: "The reporting time for purchases is within 2 working days for differences between the delivery and the purchase order, and within 24 hours for safety-critical items or vessel risks going off-hire."

Failure categories remaining: retrieval_miss x1, hallucination x2.

## Cost

- factory (intake, guard, generator, evalgen): 0 fresh model calls, 20 answered from the cache; 0s of fresh model time + 43s replayed from the cache
- agent: 1 fresh model calls, 35 answered from the cache; 127s of fresh model time + 1490s replayed from the cache
- judge: 0 fresh model calls, 20 answered from the cache; 0s of fresh model time + 64s replayed from the cache
- NOTE: this run mostly replayed identical requests from the cache, so the time above understates what a first, uncached run costs (on this CPU roughly 30-150 s per agent call).
- Total wall-clock: 5.1 min

## Known limits of this run

- Small open models (3B) on CPU: answers and judgements are noisy; one run is one sample.
- Eval cases are limited to what the documents literally say; reasoning across many documents is under-tested.
- The optimizer can only change retrieval parameters, prompt rules and temperature, not the documents or the model.
