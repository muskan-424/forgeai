# Requirements

**Brief:** An assistant for the procurement team of a ship management company. It should look up spare parts in the catalogue (prices, lead times, stock status), calculate order totals including the supplier's volume discount, and explain the approved-supplier terms and the approval and quotation rules. It must not give legal advice about contracts.

## Scope
Build a document-based assistant for the procurement team of a ship management company.

## Audience
Procurement team of a ship management company.

## Tone
professional, concise and neutral

## Must answer
- Quotations
- Volume discounts
- spare parts in the catalogue
- lead times
- stock status
- order totals
- approved-supplier terms

## Must refuse
- legal advice about contracts

## Groundability (can the documents support the brief?)
- **Quotations**: supported (every word of the topic occurs in one sentence of the documents)
- **Volume discounts**: supported (every word of the topic occurs in one sentence of the documents)
- **spare parts in the catalogue**: supported (model cautious ('partial'); topic words fully present (lexical 1.0) in the quoted sentence)
- **lead times**: supported (every word of the topic occurs in one sentence of the documents)
- **stock status**: supported (every word of the topic occurs in one sentence of the documents)
- **order totals**: supported (every word of the topic occurs in one sentence of the documents)
- **approved-supplier terms**: partial (model said 'partial', lexical coverage 0.35)

## Clarifying questions
- Q: What tone should the assistant use (for example formal, friendly, concise)?
  A: (no answer; default assumed)

## Assumptions
- Refusal 'legal advice about contracts' taken verbatim from the brief (the model's list had missed it).
- Topic 'spare parts in the catalogue' taken verbatim from the brief sentence "It should look up spare parts in the catalogue (prices, lead times, stock status), calcula" (the model's topic list had missed it).
- Topic 'lead times' taken verbatim from the brief sentence "It should look up spare parts in the catalogue (prices, lead times, stock status), calcula" (the model's topic list had missed it).
- Topic 'stock status' taken verbatim from the brief sentence "It should look up spare parts in the catalogue (prices, lead times, stock status), calcula" (the model's topic list had missed it).
- Topic 'order totals' taken verbatim from the brief sentence "It should look up spare parts in the catalogue (prices, lead times, stock status), calcula" (the model's topic list had missed it).
- Topic 'approved-supplier terms' taken verbatim from the brief sentence "It should look up spare parts in the catalogue (prices, lead times, stock status), calcula" (the model's topic list had missed it).
- tone not specified; assumed: professional, concise and neutral
