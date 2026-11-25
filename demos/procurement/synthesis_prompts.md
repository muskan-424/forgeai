# Synthesis prompts: ship-spares procurement demo

The two PDFs in `docs/` were written by Claude (Anthropic) from prompts P1-P2 below and checked by hand for internal
consistency (every supplier named in the terms exists in the catalogue; discount tiers and approval limits do not
overlap ambiguously). The catalogue was produced by prompt P3 plus a seeded script. `source/*.txt` are the generated
texts, rendered to PDF by `demos/make_pdfs.py`; the CSV is built by `make_catalogue.py` (seed 2025, reproducible).

## Shared setting

> Company: Nordhaven Ship Management, a fictional ship manager. The documents govern purchasing of **vessel spare
> parts**. Policies take effect 1 March 2025. Currency EUR. Plain, specific policy text with concrete numbers,
> thresholds, roles and exceptions; ASCII only; 250-350 words; numbered sections and "- " bullets for lists.

## P1: supplier_terms

> Write the **Approved Supplier Terms**. Four approved suppliers (Hartmann Marine GmbH: engine parts and pumps;
> Pacific Valve Co.: valves and fittings; Baltic Electric AB: electrical and navigation equipment; Orion Safety Ltd:
> lifesaving and fire-fighting equipment), each with a city and payment terms in days. Then: volume-discount tiers
> that differ by supplier (one supplier with three tiers, one with a single tier, two with none), how discounts
> apply (before freight, not combined across suppliers), delivery terms and an urgent-delivery surcharge, warranty
> length and the claim deadline, and minimum order values with a handling fee.
> Plant traps: a discount that depends on supplier AND order total; a surcharge defined as a percentage of order
> value; two different "within N days" rules (claim reporting vs repair).

## P2: procurement_procedure

> Write the **Procurement Procedure**. Cover: approval limits by order value with four named roles, quotation
> requirements by value (2 and 3 quotations), quotation validity, emergency purchases with reporting and
> retrospective-approval deadlines, parts needing a certificate of conformity, goods receipt inspection and
> discrepancy-reporting deadlines, and new-supplier screening and removal.
> Plant traps: approval ranges that meet exactly at EUR 1,001 / 10,001 / 50,001; quotation thresholds (2,000 and
> 10,000) that differ from approval thresholds; 24 h vs 48 h emergency deadlines.

## P3: spares_catalogue (templates, then script)

> For each of the four suppliers list 15 realistic vessel spare parts: a precise description (size, rating or
> material), and a plausible EUR price range, matching the supplier's category. The script `make_catalogue.py`
> expands them into 60 rows with a fixed random seed: part number (supplier prefix HM-/PV-/BE-/OS- plus four digits),
> category, supplier, unit price, lead time in days (short if in stock, 21-45 if made to order, 42-70 if backordered),
> stock status, and warranty months (6, 12 or 24).

## Brief given to the factory (`brief.txt`)

The brief needs a table lookup (catalogue prices, lead times, stock) and arithmetic (order totals with a supplier's
volume discount) but **no weather or date tool**. It also names one refusal (legal advice about contracts). The
factory must attach `csv_lookup` and `calculator` and nothing else.
