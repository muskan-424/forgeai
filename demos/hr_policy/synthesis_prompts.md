# Synthesis prompts: HR policy demo

The three documents in `docs/` were written by Claude (Anthropic) from the prompts below, then checked by hand for
internal consistency (no contradicting numbers between documents; no mention of pensions anywhere, on purpose).
`source/*.txt` are the generated texts; `docs/*.pdf` are rendered from them by `demos/make_pdfs.py`.

## Shared setting (given with every prompt)

> Company: Meridian Maritime Services, a fictional Rotterdam-based ship manager. The documents are internal
> HR policies for **shore-based** employees (office staff). Seafarers have separate agreements. Policies take effect
> 1 January 2025. Currency EUR. Write plain, specific policy text: concrete numbers, durations, thresholds,
> approval roles and exceptions. No marketing language. ASCII only. 300-450 words. Numbered sections ("1. Title"),
> short paragraphs, "- " bullets only where a list is natural.

## P1: leave_policy

> Write the company's **Leave Policy**. Cover: annual leave entitlement (with a seniority tiering), public holidays,
> how and when to request leave, carry-over rules with a deadline, sick leave pay tiers and certificate rules,
> parental leave (birth, other parent, adoption), compassionate leave (two tiers of relationship), and unpaid leave.
> Plant these retrieval traps: three different day-counts that depend on years of service; a carry-over cap *and* a
> forfeiture date; a sick-pay rule that changes at day 11; a notice period for maternity leave stated in weeks.
> **Do not mention pensions, pension contributions or retirement anywhere.**

## P2: travel_expenses_policy

> Write the **Travel and Expenses Policy**. Cover: booking lead time, flight class rules by flight length,
> a hotel cap that depends on city tier (list the cities of each tier as bullets), daily meal allowances for domestic
> versus international travel and the reduction when meals are provided, the claim deadline with a receipt rule, an
> approval threshold, the reimbursement time, private-car mileage with an approval limit, travel advances, and a short
> list of costs that are never reimbursed.
> Plant these traps: two meal-allowance amounts; two different "within N days" deadlines (claim submission vs
> reimbursement vs advance settlement); a threshold above which a department head must approve.

## P3: conduct_and_employment_policy

> Write the **Conduct and Employment Policy**. Cover: working week and core hours, overtime approval and its two pay
> multipliers, time off in lieu, remote-work limits and its probation exception, probation length and notice periods
> (before and after probation, split by years of service), gifts and conflicts of interest, a four-stage
> disciplinary procedure with the validity period of each warning, gross misconduct, the right to be accompanied,
> the appeal window, and the grievance procedure with response deadlines.
> Plant these traps: notice periods that differ by years of service; a 6-month probation and a 3-month remote-work
> restriction; warning validity periods 6/12/18 months.

## Brief given to the factory (`brief.txt`)

The brief deliberately asks for one thing the documents **cannot** support (pension contributions). The groundability
guard is expected to flag it at the requirements stage and the agent must refuse pension questions instead of guessing.
