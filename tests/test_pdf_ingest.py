import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.ingest import build_chunks, inventory, load_text, unwrap_pdf_text  # noqa: E402
from forge.textmatch import norm, split_sentences  # noqa: E402

DEMOS = Path(__file__).resolve().parent.parent / "demos"


def test_unwrap_joins_wrapped_lines_keeps_bullets_and_isolates_headings():
    raw = ("1. Booking travel\nAll business travel must be booked at least 7 days\nbefore departure. Hotels are capped:\n"
           "- Tier 1 cities: EUR 180 per night.\n- Tier 2 cities: EUR 130\nper night.\n2. Daily allowances\nMeals cost EUR 40.")
    out = unwrap_pdf_text(raw)
    assert "at least 7 days before departure." in out                        # wrapped sentence rejoined
    assert "- Tier 2 cities: EUR 130 per night." in out                      # wrapped bullet rejoined, bullet kept separate
    assert "\n- Tier 1 cities: EUR 180 per night.\n" in out
    assert "1. Booking travel\n\nAll business" in out and "\n\n2. Daily allowances\n\nMeals" in out   # headings isolated


def test_demo_pdfs_yield_sections_and_whole_sentences():
    for folder in ("hr_policy", "procurement"):
        for e in inventory(DEMOS / folder / "docs"):
            if e["type"] == "pdf":
                assert len(e["headings"]) >= 6, (e["doc"], e["headings"])
    chunks = build_chunks(DEMOS / "hr_policy" / "docs", 500, 60)
    text = norm(" ".join(c.text for c in chunks))
    assert "annual leave must be requested through the hr portal at least 14 days before" in text   # not cut at a line wrap
    sents = [s for c in chunks for s in split_sentences(c.text)]
    assert "Employees must notify their line manager before 9:00 on the first day of absence." in sents
    assert not any(s.endswith(("employment", "of", "the", "and")) for s in sents)                     # no wrap fragments
    assert any(c.heading.startswith("3. Sick leave") for c in chunks)


def test_demo_catalogue_is_reproducible_and_consistent_with_the_supplier_terms():
    import csv
    rows = list(csv.DictReader(open(DEMOS / "procurement" / "docs" / "spares_catalogue.csv", encoding="utf-8")))
    assert len(rows) == 60 and len({r["part_no"] for r in rows}) == 60
    terms = load_text(DEMOS / "procurement" / "docs" / "supplier_terms.pdf")
    assert {r["supplier"] for r in rows} == {"Hartmann Marine GmbH", "Pacific Valve Co.", "Baltic Electric AB", "Orion Safety Ltd"}
    assert all(s in terms for s in {r["supplier"] for r in rows})            # every catalogue supplier is an approved supplier
    assert all(r["stock_status"] in ("In stock", "Made to order", "Backordered") for r in rows)
    # the PDFs never mention pensions: the HR demo relies on that gap
    hr = " ".join(load_text(p) for p in (DEMOS / "hr_policy" / "docs").glob("*.pdf")).lower()
    assert "pension" not in hr and "retirement" not in hr


def test_markdown_tables_become_self_describing_rows(tmp_path):
    from forge.ingest import normalize_pipe_tables
    md = ("## Berth fees\n\nFees are charged per metre.\n\n| Vessel length | Fee per metre per day |\n|---|---|\n"
          "| up to 200 metres | EUR 2.10 |\n| over 300 metres | EUR 3.20 |\n\nA minimum charge of EUR 400 applies.\n")
    out = normalize_pipe_tables(md)
    assert "Vessel length: up to 200 metres; Fee per metre per day: EUR 2.10" in out
    assert "Vessel length: over 300 metres; Fee per metre per day: EUR 3.20" in out
    assert "|---|" not in out and "A minimum charge of EUR 400 applies." in out and "## Berth fees" in out
    (tmp_path / "fees.md").write_text(md, encoding="utf-8")
    sents = [x for c in build_chunks(tmp_path, 500, 60) for x in split_sentences(c.text)]
    assert "Vessel length: over 300 metres; Fee per metre per day: EUR 3.20" in sents          # now testable and citable
    assert normalize_pipe_tables("a | b is not a table\nplain text") == "a | b is not a table\nplain text"
