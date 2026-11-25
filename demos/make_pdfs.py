"""Renders every demos/*/source/*.txt into demos/*/docs/<name>.pdf (needs: pip install fpdf2).

Source format: first line = title, then numbered section headings ("1. Title"), paragraphs, and "- " bullet lines.
Text must be plain ASCII/Latin-1 (core PDF fonts)."""
import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).parent


def render(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    text.encode("latin-1")                                   # fail loudly on characters the core fonts cannot print
    pdf = FPDF()
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    first = True
    for raw_line in text.strip().split("\n"):
        line = raw_line.rstrip()
        if not line:
            continue
        if first:
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            first = False
        elif re.match(r"^\d+\.\s", line):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_x(18)
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(dst))


if __name__ == "__main__":
    for src in sorted(ROOT.glob("*/source/*.txt")):
        dst = src.parent.parent / "docs" / (src.stem + ".pdf")
        render(src, dst)
        print(f"{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
