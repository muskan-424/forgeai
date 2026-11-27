"""Builds the rehearsal document folder for a brief the factory has never seen.

Deliberately different from the two demos: a Word .docx (with headings that are only bold text, so no heading is
detectable), a plain-text guide with ALL-CAPS headings, and a Markdown procedure with a pipe table. The texts were
written by Claude (Anthropic) as realistic port documents, not tuned to the factory. Run: python make_docs.py"""
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).parent
DOCS = HERE / "docs"

REGULATIONS = [
    ("h", "Westerhaven Container Terminal - Terminal Regulations"),
    ("p", "These regulations apply to every vessel calling at Westerhaven Container Terminal, Maasvlakte, Rotterdam. "
          "They take effect on 1 February 2025 and replace all earlier editions."),
    ("h", "Arrival notification"),
    ("p", "The ship agent must send a pre-arrival notification to the terminal at least 72 hours before the expected time "
          "of arrival. The estimated time of arrival must be confirmed or updated no later than 24 hours before arrival. "
          "Vessels that arrive without a confirmed notification may be refused a berth window."),
    ("h", "Dangerous goods"),
    ("p", "Dangerous goods carried under the IMDG Code must be declared to the terminal at least 48 hours before arrival. "
          "Containers with class 1 explosives are not handled at Westerhaven under any circumstances."),
    ("h", "Navigation and mooring"),
    ("p", "The maximum speed inside the terminal basin is 6 knots. Tug assistance is compulsory for vessels longer than "
          "200 metres. Vessels must keep their main engines ready for immediate use while alongside during wind speeds "
          "above 14 metres per second."),
    ("h", "Waste and emissions"),
    ("p", "Ship-generated waste must be delivered to the port reception facility before departure unless the vessel holds "
          "a valid exemption. Vessels at berth for more than 2 hours must use fuel with a sulphur content of no more than "
          "0.1 percent."),
    ("h", "Inspections"),
    ("p", "The terminal may inspect any vessel's certificates before cargo operations start. A vessel with an expired "
          "statutory certificate will not be worked until a valid certificate or a flag-state extension is presented."),
]

BERTH = """# Berth booking procedure

Westerhaven Container Terminal, effective 1 February 2025.

## Requesting a berth window

Berth windows are requested through the terminal's booking portal by the ship agent. A request must be made at least
7 days before the planned arrival. The terminal confirms or rejects a request within 2 working days. Liner services with
a fixed weekly window keep their window unless they miss it three times in one quarter.

## Berth fees

Berth fees are charged per metre of vessel length per started day alongside.

| Vessel length | Fee per metre per day |
|---|---|
| up to 200 metres | EUR 2.10 |
| 200 to 300 metres | EUR 2.60 |
| over 300 metres | EUR 3.20 |

A minimum charge of EUR 400 applies to every call.

## Changes and cancellations

A berth window may be moved once free of charge if the change is requested at least 48 hours in advance. A cancellation
made less than 24 hours before the window starts is charged at 50 percent of the berth fee for one day.

## Priority

When windows conflict, the terminal gives priority first to vessels in distress, then to liner services in their fixed
window, then to other vessels in order of booking.
"""

CERTIFICATES = """CERTIFICATES GUIDE FOR SHIP AGENTS

This guide summarises which certificates the terminal checks and how long they are valid. It is a summary for agents and
does not replace the requirements of the flag state.

SHIP CERTIFICATES

The Safety Management Certificate under the ISM Code is valid for 5 years, with an intermediate verification between the
second and third anniversary. The International Ship Security Certificate under the ISPS Code is valid for 5 years. The
International Oil Pollution Prevention Certificate is valid for 5 years, with annual surveys. A certificate is treated as
expired from the day after its expiry date.

CREW CERTIFICATES

Certificates of competency issued under the STCW Convention are valid for 5 years. Seafarer medical certificates are valid
for 2 years, or for 1 year if the seafarer is under 18 years old. The terminal does not assess whether a seafarer is fit
for duty; questions about health or fitness are for the ship's doctor or a recognised medical practitioner.

RENEWAL

Agents should remind the vessel to start renewing a certificate at least 3 months before it expires. If a certificate
expires during a call, cargo operations stop until a valid certificate or a written extension from the flag state is
shown to the terminal duty officer.
"""


def make_docx(path: Path, blocks: list[tuple[str, str]]) -> None:
    """A minimal, valid .docx built by hand (no python-docx needed). Headings are only bold runs."""
    def para(kind, text):
        bold = "<w:rPr><w:b/></w:rPr>" if kind == "h" else ""
        return f'<w:p><w:r>{bold}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}"><w:body>'
                + "".join(para(k, t) for k, t in blocks) + "</w:body></w:document>")
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" '
                     'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    make_docx(DOCS / "terminal_regulations.docx", REGULATIONS)
    (DOCS / "berth_booking_procedure.md").write_text(BERTH, encoding="utf-8")
    (DOCS / "certificates_guide.txt").write_text(CERTIFICATES, encoding="utf-8")
    print("wrote", sorted(p.name for p in DOCS.iterdir()))
