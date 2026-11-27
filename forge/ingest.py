"""Document loading, chunking and inventory. Domain-agnostic."""
from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .schemas import Chunk

SUPPORTED = {".pdf", ".md", ".txt", ".csv", ".docx"}


def list_docs(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def normalize_pipe_tables(text: str) -> str:
    """Markdown pipe tables -> one self-describing line per row ('Vessel length: up to 200 metres; Fee: EUR 2.10'),
    like CSV rows. Found in a rehearsal on unseen documents: raw rows ('| up to 200 metres | EUR 2.10 |') have no
    closing punctuation, were treated as headings, and the fee table could never be tested or cited."""
    lines, out, i = text.split("\n"), [], 0
    sep = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and sep.match(lines[i + 1]):
            header = _cells(lines[i])
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|") or (i < len(lines) and lines[i].count("|") >= 2):
                row = _cells(lines[i])
                if any(row):
                    out.append("; ".join(f"{h}: {v}" for h, v in zip(header, row) if h or v))
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        return normalize_pipe_tables(path.read_text(encoding="utf-8", errors="replace"))
    if ext == ".csv":
        return path.read_text(encoding="utf-8-sig", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader
        r = PdfReader(str(path))
        return unwrap_pdf_text("\n".join((p.extract_text() or "") for p in r.pages))
    if ext == ".docx":
        with zipfile.ZipFile(path) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paras = ["".join(t.text or "" for t in p.iter(f"{ns}t")) for p in root.iter(f"{ns}p")]
        return "\n\n".join(paras)
    raise ValueError(f"unsupported file type: {path}")


def unwrap_pdf_text(raw: str) -> str:
    """PDF extraction returns one text line per VISUAL line: sentences are cut at the line wrap and headings are
    not separated from body text. Rebuild paragraphs: join wrapped lines, keep each bullet as its own line, and
    isolate heading lines (surrounded by blank lines) so the chunker can recognise sections."""
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            out.append(" ".join(buf))
            buf.clear()

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            flush()
            out.append("")
        elif _HEADING.match(s) and len(s) < 90 and not s.endswith((".", ",", ";", ":")):
            flush()
            out += ["", s, ""]
        elif re.match(r"^[-*\u2022]\s+", s):
            flush()
            buf.append(s)
        else:
            buf.append(s)
    flush()
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text)


_HEADING = re.compile(r"^(#{1,6}\s+.+|(\d+(\.\d+)*[.)]?\s+[A-Z].{2,80})|([A-Z][A-Z0-9 &/,\-]{4,80}))$")


def _split_long(par: str, limit: int) -> list[str]:
    if len(par) <= limit:
        return [par]
    sents = re.split(r"(?<=[.!?])\s+", par)
    out, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    # hard-split any sentence still too long
    final = []
    for o in out:
        while len(o) > limit * 1.5:
            final.append(o[:limit])
            o = o[limit:]
        final.append(o)
    return final


def chunk_text(doc: str, text: str, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[tuple[str, str]] = []  # (heading, paragraph)
    heading = ""
    pending: list[str] = []            # heading lines waiting to be attached to the next paragraph
    for p in paras:
        first = p.splitlines()[0].strip()
        if _HEADING.match(first) and len(p) < 200:
            heading = first.lstrip("# ").strip()
            if len(p.splitlines()) == 1:
                pending.append(heading)
                continue
        for i, piece in enumerate(_split_long(p, chunk_size)):
            if i == 0 and pending:
                piece = "\n".join(pending) + "\n" + piece
                pending = []
            units.append((heading, piece))

    chunks: list[Chunk] = []
    cur: list[tuple[str, str]] = []
    size = 0

    def flush():
        nonlocal cur, size
        if not cur:
            return
        body = "\n".join(t for _, t in cur)
        chunks.append(Chunk(id=f"{doc}#{len(chunks)}", doc=doc, text=body, heading=cur[0][0]))
        # carry trailing paragraphs as overlap
        carry, c = [], 0
        for h, t in reversed(cur):
            if c + len(t) > overlap:
                break
            carry.insert(0, (h, t))
            c += len(t)
        cur, size = carry, c

    for h, t in units:
        if size + len(t) > chunk_size and cur:
            flush()
        cur.append((h, t))
        size += len(t)
    if cur and (not chunks or any(t not in chunks[-1].text for _, t in cur)):
        flush()
    return chunks


def chunk_csv(doc: str, text: str, rows_per_chunk: int = 8) -> list[Chunk]:
    """CSV rows become self-describing chunks: header is repeated so each chunk stands alone."""
    rdr = csv.reader(io.StringIO(text))
    rows = list(rdr)
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    chunks = []
    for i in range(0, len(body), rows_per_chunk):
        lines = []
        for r in body[i:i + rows_per_chunk]:
            lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, r)))
        chunks.append(Chunk(id=f"{doc}#{len(chunks)}", doc=doc, text="\n".join(lines),
                            heading=f"rows {i + 1}-{min(i + rows_per_chunk, len(body))}"))
    return chunks


def build_chunks(folder: str | Path, chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    out: list[Chunk] = []
    base = Path(folder)
    for p in list_docs(folder):
        name = str(p.relative_to(base)).replace("\\", "/")
        text = load_text(p)
        out.extend(chunk_csv(name, text) if p.suffix.lower() == ".csv" else chunk_text(name, text, chunk_size, overlap))
    return out


def inventory(folder: str | Path) -> list[dict]:
    """Compact description of each document for intake / groundability / generator."""
    inv = []
    base = Path(folder)
    for p in list_docs(folder):
        text = load_text(p)
        name = str(p.relative_to(base)).replace("\\", "/")
        entry = {"doc": name, "type": p.suffix.lower().lstrip("."), "chars": len(text),
                 "words": len(text.split())}
        if p.suffix.lower() == ".csv":
            rdr = list(csv.reader(io.StringIO(text)))
            entry["columns"] = rdr[0] if rdr else []
            entry["rows"] = max(len(rdr) - 1, 0)
            entry["sample"] = rdr[1:3]
        else:
            heads = []
            for para in re.split(r"\n\s*\n", text):
                first = para.strip().splitlines()[0].strip() if para.strip() else ""
                if first and _HEADING.match(first) and len(para) < 200:
                    heads.append(first.lstrip("# ").strip())
            entry["headings"] = heads[:25]
            entry["opening"] = " ".join(text.split())[:300]
            paras = [x for x in re.split(r"\n\s*\n", text) if x.strip()]
            entry["avg_paragraph_chars"] = int(sum(map(len, paras)) / max(len(paras), 1))
        inv.append(entry)
    return inv


def inventory_text(inv: list[dict]) -> str:
    """One compact line per document, for prompts."""
    lines = []
    for d in inv:
        if d["type"] == "csv":
            lines.append(f"- {d['doc']} (table, {d.get('rows', '?')} rows; columns: {', '.join(d.get('columns', []))})")
        else:
            heads = "; ".join(d.get("headings", [])[:8]) or d.get("opening", "")[:120]
            lines.append(f"- {d['doc']} ({d['type']}, {d['words']} words; sections: {heads})")
    return "\n".join(lines) or "(no documents)"
