"""Deterministic text matching shared by the eval generator (to verify its own output)
and the judge (to score agent answers). No LLM involved."""
from __future__ import annotations

import re

from .retrieval import tokenize


def norm(s: str) -> str:
    s = re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', s))
    return re.sub(r"\s+", " ", s).strip().lower()


_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s):
        m = m.replace(",", "").rstrip(".")
        if "." in m:
            m = m.rstrip("0").rstrip(".")
        out.add(m)
    return out


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") else t


def content_tokens(s: str) -> set[str]:
    return {_stem(t) for t in tokenize(s)}


def fact_in_text(fact: str, text: str, thresh: float = 0.8) -> str:
    """Return how `fact` is supported by `text`: 'exact' | 'fuzzy' | '' (not supported).

    exact: normalized substring. fuzzy: every number present AND >= thresh of content words present.
    """
    nf, nt = norm(fact), norm(text)
    if not nf:
        return ""
    if nf in nt:
        return "exact"
    fn, tn = numbers(fact), numbers(text)
    if fn and not fn <= tn:
        return ""
    ft = content_tokens(fact) - fn
    if not ft:
        return "fuzzy" if fn else ""
    overlap = len(ft & content_tokens(text)) / len(ft)
    return "fuzzy" if overlap >= thresh else ""


MIN_SENTENCE_CHARS = 20


def _looks_like_heading(s: str) -> bool:
    """A short line with no closing punctuation, no key:value pair and no list separator is a heading, not a fact."""
    return (not re.search(r"[.!?]$", s) and ":" not in s and ";" not in s and len(s.split()) < 12)


def split_sentences(text: str) -> list[str]:
    """Sentences (and table rows) of a chunk; section headings that were kept in the chunk text are skipped."""
    out = []
    for line in text.splitlines():
        for s in re.split(r"(?<=[.!?])\s+", line.strip()):
            s = s.strip()
            if len(s) >= MIN_SENTENCE_CHARS and len(s.split()) >= 4 and not _looks_like_heading(s):
                out.append(s)
    return out




def coverage(text: str, *sources: str) -> float:
    """Share of `text`'s content words that occur in the source texts (1.0 = all of them are there)."""
    q = content_tokens(text)
    if not q:
        return 0.0
    src = set().union(*(content_tokens(s) for s in sources))
    return len(q & src) / len(q)


_LABEL = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+[^:.\n]{2,70}:\s*")   # '6. Unpaid leave: ' stuck on the front of a quote


def quote_verified(quote: str, texts: list[str]) -> bool:
    """Is this citation real? Either the whole quote occurs verbatim in one of the texts, or, after dropping a
    section label ('6. Unpaid leave: ') the agent glued on, EVERY sentence of it occurs verbatim. Rewording,
    inventing or changing a single sentence still fails."""
    q = norm(quote)
    if not q:
        return False
    nt = [norm(t) for t in texts]
    if any(q in t for t in nt):
        return True
    sents = [norm(s) for s in re.split(r"(?<=[.!?])\s+", _LABEL.sub("", quote, count=1).strip()) if len(s.split()) >= 3]
    return bool(sents) and all(any(s in t for t in nt) for s in sents)


def question_similarity(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


SOFT_REFUSAL = re.compile(
    r"(can'?t|cannot|can not|unable to|am not able to) (help|answer|assist|provide)|"
    r"outside (of )?(what|the scope|my)|not (covered|mentioned|contained|provided|available) in|"
    r"(no|not enough) (relevant )?information|(doesn'?t|does not|do not|don'?t) (contain|cover|say|mention|have)",
    re.I)
