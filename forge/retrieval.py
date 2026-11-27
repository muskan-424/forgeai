"""Hybrid retrieval: own BM25 + optional embeddings, min-max fused."""
from __future__ import annotations

import math
import re
from collections import Counter

from .llm import LLM
from .schemas import Chunk, RetrievalConfig

_STOP = set("a an the of to in on for and or is are was were be been it its this that with as at by from "
            "what which who how do does did can could should would i you we they my our your their".split())


def stem(t: str) -> str:
    """Light, consistent suffix stripper (not linguistically perfect, just the same on both sides of a match):
    allowances/allowance -> allowanc, claiming/claims -> claim, travelling/travel -> travel, policies/policy -> policy."""
    if not t.isalpha() or len(t) <= 3:
        return t
    for suf, repl in (("ities", ""), ("ity", ""), ("ings", ""), ("ing", ""), ("ied", "y"), ("ies", "y"), ("edly", ""), ("ed", ""),
                      ("es", ""), ("s", ""), ("ly", "")):
        if suf == "es" and t.endswith("ees"):
            continue                            # employees -> (s) -> employee, not employe
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[:-len(suf)] + repl
            if suf in ("ing", "ings", "ed") and len(t) > 3 and t[-1] == t[-2] and t[-1] not in "sz":
                t = t[:-1]                      # travelling -> travel, stopped -> stop
            break
    if t.endswith("e") and len(t) > 3:
        t = t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())
    return [stem(t) for t in toks if t not in _STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.N = len(docs)
        self.avgdl = sum(map(len, docs)) / max(self.N, 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in query:
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out.append(s)
        return out


def _maxnorm(xs: list[float]) -> list[float]:
    """BM25 scores are >= 0: scale so the best hit is 1.0 (0 stays 0, a lone chunk keeps 1.0)."""
    hi = max(xs, default=0.0)
    return [0.0] * len(xs) if hi <= 0 else [x / hi for x in xs]


def _minmax(xs: list[float]) -> list[float]:
    """Cosine scores can be negative: min-max, but equal scores are all treated as relevant."""
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    return [1.0] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class Retriever:
    def __init__(self, chunks: list[Chunk], cfg: RetrievalConfig, llm: LLM | None = None):
        self.chunks, self.cfg, self.llm = chunks, cfg, llm
        self.bm25 = BM25([tokenize(c.heading + " " + c.text) for c in chunks])
        self.emb: list[list[float]] | None = None
        if cfg.use_embeddings and llm is not None and chunks:
            self.emb = llm.embed([c.text for c in chunks])

    def top_raw_score(self, query: str) -> float:
        """Best raw (un-normalized) BM25 score; comparable across queries, unlike search() scores."""
        return max(self.bm25.scores(tokenize(query)), default=0.0)

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        k = k or self.cfg.top_k
        if not self.chunks:
            return []
        bm = _maxnorm(self.bm25.scores(tokenize(query)))
        if self.emb is not None and self.llm is not None:
            qv = self.llm.embed([query])[0]
            em = _minmax([_cos(qv, v) for v in self.emb])
            w = self.cfg.bm25_weight
            fused = [w * a + (1 - w) * b for a, b in zip(bm, em)]
        else:
            fused = bm
        order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)[:k]
        return [(self.chunks[i], fused[i]) for i in order if fused[i] >= self.cfg.min_score and fused[i] > 0]
