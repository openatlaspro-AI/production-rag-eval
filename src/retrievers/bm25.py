"""BM25 lexical retriever over the corpus (title + title_en).

Uses `rank_bm25.BM25Okapi`. Tokenization is whitespace + lowercasing on the
English title plus a CJK-aware split so Chinese titles contribute character
unigrams (BM25 needs token overlap and CJK has no spaces). Queries in this
benchmark are English, so the English title text carries most of the signal;
CJK char tokens are a cheap bonus that never hurts.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens plus individual CJK characters."""
    text = text or ""
    tokens = _WORD_RE.findall(text.lower())
    tokens.extend(_CJK_RE.findall(text))
    return tokens


def doc_text(doc: dict) -> str:
    title = doc.get("title") or ""
    title_en = doc.get("title_en") or ""
    return f"{title} {title_en}"


class BM25Retriever:
    """BM25Okapi over tokenized corpus docs.

    Parameters
    ----------
    docs:
        List of corpus doc dicts (each with `title` / `title_en`). Indices in
        the returned ranking are positions in this list.
    """

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self._tokenized = [tokenize(doc_text(d)) for d in docs]
        self.bm25 = BM25Okapi(self._tokenized)

    def scores(self, query: str):
        return self.bm25.get_scores(tokenize(query))

    def search(self, query: str, k: int = 10) -> list[int]:
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]
