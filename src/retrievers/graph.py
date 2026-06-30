"""GraphRAG-lite: entity co-occurrence graph with 1-hop expansion.

This is **not** full LLM graph construction. Entities are extracted purely
lexically (no network, no LLM):
  - multi-word Capitalized phrases and standalone Capitalized tokens in the
    English title (proper nouns: people, places, orgs), and
  - a curated keyword list of domain entities that recur in this trend corpus.

Graph construction:
  - nodes = corpus docs
  - an edge connects two docs that share >= 1 entity; edge weight = #shared
    entities.

Retrieval for a query:
  1. Extract query entities lexically.
  2. Seed = docs whose entity set intersects the query entities (lexical match).
  3. Expand 1 hop: add graph neighbours of seed docs.
  4. Score each candidate = (entity-overlap weight with the query/seeds)
     + (dense cosine similarity of the doc to the query), then rank.

The dense term is what lets the graph retriever degrade gracefully to "dense-ish"
when a query has no extractable entities, while the entity term rewards docs that
sit in a tight co-occurrence cluster around the query topic.
"""

from __future__ import annotations

import re

import networkx as nx

# Domain keywords that appear across this trend corpus and behave as entities
# even when not capitalized in the English title. Lowercased for matching.
KEYWORD_ENTITIES = {
    "china",
    "us",
    "u.s.",
    "iran",
    "taiwan",
    "trump",
    "hormuz",
    "yuan",
    "a-share",
    "a-shares",
    "hantavirus",
    "israel",
    "gaza",
    "russia",
    "ukraine",
    "tariff",
    "tariffs",
    "fifa",
    "world cup",
    "disney",
    "nasdaq",
    "oil",
    "saudi",
    "middle east",
}

_CAP_PHRASE_RE = re.compile(r"\b([A-Z][A-Za-z.&-]+(?:\s+[A-Z][A-Za-z.&-]+)*)\b")
_STOP_CAPS = {"The", "A", "An", "Of", "In", "On", "And", "Why", "How", "What"}


def extract_entities(text: str) -> set[str]:
    """Extract entity strings from text (lowercased, normalized).

    Combines capitalized proper-noun phrases with the curated keyword list.
    """
    text = text or ""
    entities: set[str] = set()

    for match in _CAP_PHRASE_RE.findall(text):
        # Strip leading sentence-start stopwords like "The"/"Why".
        words = [w for w in match.split() if w not in _STOP_CAPS]
        if not words:
            continue
        phrase = " ".join(words).lower().strip(".")
        if len(phrase) >= 2:
            entities.add(phrase)

    low = text.lower()
    for kw in KEYWORD_ENTITIES:
        if kw in low:
            entities.add(kw)

    return entities


def doc_text(doc: dict) -> str:
    title = doc.get("title") or ""
    title_en = doc.get("title_en") or ""
    return f"{title} {title_en}"


class GraphRetriever:
    """Entity co-occurrence graph retriever with 1-hop expansion.

    Parameters
    ----------
    docs:
        Corpus doc dicts.
    dense:
        A DenseRetriever used for the dense-similarity scoring term. Optional;
        if None, scoring uses entity overlap only (still functional).
    """

    def __init__(self, docs: list[dict], dense=None) -> None:
        self.docs = docs
        self.dense = dense
        self.doc_entities: list[set[str]] = [extract_entities(doc_text(d)) for d in docs]
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(self.docs)))
        # Invert: entity -> docs containing it, then connect co-occurring docs.
        entity_to_docs: dict[str, list[int]] = {}
        for i, ents in enumerate(self.doc_entities):
            for e in ents:
                entity_to_docs.setdefault(e, []).append(i)
        for e, members in entity_to_docs.items():
            if len(members) < 2:
                continue
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    a, b = members[a_idx], members[b_idx]
                    if g.has_edge(a, b):
                        g[a][b]["weight"] += 1
                        g[a][b]["entities"].add(e)
                    else:
                        g.add_edge(a, b, weight=1, entities={e})
        return g

    def search(self, query: str, k: int = 10) -> list[int]:
        q_entities = extract_entities(query)

        # 1. Seed docs by lexical entity match with the query.
        seeds = {i for i, ents in enumerate(self.doc_entities) if ents & q_entities}

        # 2. Expand 1 hop via the co-occurrence graph.
        candidates: set[int] = set(seeds)
        for s in seeds:
            candidates.update(self.graph.neighbors(s))

        # Fallback: if the query yielded no entities/seeds, fall back to dense
        # so the retriever still returns something sensible.
        if not candidates:
            if self.dense is not None:
                return self.dense.search(query, k=k)
            return list(range(min(k, len(self.docs))))

        # Precompute dense sims once if available.
        sims = self.dense.similarities(query) if self.dense is not None else None

        scored: list[tuple[float, int]] = []
        for c in candidates:
            entity_overlap = len(self.doc_entities[c] & q_entities)
            # Bonus for graph proximity to seeds (shared entities with seed docs).
            graph_bonus = 0
            for s in seeds:
                if c == s:
                    continue
                if self.graph.has_edge(c, s):
                    graph_bonus += self.graph[c][s]["weight"]
            entity_score = entity_overlap + 0.5 * graph_bonus
            dense_score = float(sims[c]) if sims is not None else 0.0
            scored.append((entity_score + dense_score, c))

        scored.sort(key=lambda t: (-t[0], t[1]))
        return [idx for _, idx in scored[:k]]
