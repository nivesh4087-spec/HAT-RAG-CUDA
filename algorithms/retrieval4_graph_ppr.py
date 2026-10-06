"""
RETRIEVAL 4: ENTITY-GRAPH RETRIEVAL WITH PERSONALISED PAGERANK  -- baseline
(GraphRAG / HippoRAG family)

Builds a passage graph over the leaf chunks once, then answers a query by
seeding a random walk at the most relevant passages and letting relevance
spread along the graph:

    offline : passages = leaf chunks
              edges    = shared named entities   (ASC 842, AWS, FIFO, Azure, ...)
                       + semantic neighbours      (cosine >= tau, top-m per node)
                       + reading-order adjacency  (chunk i <-> i+1 of the same filing)
    online  : q_hat -> cosine against all N leaves -> top seeds
              + passages that mention an entity named in the query
              -> personalised PageRank (damping d) -> top-k by PageRank mass

Same leaf vectors and encoder as the other three retrievers. Cost per query is
N cosine scores plus the walk over the whole graph; building the graph is a
one-off O(N^2) similarity pass.
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval1_hat_beam_traversal import (
    DEFAULT_TREE,
    DEMO_QUERIES,
    RetrievalResult,
    TreeIndex,
    print_result,
)


# ----------------------------------------------------------------------------
# Lightweight entity extraction
# ----------------------------------------------------------------------------
_STANDARD_RE = re.compile(r"\bASC\s?\d{3}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&]{1,}[A-Z0-9]\b|\bR&D\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*\b")


class EntityExtractor:
    """
    Pulls accounting standards, acronyms and proper-noun phrases out of text.
    A capitalised word only counts as an entity if the corpus never uses it in
    lower case -- that drops sentence-initial words like 'Total' or 'Revenue'
    without a hand-written stoplist.
    """

    def __init__(self, corpus_texts: List[str]):
        self.common: Set[str] = set()
        for t in corpus_texts:
            self.common.update(w for w in re.findall(r"\b[a-z][a-z-]+\b", t))

    def __call__(self, text: str) -> Set[str]:
        ents: Set[str] = set()
        ents.update(m.replace(" ", "").lower() for m in _STANDARD_RE.findall(text))
        ents.update(m.lower() for m in _ACRONYM_RE.findall(text) if m != "ASC")
        for phrase in _PROPER_RE.findall(text):
            if phrase == "ASC":
                continue
            words = [w for w in phrase.split() if w.lower() not in self.common]
            if words:
                ents.add(" ".join(words).lower())
        return ents


# ============================================================================
# Graph retriever
# ============================================================================
class GraphPPRRetriever:
    """
    Parameters
    ----------
    tau        : cosine threshold for a semantic edge.
    top_m      : semantic neighbours kept per passage.
    n_seeds    : dense top passages used to seed the walk.
    damping    : probability of following an edge rather than restarting.
    """

    name = "Entity-graph RAG (PPR)"
    short = "Graph PPR"

    def __init__(self, index: TreeIndex, tau: float = 0.45, top_m: int = 4, n_seeds: int = 3,
                 damping: float = 0.5, iterations: int = 30,
                 w_entity: float = 1.0, w_semantic: float = 1.0, w_sequence: float = 0.5):
        self.index = index
        self.n_seeds = n_seeds
        self.damping = damping
        self.iterations = iterations
        self.leaves = index.leaf_idx
        t0 = time.perf_counter()

        texts = [index.text(i) for i in self.leaves]
        n = len(self.leaves)
        self.extract = EntityExtractor(texts)
        self.entities: List[Set[str]] = [self.extract(t) for t in texts]
        self.entity_postings: Dict[str, List[int]] = {}
        for j, ents in enumerate(self.entities):
            for e in ents:
                self.entity_postings.setdefault(e, []).append(j)
        # an entity in half the corpus links everything to everything: drop it,
        # and weight the rest by IDF so rare shared entities count for more
        self.idf = {e: float(np.log(n / len(p))) for e, p in self.entity_postings.items()
                    if 1 < len(p) <= max(2, n // 2)}

        edges: Dict[tuple, float] = {}

        def add(a: int, b: int, w: float):
            if a != b:
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0.0) + w

        # entity edges: passages that name the same entity
        for e, w in self.idf.items():
            post = self.entity_postings[e]
            for x in range(len(post)):
                for y in range(x + 1, len(post)):
                    add(post[x], post[y], w_entity * w)
        # semantic edges: top-m neighbours above tau
        E = index.E[self.leaves]
        sim = E @ E.T
        np.fill_diagonal(sim, -1.0)
        nbrs = np.argsort(-sim, axis=1)[:, :top_m]
        for a in range(n):
            for b in nbrs[a]:
                if sim[a, b] >= tau:
                    add(a, int(b), w_semantic * float(sim[a, b]))
        # reading-order edges inside one filing
        pos = {(index.doc[i], index.nodes[i].metadata.get("chunk_index")): j
               for j, i in enumerate(self.leaves)}
        for (doc, ci), j in pos.items():
            nxt = pos.get((doc, ci + 1)) if ci is not None else None
            if nxt is not None:
                add(j, nxt, w_sequence)

        rows, cols, vals = [], [], []
        for (a, b), w in edges.items():
            rows += [a, b]
            cols += [b, a]
            vals += [w, w]
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        out = np.asarray(W.sum(axis=1)).ravel()
        out[out == 0.0] = 1.0
        self.PT = (sparse.diags(1.0 / out) @ W).T.tocsr()    # transpose of the row-stochastic walk
        self.n_edges = len(edges)
        self.build_seconds = time.perf_counter() - t0

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    def search(self, q: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        sims = self.index.score(q, self.leaves)
        n = len(self.leaves)

        personal = np.zeros(n, dtype=np.float64)
        seeds = np.argsort(-sims)[: self.n_seeds]
        personal[seeds] = np.clip(sims[seeds], 1e-6, None)
        # entity linking: passages naming an entity that the query names
        linked = 0
        for e in self.extract(query):
            for j in self.entity_postings.get(e, []):
                personal[j] += max(float(sims[j]), 1e-6)
                linked += 1
        personal /= personal.sum()

        r = personal.copy()
        for _ in range(self.iterations):
            r = (1.0 - self.damping) * personal + self.damping * (self.PT @ r)

        order = np.argsort(-r)[:k]
        elapsed = time.perf_counter() - t0
        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=[self.index.ids[self.leaves[i]] for i in order],
            scores=[float(r[i]) for i in order],
            nodes_scored=n,
            seconds=elapsed,
            trace=[f"scored all {n} leaves for seeding, {linked} entity links from the query, "
                   f"PageRank over {self.n_edges} edges"],
        )


def _cli():
    p = argparse.ArgumentParser(description="Retrieval 4: entity-graph retrieval with personalised PageRank")
    p.add_argument("--tree", default=str(DEFAULT_TREE))
    p.add_argument("--query", action="append")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    print("=" * 100)
    print(" RETRIEVAL 4: ENTITY-GRAPH RETRIEVAL WITH PERSONALISED PAGERANK")
    print("=" * 100)
    index = TreeIndex.from_json(args.tree, device=args.device)
    retriever = GraphPPRRetriever(index)
    print(f"[Graph] {len(retriever.leaves)} passages, {retriever.n_edges} edges, "
          f"{len(retriever.entity_postings)} distinct entities, built in {retriever.build_seconds:.3f}s")
    for q in args.query or DEMO_QUERIES:
        print_result(index, retriever.retrieve(q, k=args.k))
    print("=" * 100)
