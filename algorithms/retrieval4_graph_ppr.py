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
              -> personalised PageRank (damping d) over the WHOLE graph
              -> top-k by PageRank mass

Same leaf vectors, encoder and entity extractor as the other retrievers. Cost
per query is N cosine scores plus a walk over the whole graph; building the
graph is a one-off O(N^2) similarity pass.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval1_hat_beam_traversal import (
    DEFAULT_TREE,
    DEMO_QUERIES,
    RetrievalResult,
    TreeIndex,
    print_result,
    top_k,
)


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

        n = len(self.leaves)
        local = {int(i): j for j, i in enumerate(self.leaves)}
        self.local = local
        self.ents = index.entities
        edges: Dict[tuple, float] = {}

        def add(a: int, b: int, w: float):
            if a != b:
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0.0) + w

        # entity edges: passages that name the same entity, IDF-weighted. Kept as
        # the leaf x entity incidence B rather than enumerated pairs, so the walk
        # applies W_ent r = B (idf * (B^T r)) - diag * r: identical weights, and
        # it scales to corpora where pairs would number in the tens of millions.
        B_all, idf = index.entity_incidence
        self.B = B_all[self.leaves].tocsr()
        self.Bw = self.B.multiply(w_entity * idf).tocsr()
        self.ent_diag = np.asarray(self.Bw.sum(axis=1)).ravel()       # self-pairs to remove
        self.BT = self.B.T.tocsr()
        ent_pairs = int(((np.asarray(self.B.sum(axis=0)).ravel() ** 2 - np.asarray(self.B.sum(axis=0)).ravel()) / 2).sum())
        # semantic edges: top-m neighbours above tau, found in row blocks
        E = self.E_leaves = np.ascontiguousarray(index.E[self.leaves])
        for start in range(0, n, 2048):
            sim = E[start:start + 2048] @ E.T
            rows = np.arange(sim.shape[0])
            sim[rows, start + rows] = -1.0
            m = min(top_m, n - 1)
            if m <= 0:
                break
            part = np.argpartition(-sim, m - 1, axis=1)[:, :m]
            for r, nbrs in enumerate(part):
                a = start + r
                for b in nbrs[np.argsort(-sim[r, nbrs])]:
                    if sim[r, b] >= tau:
                        add(a, int(b), w_semantic * float(sim[r, b]))
        # reading-order edges inside one filing
        for i in self.leaves:
            for j in index.sequence[int(i)]:
                if local[int(i)] < local[j]:
                    add(local[int(i)], local[j], w_sequence)

        rows, cols, vals = [], [], []
        for (a, b), w in edges.items():
            rows += [a, b]
            cols += [b, a]
            vals += [w, w]
        self.W_static = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        out = np.asarray(self.W_static.sum(axis=1)).ravel() + self._entity_apply(np.ones(n))
        out[out == 0.0] = 1.0
        self.inv_out = 1.0 / out
        if n <= TreeIndex.DENSE_GRAPH_LIMIT:         # exact count of distinct edges
            from retrieval1_hat_beam_traversal import entity_weights
            union = entity_weights(self.B, idf) + self.W_static
            self.n_edges = int(union.nnz // 2)
        else:                                        # entity pairs counted, not stored
            self.n_edges = len(edges) + ent_pairs
        self.build_seconds = time.perf_counter() - t0

    def _entity_apply(self, x: np.ndarray) -> np.ndarray:
        return self.Bw @ (self.BT @ x) - self.ent_diag * x

    def _walk(self, r: np.ndarray) -> np.ndarray:
        """P^T r for the row-stochastic walk P = D^-1 W (W symmetric)."""
        y = r * self.inv_out
        return self.W_static @ y + self._entity_apply(y)

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    def search(self, q: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        sims = self.index.score(q, self.leaves) if self.index.device.startswith("cuda") else self.E_leaves @ q
        n = len(self.leaves)

        personal = np.zeros(n, dtype=np.float64)
        seeds = top_k(sims, self.n_seeds)
        personal[seeds] = np.clip(sims[seeds], 1e-6, None)
        # entity linking: passages naming an entity that the query names
        linked = 0
        for e in self.ents.extract(query):
            for i in self.ents.postings.get(e, []):
                j = self.local[i]
                personal[j] += max(float(sims[j]), 1e-6)
                linked += 1
        personal /= personal.sum()

        r = personal.copy()
        for _ in range(self.iterations):
            r = (1.0 - self.damping) * personal + self.damping * self._walk(r)

        order = top_k(r, k)
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
          f"{len(retriever.ents.postings)} distinct entities, built in {retriever.build_seconds:.3f}s")
    for q in args.query or DEMO_QUERIES:
        print_result(index, retriever.retrieve(q, k=args.k))
    print("=" * 100)
