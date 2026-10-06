"""
RETRIEVAL 2: FLAT DENSE RETRIEVAL  -- baseline (DPR / vanilla RAG)

The standard RAG retriever: ignore the tree entirely, score the query against
every leaf chunk embedding and return the k most similar.

    query q -> q_hat -> cosine against all N leaf rows -> top-k

Same chunks, same encoder and same leaf vectors as HAT-RAG (it reads the
identical TreeIndex), so any difference in the comparison comes from the
retrieval strategy alone. Cost is exactly N nodes scored per query.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrieval1_hat_beam_traversal import (
    DEFAULT_TREE,
    DEMO_QUERIES,
    RetrievalResult,
    TreeIndex,
    print_result,
)


class FlatDenseRetriever:
    """Exhaustive cosine top-k over the leaf population."""

    name = "Flat dense RAG"
    short = "Flat dense"

    def __init__(self, index: TreeIndex):
        self.index = index
        self.candidates = index.leaf_idx

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    def search(self, q: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        sims = self.index.score(q, self.candidates)
        order = np.argsort(-sims)[:k]
        elapsed = time.perf_counter() - t0
        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=[self.index.ids[self.candidates[i]] for i in order],
            scores=[float(sims[i]) for i in order],
            nodes_scored=len(self.candidates),
            seconds=elapsed,
            trace=[f"scored all {len(self.candidates)} leaves"],
        )


def _cli():
    p = argparse.ArgumentParser(description="Retrieval 2: flat dense retrieval over leaf chunks")
    p.add_argument("--tree", default=str(DEFAULT_TREE))
    p.add_argument("--query", action="append")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    print("=" * 100)
    print(" RETRIEVAL 2: FLAT DENSE RETRIEVAL (all leaves scored)")
    print("=" * 100)
    index = TreeIndex.from_json(args.tree, device=args.device)
    print(f"[Index] {index.n_leaves} leaf chunks | {index.embedder.banner()}")
    retriever = FlatDenseRetriever(index)
    for q in args.query or DEMO_QUERIES:
        print_result(index, retriever.retrieve(q, k=args.k))
    print("=" * 100)
