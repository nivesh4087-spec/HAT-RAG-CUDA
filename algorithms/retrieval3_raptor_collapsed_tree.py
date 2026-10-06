"""
RETRIEVAL 3: RAPTOR-STYLE COLLAPSED-TREE RETRIEVAL  -- baseline (Sarthi et al. 2024)

RAPTOR's preferred query mode: flatten every level of the tree into one pool
and let leaf chunks and abstract nodes compete on cosine similarity alone.

    query q -> q_hat -> cosine against ALL nodes (leaves + abstracts + root) -> top-k

It uses the same HAT tree as retrieval 1, so it gets the summary nodes for
free, but it never uses the tree's structure: no traversal, no alpha edges,
no diversity term. Cost is |V| nodes per query -- strictly more than flat.
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


class CollapsedTreeRetriever:
    """Cosine top-k over the pooled population of every tree level."""

    name = "RAPTOR collapsed tree"
    short = "RAPTOR collapsed"

    def __init__(self, index: TreeIndex, include_root: bool = True):
        self.index = index
        keep = np.ones(index.n_nodes, dtype=bool)
        if not include_root:
            keep[index.root_idx] = False
        self.candidates = np.flatnonzero(keep)

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    def search(self, q: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        sims = self.index.score(q, self.candidates)
        order = np.argsort(-sims)[:k]
        elapsed = time.perf_counter() - t0
        chosen = [self.candidates[i] for i in order]
        n_abs = sum(1 for i in chosen if not self.index.is_leaf[i])
        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=[self.index.ids[i] for i in chosen],
            scores=[float(sims[i]) for i in order],
            nodes_scored=len(self.candidates),
            seconds=elapsed,
            trace=[f"scored all {len(self.candidates)} nodes across every level; "
                   f"{n_abs} of the top-{k} are abstracts"],
        )


def _cli():
    p = argparse.ArgumentParser(description="Retrieval 3: RAPTOR-style collapsed-tree retrieval")
    p.add_argument("--tree", default=str(DEFAULT_TREE))
    p.add_argument("--query", action="append")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    print("=" * 100)
    print(" RETRIEVAL 3: RAPTOR-STYLE COLLAPSED TREE (every level pooled)")
    print("=" * 100)
    index = TreeIndex.from_json(args.tree, device=args.device)
    print(f"[Index] {index.n_nodes} nodes over {index.depth() + 1} levels | {index.embedder.banner()}")
    retriever = CollapsedTreeRetriever(index)
    for q in args.query or DEMO_QUERIES:
        print_result(index, retriever.retrieve(q, k=args.k))
    print("=" * 100)
