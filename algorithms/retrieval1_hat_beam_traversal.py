"""
RETRIEVAL 1 (ALGORITHM 4): HAT BEAM-GUIDED TOP-DOWN RETRIEVAL  -- our method

Online half of HAT-RAG. Takes the tree H built by Algorithm 3 and answers a
query by descending it instead of scoring every chunk:

    query q  -> same dense encoder as the tree -> q_hat
    ROOT     -> children -> keep the top-beta abstracts -> their children -> ... -> leaves
             + alpha cross-child edges followed sideways, so related material
               under a different parent is reachable without climbing back up
             + diversity-aware ranking  score = sim(q, v) + lambda * [doc(v) not yet in R]
    -> evidence set R (top-k leaves) -> context C(R) with provenance markers

Scoring is lazy: a candidate set is only scored when a decision depends on it
(pruning a level wider than the beam, or ranking leaves), so the nodes-scored
count reflects what the traversal actually had to look at.

This module also holds the query-time index (TreeIndex) and the result type
that the three comparison retrievers (retrieval2..4) reuse, so every method
scores the exact same vectors produced by the exact same encoder.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algo3_hierarchical_abstract_tree import (
    DEFAULT_MODEL,
    HAS_TORCH,
    DenseEmbeddingModel,
    HierarchicalAbstractTreeBuilder,
    TreeNode,
    describe_device,
    resolve_device,
)

if HAS_TORCH:
    import torch

DEFAULT_TREE = Path(__file__).resolve().parent / "hat_finance_tree_structure.json"


# ----------------------------------------------------------------------------
# Result type shared by all four retrievers
# ----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    method: str
    query: str
    node_ids: List[str]
    scores: List[float]
    nodes_scored: int          # node representations compared against the query
    seconds: float             # search time, query encoding excluded
    trace: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "query": self.query,
            "node_ids": self.node_ids,
            "scores": [round(s, 4) for s in self.scores],
            "nodes_scored": self.nodes_scored,
            "latency_ms": round(self.seconds * 1000, 4),
            "trace": self.trace,
        }


# ----------------------------------------------------------------------------
# Query-time view of a persisted tree
# ----------------------------------------------------------------------------
class TreeIndex:
    """
    Flattens the TreeNode graph into arrays a retriever can score against:
    one L2-normalised embedding row per node, levels, parent/child index lists,
    alpha cross-link adjacency and provenance. Built once, shared by every
    retriever so they all see identical vectors.
    """

    def __init__(self, nodes: Dict[str, TreeNode], root_ids: Sequence[str],
                 embedder: DenseEmbeddingModel, device: str = "cpu"):
        self.embedder = embedder
        self.device = resolve_device(device)
        self.ids: List[str] = list(nodes.keys())
        self.pos: Dict[str, int] = {nid: i for i, nid in enumerate(self.ids)}
        self.nodes: List[TreeNode] = [nodes[nid] for nid in self.ids]

        mat = np.stack([np.asarray(n.embedding, dtype=np.float32) for n in self.nodes])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-9
        self.E = mat / norms
        self._E_dev = None

        self.level = np.array([n.level for n in self.nodes], dtype=np.int64)
        self.is_leaf = self.level == 0
        self.leaf_idx = np.flatnonzero(self.is_leaf)
        self.root_idx = [self.pos[r] for r in root_ids if r in self.pos]
        self.children: List[List[int]] = [[self.pos[c.node_id] for c in n.children] for n in self.nodes]
        self.parent: List[int] = [self.pos[n.parent.node_id] if n.parent else -1 for n in self.nodes]
        self.alpha: List[List[int]] = [
            [self.pos[l["target_id"]] for l in n.cross_links if l["target_id"] in self.pos]
            for n in self.nodes
        ]
        self.doc: List[Optional[str]] = [n.doc_id for n in self.nodes]
        self.docs: List[List[str]] = [n.source_docs for n in self.nodes]

    # -- construction --------------------------------------------------------
    @classmethod
    def from_json(cls, path: Union[str, Path] = DEFAULT_TREE,
                  embedder: Optional[DenseEmbeddingModel] = None,
                  model_name: str = DEFAULT_MODEL, device: str = "cpu",
                  verbose: bool = False) -> "TreeIndex":
        embedder = embedder or DenseEmbeddingModel(model_name=model_name, device=device, verbose=verbose)
        loader = HierarchicalAbstractTreeBuilder(embedder=embedder, summarizer_mode="extractive",
                                                 device=device, verbose=False)
        data = loader.load_tree_json(str(path))
        return cls(loader.nodes, data["root_ids"], embedder, device=device)

    @classmethod
    def from_builder(cls, builder: HierarchicalAbstractTreeBuilder, device: str = "cpu") -> "TreeIndex":
        return cls(builder.nodes, [r.node_id for r in builder.root_nodes], builder.embedder, device=device)

    # -- properties ----------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return len(self.ids)

    @property
    def n_leaves(self) -> int:
        return int(self.is_leaf.sum())

    def depth(self) -> int:
        return int(self.level.max()) if len(self.level) else 0

    # -- scoring -------------------------------------------------------------
    def encode_query(self, query: str) -> np.ndarray:
        return self.embedder.encode(query, label="query")

    def score(self, q: np.ndarray, idx: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Cosine of q against the rows idx: one batched mat-vec (one kernel on CUDA)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros(0, dtype=np.float32)
        if HAS_TORCH and self.device.startswith("cuda"):
            if self._E_dev is None:
                self._E_dev = torch.as_tensor(self.E, device=self.device)
            tq = torch.as_tensor(q, device=self.device, dtype=torch.float32)
            return (self._E_dev[torch.as_tensor(idx, device=self.device)] @ tq).cpu().numpy()
        return self.E[idx] @ q

    def text(self, i: int) -> str:
        return self.nodes[i].text

    def label(self, i: int) -> str:
        """Short provenance tag: ticker/doc + chunk for leaves, node id for abstracts."""
        node = self.nodes[i]
        if node.level == 0:
            doc = (node.doc_id or "?").split("_")[0]
            return f"{doc}#c{node.metadata.get('chunk_index', '?')}"
        return node.node_id


# ----------------------------------------------------------------------------
# Diversity-aware ranking (maximal marginal relevance over provenance)
# ----------------------------------------------------------------------------
def diversity_rank(pool: Sequence[int], sims: Dict[int, float], doc_of: Sequence[Optional[str]],
                   k: int, lam: float) -> List[int]:
    """Greedy R <- R + argmax sim(q, v) + lam * [doc(v) not in docs(R)]."""
    remaining = list(pool)
    chosen: List[int] = []
    seen: Set[str] = set()
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda i: sims[i] + (lam if doc_of[i] not in seen else 0.0))
        chosen.append(best)
        remaining.remove(best)
        if doc_of[best] is not None:
            seen.add(doc_of[best])
    return chosen


# ----------------------------------------------------------------------------
# Context assembly C(R) for the generator
# ----------------------------------------------------------------------------
def assemble_context(index: TreeIndex, result: RetrievalResult, max_words: int = 400) -> str:
    """Ordered concatenation of <provenance, text> blocks under a word budget."""
    blocks: List[str] = []
    used = 0
    for nid, score in zip(result.node_ids, result.scores):
        node = index.nodes[index.pos[nid]]
        words = node.text.split()
        if used + len(words) > max_words:
            words = words[: max(0, max_words - used)]
            if not words:
                break
        if node.level == 0:
            tag = f"[{node.doc_id} | {node.metadata.get('section', '')} | sim={score:.3f}]"
        else:
            tag = f"[{node.node_id} abstract over {', '.join(node.source_docs)} | sim={score:.3f}]"
        blocks.append(f"{tag}\n{' '.join(words)}")
        used += len(words)
    return "\n\n".join(blocks)


# ============================================================================
# HAT beam-guided top-down retriever
# ============================================================================
class HATBeamRetriever:
    """
    Beam-guided descent of the HAT with alpha cross-child hops and
    diversity-aware leaf ranking.

    Parameters
    ----------
    index        : TreeIndex over the persisted tree.
    beam_width   : abstracts kept per level (beta). Wider = more recall, more cost.
    diversity    : lambda, bonus for a leaf from a document not yet in R.
    follow_alpha : follow the explicit alpha edges -- sideways between abstracts
                   during the descent, and from the strongest leaves at the end.
    """

    name = "HAT-RAG (beam traversal)"
    short = "HAT-RAG"

    def __init__(self, index: TreeIndex, beam_width: int = 3, diversity: float = 0.15,
                 follow_alpha: bool = True):
        self.index = index
        self.beam_width = max(1, int(beam_width))
        self.diversity = float(diversity)
        self.follow_alpha = follow_alpha

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    def search(self, q: np.ndarray, k: int = 5, query: str = "") -> RetrievalResult:
        t0 = time.perf_counter()
        idx = self.index
        beam = self.beam_width
        sims: Dict[int, float] = {}

        def score(cands: Sequence[int]):
            new = [c for c in cands if c not in sims]
            if new:
                for c, s in zip(new, idx.score(q, new)):
                    sims[c] = float(s)

        def top(cands: Sequence[int], n: int) -> List[int]:
            return sorted(cands, key=lambda i: -sims[i])[:n]

        trace: List[str] = []
        frontier = list(idx.root_idx)
        if len(frontier) > beam:
            score(frontier)
            frontier = top(frontier, beam)
        visited: Set[int] = set(frontier)
        leaf_pool: List[int] = []

        while frontier:
            cand: List[int] = []
            for u in frontier:
                for c in idx.children[u]:
                    if c not in visited:
                        visited.add(c)
                        cand.append(c)
            inner = [c for c in cand if not idx.is_leaf[c]]
            lateral = 0
            if self.follow_alpha:
                # sideways: an abstract under a different parent, linked by an alpha edge
                for u in list(inner):
                    for p in idx.alpha[u]:
                        if p not in visited and not idx.is_leaf[p]:
                            visited.add(p)
                            inner.append(p)
                            lateral += 1
            leaves = [c for c in cand if idx.is_leaf[c]]
            if leaves:
                score(leaves)
                leaf_pool.extend(leaves)

            if len(inner) > beam:                    # a pruning decision -> score this level
                score(inner)
                frontier = top(inner, beam)
                kept = ", ".join(f"{idx.label(i)}={sims[i]:.3f}" for i in frontier)
                trace.append(f"{len(inner)} abstracts ({lateral} via alpha) + {len(leaves)} leaves scored "
                             f"-> beam [{kept}]")
            else:                                    # nothing to prune -> expand all, no scoring
                frontier = inner
                if inner or leaves:
                    trace.append(f"{len(inner)} abstracts expanded unscored + {len(leaves)} leaves scored")

        if self.follow_alpha and leaf_pool:
            # alpha expansion from the strongest leaves: related passages that sit
            # under a parent the beam did not keep (often another company's filing)
            extra: List[int] = []
            for u in top(leaf_pool, k):
                for p in idx.alpha[u]:
                    if p not in visited and idx.is_leaf[p]:
                        visited.add(p)
                        extra.append(p)
            if extra:
                score(extra)
                leaf_pool.extend(extra)
                trace.append(f"alpha expansion: +{len(extra)} leaves from cross-links")

        chosen = diversity_rank(leaf_pool, sims, idx.doc, k, self.diversity)
        elapsed = time.perf_counter() - t0
        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=[idx.ids[i] for i in chosen],
            scores=[sims[i] for i in chosen],
            nodes_scored=len(sims),
            seconds=elapsed,
            trace=trace,
        )


# ============================================================================
# Demonstration
# ============================================================================
DEMO_QUERIES = [
    "How does Tesla establish warranty reserves?",
    "Compare the research and development spending of Apple and NVIDIA.",
    "Which companies account for leases under ASC 842?",
]


def print_result(index: TreeIndex, res: RetrievalResult, show_context: bool = False):
    print(f"\n  Q: {res.query}")
    for line in res.trace:
        print(f"     . {line}")
    print(f"     scored {res.nodes_scored}/{index.n_nodes} nodes in {res.seconds * 1000:.3f} ms")
    for rank, (nid, s) in enumerate(zip(res.node_ids, res.scores), 1):
        i = index.pos[nid]
        text = index.text(i)
        print(f"     {rank}. [{index.label(i):<10}] {s:.3f}  {text[:88]}{'...' if len(text) > 88 else ''}")
    if show_context:
        print("\n  Context C(R):")
        for line in assemble_context(index, res).splitlines():
            print(f"     {line}")


def run_hat_retrieval_demo(tree_path: str, queries: Sequence[str], k: int, beam: int,
                           lam: float, follow_alpha: bool, device: str, show_context: bool):
    print("=" * 100)
    print(" RETRIEVAL 1 / ALGORITHM 4: HAT BEAM-GUIDED TOP-DOWN RETRIEVAL")
    print("=" * 100)
    index = TreeIndex.from_json(tree_path, device=device)
    print(f"[Index] {Path(tree_path).name}: {index.n_nodes} nodes, {index.n_leaves} leaves, "
          f"depth {index.depth()}, {sum(len(a) for a in index.alpha)} alpha edges")
    print(f"[Encoder] {index.embedder.banner()}")
    print(f"[Config] beam={beam} lambda={lam} k={k} alpha-edges={'on' if follow_alpha else 'off'} "
          f"| scoring on {describe_device(index.device)}")

    retriever = HATBeamRetriever(index, beam_width=beam, diversity=lam, follow_alpha=follow_alpha)
    for q in queries:
        print_result(index, retriever.retrieve(q, k=k), show_context=show_context)
    print("=" * 100)
    return retriever


def _cli():
    p = argparse.ArgumentParser(description="Retrieval 1: HAT beam-guided top-down retrieval")
    p.add_argument("--tree", default=str(DEFAULT_TREE), help="tree JSON written by Algorithm 3")
    p.add_argument("--query", action="append", help="query text (repeatable); demo queries if omitted")
    p.add_argument("-k", type=int, default=5, help="evidence set size")
    p.add_argument("--beam", type=int, default=3, help="beam width beta")
    p.add_argument("--lam", type=float, default=0.15, help="diversity weight lambda")
    p.add_argument("--no-alpha", action="store_true", help="ignore the alpha cross-child edges")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--context", action="store_true", help="print the assembled context C(R)")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    run_hat_retrieval_demo(args.tree, args.query or DEMO_QUERIES, args.k, args.beam, args.lam,
                           not args.no_alpha, args.device, args.context)
