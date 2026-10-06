"""
RETRIEVAL 1 (ALGORITHM 4): HAT BEAM TRAVERSAL + GRAPH PROPAGATION  -- our method

Online half of HAT-RAG. Takes the tree H built by Algorithm 3 and answers a
query without scoring every chunk:

    query q  -> same dense encoder as the tree -> q_hat
    1. descend    ROOT -> children -> keep the top-beta abstracts -> ... -> leaves
                  (alpha edges let the beam step sideways to an abstract under
                  a different parent)
    2. link       leaves naming an entity the query names (ASC 842, AWS, a
                  project code ...) join the pool through an inverted index
    3. expand     from the strongest leaves along H's explicit leaf edges:
                  alpha cross-child edges + reading order inside a filing
    4. propagate  personalised PageRank over the candidate subgraph, using
                  H's own edges (alpha, reading order, shared entities)
    5. select     relevance-gated diversity: a filing not yet in R earns a
                  bonus in proportion to how relevant that filing is
    -> evidence set R (top-k leaves) -> context C(R) with provenance markers

Only the candidates the descent reaches are ever scored, so the cost stays
sub-linear in the corpus size; ranking quality comes from the graph signal
that H already stores. Steps 2-5 were added after the comparison in
compare-algos/ showed the paper's plain doc-novelty bonus ranked last; pass
**PAPER_SPEC to get the original beam + alpha + bonus version back.

This module also holds the query-time index (TreeIndex), the entity index and
the result type that the three comparison retrievers (retrieval2..4) reuse, so
every method scores the exact same vectors produced by the exact same encoder.
"""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from scipy import sparse

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
# Entity index over the leaf chunks
# ----------------------------------------------------------------------------
_STANDARD_RE = re.compile(r"\bASC\s?\d{3}\b")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&]{1,}[A-Z0-9]\b|\bR&D\b")
_PROPER_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*\b")
_MIDCAP_RE = re.compile(r"(?<=[a-z0-9,;:(]\s)[A-Z][a-zA-Z]+")


class EntityIndex:
    """
    Accounting standards, acronyms and proper-noun phrases per leaf, with an
    inverted index. No hand-written stoplist: a capitalised word is dropped if
    the corpus ever uses it in lower case, and a lone capitalised word must
    also appear capitalised mid-sentence somewhere -- so sentence-initial
    'Total', 'Operating' or 'Goodwill' never become entities, while 'Azure',
    'Megapack' or a multi-word name like 'Project A76' do. Entities found in
    more than half the leaves link everything to everything and are dropped;
    the rest carry an IDF weight.
    """

    def __init__(self, texts: Dict[int, str]):
        self.common: Set[str] = set()
        self.midcaps: Set[str] = set()
        for t in texts.values():
            self.common.update(re.findall(r"\b[a-z][a-z-]+\b", t))
            self.midcaps.update(w.lower() for w in _MIDCAP_RE.findall(t))
        self.of: Dict[int, Set[str]] = {i: self.extract(t) for i, t in texts.items()}

        n = max(1, len(texts))
        postings: Dict[str, List[int]] = {}
        for i, ents in self.of.items():
            for e in ents:
                postings.setdefault(e, []).append(i)
        self.postings = {e: p for e, p in postings.items() if len(p) <= max(2, n // 2)}
        self.idf = {e: float(np.log(n / len(p))) for e, p in self.postings.items()}

    def extract(self, text: str) -> Set[str]:
        ents: Set[str] = set()
        ents.update(m.replace(" ", "").lower() for m in _STANDARD_RE.findall(text))
        ents.update(m.lower() for m in _ACRONYM_RE.findall(text) if m != "ASC")
        for phrase in _PROPER_RE.findall(text):
            if phrase == "ASC":
                continue
            words = [w for w in phrase.split() if w.lower() not in self.common]
            if len(phrase.split()) == 1 and words and words[0].lower() not in self.midcaps:
                continue                             # only ever seen opening a sentence
            if words:
                ents.add(" ".join(words).lower())
        return ents

    def pair_weights(self) -> Dict[Tuple[int, int], float]:
        """IDF-weighted shared-entity weight for every pair of leaves that share one."""
        pairs: Dict[Tuple[int, int], float] = {}
        for e, post in self.postings.items():
            for x in range(len(post)):
                for y in range(x + 1, len(post)):
                    key = (min(post[x], post[y]), max(post[x], post[y]))
                    pairs[key] = pairs.get(key, 0.0) + self.idf[e]
        return pairs


def symmetric_matrix(pairs: Dict[Tuple[int, int], float], n: int) -> sparse.csr_matrix:
    rows, cols, vals = [], [], []
    for (a, b), w in pairs.items():
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)


# ----------------------------------------------------------------------------
# Query-time view of a persisted tree
# ----------------------------------------------------------------------------
class TreeIndex:
    """
    Flattens the TreeNode graph into arrays a retriever can score against:
    one L2-normalised embedding row per node, levels, parent/child index lists,
    alpha cross-link adjacency, reading-order adjacency and provenance. Built
    once, shared by every retriever so they all see identical vectors.
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
        self.alpha_pairs: Dict[Tuple[int, int], float] = {}
        for i, n in enumerate(self.nodes):
            for l in n.cross_links:
                j = self.pos.get(l["target_id"])
                if j is not None:
                    self.alpha_pairs[(min(i, j), max(i, j))] = float(l["score"])
        self.doc: List[Optional[str]] = [n.doc_id for n in self.nodes]
        self.docs: List[List[str]] = [n.source_docs for n in self.nodes]

        # reading-order neighbours: chunk i-1 and i+1 of the same document,
        # recovered from the provenance every leaf already carries
        slot = {(n.doc_id, n.metadata.get("chunk_index")): i
                for i, n in enumerate(self.nodes) if n.level == 0}
        self.sequence: List[List[int]] = [[] for _ in self.nodes]
        for (doc, ci), i in slot.items():
            if ci is None:
                continue
            for nb in (ci - 1, ci + 1):
                j = slot.get((doc, nb))
                if j is not None:
                    self.sequence[i].append(j)
        self._entities: Optional[EntityIndex] = None
        self._edges: Optional[Dict[str, sparse.csr_matrix]] = None
        self.edge_build_seconds = 0.0

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

    @property
    def entities(self) -> EntityIndex:
        """Built on first use, once per index (offline cost, like the alpha edges)."""
        if self._entities is None:
            self._entities = EntityIndex({int(i): self.nodes[i].text for i in self.leaf_idx})
        return self._entities

    @property
    def edges(self) -> Dict[str, sparse.csr_matrix]:
        """H's explicit edges as symmetric node x node matrices, built once:
        'alpha' (cross-child links, weight = cosine), 'sequence' (reading order,
        weight 1) and 'entity' (shared entities, IDF-weighted)."""
        if self._edges is None:
            t0 = time.perf_counter()
            seq = {(min(i, j), max(i, j)): 1.0 for i in self.leaf_idx for j in self.sequence[int(i)]}
            self._edges = {
                "alpha": symmetric_matrix(self.alpha_pairs, self.n_nodes),
                "sequence": symmetric_matrix(seq, self.n_nodes),
                "entity": symmetric_matrix(self.entities.pair_weights(), self.n_nodes),
            }
            self.edge_build_seconds = time.perf_counter() - t0
        return self._edges

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
# Ranking helpers
# ----------------------------------------------------------------------------
def diversity_rank(pool: Sequence[int], sims: Dict[int, float], doc_of: Sequence[Optional[str]],
                   k: int, lam: float, doc_weight: Optional[Dict[Optional[str], float]] = None) -> List[int]:
    """Greedy R <- R + argmax s(v) + lam * w(doc(v)) * [doc(v) not in docs(R)].

    w = 1 everywhere is the paper's plain provenance bonus; passing doc_weight
    gates it by how relevant each document is, so an off-topic filing earns no bonus."""
    remaining = list(pool)
    chosen: List[int] = []
    seen: Set[str] = set()

    def bonus(i: int) -> float:
        if doc_of[i] in seen:
            return 0.0
        return lam * (doc_weight.get(doc_of[i], 0.0) if doc_weight is not None else 1.0)

    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda i: sims[i] + bonus(i))
        chosen.append(best)
        remaining.remove(best)
        if doc_of[best] is not None:
            seen.add(doc_of[best])
    return chosen


def personalized_pagerank(W: np.ndarray, personal: np.ndarray, damping: float = 0.5,
                          tol: float = 1e-9, max_iter: int = 100) -> np.ndarray:
    """Fixed point of r = (1 - d) p + d P^T r over a symmetric weight matrix W,
    iterated until the L1 change drops below tol (the error shrinks by d per step)."""
    out = W.sum(axis=1, keepdims=True)
    out[out == 0.0] = 1.0
    PT = (W / out).T
    p = personal / personal.sum()
    restart = (1.0 - damping) * p
    r = p
    for _ in range(max_iter):
        nxt = restart + damping * (PT @ r)
        if np.abs(nxt - r).sum() < tol:
            return nxt
        r = nxt
    return r


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
            tag = f"[{node.doc_id} | {node.metadata.get('section', '')} | score={score:.3f}]"
        else:
            tag = f"[{node.node_id} abstract over {', '.join(node.source_docs)} | score={score:.3f}]"
        blocks.append(f"{tag}\n{' '.join(words)}")
        used += len(words)
    return "\n\n".join(blocks)


# ============================================================================
# HAT retriever
# ============================================================================
# The original Algorithm 4 from the paper: beam descent, alpha expansion, a
# flat +lambda bonus for any unseen document, ranking by cosine alone.
PAPER_SPEC = dict(follow_sequence=False, entity_seeding=False, ranking="similarity",
                  gate_diversity=False)


class HATBeamRetriever:
    """
    Parameters
    ----------
    index           : TreeIndex over the persisted tree.
    beam_width      : abstracts kept per level (beta). Wider = more recall, more cost.
    diversity       : lambda, bonus for a leaf from a filing not yet in R.
    follow_alpha    : use the alpha edges (sideways during the descent, and in expansion).
    follow_sequence : use reading-order edges between neighbouring chunks of a filing.
    entity_seeding  : add the leaves that name an entity the query names.
    ranking         : 'ppr' (personalised PageRank over the candidate subgraph)
                      or 'similarity' (cosine only).
    gate_diversity  : scale the lambda bonus by the filing's relevance.
    damping         : PageRank continuation probability.
    n_seeds         : top-cosine leaves that personalise the walk.
    """

    name = "HAT-RAG (beam traversal + graph propagation)"
    short = "HAT-RAG"

    def __init__(self, index: TreeIndex, beam_width: int = 3, diversity: float = 0.15,
                 follow_alpha: bool = True, follow_sequence: bool = True,
                 entity_seeding: bool = True, ranking: str = "ppr", gate_diversity: bool = True,
                 damping: float = 0.5, n_seeds: int = 3, w_sequence: float = 0.5):
        self.index = index
        self.beam_width = max(1, int(beam_width))
        self.diversity = float(diversity)
        self.follow_alpha = follow_alpha
        self.follow_sequence = follow_sequence
        self.entity_seeding = entity_seeding
        self.ranking = ranking
        self.gate_diversity = gate_diversity
        self.damping = damping
        self.n_seeds = n_seeds
        self.w_sequence = w_sequence
        self._graph = None
        if entity_seeding:
            _ = index.entities                       # build the inverted index up front
        if ranking == "ppr":                         # H's edges as one weighted matrix, built once
            edges = index.edges
            graph = edges["entity"].copy()
            if follow_alpha:
                graph = graph + edges["alpha"]
            if follow_sequence:
                graph = graph + w_sequence * edges["sequence"]
            # dense slicing is far cheaper per query while the matrix fits comfortably
            self._graph = graph.toarray() if graph.shape[0] <= 4096 else graph.tocsr()

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        return self.search(self.index.encode_query(query), k=k, query=query)

    # -- step 4: graph propagation over the candidate subgraph ---------------
    def _propagate(self, pool: List[int], sims: Dict[int, float], linked: Set[int]) -> Dict[int, float]:
        at = {i: n for n, i in enumerate(pool)}
        m = len(pool)
        if isinstance(self._graph, np.ndarray):
            W = self._graph[np.ix_(pool, pool)]
        else:
            W = self._graph[pool][:, pool].toarray()

        personal = np.zeros(m, dtype=np.float64)
        for i in sorted(pool, key=lambda i: -sims[i])[: self.n_seeds]:
            personal[at[i]] = max(sims[i], 1e-6)
        for j in linked:
            if j in at:
                personal[at[j]] += max(sims[j], 1e-6)
        r = personalized_pagerank(W, personal, self.damping)
        top = float(r.max()) or 1.0
        return {i: float(r[at[i]]) / top for i in pool}

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

        # -- step 1: beam descent ------------------------------------------
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
                trace.append(f"descend: {len(inner)} abstracts ({lateral} via alpha) + {len(leaves)} leaves "
                             f"scored -> beam [{kept}]")
            else:                                    # nothing to prune -> expand all, no scoring
                frontier = inner
                if inner or leaves:
                    trace.append(f"descend: {len(inner)} abstracts expanded unscored + {len(leaves)} leaves scored")

        # -- step 2: entity linking through the inverted index ----------------
        linked: Set[int] = set()
        if self.entity_seeding and query:
            ents = idx.entities
            named = sorted(e for e in ents.extract(query) if e in ents.postings)
            for e in named:
                linked.update(ents.postings[e])
            new = [j for j in linked if j not in visited]
            visited.update(new)
            score(new)
            leaf_pool.extend(new)
            if named:
                trace.append(f"link: query names {named} -> {len(linked)} leaves ({len(new)} new)")

        # -- step 3: expansion along H's leaf edges ----------------------------
        if (self.follow_alpha or self.follow_sequence) and leaf_pool:
            extra: List[int] = []
            n_alpha = n_seq = 0
            for u in top(leaf_pool, k):
                links = []
                if self.follow_alpha:
                    links += [(p, "a") for p in idx.alpha[u]]
                if self.follow_sequence:
                    links += [(p, "s") for p in idx.sequence[u]]
                for p, kind in links:
                    if p not in visited and idx.is_leaf[p]:
                        visited.add(p)
                        extra.append(p)
                        n_alpha += kind == "a"
                        n_seq += kind == "s"
            if extra:
                score(extra)
                leaf_pool.extend(extra)
                trace.append(f"expand: +{n_alpha} leaves via alpha edges, +{n_seq} via reading order")

        if not leaf_pool:
            return RetrievalResult(self.short, query, [], [], len(sims), time.perf_counter() - t0, trace)

        # -- step 4: rank ------------------------------------------------------
        if self.ranking == "ppr":
            ranked = self._propagate(leaf_pool, sims, linked)
            trace.append(f"propagate: PageRank over {len(leaf_pool)} candidate leaves")
        else:
            ranked = {i: sims[i] for i in leaf_pool}

        # -- step 5: relevance-gated diversity -----------------------------------
        weight = None
        if self.gate_diversity:
            doc_best: Dict[Optional[str], float] = {}
            for i in leaf_pool:
                doc_best[idx.doc[i]] = max(doc_best.get(idx.doc[i], 0.0), ranked[i])
            best = max(doc_best.values()) or 1.0
            weight = {d: max(0.0, s) / best for d, s in doc_best.items()}

        chosen = diversity_rank(leaf_pool, ranked, idx.doc, k, self.diversity, weight)
        elapsed = time.perf_counter() - t0
        return RetrievalResult(
            method=self.short,
            query=query,
            node_ids=[idx.ids[i] for i in chosen],
            scores=[ranked[i] for i in chosen],
            nodes_scored=len(sims),
            seconds=elapsed,
            trace=trace,
        )


# ============================================================================
# Demonstration
# ============================================================================
DEMO_QUERIES = [
    "What were Apple's diluted earnings per share?",
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


def run_hat_retrieval_demo(tree_path: str, queries: Sequence[str], k: int, beam: int, lam: float,
                           paper_spec: bool, device: str, show_context: bool):
    print("=" * 100)
    print(" RETRIEVAL 1 / ALGORITHM 4: HAT BEAM TRAVERSAL + GRAPH PROPAGATION")
    print("=" * 100)
    index = TreeIndex.from_json(tree_path, device=device)
    print(f"[Index] {Path(tree_path).name}: {index.n_nodes} nodes, {index.n_leaves} leaves, "
          f"depth {index.depth()}, {len(index.alpha_pairs)} alpha edges, "
          f"{len(index.entities.postings)} indexed entities")
    print(f"[Encoder] {index.embedder.banner()}")
    cfg = PAPER_SPEC if paper_spec else {}
    retriever = HATBeamRetriever(index, beam_width=beam, diversity=lam, **cfg)
    print(f"[Config] beam={beam} lambda={lam} k={k} ranking={retriever.ranking} "
          f"entity-seeding={retriever.entity_seeding} reading-order={retriever.follow_sequence} "
          f"| scoring on {describe_device(index.device)}")
    for q in queries:
        print_result(index, retriever.retrieve(q, k=k), show_context=show_context)
    print("=" * 100)
    return retriever


def _cli():
    p = argparse.ArgumentParser(description="Retrieval 1: HAT beam traversal + graph propagation")
    p.add_argument("--tree", default=str(DEFAULT_TREE), help="tree JSON written by Algorithm 3")
    p.add_argument("--query", action="append", help="query text (repeatable); demo queries if omitted")
    p.add_argument("-k", type=int, default=5, help="evidence set size")
    p.add_argument("--beam", type=int, default=3, help="beam width beta")
    p.add_argument("--lam", type=float, default=0.15, help="diversity weight lambda")
    p.add_argument("--paper-spec", action="store_true", help="the original Algorithm 4 (no linking, no PageRank)")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--context", action="store_true", help="print the assembled context C(R)")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    run_hat_retrieval_demo(args.tree, args.query or DEMO_QUERIES, args.k, args.beam, args.lam,
                           args.paper_spec, args.device, args.context)
