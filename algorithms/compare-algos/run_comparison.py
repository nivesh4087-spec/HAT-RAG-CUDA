"""
HEAD-TO-HEAD EVALUATION OF THE FOUR RETRIEVAL ALGORITHMS

    retrieval1_hat_beam_traversal   HAT-RAG: beam traversal + graph propagation (ours)
    retrieval2_flat_dense           flat dense retrieval
    retrieval3_raptor_collapsed     RAPTOR-style collapsed tree
    retrieval4_graph_ppr            entity graph + personalised PageRank

plus, as a reference row, HAT-RAG exactly as the paper specified Algorithm 4.

Benchmark A -- finance corpus (finance_data/reports, 5 x 10-K). 52 hand-labelled
    queries (single-hop / cross-document / thematic) against the HAT that
    run_finance_rag.py built (hat_finance_tree_structure.json).
Benchmark B -- synthetic cross-document corpus grown from 8 to 128 documents,
    to see how accuracy and cost behave as N grows; the finance corpus is only
    24 chunks, too small for cost differences to show.
Ablations -- every HAT-RAG component switched off in turn, beam width and
    diversity weight swept.

All methods read one TreeIndex: same chunks, same encoder, same vectors, and
the query is encoded once and handed to every method, so latency is search
time only. Results go to results/*.json; make_figures.py draws them.

    python algorithms/compare-algos/run_comparison.py
    python algorithms/compare-algos/run_comparison.py --sizes 8 16 32
"""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
ALGO_DIR = HERE.parent
sys.path.insert(0, str(ALGO_DIR))
sys.path.insert(0, str(HERE))

from retrieval1_hat_beam_traversal import DEFAULT_TREE, PAPER_SPEC, HATBeamRetriever, TreeIndex
from retrieval2_flat_dense import FlatDenseRetriever
from retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever
from retrieval4_graph_ppr import GraphPPRRetriever

from algo1_document_chunking import DocumentChunker
from algo3_hierarchical_abstract_tree import DenseEmbeddingModel, HierarchicalAbstractTreeBuilder
from synthetic_corpus import build_corpus

import numpy as np

K = 5
MATCH_FRACTION = 0.6
REPEATS = 7
RESULTS_DIR = HERE / "results"
QUERIES_PATH = HERE / "finance_queries.json"

# beta and lambda are the paper's values; everything else is retrieval1's default
HAT_CONFIG = dict(beam_width=3, diversity=0.15)
MAIN = ["HAT-RAG", "Flat dense", "RAPTOR collapsed", "Graph PPR"]
REFERENCE = "HAT-RAG (paper spec)"
METRICS = ["recall", "hit", "complete", "mrr", "doc_coverage", "precision",
           "nodes_scored", "latency_ms", "context_words"]

ABLATIONS = {
    "full HAT-RAG": {},
    "- entity linking": dict(entity_seeding=False),
    "- PageRank (cosine ranking)": dict(ranking="similarity"),
    "- reading-order edges": dict(follow_sequence=False),
    "- alpha edges": dict(follow_alpha=False),
    "- diversity": dict(diversity=0.0),
    "ungated diversity": dict(gate_diversity=False),
    "paper spec": PAPER_SPEC,
}


# ----------------------------------------------------------------------------
# Gold matching
# ----------------------------------------------------------------------------
def doc_match(node_doc: str, gold_doc: str) -> bool:
    return node_doc == gold_doc or node_doc.startswith(gold_doc + "_")


def covers(index: TreeIndex, i: int, item: Dict[str, Any]) -> bool:
    """Node i belongs to the gold filing and its own text holds >=60% of the key terms."""
    node = index.nodes[i]
    docs = [node.doc_id or ""] if node.level == 0 else node.source_docs
    if not any(doc_match(d, item["doc"]) for d in docs):
        return False
    low = node.text.lower()
    need = math.ceil(MATCH_FRACTION * len(item["terms"]))
    return sum(t.lower() in low for t in item["terms"]) >= need


def represents(index: TreeIndex, i: int, doc: str, aliases: Dict[str, List[str]]) -> bool:
    """A leaf represents its own filing; an abstract only one it actually names."""
    node = index.nodes[i]
    if node.level == 0:
        return doc_match(node.doc_id or "", doc)
    if not any(doc_match(d, doc) for d in node.source_docs):
        return False
    low = node.text.lower()
    return any(a.lower() in low for a in aliases.get(doc, []))


def validate_gold(index: TreeIndex, queries: Sequence[Dict[str, Any]]):
    """Every gold item must be coverable by some leaf, or the benchmark is broken."""
    bad = [(q["id"], g) for q in queries for g in q["gold"]
           if not any(covers(index, int(i), g) for i in index.leaf_idx)]
    if bad:
        raise ValueError(f"gold items no leaf can cover: {bad}")


def score_query(index: TreeIndex, node_ids: Sequence[str], gold: Sequence[Dict[str, Any]],
                aliases: Dict[str, List[str]]) -> Dict[str, float]:
    idxs = [index.pos[n] for n in node_ids]
    covered = [any(covers(index, i, g) for i in idxs) for g in gold]
    first = next((r for r, i in enumerate(idxs, 1) if any(covers(index, i, g) for g in gold)), None)
    gold_docs = list(dict.fromkeys(g["doc"] for g in gold))
    docs_hit = [any(represents(index, i, d, aliases) for i in idxs) for d in gold_docs]
    relevant = [any(represents(index, i, d, aliases) for d in gold_docs) for i in idxs]
    return {
        "recall": float(np.mean(covered)),
        "hit": float(any(covered)),
        "complete": float(all(covered)),
        "mrr": 1.0 / first if first else 0.0,
        "doc_coverage": float(np.mean(docs_hit)),
        "precision": float(np.mean(relevant)) if idxs else 0.0,
        "context_words": float(sum(len(index.text(i).split()) for i in idxs)),
    }


# ----------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------
def make_retrievers(index: TreeIndex, reference: bool = True) -> Dict[str, Any]:
    retrievers = {
        "HAT-RAG": HATBeamRetriever(index, **HAT_CONFIG),
        "Flat dense": FlatDenseRetriever(index),
        "RAPTOR collapsed": CollapsedTreeRetriever(index),
        "Graph PPR": GraphPPRRetriever(index),
    }
    if reference:
        retrievers[REFERENCE] = HATBeamRetriever(index, **HAT_CONFIG, **PAPER_SPEC)
    return retrievers


def timed_search(retriever, q: np.ndarray, query: str, repeats: int = REPEATS):
    result = retriever.search(q, k=K, query=query)
    times = [result.seconds]
    for _ in range(repeats - 1):
        times.append(retriever.search(q, k=K, query=query).seconds)
    return result, statistics.median(times) * 1000.0


def evaluate(index: TreeIndex, retrievers: Dict[str, Any], queries: Sequence[Dict[str, Any]],
             qvecs: np.ndarray, aliases: Dict[str, List[str]], repeats: int = REPEATS) -> List[Dict[str, Any]]:
    records = []
    for name, retriever in retrievers.items():
        retriever.search(qvecs[0], k=K, query=queries[0]["query"])          # warm-up
        for q, vec in zip(queries, qvecs):
            res, latency = timed_search(retriever, vec, q["query"], repeats)
            rec = {"method": name, "id": q["id"], "type": q["type"], "query": q["query"],
                   "node_ids": res.node_ids,
                   "retrieved": [index.label(index.pos[n]) for n in res.node_ids],
                   "nodes_scored": res.nodes_scored, "latency_ms": latency}
            rec.update(score_query(index, res.node_ids, q["gold"], aliases))
            records.append(rec)
    return records


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for method in dict.fromkeys(r["method"] for r in records):
        rows = [r for r in records if r["method"] == method]
        entry = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        entry["n_queries"] = len(rows)
        entry["by_type"] = {}
        for qtype in dict.fromkeys(r["type"] for r in rows):
            sub = [r for r in rows if r["type"] == qtype]
            entry["by_type"][qtype] = {m: float(np.mean([r[m] for r in sub])) for m in METRICS}
            entry["by_type"][qtype]["n_queries"] = len(sub)
        out[method] = entry
    return out


def paired(records: Sequence[Dict[str, Any]], a: str = "HAT-RAG") -> Dict[str, Dict[str, int]]:
    """Per-query recall of `a` against each other method: wins / ties / losses."""
    recall = {(r["method"], r["id"]): r["recall"] for r in records}
    ids = list(dict.fromkeys(r["id"] for r in records))
    out = {}
    for b in dict.fromkeys(r["method"] for r in records):
        if b == a:
            continue
        diff = [recall[(a, i)] - recall[(b, i)] for i in ids]
        out[b] = {"wins": sum(d > 1e-9 for d in diff), "ties": sum(abs(d) <= 1e-9 for d in diff),
                  "losses": sum(d < -1e-9 for d in diff)}
    return out


def ablate(index: TreeIndex, queries, qvecs, aliases, variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for name, cfg in variants.items():
        rec = evaluate(index, {name: HATBeamRetriever(index, **{**HAT_CONFIG, **cfg})},
                       queries, qvecs, aliases, repeats=3)
        s = summarize(rec)[name]
        out[name] = {"config": {k: v for k, v in cfg.items()}, **{m: s[m] for m in METRICS},
                     "by_type": s["by_type"]}
    return out


def print_table(title: str, summary: Dict[str, Dict[str, Any]]):
    print(f"\n{title}")
    print(f"  {'method':<22}{'recall':>8}{'hit':>7}{'compl':>7}{'MRR':>7}{'docCov':>8}"
          f"{'prec':>7}{'nodes':>8}{'ms':>8}")
    for m, s in summary.items():
        print(f"  {m:<22}{s['recall'] * 100:>7.1f}%{s['hit'] * 100:>6.1f}%{s['complete'] * 100:>6.1f}%"
              f"{s['mrr']:>7.3f}{s['doc_coverage'] * 100:>7.1f}%{s['precision'] * 100:>6.1f}%"
              f"{s['nodes_scored']:>8.1f}{s['latency_ms']:>8.3f}")


# ----------------------------------------------------------------------------
# Benchmark A: finance corpus
# ----------------------------------------------------------------------------
def run_finance(embedder: DenseEmbeddingModel) -> Dict[str, Any]:
    spec = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    queries, aliases = spec["queries"], spec["doc_aliases"]
    index = TreeIndex.from_json(DEFAULT_TREE, embedder=embedder)
    validate_gold(index, queries)
    qvecs = embedder.encode([q["query"] for q in queries], label="queries")

    retrievers = make_retrievers(index)
    records = evaluate(index, retrievers, queries, qvecs, aliases)
    summary = summarize(records)
    print_table(f"[A] FINANCE CORPUS  N={index.n_leaves} leaves, |V|={index.n_nodes}, "
                f"{len(queries)} queries, k={K}", summary)
    for qtype in ["single", "cross", "thematic"]:
        print_table(f"    -- {qtype}", {m: s["by_type"][qtype] for m, s in summary.items()})

    ablations = {
        "components": ablate(index, queries, qvecs, aliases, ABLATIONS),
        "beam": list(ablate(index, queries, qvecs, aliases,
                            {f"beta={b}": dict(beam_width=b) for b in range(1, 7)}).values()),
        "lambda": list(ablate(index, queries, qvecs, aliases,
                              {f"lambda={l}": dict(diversity=l)
                               for l in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]}).values()),
    }
    print("\n  HAT-RAG ablations (finance):")
    for name, a in ablations["components"].items():
        print(f"    {name:<30} recall {a['recall'] * 100:5.1f}% | coverage {a['doc_coverage'] * 100:5.1f}% "
              f"| precision {a['precision'] * 100:5.1f}% | nodes {a['nodes_scored']:.1f}")

    graph = retrievers["Graph PPR"]
    return {
        "config": {"k": K, "match_fraction": MATCH_FRACTION, "hat": HAT_CONFIG, "paper_spec": PAPER_SPEC,
                   "repeats": REPEATS, "tree": DEFAULT_TREE.name, "leaves": index.n_leaves,
                   "nodes": index.n_nodes, "depth": index.depth(), "alpha_edges": len(index.alpha_pairs),
                   "entities": len(index.entities.postings), "graph_edges": graph.n_edges,
                   "graph_build_seconds": graph.build_seconds, "encoder": embedder.stats.backend},
        "summary": summary,
        "paired": paired(records),
        "per_query": records,
        "ablations": ablations,
    }


# ----------------------------------------------------------------------------
# Benchmark B: synthetic scaling
# ----------------------------------------------------------------------------
def build_synthetic_index(n_docs: int, embedder: DenseEmbeddingModel):
    documents, queries, aliases = build_corpus(n_docs)
    chunker = DocumentChunker(chunk_size=60, chunk_overlap=10)
    chunks = chunker.process_corpus(documents)
    builder = HierarchicalAbstractTreeBuilder(
        embedder=embedder, max_levels=6, target_children=8, alpha="auto",
        summarizer_mode="extractive", device="cpu", verbose=False,
    )
    t0 = time.perf_counter()
    builder.build_tree(chunks)
    build_seconds = time.perf_counter() - t0
    return TreeIndex.from_builder(builder), queries, aliases, build_seconds


def run_scaling(embedder: DenseEmbeddingModel, sizes: Sequence[int], frontier_size: int) -> Dict[str, Any]:
    rows, frontier, components = [], {}, {}
    for n_docs in sizes:
        index, queries, aliases, build_s = build_synthetic_index(n_docs, embedder)
        validate_gold(index, queries)
        qvecs = embedder.encode([q["query"] for q in queries], label=f"synthetic_{n_docs}")
        retrievers = make_retrievers(index)
        records = evaluate(index, retrievers, queries, qvecs, aliases)
        summary = summarize(records)
        levels = [int((index.level == lv).sum()) for lv in range(index.depth() + 1)]
        rows.append({"docs": n_docs, "leaves": index.n_leaves, "nodes": index.n_nodes,
                     "depth": index.depth(), "levels": levels, "queries": len(queries),
                     "build_seconds": build_s, "graph_build_seconds": retrievers["Graph PPR"].build_seconds,
                     "summary": summary, "paired": paired(records), "per_query": records})
        print_table(f"[B] SYNTHETIC M={n_docs}: N={index.n_leaves}, |V|={index.n_nodes}, "
                    f"levels {levels}, {len(queries)} queries, tree built in {build_s:.1f}s", summary)

        if n_docs == frontier_size:
            for label, base in [("HAT-RAG", {}), (REFERENCE, PAPER_SPEC)]:
                sweep = ablate(index, queries, qvecs, aliases,
                               {f"beta={b}": dict(base, beam_width=b) for b in [1, 2, 3, 5, 8, 12, 16, 24, 32]})
                frontier[label] = [{"beam_width": b, **{m: v[m] for m in METRICS}}
                                   for b, v in zip([1, 2, 3, 5, 8, 12, 16, 24, 32], sweep.values())]
            components = ablate(index, queries, qvecs, aliases, ABLATIONS)
            print(f"  beam frontier at M={n_docs}: " + " | ".join(
                f"b={f['beam_width']}: {f['recall'] * 100:.1f}% @ {f['nodes_scored']:.0f}"
                for f in frontier["HAT-RAG"]))
            for name, a in components.items():
                print(f"    {name:<30} recall {a['recall'] * 100:5.1f}% | nodes {a['nodes_scored']:.1f}")
    return {"sizes": list(sizes), "rows": rows, "frontier_docs": frontier_size,
            "frontier": frontier, "components": components}


# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
def verdict(finance: Dict[str, Any], scaling: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision rule, fixed before the final run:
      score = 1/2 * finance recall@k + 1/2 * mean synthetic recall@k
      within one point of each other, the method that scores the smaller
      fraction of the corpus per query ranks higher.
    """
    table = {}
    for m in MAIN:
        fin = finance["summary"][m]["recall"]
        syn = [row["summary"][m]["recall"] for row in scaling["rows"]]
        costs = [finance["summary"][m]["nodes_scored"] / finance["config"]["leaves"]]
        costs += [row["summary"][m]["nodes_scored"] / row["leaves"] for row in scaling["rows"]]
        table[m] = {"finance_recall": fin, "synthetic_recall": float(np.mean(syn)) if syn else None,
                    "score": 0.5 * fin + 0.5 * (float(np.mean(syn)) if syn else fin),
                    "cost_fraction": float(np.mean(costs))}
    order = sorted(MAIN, key=lambda m: -table[m]["score"])
    changed = True
    while changed:                                   # apply the one-point tie rule
        changed = False
        for x in range(len(order) - 1):
            a, b = order[x], order[x + 1]
            if (table[a]["score"] - table[b]["score"] < 0.01
                    and table[b]["cost_fraction"] < table[a]["cost_fraction"]):
                order[x], order[x + 1] = b, a
                changed = True
    return {"rule": verdict.__doc__.strip(), "table": table, "ranking": order, "best": order[0]}


def _cli():
    p = argparse.ArgumentParser(description="Compare the four HAT-RAG retrieval algorithms")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128],
                   help="synthetic corpus sizes (documents) for the scaling study")
    p.add_argument("--frontier-size", type=int, default=64, help="corpus size for the beam frontier")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    RESULTS_DIR.mkdir(exist_ok=True)
    embedder = DenseEmbeddingModel(device="cpu", verbose=True)
    print("=" * 100)
    print(" RETRIEVAL ALGORITHM COMPARISON")
    print(f" encoder: {embedder.banner()}")
    print("=" * 100)

    finance = run_finance(embedder)
    (RESULTS_DIR / "finance.json").write_text(json.dumps(finance, indent=2), encoding="utf-8")
    scaling = run_scaling(embedder, args.sizes, args.frontier_size)
    (RESULTS_DIR / "scaling.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")

    result = verdict(finance, scaling)
    (RESULTS_DIR / "verdict.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n" + "=" * 100)
    print(" VERDICT  (1/2 finance recall + 1/2 mean synthetic recall; one-point ties -> cheaper wins)")
    for rank, m in enumerate(result["ranking"], 1):
        t = result["table"][m]
        print(f"  {rank}. {m:<18} score {t['score'] * 100:5.1f} | finance {t['finance_recall'] * 100:5.1f}% "
              f"| synthetic {t['synthetic_recall'] * 100:5.1f}% | scores {t['cost_fraction'] * 100:5.1f}% of corpus")
    print("=" * 100)
    print(f"[saved] results -> {RESULTS_DIR}")
