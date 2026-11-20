"""
Choose HAT-RAG's large-corpus settings on the MultiHop-RAG DEV split only.

The paper's beta = 3 and an uncapped entity link were set on a 32-node tree.
At 22,838 passages both choices need revisiting, but a setting picked by
looking at the test questions would inflate the test score. This script
evaluates candidate settings on the 466 dev questions (every fifth question;
never part of a reported test number), writes every variant to
results/multihop_dev_tuning.json, and prints the pick. run_multihop.py then
reports the picked setting on the 1,789 test questions.

Rule (fixed before running): highest dev recall@5; within 0.5 points, fewer
nodes scored wins.

    python algorithms/compare-algos/tune_multihop_dev.py
"""

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from algo3_hierarchical_abstract_tree import DenseEmbeddingModel
from multihop_corpus import load_articles, load_queries
from retrieval1_hat_beam_traversal import HATBeamRetriever, TreeIndex
from retrieval2_flat_dense import FlatDenseRetriever
from retrieval4_graph_ppr import GraphPPRRetriever
from run_comparison import HAT_CONFIG, RESULTS_DIR, evaluate, summarize
from run_multihop import FULL_TREE, answerable

GRID = {"beam_width": [3, 5, 8, 12], "max_link": [None, 50, 200], "n_seeds": [3, 6]}


def main():
    embedder = DenseEmbeddingModel(device="cpu", verbose=False)
    articles = load_articles()
    queries, aliases = load_queries(articles)
    index = TreeIndex.from_json(FULL_TREE, embedder=embedder)
    dev = [q for q in answerable(index, queries) if q["split"] == "dev"]
    qvecs = embedder.encode([q["query"] for q in dev], label="dev")
    print(f"[dev] {len(dev)} questions, {index.n_leaves:,} passages")

    rows = []
    for name, r in [("Flat dense", FlatDenseRetriever(index)), ("Graph PPR", GraphPPRRetriever(index))]:
        s = summarize(evaluate(index, {name: r}, dev, qvecs, aliases, repeats=1, k=5))[name]
        rows.append({"variant": name, "recall": s["recall"], "nodes_scored": s["nodes_scored"],
                     "latency_ms": s["latency_ms"]})
        print(f"  {name:<44} recall {s['recall'] * 100:5.1f}%  nodes {s['nodes_scored']:7.0f}  {s['latency_ms']:6.1f} ms")

    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, combo))
        label = "HAT-RAG " + " ".join(f"{k}={v}" for k, v in cfg.items())
        r = HATBeamRetriever(index, **{**HAT_CONFIG, **cfg})
        s = summarize(evaluate(index, {label: r}, dev, qvecs, aliases, repeats=1, k=5))[label]
        rows.append({"variant": label, "config": cfg, "recall": s["recall"], "nodes_scored": s["nodes_scored"],
                     "latency_ms": s["latency_ms"]})
        print(f"  {label:<44} recall {s['recall'] * 100:5.1f}%  nodes {s['nodes_scored']:7.0f}  {s['latency_ms']:6.1f} ms")

    hat = [r for r in rows if "config" in r]
    best = max(r["recall"] for r in hat)
    pick = min((r for r in hat if best - r["recall"] <= 0.005), key=lambda r: r["nodes_scored"])
    out = {"split": "dev", "questions": len(dev), "rule": __doc__.split("Rule")[1].split("\n\n")[0].strip(),
           "grid": {k: [str(v) for v in vs] for k, vs in GRID.items()}, "rows": rows, "pick": pick}
    (RESULTS_DIR / "multihop_dev_tuning.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[pick] {pick['variant']}  dev recall {pick['recall'] * 100:.1f}% at {pick['nodes_scored']:.0f} nodes")


if __name__ == "__main__":
    main()
