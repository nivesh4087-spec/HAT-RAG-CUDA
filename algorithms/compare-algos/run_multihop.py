"""
BENCHMARK C: MULTIHOP-RAG AT FULL SCALE (real articles, ~22,800 passages)

The finance corpus (24 chunks) is too small for cost to matter and the
synthetic corpus is generated. This benchmark answers the question an
enterprise deployment actually asks: on a large, real, multi-source document
collection, which retriever finds the evidence -- and at what cost?

    C1  full corpus      609 articles / ~22.8k leaves, 2,255 multi-document
                         questions, all four methods + the paper-spec reference,
                         recall@5 and recall@10. Reported on the held-out test
                         split (1,789 questions); the dev split (466) is the only
                         part that may inform design decisions.
    C2  growing haystack the same test questions, corpus grown from ~100 to all
                         609 articles by adding real distractor articles.
    C3  HAT-RAG ablations and beam-width sweep on the full corpus (test split).

Trees are cached in data/multihop_rag/trees/ (embeddings in .emb.npy
sidecars), so a re-run only re-scores. Results go to results/multihop.json.

    python algorithms/compare-algos/run_multihop.py                 # everything
    python algorithms/compare-algos/run_multihop.py --build-only    # just the full tree
    python algorithms/compare-algos/run_multihop.py --skip-haystack
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np

from algo3_hierarchical_abstract_tree import DenseEmbeddingModel, HierarchicalAbstractTreeBuilder
from retrieval1_hat_beam_traversal import HATBeamRetriever, TreeIndex
from multihop_corpus import TREE_DIR, article_meta, build_chunks, load_articles, load_queries
from run_comparison import (ABLATIONS, HAT_CONFIG, MAIN, METRICS, REFERENCE, RESULTS_DIR, covers,
                            evaluate, make_retrievers, paired, print_table, summarize)

BUILD = dict(max_levels=6, target_children=8, alpha="auto", summarizer_mode="extractive")
FULL_TREE = TREE_DIR / "hat_multihop_full.json"
EMBED_CACHE = TREE_DIR / "embedding_cache.npz"
K_LIST = (5, 10)
REPEATS = 3
HAYSTACK_SIZES = (100, 200, 400, 609)
TUNING = RESULTS_DIR / "multihop_dev_tuning.json"


def large_corpus_config() -> Dict[str, Any]:
    """HAT-RAG settings picked on the dev split by tune_multihop_dev.py (paper defaults if not run)."""
    if TUNING.exists():
        cfg = dict(json.loads(TUNING.read_text(encoding="utf-8"))["pick"]["config"])
        return {k: v for k, v in cfg.items()}
    return {}


HAT_LARGE = large_corpus_config()
# the latency-oriented setting from the same dev grid: broad entities capped at 200 postings
CAPPED = dict(beam_width=3, max_link=200, n_seeds=3)
SEED = 17


# ----------------------------------------------------------------------------
# Index construction (cached)
# ----------------------------------------------------------------------------
def build_index(articles, doc_ids: Sequence[str], embedder: DenseEmbeddingModel,
                cache: Path) -> Tuple[TreeIndex, Dict[str, Any]]:
    if cache.exists():
        t0 = time.perf_counter()
        index = TreeIndex.from_json(cache, embedder=embedder)
        meta = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
        print(f"[cache] {cache.name}: {index.n_nodes} nodes loaded in {time.perf_counter() - t0:.1f}s")
        return index, meta

    chunks = build_chunks(articles, list(doc_ids))
    print(f"[build] {len(doc_ids)} articles -> {len(chunks)} chunks; building the HAT ...")
    builder = HierarchicalAbstractTreeBuilder(embedder=embedder, device="cpu", verbose=False, **BUILD)
    t0 = time.perf_counter()
    builder.build_tree(chunks)
    seconds = time.perf_counter() - t0
    cache.parent.mkdir(parents=True, exist_ok=True)
    builder.save_tree_json(str(cache), sidecar=True)
    embedder.save_cache(EMBED_CACHE)
    m = builder.metrics
    meta = {"docs": len(doc_ids), "chunks": len(chunks), "nodes": len(builder.nodes),
            "build_seconds": seconds, "embed_seconds": m.embed_seconds, "cluster_seconds": m.cluster_seconds,
            "summarize_seconds": m.summarize_seconds, "crosslink_seconds": m.crosslink_seconds,
            "levels": builder.level_counts(), "cross_links": m.cross_links}
    cache.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[build] {len(builder.nodes)} nodes in {seconds:.1f}s "
          f"(embed {m.embed_seconds:.0f}s, cluster {m.cluster_seconds:.0f}s, "
          f"summaries {m.summarize_seconds:.0f}s, links {m.crosslink_seconds:.0f}s)")
    return TreeIndex.from_builder(builder), meta


def seed_cache(embedder: DenseEmbeddingModel, index: TreeIndex) -> None:
    """Leaf texts are encoded verbatim (no section prefix), so a built tree's
    leaf vectors can stand in for re-encoding the same chunks at other sizes."""
    import hashlib
    for i in index.leaf_idx:
        key = hashlib.blake2b(index.nodes[int(i)].text.encode("utf-8"), digest_size=16).hexdigest()
        embedder._cache.setdefault(key, index.E[int(i)])


def answerable(index: TreeIndex, queries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Questions whose every evidence item is held by some leaf of this index."""
    by_doc: Dict[str, List[int]] = {}
    for i in index.leaf_idx:
        by_doc.setdefault(index.doc[int(i)], []).append(int(i))
    return [q for q in queries
            if all(any(covers(index, i, g) for i in by_doc.get(g["doc"], [])) for g in q["gold"])]


def strip(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keep = ("method", "id", "type", "recall", "hit", "complete", "mrr", "nodes_scored", "latency_ms", "retrieved")
    return [{k: r[k] for k in keep} for r in records]


def by_split(records, queries) -> Dict[str, Dict[str, Any]]:
    split = {q["id"]: q["split"] for q in queries}
    return {s: summarize([r for r in records if split[r["id"]] == s]) for s in ("test", "dev")}


# ----------------------------------------------------------------------------
# C1: full corpus
# ----------------------------------------------------------------------------
def run_full(articles, queries, aliases, embedder, checkpoint=None) -> Dict[str, Any]:
    meta_docs = list(article_meta(articles))
    index, build = build_index(articles, meta_docs, embedder, FULL_TREE)
    seed_cache(embedder, index)
    qs = answerable(index, queries)
    print(f"[C1] {index.n_leaves} leaves, {index.n_nodes} nodes, depth {index.depth()}, "
          f"{len(qs)}/{len(queries)} questions answerable from the chunks")
    qvecs = embedder.encode([q["query"] for q in qs], label="multihop_queries")
    t0 = time.perf_counter()
    retrievers = make_retrievers(index, hat=HAT_LARGE)
    setup = time.perf_counter() - t0
    out: Dict[str, Any] = {"index": {"leaves": index.n_leaves, "nodes": index.n_nodes, "depth": index.depth(),
                                     "levels": [int((index.level == lv).sum()) for lv in range(index.depth() + 1)],
                                     "alpha_edges": len(index.alpha_pairs),
                                     "entities": len(index.entities.postings),
                                     "graph_edges": retrievers["Graph PPR"].n_edges,
                                     "graph_build_seconds": retrievers["Graph PPR"].build_seconds,
                                     "retriever_setup_seconds": setup, **build},
                           "queries": len(qs), "by_k": {}}
    test = [q for q in qs if q["split"] == "test"]
    for k in K_LIST:
        recs = evaluate(index, retrievers, qs, qvecs, aliases, repeats=REPEATS if k == K_LIST[0] else 1, k=k)
        splits = by_split(recs, qs)
        test_recs = [r for r in recs if r["id"] in {q["id"] for q in test}]
        out["by_k"][str(k)] = {"summary": splits["test"], "dev_summary": splits["dev"],
                               "paired": paired(test_recs), "per_query": strip(test_recs)}
        print_table(f"[C1] MULTIHOP-RAG full corpus, test split ({len(test)} q), k={k}", splits["test"])
        for qtype in ["inference", "comparison", "temporal"]:
            print_table(f"    -- {qtype}", {m: s["by_type"][qtype] for m, s in splits["test"].items()})

    if checkpoint:
        checkpoint(out)
    # C3: ablations + beam sweep, test split, k = 5
    tvecs = np.stack([qvecs[i] for i, q in enumerate(qs) if q["split"] == "test"])
    def variant(cfg, label):
        rec = evaluate(index, {label: HATBeamRetriever(index, **{**HAT_CONFIG, **HAT_LARGE, **cfg})},
                       test, tvecs, aliases, repeats=1, k=5)
        s = summarize(rec)[label]
        return {m: s[m] for m in METRICS}
    variants = dict(ABLATIONS)
    variants["capped links (beta=3, max_link=200)"] = CAPPED
    variants["paper spec"] = {**ABLATIONS["paper spec"], "beam_width": 3, "n_seeds": 3}
    out["ablations"] = {name: variant(cfg, name) for name, cfg in variants.items()}
    print("\n  [C3] HAT-RAG ablations (MultiHop-RAG, test, k=5)")
    for name, a in out["ablations"].items():
        print(f"    {name:<30} recall {a['recall'] * 100:5.1f}% | nodes {a['nodes_scored']:.0f}")
    out["beam"] = [{"beam_width": b, **variant(dict(beam_width=b), f"beta={b}")} for b in (1, 2, 3, 5, 8, 16)]
    print("  [C3] beam: " + " | ".join(f"b={r['beam_width']}: {r['recall'] * 100:.1f}% @ {r['nodes_scored']:.0f}"
                                       for r in out["beam"]))
    return out


# ----------------------------------------------------------------------------
# C2: growing haystack
# ----------------------------------------------------------------------------
def haystack_plan(articles, queries, sizes) -> Tuple[List[str], List[Dict[str, Any]]]:
    """A document order whose first sizes[0] articles hold every evidence
    article of a fixed question set; later articles are real distractors."""
    rng = random.Random(SEED)
    docs = list(article_meta(articles))
    test = [q for q in queries if q["split"] == "test"]
    rng.shuffle(test)
    core: List[str] = []
    for q in test:
        need = [g["doc"] for g in q["gold"] if g["doc"] not in core]
        if len(core) + len(need) <= sizes[0]:
            core += list(dict.fromkeys(need))
    rest = [d for d in docs if d not in core]
    rng.shuffle(rest)
    order = core + rest
    fixed = [q for q in test if all(g["doc"] in set(core) for g in q["gold"])]
    return order, fixed


def run_haystack(articles, queries, aliases, embedder, sizes) -> Dict[str, Any]:
    order, fixed = haystack_plan(articles, queries, sizes)
    print(f"\n[C2] growing haystack: {len(fixed)} fixed test questions, sizes {list(sizes)} articles")
    qvecs = embedder.encode([q["query"] for q in fixed], label="haystack_queries")
    rows = []
    for size in sizes:
        cache = FULL_TREE if size >= len(order) else TREE_DIR / f"hat_multihop_{size}.json"
        index, build = build_index(articles, order[:size], embedder, cache)
        qs = answerable(index, fixed)
        assert len(qs) == len(fixed), "haystack questions must stay answerable at every size"
        retrievers = make_retrievers(index, hat=HAT_LARGE)
        retrievers["HAT-RAG (capped links)"] = HATBeamRetriever(index, **{**HAT_CONFIG, **CAPPED})
        recs = evaluate(index, retrievers, fixed, qvecs, aliases, repeats=REPEATS, k=5)
        summary = summarize(recs)
        rows.append({"docs": size, "leaves": index.n_leaves, "nodes": index.n_nodes, "depth": index.depth(),
                     "build_seconds": build["build_seconds"], "summary": summary, "paired": paired(recs)})
        print_table(f"[C2] {size} articles: N={index.n_leaves}, |V|={index.n_nodes}", summary)
    return {"sizes": list(sizes), "queries": len(fixed), "rows": rows}


def _cli():
    p = argparse.ArgumentParser(description="MultiHop-RAG benchmark for the four retrievers")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--skip-haystack", action="store_true")
    p.add_argument("--sizes", type=int, nargs="+", default=list(HAYSTACK_SIZES))
    return p.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _cli()
    embedder = DenseEmbeddingModel(device="cpu", verbose=True)
    print(f"[cache] {embedder.load_cache(EMBED_CACHE)} cached embeddings")
    articles = load_articles()
    queries, aliases = load_queries(articles)
    if args.build_only:
        build_index(articles, list(article_meta(articles)), embedder, FULL_TREE)
        sys.exit(0)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"[config] HAT-RAG large-corpus setting (dev-picked): {HAT_LARGE or 'paper defaults'}")
    result = {"dataset": "MultiHop-RAG (Tang & Yang, COLM 2024)", "k": list(K_LIST),
              "hat": {**HAT_CONFIG, **HAT_LARGE}, "capped": CAPPED,
              "methods": MAIN + [REFERENCE]}
    path = RESULTS_DIR / "multihop.json"

    def checkpoint(full):
        result["full"] = full
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["full"] = run_full(articles, queries, aliases, embedder, checkpoint)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    if not args.skip_haystack:
        result["haystack"] = run_haystack(articles, queries, aliases, embedder, args.sizes)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[saved] {path}")
