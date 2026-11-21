"""
CPU vs. CUDA timing for the pipeline and the four retrievers.

Runs the same synthetic scaling benchmark as run_comparison.py, but with the
encoder, the spherical k-means and the TreeIndex scoring kernel all placed on
one device. Run it once per device with the same PyTorch build:

    python run_device_benchmark.py --device cuda
    python run_device_benchmark.py --device cpu

Each run writes results/device_<device>.json. Recall is recorded too, so the
two runs can be checked for agreement (fp16 encoding on CUDA can move a few
cosines by ~1e-3).
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch

from run_comparison import (HERE, K, REPEATS, build_corpus, DocumentChunker, DenseEmbeddingModel,
                            HierarchicalAbstractTreeBuilder, TreeIndex, evaluate, make_retrievers,
                            summarize, validate_gold)
from algo3_hierarchical_abstract_tree import describe_device


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_size(n_docs, embedder, device):
    documents, queries, aliases = build_corpus(n_docs)
    chunks = DocumentChunker(chunk_size=60, chunk_overlap=10).process_corpus(documents)
    embedder._cache.clear()
    builder = HierarchicalAbstractTreeBuilder(
        embedder=embedder, max_levels=6, target_children=8, alpha="auto",
        summarizer_mode="extractive", device=device, verbose=False,
    )
    sync(device); t0 = time.perf_counter()
    builder.build_tree(chunks)
    sync(device); build_s = time.perf_counter() - t0

    index = TreeIndex.from_builder(builder, device=device)
    validate_gold(index, queries)
    qvecs = embedder.encode([q["query"] for q in queries], label=f"bench_{n_docs}")
    summary = summarize(evaluate(index, make_retrievers(index, reference=False), queries, qvecs, aliases))
    m = builder.metrics
    return {
        "docs": n_docs, "leaves": index.n_leaves, "nodes": index.n_nodes, "queries": len(queries),
        "build_seconds": round(build_s, 3),
        "embed_seconds": round(m.embed_seconds, 3), "cluster_seconds": round(m.cluster_seconds, 3),
        "summarize_seconds": round(m.summarize_seconds, 3), "crosslink_seconds": round(m.crosslink_seconds, 3),
        "methods": {name: {"recall": s["recall"], "nodes_scored": s["nodes_scored"],
                           "latency_ms": s["latency_ms"]} for name, s in summary.items()},
    }


def run_kernel(device, dim=384, counts=(1_000, 10_000, 100_000, 500_000), reps=20):
    """The TreeIndex.score kernel: one mat-vec against a device-resident matrix."""
    rng = np.random.default_rng(7)
    rows = []
    for n in counts:
        E = torch.as_tensor(rng.standard_normal((n, dim), dtype=np.float32), device=device)
        E /= E.norm(dim=-1, keepdim=True)
        q = torch.as_tensor(rng.standard_normal(dim, dtype=np.float32), device=device)
        for _ in range(3):
            (E @ q).cpu()
        ts = []
        for _ in range(reps):
            sync(device); t0 = time.perf_counter()
            (E @ q).cpu()
            sync(device); ts.append((time.perf_counter() - t0) * 1000)
        ms = statistics.median(ts)
        rows.append({"n_vectors": n, "ms": round(ms, 4), "Mvec_per_s": round(n / ms / 1000, 2)})
        print(f"  [kernel] n={n:>9,}: {ms:8.3f} ms", flush=True)
        del E
    return rows


def run_encoder(embedder, device, n_docs=128):
    documents, _, _ = build_corpus(n_docs)
    texts = [c["text"] for c in DocumentChunker(chunk_size=60, chunk_overlap=10).process_corpus(documents)]
    embedder._cache.clear()
    embedder.encode(texts[:256], label="warmup")
    embedder._cache.clear()
    sync(device); t0 = time.perf_counter()
    embedder.encode(texts, label="encoder_bench")
    sync(device); s = time.perf_counter() - t0
    return {"chunks": len(texts), "seconds": round(s, 3), "chunks_per_s": round(len(texts) / s, 1),
            "batch_size": embedder.batch_size, "fp16": embedder.use_fp16}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", help="cpu | cuda")
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = p.parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")

    embedder = DenseEmbeddingModel(device=device, verbose=False)
    out = {"device": describe_device(device), "torch": torch.__version__,
           "cuda": torch.version.cuda, "k": K, "repeats": REPEATS}
    print(f"[env] {out}", flush=True)
    out["encoder"] = run_encoder(embedder, device)
    print(f"  [encoder] {out['encoder']}", flush=True)
    out["kernel"] = run_kernel(device)
    out["scaling"] = []
    for n in args.sizes:
        row = run_size(n, embedder, device)
        out["scaling"].append(row)
        print(f"  [M={n}] N={row['leaves']} build={row['build_seconds']}s "
              f"(embed {row['embed_seconds']}s, cluster {row['cluster_seconds']}s) | " +
              " ".join(f"{k}: {v['recall']:.3f}/{v['latency_ms']:.2f}ms" for k, v in row["methods"].items()),
              flush=True)
    path = HERE / "results" / f"device_{device.split(':')[0]}.json"
    path.write_text(json.dumps(out, indent=2))
    print("written", path)


if __name__ == "__main__":
    main()
