# HAT-RAG — Knowledge Construction and Retrieval

Standalone implementation of the Hierarchical Abstract Tree for Cross-Document RAG
(HAT-RAG). This directory builds the tree `H` from a document corpus (offline half), and
retrieves evidence from it with four interchangeable retrieval algorithms (online half),
compared head to head in [`compare-algos/`](./compare-algos/).

```
DOCUMENT CORPUS
      |
      +--> ALGORITHM 1: document chunking   -> chunk collection C
      |
      +--> dense embedding of the chunks    -> embedding matrix E
      |
      +--> ALGORITHM 3: HAT construction    -> tree H
                (k-means clustering -> cluster abstracts -> LLM summarisation)
                + explicit alpha cross-child relationships
      |
QUERY +--> RETRIEVAL 1 (Algorithm 4): HAT beam traversal + graph propagation -> evidence R
           (retrieval 2-4: flat dense, RAPTOR collapsed tree, graph PageRank -- baselines)
```

The dense encoder lives inside Algorithm 3 rather than in a module of its own. The
query-time index (`TreeIndex`) lives in retrieval 1 and is shared by all four
retrievers, so every method scores the same vectors.

---

## Does this need a GPU?

**No.** Everything here runs on plain CPU — that is how every number in this README
was produced (torch CPU build, no CUDA device). The full finance corpus builds in
about 45 seconds end to end, and roughly 1 second with `--summarizer extractive`.

CUDA is an optional accelerator, not a requirement:

| Stage | CPU today | What CUDA would change |
|---|---|---|
| Embedding | ~20-45 texts/sec | Larger batches + fp16; the win shows up at thousands of chunks |
| k-means | 3–9 ms per level | Already negligible at this size; the assignment step is one matmul |
| LLM summarisation | 3–10 s per abstract node | The only real bottleneck — a GPU cuts it to well under a second |

Every tensor operation is written against a `device` argument that already resolves
to `"cuda"` when a GPU is present (`resolve_device`, `SphericalKMeans._fit_torch`,
fp16 weights in `DenseEmbeddingModel`), so a CUDA port is a device switch and
kernel-level tuning, not a rewrite. Until then `--device cpu` is the working default.

---

## Files

| File | Role |
|---|---|
| [`algo1_document_chunking.py`](./algo1_document_chunking.py) | **Algorithm 1.** Sentence-boundary sliding-window chunking with overlap, heading/section detection and SHA-256 chunk fingerprints. |
| [`algo3_hierarchical_abstract_tree.py`](./algo3_hierarchical_abstract_tree.py) | **Algorithm 3.** Dense encoder, spherical k-means, abstractive LLM summarisation, recursive tree assembly, alpha cross-child edges and JSON persistence. |
| [`run_demo.py`](./run_demo.py) | Full pipeline on a small engineering-research corpus. |
| [`run_finance_rag.py`](./run_finance_rag.py) | Full pipeline on the SEC 10-K corpus in [`../finance_data/reports/`](../finance_data/reports/) (AAPL, MSFT, NVDA, AMZN, TSLA). |
| `hat_tree_structure.json` / `hat_finance_tree_structure.json` | Serialized trees (nodes, embeddings, cross-links, build metrics). |
| [`retrieval1_hat_beam_traversal.py`](./retrieval1_hat_beam_traversal.py) | **Retrieval 1 / Algorithm 4 (ours, best of the four).** Beam descent of `H`, entity linking, expansion along alpha and reading-order edges, personalised PageRank over the candidate subgraph, relevance-gated diversity. Also holds the shared `TreeIndex`. |
| [`retrieval2_flat_dense.py`](./retrieval2_flat_dense.py) | **Retrieval 2.** Flat dense baseline: cosine against every leaf. |
| [`retrieval3_raptor_collapsed_tree.py`](./retrieval3_raptor_collapsed_tree.py) | **Retrieval 3.** RAPTOR-style collapsed tree: every level pooled, cosine top-k. |
| [`retrieval4_graph_ppr.py`](./retrieval4_graph_ppr.py) | **Retrieval 4.** Entity-graph baseline: personalised PageRank over a whole-corpus passage graph. |
| [`compare-algos/`](./compare-algos/) | Benchmark queries, evaluation harness, results, figures and the comparison report. |

---

## Running

```bash
python algorithms/run_finance_rag.py                           # full pipeline, LLM abstracts
python algorithms/run_finance_rag.py --summarizer extractive   # ~1s, no LLM
python algorithms/run_demo.py                                  # engineering corpus
python algorithms/algo1_document_chunking.py                   # Algorithm 1 alone
python algorithms/algo3_hierarchical_abstract_tree.py          # Algorithm 3 alone

python algorithms/retrieval1_hat_beam_traversal.py --context   # retrieve from the finance tree
python algorithms/retrieval1_hat_beam_traversal.py --query "How does Tesla set its warranty reserves?"
python algorithms/retrieval2_flat_dense.py                     # each baseline the same way
python algorithms/compare-algos/run_comparison.py              # full comparison -> results/
python algorithms/compare-algos/make_figures.py                # -> figures/, results/tables.md
```

Useful flags (both runners): `--levels`, `--children`, `--alpha`, `--max-cross-links`,
`--summarizer {auto,llm,extractive}`, `--model`, `--device`, `--chunk-size`, `--chunk-overlap`.
Retrieval 1: `--beam`, `--lam`, `-k`, `--paper-spec` (the original Algorithm 4), `--context`.

---

## Dense encoder (inside Algorithm 3)

Replaces the earlier random-projection placeholder, so cosine similarity carries actual
semantics — Apple's and Microsoft's balance-sheet passages score 0.62 against each other,
and the strongest cross-company link in the built tree reaches 0.69.

- `sentence-transformers/all-MiniLM-L6-v2`, d = 384, L2-normalised output, so every
  downstream dot product *is* cosine similarity.
- Batched, with an embedding cache — the tree builder re-encodes the leaf set for free.
- `device="auto"` picks CUDA when present (4× batch size, fp16 weights), CPU otherwise.
- If the transformer stack or the weights are missing, it degrades to a deterministic
  feature-hashing encoder (word uni/bigrams + char 4-grams) instead of failing.

## Algorithm 3 — HAT construction

| Stage | Implementation |
|---|---|
| Clustering | Spherical **k-means** with k-means++ seeding; one `(n×d)@(d×k)` matmul per iteration via torch, empty clusters re-seeded from the worst-fitting point. Falls back to numpy when torch is absent. |
| Cluster count | Adaptive: `k = ceil(n / target_children)` per level, instead of a fixed branching factor. |
| Summarisation | Abstractive **LLM** (`facebook/bart-large-cnn`) writes each abstract node; falls back to centroid-based extractive sentence selection when the model is unavailable or `--summarizer extractive` is passed. |
| Parent embeddings | The generated abstract is embedded by the same encoder — parents live in the same vector space as leaves. |
| Singleton collapse | A one-member cluster promotes its child to the next level instead of creating a chain node that only paraphrases itself. |
| Root | Levels are built until one node remains, or a global root is synthesised over whatever is left at `max_levels`. |
| **Alpha cross-child edges** | The dashed edges of the HAT: per level, each node keeps up to `max_cross_links` peers with cosine ≥ alpha that sit under a **different parent**, tagged `cross_document` or `cross_cluster`. `--alpha auto` sets the threshold at mean + 1σ of that level's similarity distribution (clamped to [0.35, 0.90]). |

Every build reports per-level cohesion, k-means iterations, cross-document cluster
counts, phase timings and compression ratio; `save_tree_json` persists the config,
encoder stats, metrics, node embeddings and cross-links.

### Typical finance-corpus build (CPU)

```
24 leaf chunks (5 companies) -> 5 L1 abstracts -> 2 L2 abstracts -> 1 root   (24x compression)
29 explicit alpha edges, 17 of them between different companies
embed 0.33s | cluster 0.06s | summarize 43.8s (LLM) | crosslink 0.01s | total 44.2s
```

---

## Retrieval — four algorithms compared

| Rank | Retriever | Finance recall@5 | Synthetic recall@5 (8–128 docs) | Share of corpus scored |
|---:|---|---:|---:|---:|
| 1 | HAT-RAG (retrieval 1) | 85.0% | 96.7% | 39% |
| 2 | Graph PPR (retrieval 4) | 83.4% | 93.8% | 100% |
| 3 | Flat dense (retrieval 2) | 83.9% | 88.6% | 100% |
| 4 | RAPTOR collapsed (retrieval 3) | 74.6% | 85.1% | 118% |

Method, figures, ablations and limitations: [`compare-algos/README.md`](./compare-algos/README.md).

## Scope

Implemented: corpus → chunks → embeddings → hierarchical abstract tree, persisted to JSON;
query → evidence set with provenance (`assemble_context` builds the generator context).
Not implemented yet: answer generation from the retrieved context.
