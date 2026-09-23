# HAT-RAG — Offline / Knowledge Construction

Standalone implementation of the **indexing half** of the Hierarchical Abstract Tree
for Cross-Document RAG (HAT-RAG). This directory builds the tree `H` from a document
corpus. **Query-time retrieval is deliberately not implemented here.**

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
```

Two algorithm files only: chunking and tree construction. The dense encoder lives
inside Algorithm 3 rather than in a module of its own.

---

## Does this need a GPU?

**No.** Everything here runs on plain CPU — that is how every number in this README
was produced (torch CPU build, no CUDA device). The full finance corpus builds in
about 33 seconds end to end, and roughly 1 second with `--summarizer extractive`.

CUDA is an optional accelerator, not a requirement:

| Stage | CPU today | What CUDA would change |
|---|---|---|
| Embedding | ~36 texts/sec | Larger batches + fp16; the win shows up at thousands of chunks |
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

---

## Running

```bash
python algorithms/run_finance_rag.py                           # full pipeline, LLM abstracts
python algorithms/run_finance_rag.py --summarizer extractive   # ~1s, no LLM
python algorithms/run_demo.py                                  # engineering corpus
python algorithms/algo1_document_chunking.py                   # Algorithm 1 alone
python algorithms/algo3_hierarchical_abstract_tree.py          # Algorithm 3 alone
```

Useful flags (both runners): `--levels`, `--children`, `--alpha`, `--max-cross-links`,
`--summarizer {auto,llm,extractive}`, `--model`, `--device`, `--chunk-size`, `--chunk-overlap`.

---

## Dense encoder (inside Algorithm 3)

Replaces the earlier random-projection placeholder, so cosine similarity carries actual
semantics — Apple's and NVIDIA's balance-sheet passages score 0.70 against each other.

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
embed 0.22s | cluster 0.01s | summarize 32.3s (LLM) | crosslink 0.00s
```

---

## Scope

Implemented: corpus → chunks → embeddings → hierarchical abstract tree, persisted to JSON.
Not implemented (by design): query embedding, tree traversal, retrieval, reranking and
answer generation.
