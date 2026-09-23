# HAT-RAG — Offline / Knowledge Construction

Standalone implementation of the **indexing half** of the Hierarchical Abstract Tree
for Cross-Document RAG (HAT-RAG). This directory builds the tree `H` from a document
corpus. **Query-time retrieval is deliberately not implemented here.**

```
DOCUMENT CORPUS
      |
      +--> ALGORITHM 1: document chunking              -> chunk collection C
      |
      +--> ALGORITHM 2: GPU-accelerated dense embedding -> embedding matrix E
      |
      +--> ALGORITHM 3: HAT construction                -> tree H
                (GPU k-means -> cluster abstracts -> LLM summarisation)
                + explicit alpha cross-child relationships
```

---

## Files

| File | Role |
|---|---|
| [`algo1_document_chunking.py`](./algo1_document_chunking.py) | **Algorithm 1.** Sentence-boundary sliding-window chunking with overlap, heading/section detection and SHA-256 chunk fingerprints. |
| [`algo2_gpu_dense_embedding.py`](./algo2_gpu_dense_embedding.py) | **Algorithm 2.** Batched transformer encoder (`all-MiniLM-L6-v2`, d=384) with CUDA auto-detection, fp16 on GPU, embedding cache and throughput stats. |
| [`algo3_hierarchical_abstract_tree.py`](./algo3_hierarchical_abstract_tree.py) | **Algorithm 3.** Spherical GPU k-means clustering, abstractive LLM summarisation, recursive tree assembly, alpha cross-child edges and JSON persistence. |
| [`run_demo.py`](./run_demo.py) | Full 1 → 2 → 3 pipeline on a small engineering-research corpus. |
| [`run_finance_rag.py`](./run_finance_rag.py) | Full pipeline on the SEC 10-K corpus in [`../finance_data/reports/`](../finance_data/reports/) (AAPL, MSFT, NVDA, AMZN, TSLA). |
| `hat_tree_structure.json` / `hat_finance_tree_structure.json` | Serialized trees (nodes, embeddings, cross-links, build metrics). |

---

## Running

```bash
python algorithms/run_finance_rag.py                      # full pipeline, LLM abstracts
python algorithms/run_finance_rag.py --summarizer extractive   # ~1s, no LLM
python algorithms/run_finance_rag.py --device cuda --alpha 0.65
python algorithms/run_demo.py                             # engineering corpus
python algorithms/algo2_gpu_dense_embedding.py            # Algorithm 2 alone
python algorithms/algo3_hierarchical_abstract_tree.py     # Algorithm 3 alone
```

Useful flags (both runners): `--levels`, `--children`, `--alpha`, `--max-cross-links`,
`--summarizer {auto,llm,extractive}`, `--model`, `--device`, `--chunk-size`, `--chunk-overlap`.

---

## Algorithm 2 — GPU-accelerated dense embedding

Replaces the earlier random-projection placeholder encoder with a real sentence
transformer, so cosine similarity carries actual semantics (Apple's and NVIDIA's
balance-sheet passages score 0.70 against each other).

- `device="auto"` → CUDA when available, otherwise CPU; batch size is raised 4× and
  weights switch to fp16 on GPU.
- Outputs are L2-normalised, so every downstream dot product *is* cosine similarity.
- Repeated texts are served from an embedding cache (the tree builder re-encodes the
  leaf set for free).
- If the transformer stack or the weights are missing, the module degrades to a
  deterministic feature-hashing encoder (word uni/bigrams + char 4-grams) rather than
  failing — shared vocabulary still produces shared dimensions.

## Algorithm 3 — HAT construction

| Stage | Implementation |
|---|---|
| Clustering | Spherical **k-means** with k-means++ seeding, run as a single `(n×d)@(d×k)` matmul per iteration on torch (CUDA when present). Empty clusters are re-seeded from the worst-fitting point. |
| Cluster count | Adaptive: `k = ceil(n / target_children)` per level, instead of a fixed branching factor. |
| Summarisation | Abstractive **LLM** (`facebook/bart-large-cnn`) writes each abstract node; falls back to centroid-based extractive sentence selection when the model is unavailable or `--summarizer extractive` is passed. |
| Parent embeddings | The generated abstract is embedded by Algorithm 2 — parents live in the same vector space as leaves. |
| Singleton collapse | A one-member cluster promotes its child to the next level instead of creating a chain node that only paraphrases itself. |
| Root | Levels are built until one node remains, or a global root is synthesised over whatever remains at `max_levels`. |
| **Alpha cross-child edges** | The dashed edges of the HAT: per level, each node keeps up to `max_cross_links` peers with cosine ≥ alpha that sit under a **different parent**, tagged `cross_document` or `cross_cluster`. `--alpha auto` sets the threshold at mean + 1σ of that level's similarity distribution (clamped to [0.35, 0.90]). |

Every build reports per-level cohesion, k-means iterations, cross-document cluster
counts, phase timings and compression ratio; `save_tree_json` persists the config,
embedding-model stats, metrics, node embeddings and cross-links.

### Typical finance-corpus build

```
24 leaf chunks (5 companies) -> 5 L1 abstracts -> 2 L2 abstracts -> 1 root   (24x compression)
29 explicit alpha edges, 17 of them between different companies
embed 0.22s | cluster 0.01s | summarize 32.3s (LLM, CPU) | crosslink 0.00s
```

---

## Scope

Implemented: corpus → chunks → embeddings → hierarchical abstract tree, persisted to JSON.
Not implemented (by design): query embedding, tree traversal, retrieval, reranking and
answer generation.
