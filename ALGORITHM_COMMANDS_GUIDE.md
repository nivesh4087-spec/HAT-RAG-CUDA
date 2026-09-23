# HAT-RAG Algorithms Execution Guide: Commands, Outputs & Architecture

This guide explains how to run the standalone offline indexing pipeline located in the [`algorithms/`](./algorithms/) directory, details what each command does, explains every part of the terminal output, and breaks down the architectural rationale behind this design.

---

## 1. Why Is the Pipeline Designed to Run Like This?

### A. Decoupling Offline Knowledge Construction from Online Query Retrieval
In standard RAG systems, ingestion and retrieval are often tangled together. HAT-RAG separates them strictly into two halves:
1. **Offline Knowledge Construction (Indexing)**: Consumes raw text documents, executes semantic chunking (Algorithm 1), embeds passages into dense vector space, performs hierarchical spherical k-means clustering, synthesizes multi-tier abstract nodes (Algorithm 3), establishes cross-document lateral $\alpha$-links, and serializes the complete topology into JSON.
2. **Online Query Retrieval**: Loads the pre-computed tree $H$ and traverses it from root to leaf in logarithmic time $\mathcal{O}(k \log N)$ during user queries.

**Why this separation is essential:**
- **Inference Speed**: Generating abstractive LLM summaries and calculating k-means clusters takes seconds to minutes. At query time, a user expects responses in **sub-second latency** (< 100 ms). Query traversal cannot wait for a tree to be built on the fly; the tree must already exist in persistent storage.
- **Resource Efficiency**: Indexing is computationally heavy (dense embedding, clustering, summarization), whereas retrieval requires only top-down vector dot-products. Separating them means offline construction can run on batch machines or background CPU jobs without tying up the query API server.
- **Inspectability & Auditability**: Exporting the tree to a serialized structure (`hat_finance_tree_structure.json`) allows data engineers and researchers to inspect cluster cohesion, verify semantic groupings, and audit cross-document edges before serving queries.

### B. Why Two Distinct Algorithms (Algorithm 1 & Algorithm 3)?
- **Algorithm 1 (`algo1_document_chunking.py`)**: 
  - Raw documents are noisy. Fixed-character chunking splits sentences in half (e.g. cutting `$383,285 million` into `$383,` and `285 million`), ruining embedding quality.
  - Algorithm 1 splits strictly on sentence boundaries, preserves section headings (`# Section 1...`), carries multi-sentence overlaps, and tags each chunk with a deterministic SHA-256 fingerprint.
- **Algorithm 3 (`algo3_hierarchical_abstract_tree.py`)**:
  - Takes the clean chunk collection $C$ from Algorithm 1, passes it through a 384-dimensional dense encoder (`all-MiniLM-L6-v2`), and recursively clusters them using **Spherical k-Means** with k-means++ seeding.
  - For each cluster, an abstract summary is generated and embedded into the **exact same vector space** as the leaf chunks. This ensures queries can be compared directly against high-level summaries and leaf passages with identical mathematical operations.

### C. Why Explicit $\alpha$ Cross-Child Edges?
In a conventional tree hierarchy (like RAPTOR), two related pieces of evidence that sit in different branches can only be connected by traversing up to a distant common ancestor (like the root). By the time you reach the root, fine-grained details are lost in high-level summaries.
- **$\alpha$-links** are horizontal/lateral edges connecting nodes that sit under **different parents or different source documents** whose cosine similarity exceeds a threshold $\alpha$.
- For example, Apple's balance-sheet passage and Tesla's capital-expenditure passage can be linked directly across branches, enabling multi-hop cross-document reasoning.

### D. Why Extractive vs. LLM Summarizer Modes?
- **Extractive Mode (`--summarizer extractive`)**: Identifies the geometric centroid of each cluster and selects the most representative sentences. It runs in **under 1 second** on plain CPU, requires zero model downloads, and uses minimal RAM. Perfect for development, debugging, and continuous integration.
- **LLM Mode (`--summarizer llm`)**: Uses `facebook/bart-large-cnn` to synthesize fluent, abstractive paragraphs across multiple documents. It provides natural-language summaries for high-level overviews.

---

## 2. All Available Commands & What Each Does

All commands are run from the project root (`HAT-RAG-CUDA`):

```powershell
# 1. Full Corporate Finance Pipeline (Fast Extractive Summaries ~1s)
python algorithms/run_finance_rag.py --summarizer extractive

# 2. Full Corporate Finance Pipeline (Abstractive LLM Summaries ~40s)
python algorithms/run_finance_rag.py --summarizer llm

# 3. Full Engineering Demo Pipeline (Fast Extractive Summaries ~0.2s)
python algorithms/run_demo.py --summarizer extractive

# 4. Full Engineering Demo Pipeline (Default LLM Summaries)
python algorithms/run_demo.py

# 5. Algorithm 1 Alone (Document Chunking & SHA-256 Fingerprinting)
python algorithms/algo1_document_chunking.py

# 6. Algorithm 3 Alone (Spherical k-Means Tree Construction)
python algorithms/algo3_hierarchical_abstract_tree.py --summarizer extractive
```

---

## 3. Dissecting the Output of Each Command

### Command 1: `python algorithms/run_finance_rag.py --summarizer extractive`
Runs the complete knowledge construction pipeline over 5 real SEC 10-K corporate filings (`AAPL`, `AMZN`, `MSFT`, `NVDA`, `TSLA`) located in `finance_data/reports/`.

#### Output Walkthrough:

```text
====================================================================================================
 HAT-RAG OFFLINE KNOWLEDGE CONSTRUCTION -- FINANCIAL & ACCOUNTING CORPUS
 Algorithm 1 (document chunking) -> Algorithm 3 (hierarchical abstract tree)
====================================================================================================

[Document Corpus] 5 corporate 10-K reports:
  - AAPL_Apple_Financial_Report_2024.txt          (259 words)
  - AMZN_Amazon_Financial_Report_2024.txt         (232 words)
  - MSFT_Microsoft_Financial_Report_2024.txt      (257 words)
  - NVDA_NVIDIA_Financial_Report_2024.txt         (253 words)
  - TSLA_Tesla_Financial_Report_2024.txt          (240 words)
```
> **What this means**: The runner discovers all 10-K reports in `finance_data/reports/` and reads their raw text.

```text
[PHASE 1] ALGORITHM 1: DOCUMENT CHUNKING -> chunk collection C
  > AAPL_Apple_Financial_Report_2024.txt          ->   5 chunks
  > AMZN_Amazon_Financial_Report_2024.txt         ->   4 chunks
  > MSFT_Microsoft_Financial_Report_2024.txt      ->   5 chunks
  > NVDA_NVIDIA_Financial_Report_2024.txt         ->   5 chunks
  > TSLA_Tesla_Financial_Report_2024.txt          ->   5 chunks

  |C| = 24 chunks | avg 46.5 words/chunk | target 60 overlap 15
```
> **What this means**: Algorithm 1 chunks the documents. 5 documents produce exactly 24 leaf passages ($C$), with an average of 46.5 words per chunk, respecting sentence boundaries and sections.

```text
[PHASE 2] DENSE EMBEDDING OF C -> matrix E (leaf vectors for Algorithm 3)
  backend=sentence-transformers::sentence-transformers/all-MiniLM-L6-v2 | device=CPU (12 threads) | dim=384 | batch=32 | fp16=False
  E shape = (24, 384) | 90.9 texts/sec on CPU (12 threads)
```
> **What this means**: The 24 passages are converted into 384-dimensional unit vectors ($E \in \mathbb{R}^{24 \times 384}$) using `all-MiniLM-L6-v2`.

```text
[PHASE 3] ALGORITHM 3: HIERARCHICAL ABSTRACT TREE CONSTRUCTION -> H
  Summarizer: extractive::centroid-sentence-selection
  [L1] k-means: n=24 -> k=6 | backend=torch/cpu | iters=4 | cohesion=0.829 | 2.0 ms
  [L2] k-means: n=6 -> k=2  | backend=torch/cpu | iters=3 | cohesion=0.787 | 2.0 ms
  [L3] k-means: n=2 -> k=1  | backend=torch/cpu | iters=3 | cohesion=0.776 | 1.0 ms
  [alpha>=0.577] level 0: 27 explicit cross-child edges over 24 nodes
```
> **What this means**:
> - **L1**: 24 leaf nodes are clustered into 6 groups using spherical k-means. Convergence took 4 iterations with average cluster cohesion of 0.829.
> - **L2**: 6 L1 abstracts are clustered into 2 groups (cohesion 0.787).
> - **L3**: The 2 L2 abstracts form the global ROOT node (cohesion 0.776).
> - **$\alpha$-links**: 27 cross-child lateral connections were established where similarity exceeded $\alpha = 0.577$.

```text
====================================================================================================
 HIERARCHICAL ABSTRACT TREE (H) -- TOPOLOGY
====================================================================================================
`-- [L3] ROOT (2 children | 5 docs | cohesion 0.776) topics: million, equity, expensed...
    |-- [L2] L2_A0 (2 children | 3 docs | cohesion 0.821)
    `-- [L2] L2_A1 (4 children | 5 docs | cohesion 0.732)
        |-- [L1] L1_A0 (8 children | 4 docs | cohesion 0.803)
        |   |-- [L0] AAPL_...txt_c0 (doc: AAPL...)
        |   |-- [L0] AAPL_...txt_c1
        |   |     ~~> alpha-link TSLA_...txt_c1 (cos=0.6376, cross_document)
```
> **What this means**: The tree hierarchy visualized:
> - `[L3]` Global Root.
> - `[L2]` Mid-level abstract groupings.
> - `[L1]` Cluster abstracts.
> - `[L0]` Original raw passage chunks.
> - `~~> alpha-link`: Direct lateral link between Apple and Tesla chunks (cosine similarity 0.6376) despite living under different branches.

```text
[Persistence] H serialized to hat_finance_tree_structure.json (273 KB)
[Persistence] Round trip verified: 32 nodes, 1 root(s), 27 alpha edges
```
> **What this means**: The built tree is exported to `hat_finance_tree_structure.json` and immediately reloaded to guarantee JSON serialization integrity.

---

### Command 2: `python algorithms/run_finance_rag.py --summarizer llm`
Runs the exact same pipeline on the finance corpus, but uses `facebook/bart-large-cnn` to compose natural-language abstractive summaries for every non-leaf node.

#### Output Differences from Command 1:
- **Summarizer**: Displays `backend=transformers::facebook/bart-large-cnn`.
- **Abstract Texts**: Instead of extracting existing sentences, internal nodes (`L1_A0`, `L2_A0`, `ROOT`) contain synthesized paragraphs authored by the BART model.
- **Timing**: Takes ~40 seconds on CPU because generating sentences with an autoregressive transformer requires beam search decoding.
- **$\alpha$-links**: Auto-calibrated threshold produces 29 cross-child links (17 between different companies).

---

### Command 3: `python algorithms/run_demo.py --summarizer extractive`
Runs the full pipeline on a compact 3-document engineering research corpus (Thermal Dynamics, Motor Diagnostics, Energy Optimization).

#### Output Highlights:
- Ingests 3 documents $\rightarrow$ creates 6 leaf chunks ($C$).
- Builds a 3-tier hierarchy:
  - 6 leaf chunks (`L0`) $\rightarrow$ 2 cluster abstracts (`L1`) $\rightarrow$ 1 global root (`L2`).
- Total nodes: 9 (Compression ratio 6.0x).
- Serialization: Saves to `hat_tree_structure.json`.
- Runtime: **0.22 seconds**.

---

### Command 4: `python algorithms/run_demo.py`
Runs the demo corpus using the abstractive BART summarizer by default.
- Shows pipeline audit:
  - `Algorithm 1 (chunking): PASSED`
  - `Dense embedding: PASSED`
  - `Algorithm 3 (HAT construction): PASSED`
  - `JSON persistence round trip: PASSED`

---

### Command 5: `python algorithms/algo1_document_chunking.py`
Tests **Algorithm 1 alone** without loading PyTorch or embedding models.
- Demonstrates sentence splitting regex, section boundary detection, and SHA-256 chunk fingerprinting.
- Shows chunk hash uniqueness: `37/37 (100% Unique Fingerprints)`.
- Runtime: Instantaneous (< 0.05 seconds).

---

### Command 6: `python algorithms/algo3_hierarchical_abstract_tree.py --summarizer extractive`
Tests **Algorithm 3 alone** on pre-defined synthetic chunks.
- Validates spherical k-means clustering, cluster centroid cosine alignment, singleton collapse, and export to `hat_tree_structure.json`.

---

## 4. Key Metrics Explained

When running these commands, several quantitative metrics appear in the summary blocks:

| Metric | What It Means | Why It Matters |
|---|---|---|
| **Cohesion** | Average cosine similarity of all nodes in a cluster to their cluster centroid. | Values around **0.75 – 0.85** indicate tight, coherent semantic groupings. If clustering were random, this value would drop below 0.3. |
| **Compression (L0 $\rightarrow$ Root)** | Ratio of leaf chunks to root nodes (e.g. `24.0x`). | Measures how effectively the hierarchy condenses raw information into high-level representations for top-down traversal. |
| **Cross-Document Clusters** | Number of abstract clusters that contain passages from more than one company/document. | Proves that the tree pools knowledge globally across documents rather than building isolated per-document trees. |
| **Explicit $\alpha$ Edges** | Number of lateral connections between nodes under different parents. | Quantifies cross-branch relationships preserved in the index for multi-hop retrieval. |
| **Singleton Collapse** | Number of single-child clusters promoted directly to the next level. | Avoids creating redundant chain nodes that merely paraphrase a single child. |

---

## 5. Artifacts Produced

Running these commands creates and updates two JSON artifacts in [`algorithms/`](./algorithms/):

1. **`hat_finance_tree_structure.json`**:
   - Built by `run_finance_rag.py`.
   - Contains 32 nodes, 384-dimensional embeddings, 27-29 cross-company $\alpha$-links, cluster centroids, and build metadata for SEC 10-K filings.
2. **`hat_tree_structure.json`**:
   - Built by `run_demo.py` and `algo3_hierarchical_abstract_tree.py`.
   - Contains the 9-node tree for the engineering research corpus.

These JSON structures are directly consumable by the online retrieval engine and REST API.
