# Project Files — What Is Actually Built (50% Milestone)

This document describes the files that make up the **completed half** of HAT-RAG: the
offline knowledge-construction pipeline that turns a raw document corpus into a
persisted Hierarchical Abstract Tree.

The other half — query-time retrieval, traversal, reranking and answer generation — is
**not implemented yet**, and nothing in the files below pretends otherwise.

---

## 1. Scope: what "50%" means here

HAT-RAG has two halves that can be built and judged independently:

| Half | Question it answers | Status |
|---|---|---|
| **Offline / knowledge construction** | *How is the corpus turned into a searchable hierarchy?* | **Done** — this document |
| Online / retrieval | *Given a query, which nodes are visited and in what order?* | Not started |

This split is deliberate. The tree is the substrate that any retrieval policy runs on,
so its quality bounds everything downstream — a retriever cannot recover information
that the index never captured. Building and validating the tree first means the
retrieval work later has a fixed, measurable foundation instead of a moving target.

---

## 2. Files that contribute

Everything in this milestone lives in two directories: `algorithms/` (the pipeline) and
`finance_data/` (the corpus it runs on).

### `algorithms/` — the pipeline

| File | Lines | What it contributes | Why it matters |
|---|---|---|---|
| **`algo1_document_chunking.py`** | 195 | **Algorithm 1.** Splits raw documents into overlapping, sentence-aligned chunks; detects section headings; attaches section labels, word counts and SHA-256 fingerprints. | Every node at the bottom of the tree is one of these chunks. If a chunk straddles two unrelated topics or cuts a sentence in half, that damage propagates into the embeddings, the clusters and every abstract above it. This is the file that decides what a "unit of evidence" is. |
| **`algo3_hierarchical_abstract_tree.py`** | 1,256 | **Algorithm 3** plus the dense encoder it depends on. Embeds chunks, clusters them with spherical k-means, writes an LLM abstract per cluster, recurses to a single root, adds explicit α cross-child edges, reports build metrics and serialises the tree. | This is the heart of the project. It is the file that actually produces `H`, and the only place where the *hierarchical* and *cross-document* claims of HAT-RAG are either earned or not. |
| **`run_finance_rag.py`** | 162 | End-to-end pipeline on the 5-company financial corpus, with CLI flags for every parameter, a phase-by-phase trace and a cross-document structure report. | The primary reproducible experiment. It is what a reviewer runs to confirm the numbers in this repo, and the report it prints is the evidence for the cross-document claim. |
| **`run_demo.py`** | 152 | Same pipeline on a small self-contained engineering corpus embedded in the script. | Runs with no external data at all, so the pipeline can be demonstrated or debugged in isolation from the finance corpus. Also acts as a smoke test for the whole path. |
| **`hat_finance_tree_structure.json`** | — | The serialised finance tree: 32 nodes, all 384-d embeddings, parent/child links, α edges, build config and metrics. | The actual deliverable artifact. It is a complete, inspectable index — a reviewer can verify every claim in this repo by reading this file, without running anything. |
| **`hat_tree_structure.json`** | — | Same, for the demo corpus (9 nodes). | Small enough to read end to end by hand when checking the schema. |
| **`README.md`** | — | Operating manual for `algorithms/`: how to run it, what each stage does, what the flags mean, and an explicit statement that no GPU is required. | Keeps the how-to-run knowledge next to the code instead of in someone's head. |

### `finance_data/` — the corpus

| File | What it contributes | Why it matters |
|---|---|---|
| **`reports/*.txt`** (5 files) | Form 10-K style annual reports for Apple, Microsoft, NVIDIA, Amazon and Tesla — each ~230–260 words across 3 labelled sections (income statement, balance sheet, accounting policies). | The corpus is chosen, not incidental. Five companies describing *the same financial concepts in different words and different numbers* is exactly the condition under which a cross-document index either works or collapses. A corpus of five unrelated topics could not test the claim at all. |
| **`reports/finance_manifest.json`** | Ticker, company name, filing type, period, word count and line count per report. | Makes the corpus self-describing and auditable — a reader can tell what was ingested without opening five text files. |
| **`download_finance_data.py`** | Script that generates/fetches the report set. | Reproducibility: the corpus can be rebuilt rather than being an unexplained blob of text in the repo. |

### Supporting

| File | Contribution |
|---|---|
| `md/git_commit.md` | Standing rules for commit granularity and message style, so the history reads as incremental development. Git-ignored, so it never enters the history itself. |
| `.gitignore` | Keeps caches, build output and paper-drafting assets out of the repository. |

---

## 3. Files **not** part of this milestone

The repository root also contains an earlier full-system prototype — `app.py`,
`run_app.py`, `demo_hat_rag.py`, `run_tests.py`, `src/`, `hat_rag/`, `tests/` — and the
root `README.md` that describes the complete intended platform (FastAPI service,
Streamlit UI, CUDA traversal, retrieval).

That material describes the **target system**, not the work delivered here. The 50%
milestone is `algorithms/` + `finance_data/` only. When the two disagree, the
`algorithms/` code is the truth: it is the part that runs, is measured, and is
reproducible today.

---

## 4. How the files fit together

```
finance_data/reports/*.txt            5 raw documents
        |
        v
algo1_document_chunking.py            Algorithm 1
        |                             -> 24 chunks, each with section + hash
        v
algo3_hierarchical_abstract_tree.py   DenseEmbeddingModel
        |                             -> E, a 24 x 384 matrix of unit vectors
        |
        |                             SphericalKMeans + AbstractSummarizer
        |                             -> 5 L1 abstracts -> 2 L2 -> 1 ROOT
        |                             -> 29 explicit alpha cross-child edges
        v
hat_finance_tree_structure.json       the persisted tree H
```

`run_finance_rag.py` and `run_demo.py` are thin drivers over this path — they own no
algorithmic logic, only corpus loading, CLI parsing and reporting. That separation is
intentional: the algorithms must be usable as a library by the retrieval half later,
not only through a demo script.

---

## 5. Reproducing the milestone

```bash
python algorithms/run_finance_rag.py                           # full build, LLM abstracts (~45 s, CPU)
python algorithms/run_finance_rag.py --summarizer extractive   # same tree shape, no LLM (~1 s)
python algorithms/run_demo.py                                  # self-contained corpus
```

No GPU is required; every measurement in this repository was produced on CPU. The build
is seeded (`seed=42`), so clustering is deterministic and two runs of the same command
produce the same tree topology.

For what the algorithms actually do and why they are built the way they are, see
[`ALGORITHMS.md`](./ALGORITHMS.md).
