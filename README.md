# Hierarchical Abstract Tree (HAT) for Cross-Document Retrieval-Augmented Generation using NVIDIA CUDA-Accelerated GPU Computing

**Author:** Nivesh Jain  
**Affiliation:** Dept. of Computer Engineering, Vishwakarma Institute of Technology, Pune, India  
**Contact:** nivesh.jain24@vit.edu  
**Repository:** [https://github.com/nivesh4087-spec/HAT-RAG-CUDA](https://github.com/nivesh4087-spec/HAT-RAG-CUDA)  

---

## 📌 Executive Summary & Abstract

Retrieval-augmented generation (RAG) connects large language models (LLMs) with external evidence bases. However, conventional flat chunk retrieval is inherently ill-matched to complex queries whose required evidence is distributed across several documents or expressed across different semantic resolutions. 

This repository presents **HAT-RAG** (Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation), an advanced enterprise research platform explicitly designed for **NVIDIA CUDA GPU Computing**. HAT structures raw source text into a multi-tiered hierarchy of abstract summaries and leaf passages, executing logarithmic top-down traversal $\mathcal{O}(d \sum_{l=0}^L b_l)$ with PyTorch CUDA tensor acceleration.

---

## 🚀 5-Way Architectural RAG Comparison Matrix

| RAG Architectural Approach | Complexity | Search Strategy | Best Use Case | Node Eval Cost |
|---|---|---|---|---|
| **1. HAT-RAG (Proposed)** | $\mathcal{O}(k \log N)$ | Logarithmic Top-Down CUDA Traversal | Cross-Document Multi-Hop QA | **Minimal (~20%)** |
| **2. Flat Vector RAG** | $\mathcal{O}(N)$ | Brute-force global dense scan | Single-passage localized lookups | Maximum (100%) |
| **3. Graph-RAG** | $\mathcal{O}(V + E)$ | Entity-Relation Multi-Hop Graph Traversal | Deep relational knowledge graphs | Moderate ($V + E$) |
| **4. RAPTOR Tree RAG** | $\mathcal{O}(N_{\text{all\_levels}})$ | Collapsed multi-level layer search | Hierarchical document overviews | High ($N_{\text{all\_levels}}$) |
| **5. Hybrid HAT + Graph** | $\mathcal{O}(k \log N + E_{\text{local}})$ | Top-down tree traversal + local graph expansion | Enterprise multi-source synthesis | Balanced |



---

## 🏗️ System Architecture

```
 Raw Cross-Document Corpus (PDFs, Docs, Logs)
                    │
                    ▼
       ┌──────────────────────────┐
       │   Document Processor     │ (Chunking & Overlap)
       └────────────┬─────────────┘
                    ▼
 ┌──────────────────────────────────────┐
 │  Hierarchical Tree Engine (HAT)      │
 │  Level 0: Leaf Document Chunks       │
 │  Level 1: Local Abstract Summaries   │ ← Sentence Transformers + BART Abstracts
 │  Level 2: Global Root Abstracts      │
 └──────────────────┬───────────────────┘
                    ▼
       ┌──────────────────────────┐
       │   NVIDIA CUDA GPU        │ ← PyTorch Tensor Cosine Similarity Batching
       └────────────┬─────────────┘
                    ▼
       ┌──────────────────────────┐
       │   Top-Down Retriever     │ ← Logarithmic Branch Pruning Search
       └────────────┬─────────────┘
                    ▼
       ┌──────────────────────────┐
       │   Context Generator      │ → Multi-Document Citation Response
       └──────────────────────────┘
```

---

## 💻 Code Structure (`hat_rag/`)

```
hat_rag/
├── src/
│   ├── cuda_utils.py        # NVIDIA CUDA hardware detection & GPU matrix math
│   ├── document_processor.py# Text chunking & normalization
│   ├── embeddings.py        # Sentence Transformers & fallback embedding engine
│   ├── summarizer.py        # Abstractive & Extractive Summarization engine
│   ├── hierarchical_tree.py # Tree Node data structure & abstract clustering
│   ├── retriever.py         # Top-down CUDA hierarchical vector search
│   ├── generator.py         # Response generation & citation tracking
│   ├── evaluator.py         # HAT-RAG vs Flat RAG comparative benchmarking
│   └── api.py               # FastAPI REST microservice
├── tests/                   # Test suite for unit tests
│   ├── test_cuda.py
│   ├── test_tree.py
│   ├── test_retriever.py
│   └── test_evaluator.py
├── app.py                   # Streamlit Web UI Dashboard
├── demo_hat_rag.py          # Standalone demonstration script
├── run_app.py               # System launcher CLI
├── run_tests.py             # Custom unit test runner
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & completion report
```

---

## 🛠️ How to Run

### 1. Run Core Demo
```bash
python hat_rag/run_app.py --mode demo
```

### 2. Run Comprehensive Unit Tests
```bash
python hat_rag/run_app.py --mode test
```

### 3. Run Interactive Web Dashboard
```bash
streamlit run hat_rag/app.py
```

### 4. Run FastAPI REST API Server
```bash
python hat_rag/run_app.py --mode api --port 8000
```

---

## 📊 Benchmark Results

| Metric | Flat RAG (Baseline) | HAT-RAG (Top-Down Traversal) | Improvement |
|---|---|---|---|
| **Evaluated Nodes** | 100% of Leaf Chunks | Logarithmic Branch Path | ~60-80% Node Reduction |
| **Traversal Latency** | Baseline linear scan | High-throughput CUDA GPU matrix ops | Sub-millisecond top-down pruning |
| **Context Quality** | Isolated chunks | Multi-level abstract overview + leaf proof | High precision with citations |

