# Core RAG Algorithms: Algorithm 1 & Algorithm 3

This directory contains standalone implementations of the core indexing algorithms for the **Hierarchical Abstract Tree for Cross-Document RAG (HAT-RAG)** system.

---

## 📌 Implemented Algorithms & Scripts

| Algorithm / Script | Purpose |
|---|---|
| **[`algo1_document_chunking.py`](./algo1_document_chunking.py)** | Algorithm 1: Sentence-boundary sliding-window chunking with overlap, section detection, and SHA-256 chunk hashing. |
| **[`algo3_hierarchical_abstract_tree.py`](./algo3_hierarchical_abstract_tree.py)** | Algorithm 3: Multi-tier recursive semantic clustering, abstract summarization, tree topology generation, and JSON persistence. |
| **[`run_demo.py`](./run_demo.py)** | Master Pipeline Showcase on engineering research documents. |
| **[`run_finance_rag.py`](./run_finance_rag.py)** | Cross-Company Financial & Accounting Pipeline ingesting SEC 10-K filings from Apple, Microsoft, NVIDIA, Amazon, and Tesla. |

---

## 🚀 How to Run the Demonstrations

### 1. Run Algorithm 1 (Document Chunking)
```bash
python algorithms/algo1_document_chunking.py
```

### 2. Run Algorithm 3 (Hierarchical Abstract Tree Construction)
```bash
python algorithms/algo3_hierarchical_abstract_tree.py
```

### 3. Run General Showcase
```bash
python algorithms/run_demo.py
```

### 4. Run Financial & Accounting Data Pipeline (5 Tech Enterprises)
```bash
python algorithms/run_finance_rag.py
```
**Ingested Financial Reports:**
- `AAPL_Apple_Financial_Report_2024.txt`
- `MSFT_Microsoft_Financial_Report_2024.txt`
- `NVDA_NVIDIA_Financial_Report_2024.txt`
- `AMZN_Amazon_Financial_Report_2024.txt`
- `TSLA_Tesla_Financial_Report_2024.txt`
