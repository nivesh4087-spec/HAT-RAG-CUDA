# Hierarchical Abstract Tree (HAT) for Cross-Document Retrieval-Augmented Generation using NVIDIA CUDA-Accelerated GPU Computing

---

## 📌 Project Overview & 50% Milestone Report

This repository contains the implementation of **Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation (HAT-RAG)** accelerated via **NVIDIA CUDA GPU Computing**.

HAT-RAG solves the context fragmentation and high-latency challenges in standard flat RAG pipelines when dealing with complex, multi-document enterprise knowledge bases. By structuring documents into a multi-tiered tree of abstracts and fine-grained passages, HAT enables top-down logarithmic retrieval over millions of tokens.

---

## 🚀 50% Project Progress Summary

| Module / Component | Status | Description |
|---|---|---|
| **1. CUDA Hardware Utilities (`cuda_utils.py`)** | ✅ Completed | Detects NVIDIA GPUs, handles CUDA memory, and accelerates vector batch cosine similarity operations via PyTorch GPU tensors (with CPU fallback). |
| **2. Document Ingestion & Chunking (`document_processor.py`)** | ✅ Completed | Parses multi-source raw documents into overlapping fine-grained chunk representations. |
| **3. Hierarchical Tree Builder (`hierarchical_tree.py`)** | ✅ Completed | Constructs multi-level abstract summary trees using recursive vector clustering (K-Means/GMM) and cluster summarization. |
| **4. Multi-Level Tree Retriever (`retriever.py`)** | ✅ Completed | Implements CUDA-accelerated top-down branch traversal to select relevant abstract clusters down to leaf contexts. |
| **5. Generator & Synthesizer (`generator.py`)** | ✅ Completed | Synthesizes retrieved cross-document hierarchical contexts into final LLM answers. |
| **6. End-to-End Core Demo (`demo_hat_rag.py`)** | ✅ Completed | Functional demonstration script validating document processing, tree building, retrieval, and response synthesis. |
| **7. Web UI / Dashboard** | ⏳ Planned (50-100%) | Interactive Streamlit/Gradio visualization for inspecting tree nodes and context retrieval. |
| **8. Fine-Tuned Local LLM Integration** | ⏳ Planned (50-100%) | Integration with fine-tuned LLaMA-3 / Mistral model on NVIDIA CUDA hardware. |

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
 │  Level 1: Local Abstract Summaries   │ ← GPU Accelerated Vector Embeddings
 │  Level 2: Global Root Abstracts      │
 └──────────────────┬───────────────────┘
                    ▼
       ┌──────────────────────────┐
       │   NVIDIA CUDA GPU        │ ← Matrix Multiplication & Cosine Similarity Batching
       └────────────┬─────────────┘
                    ▼
       ┌──────────────────────────┐
       │   Top-Down Retriever     │ ← Logarithmic Branch Pruning Search
       └────────────┬─────────────┘
                    ▼
       ┌──────────────────────────┐
       │   Context Generator      │ → Final RAG Output
       └──────────────────────────┘
```

---

## 💻 Code Structure (`hat_rag/`)

```
hat_rag/
├── data/                    # Sample raw document inputs
├── models/                  # Saved embedding / summarization models
├── src/
│   ├── cuda_utils.py        # NVIDIA CUDA hardware detection & GPU matrix math
│   ├── document_processor.py# Text chunking & normalization
│   ├── hierarchical_tree.py # Tree Node data structure & abstract clustering
│   ├── retriever.py         # Top-down CUDA hierarchical vector search
│   └── generator.py         # Response generation from retrieved tree contexts
├── demo_hat_rag.py          # Functional 50% milestone pipeline demo
├── requirements.txt         # Project dependencies
└── README.md                # Documentation & progress report
```

---

## 🛠️ How to Run the Demo

### 1. Execute the Pipeline
Run the 50% core demonstration pipeline using Python:

```bash
python hat_rag/demo_hat_rag.py
```

### 2. Sample Output
```
==========================================================================
Hierarchical Abstract Tree (HAT) RAG - CUDA Accelerated Engine Demo
==========================================================================
[Hardware Setup] Active Device: NVIDIA GeForce RTX / PyTorch CUDA
[Doc Processing] Processed 3 documents into 6 leaf chunks.
[HAT Engine] Constructing Hierarchical Abstract Tree...
[HAT Engine] Tree Built Successfully! Total nodes: 10, Root clusters: 2

[Retrieval] Query: 'How to prevent motor failure and optimize ceiling fan cooling performance?'
[Retrieval] Executing CUDA-Accelerated Top-Down Tree Traversal...

=== HAT-RAG Response ===
Query: How to prevent motor failure and optimize ceiling fan cooling performance?
Retrieved Context (2 nodes from Hierarchical Tree):
[doc2_maintenance.txt_chunk_0] (Level 0): Tool wear in stamping fan blades causes motor alignment errors...
[doc1_cooling.txt_chunk_1] (Level 0): downward airflow, reducing effective temperature...

Answer: Based on cross-document abstract hierarchy, the information suggests that...
==========================================================================
[SUCCESS] 50% Project Core Pipeline Completed Successfully.
```

---

## 🎯 Next Steps (Path to 100% Completion)
1. **CUDA Kernel Optimization**: Custom PyTorch CUDA kernels for ultra-low latency tree traversal.
2. **Interactive Visual Dashboard**: Graph visualization of the Hierarchical Abstract Tree nodes.
3. **Enterprise Evaluation Suite**: Benchmark RAG accuracy (ROUGE, BLEU, Context Precision) against flat RAG.
