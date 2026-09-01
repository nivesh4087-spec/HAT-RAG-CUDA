# -*- coding: utf-8 -*-

PAPERS_P1 = [
    {
        "num": 1,
        "id": "2401.18059",
        "topic": "Tree-RAG",
        "title": "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
        "authors": "Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, Christopher D. Manning (Stanford University)",
        "synthesis": "RAPTOR introduces recursive clustering and abstractive summarization of text chunks, building a multi-tier tree structure. Standard RAG systems retrieve only small, fixed-size contiguous text passages, missing high-level thematic context. RAPTOR addresses this by building trees of abstract summaries so retrieval can query both macro summaries and micro leaf passages.",
        "mechanism": "1. **Leaf Partitioning**: Splits text into 100-word leaf passages.\n2. **GMM Vector Clustering**: Embeds passages and applies Soft Gaussian Mixture Models (GMM) with BIC dimensionality reduction.\n3. **Abstractive Summarization**: Uses an LLM to generate summary nodes for each cluster.\n4. **Recursive Assembly**: Repeats clustering and summarization recursively until reaching the root.",
        "results": "RAPTOR set new SOTA performance on QASPER, NarrativeQA (+20% accuracy), and QuALITY benchmarks. Demonstrated that retrieving high-level summary nodes provides critical context for complex multi-hop queries.",
        "hat_relevance": "Direct theoretical foundation for **HAT-RAG**. Implemented in `hat_rag/src/hierarchical_tree.py` using K-Means vector clustering and HuggingFace BART summarizers accelerated with PyTorch CUDA GPU tensors."
    },
    {
        "num": 2,
        "id": "2404.16130",
        "topic": "Graph-RAG",
        "title": "From Local to Global Graph-Based Retrieval-Augmented Generation",
        "authors": "Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)",
        "synthesis": "Microsoft GraphRAG solves the limitation of standard vector RAG when answering global dataset-wide queries ('What are the main themes in this corpus?'). It extracts entity-relation knowledge graphs from raw text, partitions them into multi-level communities, and pre-generates hierarchical community summaries.",
        "mechanism": "1. **Graph Extraction**: Uses LLMs to extract entities, types, and directed relationships.\n2. **Leiden Community Detection**: Clusters graph nodes into hierarchical community partitions.\n3. **Community Summarization**: Summarizes each community at multiple granularities.\n4. **Global Query Answering**: Synthesizes responses by aggregating intermediate answers from community summaries.",
        "results": "GraphRAG significantly outperformed standard RAG on global sensemaking queries, demonstrating higher comprehensiveness (+35%), diversity (+28%), and empowerment ratings.",
        "hat_relevance": "Implemented as **Approach 3 (Graph-RAG)** in `hat_rag/src/multi_approach.py` to compare knowledge graph multi-hop traversal against HAT-RAG's tree traversal."
    },
    {
        "num": 3,
        "id": "2408.08921",
        "topic": "Hierarchical-RAG",
        "title": "HiRAG: Hierarchical Information Retrieval-Augmented Generation",
        "authors": "Research Community (ArXiv:2408.08921)",
        "synthesis": "HiRAG proposes a hierarchical indexing scheme that explicitly separates document structure into root macro-abstracts, sub-topic abstracts, and micro-leaf passages. It optimizes context retrieval by selecting context paths down the tree.",
        "mechanism": "Constructs a strict 3-tier tree (Global Root -> Sub-Cluster Abstracts -> Leaf Passages). Applies path scoring where parent node similarity weights child node inclusion probability.",
        "results": "Reduces context redundancy by 45% while improving answer accuracy by 14% on open-domain multi-document QA tasks.",
        "hat_relevance": "HiRAG's strict 3-tier node schema (Level 0 Leaf, Level 1 Cluster Abstract, Level 2 Root Abstract) directly defines the node hierarchy used in `hat_rag/src/hierarchical_tree.py`."
    },
    {
        "num": 4,
        "id": "2410.05779",
        "topic": "Tree-RAG",
        "title": "Tree-of-Thought Prompting Meets Retrieval Augmented Generation",
        "authors": "Research Community (ArXiv:2410.05779)",
        "synthesis": "Combines Tree-of-Thought (ToT) reasoning with RAG traversal. Rather than performing a single retrieval pass, the agent explores multiple tree branches, evaluating candidate retrieved context paths using heuristic self-evaluation.",
        "mechanism": "Branch-and-bound search over hierarchical retrieval indices. Evaluates state quality at each tree node and backtracks if retrieval score drops below threshold.",
        "results": "Achieves superior reasoning accuracy on multi-step logic and financial document analysis datasets compared to single-pass RAG.",
        "hat_relevance": "Informs the logarithmic top-down branch traversal search algorithm in `hat_rag/src/retriever.py`."
    }
]
