# -*- coding: utf-8 -*-

PAPERS_P3 = [
    {
        "num": 10,
        "id": "2410.18057",
        "topic": "Multi-Doc",
        "title": "MemoRAG: Moving Towards Next-Generation RAG via Memory-Augmented LLMs",
        "authors": "Research Community (ArXiv:2410.18057)",
        "synthesis": "MemoRAG introduces a dual-system memory architecture. A lightweight long-memory model compresses long documents into global memory representations, which generate draft answers to guide targeted dense retrieval.",
        "mechanism": "Global memory model builds a global abstract representation of ultra-long documents. When queried, it generates 'clue' concepts to retrieve precise leaf passages.",
        "results": "Achieved superior recall and synthesis quality on ultra-long multi-document benchmarks (over 100K tokens).",
        "hat_relevance": "Inspired the caching of global root abstract summaries in GPU VRAM within `hat_rag/src/cuda_utils.py`."
    },
    {
        "num": 11,
        "id": "2410.08012",
        "topic": "Graph-RAG",
        "title": "LightRAG: Simple and Fast Knowledge Graph-Based Retrieval-Augmented Generation",
        "authors": "Zirui Zhai et al. (HKU / Research Community)",
        "synthesis": "LightRAG optimizes GraphRAG by introducing dual-level retrieval (low-level entity/relation search and high-level global topic search) and incremental graph indexing, reducing graph processing latency by 10x.",
        "mechanism": "1. Dual-level entity graph indexing (entities + relationship triples).\n2. Dual-level vector search over entity names and high-level summaries.\n3. Incremental node merging without full graph rebuilds.",
        "results": "Achieved comparable or superior accuracy to Microsoft GraphRAG with 90% reduction in indexing compute and API cost.",
        "hat_relevance": "Used to optimize entity-relation extraction and fast community search in Approach 3 (`multi_approach.py`)."
    },
    {
        "num": 12,
        "id": "2404.10642",
        "topic": "Multi-Doc",
        "title": "LongRAG: Enhancing Retrieval-Augmented Generation for Long Context",
        "authors": "Research Community (ArXiv:2404.10642)",
        "synthesis": "LongRAG shifts the retrieval unit from small 100-512 token chunks to long 4,000+ token document units. It demonstrates that long retrieval units preserve semantic completeness and reduce retriever search space.",
        "mechanism": "1. Long retrieval unit formulation (combining contiguous sections into ~4K token passages).\n2. Coarse-grained dense retrieval followed by targeted LLM extraction.",
        "results": "Outperformed short-chunk RAG on long-context QA datasets (NQ, HotpotQA) with 70% fewer retrieved units.",
        "hat_relevance": "Informed chunk size selection and overlapping window parameters in `hat_rag/src/document_processor.py`."
    },
    {
        "num": 13,
        "id": "2310.03025",
        "topic": "Context-RAG",
        "title": "Dense X Retrieval: What Retrieval Granularity Should We Use?",
        "authors": "Tong Chen et al. (Princeton University)",
        "synthesis": "Investigates the optimal text unit granularity for dense retrieval (passages, sentences, or propositions). Introduces 'Propositional Retrieval', breaking text into self-contained atomic factual propositions.",
        "mechanism": "1. Text decomposition into independent propositions using LLMs.\n2. Vector indexing at proposition level.\n3. Retrieval aggregates atomic propositions for context synthesis.",
        "results": "Propositions significantly outperformed traditional 100-word passages and sentences in dense retrieval recall across 5 QA datasets.",
        "hat_relevance": "Directly guides text chunking and granularity options in `hat_rag/src/document_processor.py`."
    },
    {
        "num": 14,
        "id": "2212.10496",
        "topic": "Dense-RAG",
        "title": "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)",
        "authors": "Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan (CMU / University of Waterloo)",
        "synthesis": "HyDE (Hypothetical Document Embeddings) uses an LLM to generate a hypothetical answer document for an incoming query, then embeds the hypothetical document to search for real matching passages.",
        "mechanism": "Query -> LLM generates hypothetical document -> Encoder converts to vector -> Vector search retrieves real passages.",
        "results": "Outperformed state-of-the-art zero-shot dense retrievers on BEIR benchmarks without requiring fine-tuning.",
        "hat_relevance": "Implemented as a query expansion module option in `hat_rag/src/retriever.py`."
    }
]
