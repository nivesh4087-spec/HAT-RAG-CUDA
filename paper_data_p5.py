# -*- coding: utf-8 -*-

PAPERS_P5 = [
    {
        "num": 20,
        "id": "2401.07883",
        "topic": "Modular-RAG",
        "title": "Modular RAG: Towards an Advanced RAG Architecture",
        "authors": "Yunfan Gao et al. (Fudan University)",
        "synthesis": "Modular RAG unbundles traditional monolithic RAG into independent, reconfigurable modules (Search, Routing, Rewrite, Predict, Read), enabling customized RAG workflows for diverse application domains.",
        "mechanism": "Decoupled module interface with standard tensor/text data pipelines, allowing runtime module swapping.",
        "results": "Demonstrated higher flexibility and superior performance across varied domain benchmarks.",
        "hat_relevance": "Direct architecture design for `hat_rag/src/` (processor, tree, retriever, generator, evaluator)."
    },
    {
        "num": 21,
        "id": "2402.16840",
        "topic": "Graph-RAG",
        "title": "Knowledge Graph-Augmented Language Models: A Survey",
        "authors": "Research Community (ArXiv:2402.16840)",
        "synthesis": "Comprehensive survey on combining Knowledge Graphs (KGs) with LLMs for pre-training, fine-tuning, and retrieval-augmented inference.",
        "mechanism": "Analyzes KG construction, entity alignment, graph neural network (GNN) embeddings, and multi-hop graph retrieval.",
        "results": "Established theoretical taxonomy for graph-augmented reasoning.",
        "hat_relevance": "Provides graph traversal theories for Approach 3 in `multi_approach.py`."
    },
    {
        "num": 22,
        "id": "2405.04517",
        "topic": "Structured RAG",
        "title": "StructRAG: Boosting Knowledge Intensive Reasoning via Structured Text Aggregation",
        "authors": "Research Community (ArXiv:2405.04517)",
        "synthesis": "StructRAG dynamically converts unstructured multi-document text into structured representations (trees, tables, graphs) based on query intent prior to LLM reasoning.",
        "mechanism": "1. Intent classifier selects optimal structure format.\n2. Information extraction pipeline constructs structured text views.\n3. LLM executes structured reasoning over structured views.",
        "results": "Outperformed flat RAG by +18% on complex tabular and multi-document reasoning datasets.",
        "hat_relevance": "Validates HAT-RAG's structured abstract tree model for document knowledge aggregation."
    },
    {
        "num": 23,
        "id": "2305.14283",
        "topic": "Reasoning",
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "authors": "Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (Princeton / Google DeepMind)",
        "synthesis": "Introduces Tree of Thoughts (ToT), enabling LLMs to explore multiple reasoning paths, evaluate choices at each step, and backtrack when necessary using BFS or DFS algorithms.",
        "mechanism": "1. Problem decomposition into thought steps.\n2. Generation of multiple candidate thoughts.\n3. Heuristic evaluation of thought states.\n4. Search tree traversal (BFS/DFS) with backtracking.",
        "results": "Solves complex reasoning tasks (Game of 24, Creative Writing, Mini Crosswords) where standard Chain-of-Thought fails dramatically.",
        "hat_relevance": "Conceptual origin for top-down tree branch search and score-guided pruning in `hat_rag/src/retriever.py`."
    },
    {
        "num": 24,
        "id": "2305.04091",
        "topic": "Context-RAG",
        "title": "In-Context Retrieval-Augmented Language Models",
        "authors": "Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Shaham, Amir Globerson, Jonathan Berant, Yoav Shoham (Tel Aviv University / AI21 Labs)",
        "synthesis": "Investigates in-context RALM, showing how off-the-shelf language models can effectively leverage retrieved documents inserted directly into the prompt without model retraining.",
        "mechanism": "Pre-retrieval query rewriting, document scoring, and prompt formatting templates for zero-shot context injection.",
        "results": "Demonstrated consistent LM perplexity improvements across diverse benchmarks without fine-tuning generator weights.",
        "hat_relevance": "Guides prompt template construction and context injection in `hat_rag/src/generator.py`."
    }
]
