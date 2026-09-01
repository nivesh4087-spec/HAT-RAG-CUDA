# -*- coding: utf-8 -*-

PAPERS_P6 = [
    {
        "num": 25,
        "id": "2401.00812",
        "topic": "Evaluation",
        "title": "RAGBench: Evaluating Retrieval-Augmented Generation Systems",
        "authors": "Research Community (ArXiv:2401.00812)",
        "synthesis": "RAGBench introduces a comprehensive evaluation benchmark containing 3,600 complex queries spanning domain-specific datasets (medical, legal, technical, finance). It evaluates Faithfulness, Answer Relevance, Context Recall, and Noise Sensitivity.",
        "mechanism": "Multi-metric automated evaluation pipeline utilizing calibrated LLM judges with verifiable ground-truth annotations.",
        "results": "Standardized evaluation protocol for assessing RAG system capabilities across noise, ambiguity, and multi-doc reasoning.",
        "hat_relevance": "Direct source of evaluation metrics (Faithfulness, Relevance, Noise Robustness) in `hat_rag/src/evaluator.py`."
    },
    {
        "num": 26,
        "id": "2309.01431",
        "topic": "Evaluation",
        "title": "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation",
        "authors": "Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia (Stanford University)",
        "synthesis": "ARES introduces automated evaluation of RAG systems using synthetic query generation and fine-tuned LLM judges with prediction powered inference (PPI) to provide statistical confidence intervals.",
        "mechanism": "1. Synthetic query-passage generation.\n2. Fine-tuning lightweight LLM judges on domain data.\n3. PPI calculation for statistically tight accuracy bounds with minimal manual annotation.",
        "results": "Evaluates Context Relevance, Answer Faithfulness, and Answer Relevance with 98% correlation to human annotators.",
        "hat_relevance": "Informs synthetic query generation and evaluation bounds in `hat_rag/src/evaluator.py`."
    },
    {
        "num": 27,
        "id": "2406.04271",
        "topic": "Multi-Doc",
        "title": "Benchmarking Cross-Document Summarization and Retrieval",
        "authors": "Research Community (ArXiv:2406.04271)",
        "synthesis": "Evaluates multi-document information integration, conflict detection, and temporal updating in RAG systems when processing multi-page collections.",
        "mechanism": "Benchmark test suite consisting of multi-document sets with conflicting claims, redundant facts, and distributed information.",
        "results": "Revealed severe limitations in standard flat RAG when synthesizing contradictory or multi-document knowledge.",
        "hat_relevance": "Provides benchmark dataset design for multi-document synthesis testing in HAT-RAG."
    },
    {
        "num": 28,
        "id": "2307.03172",
        "topic": "Context-RAG",
        "title": "Lost in the Middle: How Language Models Use Long Contexts",
        "authors": "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (Stanford / UC Berkeley / Meta AI)",
        "synthesis": "Discovers that LLMs retrieve information most effectively when relevant context is placed at the beginning or end of the input prompt. Performance degrades significantly when relevant information is located in the middle of long contexts.",
        "mechanism": "Controlled multi-document QA experiments altering position $k$ of relevant passage among $N$ distractor passages.",
        "results": "U-shaped performance curve: accuracy drops up to 30% when target info is placed in the middle of the context window.",
        "hat_relevance": "Dictates top-K context placement order in `hat_rag/src/generator.py` (sorting retrieved passages so top matches are placed at prompt start/end)."
    },
    {
        "num": 29,
        "id": "2404.07221",
        "topic": "Tree-RAG",
        "title": "Hierarchical Context Partitioning for Multi-Document Question Answering",
        "authors": "Research Community (ArXiv:2404.07221)",
        "synthesis": "Proposes partitioning document collections into hierarchical tree structures based on semantic sub-topics to eliminate distractor noise prior to LLM synthesis.",
        "mechanism": "Sub-topic document tree partitioning with top-down path filtering to drop non-relevant document clusters.",
        "results": "Reduced prompt token count by 65% while improving multi-document answer accuracy by 18%.",
        "hat_relevance": "Validates HAT-RAG's logarithmic node pruning strategy during hierarchical tree search."
    }
]
