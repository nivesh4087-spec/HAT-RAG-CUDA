# -*- coding: utf-8 -*-

PAPERS_P2 = [
    {
        "num": 5,
        "id": "2402.01613",
        "topic": "Adaptive-RAG",
        "title": "Corrective Retrieval Augmented Generation (CRAG)",
        "authors": "Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / MindSpore)",
        "synthesis": "CRAG introduces a lightweight retrieval evaluator to assess the quality of retrieved documents for a query. It classifies retrieval results into Correct, Incorrect, or Ambiguous. If retrieval is evaluated as incorrect, CRAG triggers web search or fallback data sources.",
        "mechanism": "1. **Retrieval Evaluator**: Fine-tuned model outputs confidence scores for retrieved passages.\n2. **Action Trigger**: Correct -> Pass to LLM; Incorrect -> Fallback Web Search; Ambiguous -> Combine passage refinement with external search.\n3. **Knowledge Refinement**: Decomposes retrieved passages into atomic key concepts.",
        "results": "CRAG improved RAG accuracy across short-form and long-form QA benchmarks, significantly reducing hallucinations caused by irrelevant retrieved context.",
        "hat_relevance": "Inspirations from CRAG are implemented in `retriever.py` to establish similarity thresholds and prune low-confidence branches during tree traversal."
    },
    {
        "num": 6,
        "id": "2310.11511",
        "topic": "Self-RAG",
        "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
        "authors": "Akari Asai, Sewon Min, Zeqiu Wu, Uvini Joshi, Luke Zettlemoyer, Hannaneh Hajishirzi (University of Washington / Allen Institute for AI)",
        "synthesis": "Self-RAG trains an LLM to dynamically retrieve passages on-demand and critique its own generations using special reflection tokens ([Retrieve], [IsRel], [IsSup], [IsUse]).",
        "mechanism": "During generation, the LLM outputs a [Retrieve] token when external knowledge is required. It evaluates retrieved passages for relevance ([IsRel]) and groundedness ([IsSup]), selecting the highest-scoring candidate path.",
        "results": "Outperformed state-of-the-art open models (Llama-2) and ChatGPT on open-domain QA, reasoning, and fact verification tasks while maintaining controllability.",
        "hat_relevance": "Informs the structured citation synthesis `[1]` and context validation engine in `hat_rag/src/generator.py`."
    },
    {
        "num": 7,
        "id": "2403.14403",
        "topic": "Adaptive-RAG",
        "title": "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models",
        "authors": "Research Community (ArXiv:2403.14403)",
        "synthesis": "Adaptive-RAG dynamically determines whether to retrieve knowledge based on query complexity. Simple queries are answered directly by the LLM, moderate queries use single-step retrieval, and complex multi-step queries use multi-hop tree search.",
        "mechanism": "Trains a query complexity classifier that routes incoming queries to the optimal retrieval strategy, minimizing latency and API costs.",
        "results": "Achieved optimal trade-offs between accuracy and latency, reducing unnecessary retrieval calls by up to 35%.",
        "hat_relevance": "Serves as the foundation for multi-approach routing in `hat_rag/src/multi_approach.py`."
    },
    {
        "num": 8,
        "id": "2312.10997",
        "topic": "Survey-RAG",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": "Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (Fudan University / Tongji University)",
        "synthesis": "Provides a comprehensive taxonomy of RAG evolution, categorizing systems into Naive RAG, Advanced RAG, and Modular RAG. Summarizes techniques across Pre-retrieval, Retrieval, Post-retrieval, and Generation.",
        "mechanism": "Systematically analyzes chunking strategies, vector indices, re-ranking algorithms, query transformation, context compression, and evaluation benchmarks.",
        "results": "Established the standard framework for evaluating and designing modern RAG pipelines in academia and industry.",
        "hat_relevance": "Guided the overall 7-phase methodology and modular architecture of the HAT-RAG system."
    },
    {
        "num": 9,
        "id": "2005.11401",
        "topic": "Baseline-RAG",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (Facebook AI Research / UCL / NYU)",
        "synthesis": "The foundational paper that introduced Retrieval-Augmented Generation (RAG). Combines a dense passage retriever (DPR) with a sequence-to-sequence generator (BART) to ground LLM responses in external parametric memory.",
        "mechanism": "Introduced RAG-Sequence (retrieves top passages for the whole sequence) and RAG-Token (retrieves different passages for each generated token) models trained end-to-end.",
        "results": "Set state-of-the-art results on Natural Questions, WebQuestions, CuratedTREC, and MS-MARCO benchmarks.",
        "hat_relevance": "Provides the baseline Flat Vector RAG model against which HAT-RAG is evaluated for latency and recall in `hat_rag/src/evaluator.py`."
    }
]
