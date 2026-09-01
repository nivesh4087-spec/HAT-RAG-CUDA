# -*- coding: utf-8 -*-
import os
from paper_data_p1 import PAPERS_P1
from paper_data_p2 import PAPERS_P2
from paper_data_p3 import PAPERS_P3
from paper_data_p4 import PAPERS_P4
from paper_data_p5 import PAPERS_P5
from paper_data_p6 import PAPERS_P6

ALL_PAPERS = PAPERS_P1 + PAPERS_P2 + PAPERS_P3 + PAPERS_P4 + PAPERS_P5 + PAPERS_P6

def build_paper_markdown(p):
    num_str = "%02d" % p['num']
    return (
        "### Paper " + num_str + ": " + p['title'] + "\n" +
        "- **ArXiv ID**: [" + p['id'] + "](https://arxiv.org/abs/" + p['id'] + ") | **Category**: `" + p['topic'] + "`\n" +
        "- **Local PDF File**: `Paper_" + num_str + "_" + p['id'] + ".pdf`\n" +
        "- **Authors**: " + p['authors'] + "\n\n" +
        "#### What the Paper Says (Detailed Synthesis)\n" +
        p['synthesis'] + "\n\n" +
        "#### Core Mechanism & Architectural Innovations\n" +
        p['mechanism'] + "\n\n" +
        "#### Key Experimental Results & Benchmarks\n" +
        p['results'] + "\n\n" +
        "#### Direct Relevance & Application to HAT-RAG\n" +
        p['hat_relevance'] + "\n\n" +
        "---\n\n"
    )

def generate_doc():
    lines = []
    lines.append("# Deep-Dive Research Papers Documentation: 29 Seminal RAG & CUDA Publications\n\n---\n\n")
    lines.append("## Executive Summary\n\n")
    lines.append("This document provides a comprehensive literature synthesis and technical analysis of the **29 reference research papers** stored in `hat_rag/papers/`. These papers form the theoretical, mathematical, and algorithmic foundation of the **Hierarchical Abstract Tree Retrieval-Augmented Generation (HAT-RAG)** system and its CUDA-accelerated vector retrieval engine.\n\n")
    lines.append("The 29 papers are categorized into **6 Strategic Research Pillars**:\n")
    lines.append("1. **Tree & Hierarchical RAG (Papers 1, 3, 4, 23, 29)**: Multi-level tree indexing, recursive abstract summarization, top-down branch pruning.\n")
    lines.append("2. **Graph-Based RAG (Papers 2, 11, 21, 22)**: Entity-relation knowledge graphs, multi-hop community summaries, structured reasoning.\n")
    lines.append("3. **Adaptive & Self-Reflective RAG (Papers 5, 6, 7, 20)**: Corrective retrieval (CRAG), self-reflection tokens (Self-RAG), dynamic routing.\n")
    lines.append("4. **Dense Retrieval & Query Granularity (Papers 9, 13, 14, 15, 24)**: Dense Passage Retrieval (DPR), HyDE hypothetical queries, proposition-level chunking.\n")
    lines.append("5. **GPU CUDA Acceleration & Model Efficiency (Papers 16, 17, 18, 19)**: PyTorch/Faiss GPU vector kernels, 4-bit QLoRA, speculative inference.\n")
    lines.append("6. **Long-Context Dynamics & Evaluation Frameworks (Papers 8, 10, 12, 25, 26, 27, 28)**: \"Lost in the Middle\" context bias, LongRAG, ARES/RAGBench evaluation suites.\n\n---\n\n")

    lines.append("## Quick Taxonomy Matrix of All 29 Research Papers\n\n")
    lines.append("| # | ArXiv ID | Title | Topic Category | Primary Theoretical Finding | HAT-RAG System Integration |\n")
    lines.append("|---|---|---|---|---|---|\n")

    for p in ALL_PAPERS:
        short_title = p['title'][:40] + "..." if len(p['title']) > 40 else p['title']
        clean_synth = p['synthesis'].replace('\n', ' ')
        clean_synth = clean_synth[:75] + "..." if len(clean_synth) > 75 else clean_synth
        clean_rel = p['hat_relevance'].replace('\n', ' ')
        clean_rel = clean_rel[:75] + "..." if len(clean_rel) > 75 else clean_rel
        num_str = "%02d" % p['num']
        pid = p['id']
        ptopic = p['topic']
        lines.append("| " + num_str + " | [" + pid + "](https://arxiv.org/abs/" + pid + ") | " + short_title + " | `" + ptopic + "` | " + clean_synth + " | " + clean_rel + " |\n")

    lines.append("\n---\n\n## In-Depth Analysis of All 29 Research Papers\n\n")

    for p in ALL_PAPERS:
        lines.append(build_paper_markdown(p))

    lines.append("\n## Summary & Architectural Takeaways for HAT-RAG\n\n")
    lines.append("1. **Hierarchical Superiority**: Papers 1, 3, and 29 confirm that multi-tier tree abstractions eliminate distractor noise by up to 65% and outperform flat vector retrieval on long multi-document reasoning.\n")
    lines.append("2. **CUDA GPU Hardware Scaling**: Papers 16 and 17 prove that in-memory GPU PyTorch tensor matrix calculations ($Q \\times D^T$) achieve sub-millisecond retrieval latency for million-vector corpora.\n")
    lines.append("3. **Context Ordering Optimization**: Paper 28 ('Lost in the Middle') mandates ordering top-K retrieved context so highest relevance chunks sit at prompt boundaries, avoiding degradation in LLM attention.\n")
    lines.append("4. **Adaptive Routing Flexibility**: Papers 5, 7, and 20 validate modular multi-approach routing, allowing HAT-RAG to dynamically fallback between Flat, Tree, and Graph retrieval modes.\n")

    return "".join(lines)

if __name__ == "__main__":
    content = generate_doc()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    targets = [
        os.path.join(base_dir, "RESEARCH_PAPERS_SUMMARY_DOCUMENTATION.md"),
        os.path.join(base_dir, "docs", "RESEARCH_PAPERS_SUMMARY_DOCUMENTATION.md")
    ]
    for target in targets:
        dirname = os.path.dirname(target)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully generated " + str(target) + " (" + str(len(content)) + " characters, " + str(len(ALL_PAPERS)) + " papers)")
