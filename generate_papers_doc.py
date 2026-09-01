import os, json

def build_paper_entry(p):
    return f"""### Paper {p['num']:02d}: {p['title']}
- **ArXiv ID**: [{p['id']}](https://arxiv.org/abs/{p['id']}) | **Category**: `{p['topic']}`
- **Local File**: `Paper_{p['num']:02d}_{p['id']}.pdf`
- **Authors**: {p['authors']}

#### 📖 What the Paper Says (Detailed Synthesis)
{p['synthesis']}

#### 🔬 Core Mechanism & Architectural Innovations
{p['mechanism']}

#### 📊 Key Experimental Results & Benchmarks
{p['results']}

#### 🎯 Direct Relevance & Application to HAT-RAG
{p['hat_relevance']}

---
"""
