import os

def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {len(text)} bytes to {path}")

def build_paper_markdown(p):
    return f"""### Paper {p['num']:02d}: {p['title']}
- **ArXiv ID**: [{p['id']}](https://arxiv.org/abs/{p['id']}) | **Category**: `{p['topic']}`
- **Local PDF File**: `Paper_{p['num']:02d}_{p['id']}.pdf`
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
