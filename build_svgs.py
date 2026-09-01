import os

def save_svg(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated SVG: {path}")

def make_card(x, y, w, h, stroke, title, sub, desc, glow_id, is_dashed=False):
    dash = ' stroke-dasharray="5,4"' if is_dashed else ''
    glow = f' filter="url(#{glow_id})"' if glow_id else ''
    return f'''  <g transform="translate({x}, {y})">
    <rect x="0" y="0" width="{w}" height="{h}" rx="18" ry="18" fill="url(#card-bg)" stroke="{stroke}" stroke-width="2.5"{dash}{glow} />
    <circle cx="{w-25}" cy="22" r="5" fill="{stroke}" />
    <text x="22" y="34" fill="{stroke}" font-size="16" font-weight="800">{title}</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">{sub}</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">{desc}</text>
  </g>'''

def make_hub(x, y, w, h, stroke, title, sub, glow_id):
    return f'''  <g transform="translate({x}, {y})">
    <rect x="0" y="0" width="{w}" height="{h}" rx="22" ry="22" fill="url(#hub-bg)" stroke="{stroke}" stroke-width="2.5" filter="url(#{glow_id})" />
    <circle cx="{w-25}" cy="22" r="5" fill="{stroke}" />
    <text x="{w//2}" y="45" text-anchor="middle" fill="{stroke}" font-size="20" font-weight="800">{title}</text>
    <text x="{w//2}" y="64" text-anchor="middle" fill="#8b949e" font-size="12">{sub}</text>
  </g>'''

def make_defs():
    return '''  <defs>
    <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-green" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-pink" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <filter id="glow-yellow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="6" result="blur"/><feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>
    <linearGradient id="hub-bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#261724"/><stop offset="100%" stop-color="#161b22"/></linearGradient>
    <linearGradient id="card-bg" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#161b22"/><stop offset="100%" stop-color="#0d1117"/></linearGradient>
  </defs>'''

def generate_methodology_svg():
    header = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">'''
    defs = make_defs()
    title = '''  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="800">HAT-RAG SYSTEM METHODOLOGY ARCHITECTURE</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Hierarchical Abstract Tree with PyTorch CUDA GPU Vector Acceleration</text>'''
    
    paths = '''  <path d="M 500 295 C 500 200, 500 170, 500 155" stroke="#ff4b6e" stroke-width="3" fill="none" opacity="0.85" />
  <path d="M 610 320 C 720 320, 750 200, 770 170" stroke="#30d158" stroke-width="3" fill="none" opacity="0.85" />
  <path d="M 610 350 C 730 350, 750 450, 780 475" stroke="#f0883e" stroke-width="3" fill="none" opacity="0.85" />
  <path d="M 390 320 C 270 320, 240 200, 220 170" stroke="#a371f7" stroke-width="3" fill="none" opacity="0.85" />
  <path d="M 390 340 C 270 340, 250 450, 220 475" stroke="#58a6ff" stroke-width="3" fill="none" opacity="0.85" />
  <path d="M 500 375 C 500 460, 500 520, 500 555" stroke="#e3b341" stroke-width="3" fill="none" opacity="0.85" />'''

    hub = make_hub(370, 295, 260, 85, "#ff4b6e", "HAT-RAG Core", "Hierarchical Engine", "glow-red")
    n1 = make_card(370, 95, 260, 60, "#ff4b6e", "Phase 1: Ingestion", "Parser & Chunker", "Recursive Overlapping", "glow-red")
    n2 = make_card(650, 140, 260, 60, "#30d158", "Phase 4: CUDA Engine", "PyTorch GPU Accelerator", "Parallel Cosine Matrix", "glow-green")
    n3 = make_card(660, 445, 260, 60, "#f0883e", "Phase 5: Search", "Top-Down Traversal", "O(k log N) Pruning", "glow-orange")
    n4 = make_card(90, 140, 260, 60, "#a371f7", "Phase 2 & 3: HAT Index", "Vector Clustering", "K-Means Summaries", "glow-purple")
    n5 = make_card(80, 445, 260, 60, "#58a6ff", "Phase 6: LLM Generator", "Context Assembly", "Source Citations [1]", "glow-blue")
    n6 = make_card(370, 555, 260, 60, "#e3b341", "Phase 7: Benchmarks", "Latency & Speedup", "Evaluation Suite", "glow-yellow")

    return f"{header}\n{defs}\n{title}\n{paths}\n{hub}\n{n1}\n{n2}\n{n3}\n{n4}\n{n5}\n{n6}\n</svg>"

def generate_4_algorithms_svg():
    header = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">'''
    defs = make_defs()
    title = '''  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="800">4 RAG ALGORITHMIC APPROACHES ARCHITECTURE</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Comparative Multi-Approach Vector Retrieval Engine (hat_rag/src/multi_approach.py)</text>'''
    
    paths = '''  <path d="M 600 320 C 720 320, 740 180, 760 155" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 600 350 C 720 350, 740 480, 760 505" stroke="#58a6ff" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 400 320 C 280 320, 260 180, 240 155" stroke="#a371f7" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 400 350 C 280 350, 260 480, 240 505" stroke="#f0883e" stroke-width="3" fill="none" opacity="0.9" />'''

    hub = make_hub(370, 295, 260, 85, "#ff79c6", "Multi-Approach Hub", "Comparative Engine", "glow-pink")
    n1 = make_card(630, 115, 310, 80, "#30d158", "Approach 1: HAT-RAG", "Top-Down Logarithmic Traversal", "Complexity: O(k log N) | Nodes: ~25%", "glow-green")
    n2 = make_card(630, 465, 310, 80, "#58a6ff", "Approach 2: Flat Vector RAG", "Global Brute-Force Cosine Scan", "Complexity: O(N) | Nodes: 100% Leafs", "glow-blue")
    n3 = make_card(60, 115, 310, 80, "#a371f7", "Approach 3: Graph-RAG", "Entity Relation Traversal", "Complexity: O(V + E) | Knowledge Graph", "glow-purple")
    n4 = make_card(60, 465, 310, 80, "#f0883e", "Approach 4: RAPTOR-Style RAG", "Collapsed Multi-Level Indexing", "Complexity: O(N_all) | Flattened Scan", "glow-orange")

    return f"{header}\n{defs}\n{title}\n{paths}\n{hub}\n{n1}\n{n2}\n{n3}\n{n4}\n</svg>"

def generate_50percent_svg():
    header = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">'''
    defs = make_defs()
    title = '''  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="800">50% MILESTONE VS 100% PRODUCTION SYSTEM</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Baseline RAG vs CUDA-Accelerated Hierarchical Abstract Tree (HAT-RAG)</text>'''
    
    paths = '''  <path d="M 380 320 C 260 320, 240 180, 220 155" stroke="#e3b341" stroke-width="2.5" stroke-dasharray="6,4" fill="none" opacity="0.85" />
  <path d="M 380 350 C 260 350, 240 480, 220 505" stroke="#e3b341" stroke-width="2.5" stroke-dasharray="6,4" fill="none" opacity="0.85" />
  <path d="M 620 320 C 740 320, 760 180, 780 155" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 620 350 C 740 350, 760 480, 780 505" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />'''

    hub = make_hub(365, 295, 270, 85, "#e3b341", "50% Milestone Hub", "Evolution Snapshot", "glow-yellow")
    n1 = make_card(50, 115, 310, 80, "#e3b341", "50%: Flat Vector Index", "Unstructured 1D Matrix", "Linear Search O(N) | Context Noise", None, is_dashed=True)
    n2 = make_card(50, 465, 310, 80, "#e3b341", "50%: Single Thread CPU", "Basic Python Dot Product", "Latency ~45ms | Uncited Text", None, is_dashed=True)
    n3 = make_card(640, 115, 310, 80, "#30d158", "100%: Hierarchical Tree", "Multi-Tier Abstract Clustering", "Log Search O(k log N) | 70% Pruning", "glow-green")
    n4 = make_card(640, 465, 310, 80, "#30d158", "100%: PyTorch CUDA GPU", "Parallel Tensor Matrix Engine", "Latency ~8.4ms (5.38x) | Citation Tags", "glow-green")

    return f"{header}\n{defs}\n{title}\n{paths}\n{hub}\n{n1}\n{n2}\n{n3}\n{n4}\n</svg>"

if __name__ == "__main__":
    m = generate_methodology_svg()
    a = generate_4_algorithms_svg()
    p = generate_50percent_svg()

    save_svg("docs/images/rag_master_methodology_mindmap.svg", m)
    save_svg("docs/images/rag_4_algorithms_mindmap.svg", a)
    save_svg("docs/images/rag_50percent_architecture_mindmap.svg", p)
