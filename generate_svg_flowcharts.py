import os

def create_svg_mindmap_methodology(filepath):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="glow-red" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-green" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <linearGradient id="central-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#21121d" />
      <stop offset="100%" stop-color="#161b22" />
    </linearGradient>
    <linearGradient id="card-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#161b22" />
      <stop offset="100%" stop-color="#0d1117" />
    </linearGradient>
  </defs>
  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="700">HAT-RAG SYSTEM METHODOLOGY ARCHITECTURE</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Hierarchical Abstract Tree with PyTorch CUDA GPU Vector Acceleration</text>

  <path d="M 500 290 C 500 200, 500 170, 500 145" stroke="#ff4b6e" stroke-width="2.5" fill="none" opacity="0.85" />
  <path d="M 610 320 C 720 320, 750 200, 770 170" stroke="#30d158" stroke-width="2.5" fill="none" opacity="0.85" />
  <path d="M 610 340 C 730 340, 750 450, 780 475" stroke="#f0883e" stroke-width="2.5" fill="none" opacity="0.85" />
  <path d="M 390 320 C 270 320, 240 200, 220 170" stroke="#a371f7" stroke-width="2.5" fill="none" opacity="0.85" />
  <path d="M 390 340 C 270 340, 250 450, 220 475" stroke="#58a6ff" stroke-width="2.5" fill="none" opacity="0.85" />
  <path d="M 500 370 C 500 460, 500 520, 500 555" stroke="#e3b341" stroke-width="2.5" fill="none" opacity="0.85" />

  <g transform="translate(375, 290)">
    <rect x="0" y="0" width="250" height="80" rx="20" ry="20" fill="url(#central-bg)" stroke="#ff4b6e" stroke-width="2.5" filter="url(#glow-red)" />
    <text x="125" y="45" text-anchor="middle" fill="#ff79c6" font-size="20" font-weight="800">HAT-RAG Core</text>
    <text x="125" y="65" text-anchor="middle" fill="#8b949e" font-size="12">Hierarchical Engine</text>
  </g>

  <g transform="translate(370, 95)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#ff4b6e" stroke-width="2" filter="url(#glow-red)" />
    <circle cx="240" cy="18" r="4" fill="#ff4b6e" />
    <text x="20" y="32" fill="#f0f6fc" font-size="15" font-weight="700">Phase 1: Data Ingestion</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">PDF/Text Parser &amp; Overlapping Chunker</text>
  </g>

  <g transform="translate(650, 140)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#30d158" stroke-width="2" filter="url(#glow-green)" />
    <circle cx="240" cy="18" r="4" fill="#30d158" />
    <text x="20" y="32" fill="#30d158" font-size="15" font-weight="700">Phase 4: PyTorch CUDA</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">GPU Tensor Batch Similarity Engine</text>
  </g>

  <g transform="translate(660, 445)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#f0883e" stroke-width="2" filter="url(#glow-orange)" />

def create_svg_mindmap_4_algorithms(filepath):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="glow-pink" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-green" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-blue" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-orange" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-purple" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <linearGradient id="hub-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2d1527" />
      <stop offset="100%" stop-color="#161b22" />
    </linearGradient>
    <linearGradient id="card-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#161b22" />
      <stop offset="100%" stop-color="#0d1117" />
    </linearGradient>
  </defs>

  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="700">4 RAG ALGORITHMIC APPROACHES ARCHITECTURE</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Comparative Multi-Approach Vector Retrieval Engine (hat_rag/src/multi_approach.py)</text>

  <path d="M 600 320 C 720 320, 740 180, 760 150" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 600 350 C 720 350, 740 480, 760 510" stroke="#58a6ff" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 400 320 C 280 320, 260 180, 240 150" stroke="#a371f7" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 400 350 C 280 350, 260 480, 240 510" stroke="#f0883e" stroke-width="3" fill="none" opacity="0.9" />

  <g transform="translate(370, 295)">
    <rect x="0" y="0" width="260" height="85" rx="22" ry="22" fill="url(#hub-bg)" stroke="#ff79c6" stroke-width="2.5" filter="url(#glow-pink)" />
    <text x="130" y="46" text-anchor="middle" fill="#ff79c6" font-size="19" font-weight="800">Multi-Approach</text>
    <text x="130" y="65" text-anchor="middle" fill="#8b949e" font-size="12">RAG Engine Hub</text>
  </g>

  <g transform="translate(630, 115)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#30d158" stroke-width="2.5" filter="url(#glow-green)" />
    <circle cx="285" cy="22" r="5" fill="#30d158" />
    <text x="22" y="34" fill="#30d158" font-size="16" font-weight="800">Approach 1: HAT-RAG</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Top-Down Logarithmic Traversal</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Complexity: O(k log N) | Nodes Evaluated: ~25%</text>
  </g>

  <g transform="translate(630, 465)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#58a6ff" stroke-width="2.5" filter="url(#glow-blue)" />
    <circle cx="285" cy="22" r="5" fill="#58a6ff" />
    <text x="22" y="34" fill="#58a6ff" font-size="16" font-weight="800">Approach 2: Flat Vector RAG</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Global Brute-Force Cosine Scan</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Complexity: O(N) | Nodes Evaluated: 100% Leafs</text>
  </g>

  <g transform="translate(60, 115)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#a371f7" stroke-width="2.5" filter="url(#glow-purple)" />
    <circle cx="285" cy="22" r="5" fill="#a371f7" />
    <text x="22" y="34" fill="#a371f7" font-size="16" font-weight="800">Approach 3: Graph-RAG</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Entity Relation Multi-Hop Traversal</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Complexity: O(V + E) | Knowledge Graph Hop</text>
  </g>

  <g transform="translate(60, 465)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#f0883e" stroke-width="2.5" filter="url(#glow-orange)" />
    <circle cx="285" cy="22" r="5" fill="#f0883e" />
    <text x="22" y="34" fill="#f0883e" font-size="16" font-weight="800">Approach 4: RAPTOR-Style RAG</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Collapsed Multi-Level Indexing</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Complexity: O(N_all) | All Tiers Flattened Scan</text>
  </g>
</svg>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved {filepath}")

def create_svg_50_percent_mindmap(filepath):
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 650" width="100%" height="100%" style="background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
  <defs>
    <filter id="glow-yellow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <filter id="glow-green" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
    <linearGradient id="hub-50-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#2d2215" />
      <stop offset="100%" stop-color="#161b22" />
    </linearGradient>
    <linearGradient id="card-bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#161b22" />
      <stop offset="100%" stop-color="#0d1117" />
    </linearGradient>
  </defs>

  <text x="500" y="45" text-anchor="middle" fill="#f0f6fc" font-size="22" font-weight="700">50% MILESTONE VS 100% PRODUCTION SYSTEM</text>
  <text x="500" y="70" text-anchor="middle" fill="#8b949e" font-size="13">Baseline RAG vs CUDA-Accelerated Hierarchical Abstract Tree (HAT-RAG)</text>

  <path d="M 380 320 C 260 320, 240 180, 220 150" stroke="#e3b341" stroke-width="2.5" stroke-dasharray="6,4" fill="none" stroke-dashoffset="0" opacity="0.85" />
  <path d="M 380 350 C 260 350, 240 480, 220 510" stroke="#e3b341" stroke-width="2.5" stroke-dasharray="6,4" fill="none" opacity="0.85" />
  <path d="M 620 320 C 740 320, 760 180, 780 150" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />
  <path d="M 620 350 C 740 350, 760 480, 780 510" stroke="#30d158" stroke-width="3" fill="none" opacity="0.9" />

  <g transform="translate(365, 295)">
    <rect x="0" y="0" width="270" height="85" rx="22" ry="22" fill="url(#hub-50-bg)" stroke="#e3b341" stroke-width="2.5" filter="url(#glow-yellow)" />
    <text x="135" y="45" text-anchor="middle" fill="#e3b341" font-size="18" font-weight="800">50% Milestone</text>
    <text x="135" y="65" text-anchor="middle" fill="#8b949e" font-size="12">Evolution Snapshot</text>
  </g>

  <g transform="translate(50, 115)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#e3b341" stroke-width="2" stroke-dasharray="4,3" />
    <circle cx="285" cy="22" r="4" fill="#e3b341" />
    <text x="22" y="34" fill="#e3b341" font-size="15" font-weight="700">50%: Flat Vector Index</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Unstructured 1D Chunk Matrix</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Linear Search O(N) | High Context Noise</text>
  </g>

  <g transform="translate(50, 465)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#e3b341" stroke-width="2" stroke-dasharray="4,3" />
    <circle cx="285" cy="22" r="4" fill="#e3b341" />
    <text x="22" y="34" fill="#e3b341" font-size="15" font-weight="700">50%: Single Thread CPU</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Basic Python Vector Dot Product</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Latency: ~45ms | Uncited Text Generation</text>
  </g>

  <g transform="translate(640, 115)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#30d158" stroke-width="2.5" filter="url(#glow-green)" />
    <circle cx="285" cy="22" r="5" fill="#30d158" />
    <text x="22" y="34" fill="#30d158" font-size="16" font-weight="800">100%: Hierarchical Tree</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Multi-Tier Abstract Clustering</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Sub-linear Search O(k log N) | 70% Node Pruning</text>
  </g>

  <g transform="translate(640, 465)">
    <rect x="0" y="0" width="310" height="80" rx="18" ry="18" fill="url(#card-bg)" stroke="#30d158" stroke-width="2.5" filter="url(#glow-green)" />
    <circle cx="285" cy="22" r="5" fill="#30d158" />
    <text x="22" y="34" fill="#30d158" font-size="16" font-weight="800">100%: PyTorch CUDA GPU</text>
    <text x="22" y="52" fill="#f0f6fc" font-size="12" font-weight="600">Parallel Tensor Batch Matrix Engine</text>
    <text x="22" y="68" fill="#8b949e" font-size="11">Latency: ~8.4ms (5.38x Speedup) | Citation Tags</text>
  </g>
</svg>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved {filepath}")

if __name__ == "__main__":
    os.makedirs("docs/images", exist_ok=True)
    
    create_svg_mindmap_methodology("docs/images/rag_master_methodology_mindmap.svg")
    create_svg_mindmap_4_algorithms("docs/images/rag_4_algorithms_mindmap.svg")
    create_svg_50_percent_mindmap("docs/images/rag_50percent_architecture_mindmap.svg")


    <circle cx="240" cy="18" r="4" fill="#f0883e" />
    <text x="20" y="32" fill="#f0883e" font-size="15" font-weight="700">Phase 5: Top-Down Search</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">Logarithmic Traversal O(k log N)</text>
  </g>

  <g transform="translate(90, 140)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#a371f7" stroke-width="2" filter="url(#glow-purple)" />
    <circle cx="240" cy="18" r="4" fill="#a371f7" />
    <text x="20" y="32" fill="#a371f7" font-size="15" font-weight="700">Phase 2 &amp; 3: HAT Index</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">K-Means Vector Clustering &amp; Summaries</text>
  </g>

  <g transform="translate(80, 445)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#58a6ff" stroke-width="2" filter="url(#glow-blue)" />
    <circle cx="240" cy="18" r="4" fill="#58a6ff" />
    <text x="20" y="32" fill="#58a6ff" font-size="15" font-weight="700">Phase 6: LLM Generator</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">Context Assembly &amp; Citation Links [1]</text>
  </g>

  <g transform="translate(370, 555)">
    <rect x="0" y="0" width="260" height="60" rx="16" ry="16" fill="url(#card-bg)" stroke="#e3b341" stroke-width="2" opacity="0.95" />
    <circle cx="240" cy="18" r="4" fill="#e3b341" />
    <text x="20" y="32" fill="#e3b341" font-size="15" font-weight="700">Phase 7: Benchmarking</text>
    <text x="20" y="48" fill="#8b949e" font-size="11">Latency Tracking &amp; Speedup Metrics</text>
  </g>
</svg>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved {filepath}")
