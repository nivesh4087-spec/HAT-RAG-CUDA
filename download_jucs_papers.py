import urllib.request
import os

os.makedirs('docs', exist_ok=True)
os.makedirs('papers', exist_ok=True)

papers_to_download = [
    {
        'id': '207845',
        'title': 'Optimization of Parallel Number Theoretic Transform Algorithms for Multi-Core Digital Signal Processors',
        'pdf_url': 'https://lib.jucs.org/article/207845/download/pdf/',
        'file_doc': 'docs/JUCS_2026_Parallel_Optimization_NTT.pdf',
        'file_paper': 'papers/JUCS_2026_Parallel_Optimization_NTT.pdf'
    },
    {
        'id': '151691',
        'title': 'Using Generative Artificial Intelligence to Improve User Engagement in Content Marketing',
        'pdf_url': 'https://lib.jucs.org/article/151691/download/pdf/',
        'file_doc': 'docs/JUCS_2026_Generative_AI_RAG.pdf',
        'file_paper': 'papers/JUCS_2026_Generative_AI_RAG.pdf'
    }
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for p in papers_to_download:
    print(f"Downloading {p['id']}: {p['title']}...")
    try:
        req = urllib.request.Request(p['pdf_url'], headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
            print(f"  Downloaded {len(data)} bytes")
            with open(p['file_doc'], 'wb') as f:
                f.write(data)
            with open(p['file_paper'], 'wb') as f:
                f.write(data)
    except Exception as e:
        print(f"  Error downloading {p['id']}: {e}")
