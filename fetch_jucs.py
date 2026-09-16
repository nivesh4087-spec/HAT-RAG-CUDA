import urllib.request
import re
from bs4 import BeautifulSoup

candidates = [
    '207845', # Optimization of Parallel Number Theoretic Transform Algorithms for Multi-Core Architecture
    '151691', # Using Generative Artificial Intelligence to Improve User Engagement in Content Management Systems
    '207303', # A Spatio-Temporal Transformer-Based Approach for Network Attack Detection
    '207783'  # Malicious Code Detection Based on AST Multidimensional Features
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for aid in candidates:
    url = f"https://lib.jucs.org/article/{aid}/"
    try:
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=5).read().decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        meta_title = soup.find('meta', {'name': 'DC.title'}) or soup.find('meta', {'property': 'og:title'})
        title = meta_title['content'] if meta_title else ''
        if not title:
            h1 = soup.find('h1')
            title = h1.text.strip() if h1 else 'Unknown Title'
            
        # Extract abstract
        meta_abs = soup.find('meta', {'name': 'DC.description'}) or soup.find('meta', {'name': 'description'})
        abstract = meta_abs['content'] if meta_abs else ''
        
        pdf_url = f"https://lib.jucs.org/article/{aid}/download/pdf/"
        
        print(f"ID: {aid}")
        print(f"Title: {title}")
        print(f"Abstract: {abstract[:150]}...")
        print(f"PDF URL: {pdf_url}\n")
    except Exception as e:
        print(f"Error fetching {aid}: {e}")







