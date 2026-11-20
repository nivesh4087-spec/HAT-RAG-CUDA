"""
MultiHop-RAG: the large, real-world benchmark.

Tang & Yang, "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for
Multi-Hop Queries" (COLM 2024), https://arxiv.org/abs/2401.15391
Data: https://huggingface.co/datasets/yixuantt/MultiHopRAG (ODC-BY licence)

    609 news articles from 49 publishers (TechCrunch, The Verge, Fortune, BBC,
    ...), 1.06M words. At this repository's chunk size that is ~21,000 leaf
    passages -- 13x the largest synthetic corpus and ~900x the 10-K corpus.
    2,556 questions; 2,255 have evidence spread over 2-4 different articles
    (inference / comparison / temporal), 301 are unanswerable (null_query)
    and are left out of retrieval scoring.

Every chunk carries a short header "[publisher, Mon YYYY] title." so the
article metadata the questions refer to ("...as reported by TechCrunch in
October 2023") is visible to every retriever equally. Gold items use the same
format as finance_queries.json, with a "fact" (the evidence sentence the
dataset labels) instead of key terms:
    {"doc": doc_id, "fact": "..."}
"""

import json
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "multihop_rag"
RAW_DIR = DATA_DIR / "raw"
TREE_DIR = DATA_DIR / "trees"
HF_BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/"
FILES = ["corpus.json", "MultiHopRAG.json"]

CHUNK_SIZE = 60
CHUNK_OVERLAP = 10


def download(force: bool = False) -> Path:
    """Fetch the two dataset files (~12 MB) into data/multihop_rag/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = RAW_DIR / name
        if path.exists() and path.stat().st_size > 1_000_000 and not force:
            continue
        print(f"[download] {HF_BASE}{name}")
        req = urllib.request.Request(HF_BASE + name, headers={"User-Agent": "hat-rag-benchmark"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    return RAW_DIR


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:16] or "src"


def _month(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%b %Y")
    except (ValueError, AttributeError):
        return ""


def load_articles() -> List[Dict[str, Any]]:
    download()
    return json.loads((RAW_DIR / "corpus.json").read_text(encoding="utf-8"))


def article_meta(articles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """doc_id -> {title, source, published_at, month, url, category}."""
    meta = {}
    for i, a in enumerate(articles):
        doc_id = f"mh{i:03d}_{_slug(a['source'])}"
        meta[doc_id] = {"title": a["title"].strip(), "source": a["source"], "published_at": a["published_at"],
                        "month": _month(a["published_at"]), "url": a["url"], "category": a["category"]}
    return meta


def header(m: Dict[str, Any]) -> str:
    return f"[{m['source']}, {m['month']}] {m['title']}."


def build_chunks(articles: List[Dict[str, Any]], doc_ids: Optional[List[str]] = None,
                 chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Algorithm 1 over the selected articles, then the metadata header per chunk."""
    from algo1_document_chunking import DocumentChunker

    meta = article_meta(articles)
    by_id = {d: a for d, a in zip(meta, articles)}
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Dict[str, Any]] = []
    for doc_id in (doc_ids or list(meta)):
        head = header(meta[doc_id])
        for c in chunker.chunk_document(by_id[doc_id]["body"], doc_id):
            c["text"] = f"{head} {c['text']}"
            c["section"] = ""                        # no headings in news bodies
            chunks.append(c)
    return chunks


def load_queries(articles: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """(queries, doc_aliases). Null queries are dropped; split = dev for every
    fifth question (used for any design decision), test for the rest."""
    meta = article_meta(articles)
    by_url = {m["url"]: d for d, m in meta.items()}
    by_title = {(m["title"], m["source"]): d for d, m in meta.items()}
    raw = json.loads((RAW_DIR / "MultiHopRAG.json").read_text(encoding="utf-8"))

    queries = []
    for n, q in enumerate(raw):
        if not q["evidence_list"]:
            continue
        gold = []
        for ev in q["evidence_list"]:
            doc = by_url.get(ev.get("url")) or by_title.get((ev["title"].strip(), ev["source"]))
            if doc is None:
                break
            gold.append({"doc": doc, "fact": ev["fact"]})
        else:
            queries.append({
                "id": f"M{n:04d}",
                "type": q["question_type"].replace("_query", ""),
                "query": q["query"],
                "answer": q["answer"],
                "gold": gold,
                "split": "dev" if n % 5 == 0 else "test",
            })
    aliases = {d: [m["title"]] for d, m in meta.items()}
    return queries, aliases


def stats() -> Dict[str, Any]:
    articles = load_articles()
    queries, _ = load_queries(articles)
    words = sum(len(a["body"].split()) for a in articles)
    return {"articles": len(articles), "words": words,
            "sources": len({a["source"] for a in articles}),
            "queries": len(queries),
            "dev": sum(q["split"] == "dev" for q in queries),
            "test": sum(q["split"] == "test" for q in queries)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    print(json.dumps(stats(), indent=2))
    arts = load_articles()
    ch = build_chunks(arts)
    print(f"chunks: {len(ch)} | mean words {sum(len(c['text'].split()) for c in ch) / len(ch):.1f}")
    print(ch[3]["text"][:300])
