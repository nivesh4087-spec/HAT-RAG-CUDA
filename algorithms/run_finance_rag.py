"""
OFFLINE / KNOWLEDGE CONSTRUCTION PIPELINE -- CORPORATE FINANCE CORPUS

Document corpus -> Algorithm 1 (chunking) -> Algorithm 3 (HAT construction)
-> serialized Hierarchical Abstract Tree H.

Indexing only: no query-time retrieval in this pipeline. Runs on CPU; --device
cuda is accepted but not required.

    python algorithms/run_finance_rag.py                    # abstractive LLM abstracts
    python algorithms/run_finance_rag.py --summarizer extractive   # fast, no LLM
    python algorithms/run_finance_rag.py --alpha 0.65
"""

import argparse
import glob
import os
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from algo1_document_chunking import DocumentChunker
from algo3_hierarchical_abstract_tree import (
    DEFAULT_MODEL,
    DEFAULT_SUMMARIZER_MODEL,
    DenseEmbeddingModel,
    HierarchicalAbstractTreeBuilder,
    describe_device,
)


def load_finance_corpus():
    reports_dir = current_dir.parent / "finance_data" / "reports"
    corpus = {}
    for fpath in sorted(glob.glob(str(reports_dir / "*.txt"))):
        with open(fpath, "r", encoding="utf-8") as f:
            corpus[os.path.basename(fpath)] = f.read()
    return corpus


def run_finance_pipeline(args):
    print("=" * 100)
    print(" HAT-RAG OFFLINE KNOWLEDGE CONSTRUCTION -- FINANCIAL & ACCOUNTING CORPUS")
    print(" Algorithm 1 (document chunking) -> Algorithm 3 (hierarchical abstract tree)")
    print("=" * 100)

    corpus = load_finance_corpus()
    if not corpus:
        print("[Error] No financial report files found in finance_data/reports/")
        return None

    print(f"\n[Document Corpus] {len(corpus)} corporate 10-K reports:")
    for fname, text in corpus.items():
        print(f"  - {fname:<45} ({len(text.split())} words)")

    # ---------------------------------------------------------------- PHASE 1
    print("\n" + "-" * 100)
    print("[PHASE 1] ALGORITHM 1: DOCUMENT CHUNKING -> chunk collection C")
    chunker = DocumentChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    all_chunks = []
    for doc_id, text in corpus.items():
        doc_chunks = chunker.chunk_document(text, doc_id)
        all_chunks.extend(doc_chunks)
        print(f"  > {doc_id:<45} -> {len(doc_chunks):>3} chunks")

    avg_tokens = sum(c["token_count"] for c in all_chunks) / max(1, len(all_chunks))
    print(f"\n  |C| = {len(all_chunks)} chunks | avg {avg_tokens:.1f} words/chunk | "
          f"target {chunker.chunk_size} overlap {chunker.chunk_overlap}")
    sample = all_chunks[0]
    print(f"  Sample: [{sample['chunk_id']}] section='{sample['section']}' hash={sample['hash']}")

    # ---------------------------------------------------------------- PHASE 2
    print("\n" + "-" * 100)
    print("[PHASE 2] DENSE EMBEDDING OF C -> matrix E (leaf vectors for Algorithm 3)")
    embedder = DenseEmbeddingModel(model_name=args.model, device=args.device)
    print(f"  {embedder.banner()}")
    E = embedder.encode_chunks(all_chunks)
    print(f"  E shape = {E.shape} | {embedder.stats.throughput:.1f} texts/sec on "
          f"{describe_device(embedder.device)} | load {embedder.stats.load_seconds:.1f}s")

    # ---------------------------------------------------------------- PHASE 3
    print("\n" + "-" * 100)
    print("[PHASE 3] ALGORITHM 3: HIERARCHICAL ABSTRACT TREE CONSTRUCTION -> H")
    builder = HierarchicalAbstractTreeBuilder(
        embedder=embedder,                      # cached leaf embeddings are reused
        max_levels=args.levels,
        target_children=args.children,
        alpha=args.alpha,
        max_cross_links=args.max_cross_links,
        summarizer_mode=args.summarizer,
        summarizer_model=args.summarizer_model,
        device=args.device,
    )
    print(f"  Summarizer: {builder.summarizer.banner()}")
    if builder.summarizer.mode == "llm":
        print("  (abstractive summarisation on CPU takes a few seconds per abstract node)")
    builder.build_tree(all_chunks)

    builder.display_tree(max_leaves_per_parent=args.show_leaves)

    json_path = current_dir / "hat_finance_tree_structure.json"
    builder.save_tree_json(str(json_path), include_embeddings=not args.no_embeddings)
    size_kb = json_path.stat().st_size / 1024
    print(f"\n[Persistence] H serialized to {json_path.name} ({size_kb:.0f} KB)")

    verify = HierarchicalAbstractTreeBuilder(embedder=embedder, summarizer_mode="extractive", verbose=False)
    verify.load_tree_json(str(json_path))
    links = sum(len(n.cross_links) for n in verify.nodes.values())
    print(f"[Persistence] Round trip verified: {len(verify.nodes)} nodes, "
          f"{len(verify.root_nodes)} root(s), {links} alpha edges")

    builder.print_metrics()

    # cross-document evidence: the property the HAT exists for
    cross_doc_abstracts = [
        n for n in builder.nodes.values()
        if n.level > 0 and n.metadata.get("is_cross_document")
    ]
    print("\n" + "=" * 100)
    print(" CROSS-DOCUMENT STRUCTURE")
    print("=" * 100)
    print(f"  Abstract nodes spanning >1 company : {len(cross_doc_abstracts)}")
    for n in sorted(cross_doc_abstracts, key=lambda x: -len(x.source_docs))[:5]:
        tickers = ", ".join(d.split("_")[0] for d in n.source_docs)
        print(f"    {n.node_id:<10} L{n.level} | {len(n.source_docs)} companies ({tickers})")
        print(f"      topics: {', '.join(n.metadata.get('keywords', [])[:6])}")
    print(f"  Explicit alpha cross-child edges   : {builder.metrics.cross_links} "
          f"({builder.metrics.cross_doc_links} between different companies)")
    for n in builder.nodes.values():
        for link in n.cross_links[:1]:
            if link["relation"] == "cross_document":
                peer = builder.nodes[link["target_id"]]
                print(f"    {n.node_id[:34]:<34} ~~{link['score']}~~> {peer.node_id[:34]}")
                break
    print("=" * 100)
    return builder


def _cli():
    p = argparse.ArgumentParser(description="HAT-RAG offline construction on the finance corpus")
    p.add_argument("--chunk-size", type=int, default=60, help="target words per chunk")
    p.add_argument("--chunk-overlap", type=int, default=15, help="overlap words between chunks")
    p.add_argument("--levels", type=int, default=3, help="max abstract levels above the leaves")
    p.add_argument("--children", type=int, default=4, help="target children per abstract node")
    p.add_argument("--alpha", default="auto", help="cross-child cosine threshold: float or 'auto'")
    p.add_argument("--max-cross-links", type=int, default=3, help="alpha edges kept per node")
    p.add_argument("--summarizer", choices=["auto", "llm", "extractive"], default="auto")
    p.add_argument("--summarizer-model", default=DEFAULT_SUMMARIZER_MODEL)
    p.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers embedding model")
    p.add_argument("--device", default="auto", help="cpu is the default; cuda used only if present")
    p.add_argument("--show-leaves", type=int, default=2, help="leaf chunks printed per parent")
    p.add_argument("--no-embeddings", action="store_true", help="omit vectors from the JSON export")
    args = p.parse_args()
    if args.alpha != "auto":
        args.alpha = float(args.alpha)
    return args


if __name__ == "__main__":
    run_finance_pipeline(_cli())
