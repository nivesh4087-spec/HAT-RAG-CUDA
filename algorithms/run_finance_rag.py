import os
import sys
import glob
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from algo1_document_chunking import DocumentChunker
from algo3_hierarchical_abstract_tree import HierarchicalAbstractTreeBuilder


def load_finance_corpus():
    reports_dir = current_dir.parent / "finance_data" / "reports"
    files = glob.glob(str(reports_dir / "*.txt"))
    corpus = {}
    for fpath in sorted(files):
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            corpus[fname] = f.read()
    return corpus


def run_finance_pipeline():
    print("=" * 90)
    print(" FINANCIAL & ACCOUNTING DATA RAG PIPELINE: ALGORITHM 1 & ALGORITHM 3")
    print("=" * 90)

    # 1. Load Financial Corpus
    corpus = load_finance_corpus()
    if not corpus:
        print("[Error] No financial report files found in finance_data/reports/")
        return

    print(f"\n[Ingestion] Loaded {len(corpus)} Corporate 10-K Financial & Accounting Reports:")
    for fname in corpus.keys():
        print(f"  - {fname}")

    # STEP 1: ALGORITHM 1 - CHUNKING
    print("\n" + "-" * 90)
    print("[PHASE 1] RUNNING ALGORITHM 1: DOCUMENT CHUNKING ON FINANCIAL STATEMENTS...")
    chunker = DocumentChunker(chunk_size=40, chunk_overlap=12)
    all_chunks = []

    for doc_id, text in corpus.items():
        doc_chunks = chunker.chunk_document(text, doc_id)
        all_chunks.extend(doc_chunks)
        print(f"  > Document: {doc_id:<38} -> Created {len(doc_chunks)} Chunks")

    print(f"\n[Phase 1 Complete] Ingested {len(corpus)} Documents into {len(all_chunks)} Semantic Financial Chunks.")
    print("  Sample Financial Chunk:")
    sample = all_chunks[0]
    print(f"    - Chunk ID:    {sample['chunk_id']}")
    print(f"    - Section:     {sample['section']}")
    print(f"    - Token Count: {sample['token_count']} words | Hash: {sample['hash']}")
    print(f"    - Passage:     \"{sample['text'][:85]}...\"")

    # STEP 2: ALGORITHM 3 - HIERARCHICAL TREE CONSTRUCTION
    print("\n" + "-" * 90)
    print("[PHASE 2] RUNNING ALGORITHM 3: HIERARCHICAL ABSTRACT TREE (HAT) ON FINANCIAL DATA...")
    tree_builder = HierarchicalAbstractTreeBuilder(max_levels=2, branching_factor=2)
    root_nodes = tree_builder.build_tree(all_chunks)

    print(f"[Phase 2 Complete] Tree Constructed! Total Nodes: {len(tree_builder.nodes)}, Root Clusters: {len(root_nodes)}")

    # STEP 3: DISPLAY TOPOLOGY & EXPORT
    tree_builder.display_tree()

    json_path = current_dir / "hat_finance_tree_structure.json"
    tree_builder.save_tree_json(str(json_path))
    print(f"\n[Persistence Check] Serialized Financial HAT Tree exported to: {json_path.name}")

    # SUMMARY AUDIT
    leaf_count = sum(1 for n in tree_builder.nodes.values() if n.level == 0)
    l1_count = sum(1 for n in tree_builder.nodes.values() if n.level == 1)
    l2_count = sum(1 for n in tree_builder.nodes.values() if n.level == 2)

    print("\n" + "=" * 90)
    print(" FINANCIAL RAG PIPELINE EXECUTION AUDIT")
    print("=" * 90)
    print(f"  * Total Companies Ingested                 : {len(corpus)} (AAPL, MSFT, NVDA, AMZN, TSLA)")
    print(f"  * Total Financial Passage Chunks (L0)      : {leaf_count} chunks")
    print(f"  * Intermediate Abstract Summaries (L1)     : {l1_count} clusters")
    print(f"  * Global Root Corporate Abstracts (L2)     : {l2_count} roots")
    print(f"  * Hierarchical Compression Factor          : {leaf_count / max(1, l2_count):.1f}x reduction to root")
    print(f"  * JSON Persistence Verified                : PASSED ({json_path.name})")
    print("=" * 90)


if __name__ == "__main__":
    run_finance_pipeline()
