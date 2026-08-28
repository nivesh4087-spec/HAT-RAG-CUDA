import pytest
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree

def test_document_processor():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
    docs = {"doc1": "Hello world testing document processor chunking functionality."}
    chunks = processor.process_documents(docs)
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == "doc1"

def test_tree_builder():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
    docs = {
        "doc1": "First document text for testing tree generation.",
        "doc2": "Second document text for testing tree generation."
    }
    chunks = processor.process_documents(docs)
    tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
    roots = tree.build_tree(chunks)
    
    assert len(roots) > 0
    assert len(tree.nodes) > len(chunks)
