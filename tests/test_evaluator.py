import pytest
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.evaluator import RAGEvaluator

def test_evaluator():
    processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
    docs = {
        "doc1": "Test document one for evaluator benchmark testing.",
        "doc2": "Test document two for evaluator benchmark testing."
    }
    chunks = processor.process_documents(docs)
    tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
    tree.build_tree(chunks)
    
    evaluator = RAGEvaluator(tree)
    res = evaluator.evaluate_query("test query")
    assert "comparison" in res
    assert "speedup_factor" in res["comparison"]
