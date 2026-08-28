import pytest
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.retriever import HierarchicalRetriever
from hat_rag.src.generator import HATGenerator

def test_retriever_and_generator():
    processor = DocumentProcessor(chunk_size=15, chunk_overlap=3)
    docs = {"doc1": "Airflow optimization reduces hallway temperatures and improves energy output."}
    chunks = processor.process_documents(docs)
    
    tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
    tree.build_tree(chunks)
    
    retriever = HierarchicalRetriever(tree)
    nodes, stats = retriever.top_down_search("airflow optimization", top_k=1)
    assert len(nodes) > 0
    assert "execution_time_ms" in stats
    
    generator = HATGenerator()
    resp = generator.generate_response("airflow optimization", nodes, search_stats=stats)
    assert "answer" in resp
    assert len(resp["citations"]) > 0
