"""
Custom Test Runner for HAT-RAG using standard unittest framework
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hat_rag.src.cuda_utils import check_cuda_availability, gpu_batch_cosine_similarity, benchmark_cuda_vs_cpu
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.embeddings import EmbeddingEngine
from hat_rag.src.summarizer import AbstractSummarizer
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.retriever import HierarchicalRetriever
from hat_rag.src.generator import HATGenerator
from hat_rag.src.evaluator import RAGEvaluator
from hat_rag.src.multi_approach import MultiApproachEngine

class TestHATRAG(unittest.TestCase):

    def test_cuda_utils(self):
        status = check_cuda_availability()
        self.assertIsInstance(status, dict)
        self.assertIn("cuda_available", status)
        
        q = [1.0, 0.0, 0.0]
        docs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        sims, elapsed = gpu_batch_cosine_similarity(q, docs)
        self.assertEqual(len(sims), 2)
        self.assertGreater(sims[0], sims[1])
        
        bench = benchmark_cuda_vs_cpu(vector_count=50, dim=16)
        self.assertEqual(bench["vector_count"], 50)

    def test_embeddings_and_summarizer(self):
        engine = EmbeddingEngine(dim=32)
        vec = engine.encode("Sample test sentence")
        self.assertIsNotNone(vec)
        
        summarizer = AbstractSummarizer()
        summary = summarizer.summarize_cluster(["Passage one text.", "Passage two text."])
        self.assertTrue(summary.startswith("ABSTRACT SUMMARY:"))

    def test_tree_and_retriever(self):
        processor = DocumentProcessor(chunk_size=15, chunk_overlap=3)
        docs = {
            "doc1": "Airflow optimization reduces hallway temperatures.",
            "doc2": "Regular motor torque inspection prevents overheating."
        }
        chunks = processor.process_documents(docs)
        self.assertGreater(len(chunks), 0)
        
        tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
        roots = tree.build_tree(chunks)
        self.assertGreater(len(roots), 0)
        
        retriever = HierarchicalRetriever(tree)
        nodes, stats = retriever.top_down_search("motor inspection", top_k=1)
        self.assertGreater(len(nodes), 0)
        
        generator = HATGenerator()
        resp = generator.generate_response("motor inspection", nodes, search_stats=stats)
        self.assertIn("answer", resp)
        self.assertGreater(len(resp["citations"]), 0)

    def test_multi_approach_engine(self):
        processor = DocumentProcessor(chunk_size=10, chunk_overlap=2)
        docs = {"doc1": "Document text for testing multi-approach search engine."}
        chunks = processor.process_documents(docs)
        tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
        tree.build_tree(chunks)
        
        multi_engine = MultiApproachEngine(tree)
        res = multi_engine.compare_all("test query")
        self.assertEqual(len(res["approaches"]), 4)
        self.assertIn("Approach 1", res["approaches"][0]["stats"]["approach"])
        self.assertIn("Approach 2", res["approaches"][1]["stats"]["approach"])
        self.assertIn("Approach 3", res["approaches"][2]["stats"]["approach"])
        self.assertIn("Approach 4", res["approaches"][3]["stats"]["approach"])



if __name__ == "__main__":
    unittest.main()
