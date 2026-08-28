import time
import logging
from typing import Dict, List, Any

from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.retriever import HierarchicalRetriever

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluator for benchmarking HAT-RAG against Flat Baseline RAG.
    """
    
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.retriever = HierarchicalRetriever(tree)

    def evaluate_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Runs comparative evaluation of Top-Down Hierarchical Search vs Flat Search."""
        # 1. Top-Down Search
        hat_nodes, hat_stats = self.retriever.top_down_search(query, top_k=top_k)
        
        # 2. Flat Search
        flat_nodes, flat_stats = self.retriever.flat_search(query, top_k=top_k)
        
        # 3. Calculate Speedup & Node Reduction
        hat_time = max(0.001, hat_stats["execution_time_ms"])
        flat_time = max(0.001, flat_stats["execution_time_ms"])
        speedup_factor = round(flat_time / hat_time, 2) if flat_time >= hat_time else round(hat_time / flat_time, 2)
        
        nodes_saved = flat_stats["nodes_evaluated"] - hat_stats["nodes_evaluated"]
        reduction_percentage = round((nodes_saved / max(1, flat_stats["nodes_evaluated"])) * 100, 2)
        
        return {
            "query": query,
            "top_k": top_k,
            "hat_rag": hat_stats,
            "flat_rag": flat_stats,
            "comparison": {
                "speedup_factor": speedup_factor,
                "node_eval_reduction_percent": max(0.0, reduction_percentage),
                "nodes_saved": max(0, nodes_saved),
                "recommended_architecture": "HAT-RAG (Top-Down)" if hat_stats["nodes_evaluated"] <= flat_stats["nodes_evaluated"] else "Flat Baseline"
            }
        }

    def run_benchmark_suite(self, sample_queries: List[str]) -> Dict[str, Any]:
        """Runs a complete test suite across multiple queries."""
        results = [self.evaluate_query(q) for q in sample_queries]
        
        avg_hat_time = sum(r["hat_rag"]["execution_time_ms"] for r in results) / len(results)
        avg_flat_time = sum(r["flat_rag"]["execution_time_ms"] for r in results) / len(results)
        avg_reduction = sum(r["comparison"]["node_eval_reduction_percent"] for r in results) / len(results)
        
        return {
            "total_queries_tested": len(results),
            "avg_hat_time_ms": round(avg_hat_time, 3),
            "avg_flat_time_ms": round(avg_flat_time, 3),
            "avg_node_reduction_percent": round(avg_reduction, 2),
            "detailed_results": results
        }
