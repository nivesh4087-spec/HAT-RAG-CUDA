import time
import math
from typing import List, Dict, Tuple, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from hat_rag.src.cuda_utils import gpu_batch_cosine_similarity
from hat_rag.src.embeddings import EmbeddingEngine
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree, TreeNode

class HierarchicalRetriever:
    """
    Retriever for Hierarchical Abstract Trees (HAT).
    Supports both Top-Down Traversal Search and Flat Search for Benchmarking.
    """
    
    def __init__(self, tree: HierarchicalAbstractTree, embedding_engine: Optional[EmbeddingEngine] = None):
        self.tree = tree
        self.embedding_engine = embedding_engine or tree.embedding_engine

    def top_down_search(self, query: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        """
        Traverses tree from top root abstracts down to leaf nodes based on similarity scoring.
        """
        start_time = time.perf_counter()
        query_vec = self.embedding_engine.encode(query)
        current_candidates = self.tree.root_nodes
        nodes_evaluated = len(current_candidates)
        
        while current_candidates:
            if current_candidates[0].level == 0:
                break
                
            next_level_nodes = []
            for node in current_candidates:
                next_level_nodes.extend(node.children)
                
            if not next_level_nodes:
                break
                
            nodes_evaluated += len(next_level_nodes)
            candidate_embeddings = [n.embedding for n in next_level_nodes]
            if HAS_NUMPY:
                candidate_embeddings = np.array(candidate_embeddings)
                
            similarities, _ = gpu_batch_cosine_similarity(query_vec, candidate_embeddings)
            
            if isinstance(similarities, list):
                indexed_sims = list(enumerate(similarities))
                indexed_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in indexed_sims[:top_k]]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
            current_candidates = [next_level_nodes[i] for i in top_indices]
            
        final_nodes = current_candidates[:top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        stats = {
            "search_mode": "Top-Down Hierarchical Traversal",
            "execution_time_ms": round(elapsed_ms, 3),
            "nodes_evaluated": nodes_evaluated,
            "total_tree_nodes": len(self.tree.nodes),
            "efficiency_ratio": round((1.0 - (nodes_evaluated / max(1, len(self.tree.nodes)))) * 100, 2)
        }
        return final_nodes, stats

    def flat_search(self, query: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        """
        Flat linear search across ALL leaf nodes (Standard RAG baseline).
        """
        start_time = time.perf_counter()
        query_vec = self.embedding_engine.encode(query)
        
        leaf_nodes = [n for n in self.tree.nodes.values() if n.level == 0]
        if not leaf_nodes:
            return [], {"search_mode": "Flat Search Baseline", "execution_time_ms": 0.0, "nodes_evaluated": 0}
            
        candidate_embeddings = [n.embedding for n in leaf_nodes]
        if HAS_NUMPY:
            candidate_embeddings = np.array(candidate_embeddings)
            
        similarities, _ = gpu_batch_cosine_similarity(query_vec, candidate_embeddings)
        
        if isinstance(similarities, list):
            indexed_sims = list(enumerate(similarities))
            indexed_sims.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in indexed_sims[:top_k]]
        else:
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
        final_nodes = [leaf_nodes[i] for i in top_indices]
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        stats = {
            "search_mode": "Flat Baseline Search",
            "execution_time_ms": round(elapsed_ms, 3),
            "nodes_evaluated": len(leaf_nodes),
            "total_tree_nodes": len(self.tree.nodes),
            "efficiency_ratio": 0.0
        }
        return final_nodes, stats



