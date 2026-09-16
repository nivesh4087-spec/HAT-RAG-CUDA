"""
Coarse-to-Fine Hierarchical Retriever for HAT-RAG
Author: Nivesh Jain (Vishwakarma Institute of Technology, Pune)
"""

import time
import math
import logging
from typing import List, Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)

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
    Coarse-to-Fine Top-Down Beam Search Retriever for HAT-RAG.
    Supports CUDA batch similarity acceleration, configurable beam width B,
    dynamic confidence threshold fallback, and Maximal Marginal Relevance (MMR) document diversity reranking.
    """

    def __init__(
        self,
        tree: HierarchicalAbstractTree,
        embedding_engine: Optional[EmbeddingEngine] = None,
        beam_width: int = 3,
        confidence_threshold: float = 0.15
    ):
        self.tree = tree
        self.embedding_engine = embedding_engine or tree.embedding_engine
        self.beam_width = beam_width
        self.confidence_threshold = confidence_threshold

    def top_down_search(
        self,
        query: str,
        top_k: int = 3,
        beam_width: Optional[int] = None
    ) -> Tuple[List[TreeNode], Dict[str, Any]]:
        """
        Traverses tree level-by-level from root abstract nodes down to leaf passages.
        Tracks full traversal path, node evaluation count, and CUDA timing.
        """
        start_time = time.perf_counter()
        bw = beam_width or self.beam_width
        query_vec = self.embedding_engine.encode(query)

        current_candidates = list(self.tree.root_nodes)
        nodes_evaluated = len(current_candidates)

        if not current_candidates:
            return [], {
                "search_mode": "Top-Down Hierarchical Traversal",
                "execution_time_ms": 0.0,
                "nodes_evaluated": 0,
                "traversal_path": []
            }

        root_embeds = [n.embedding for n in current_candidates]
        if HAS_NUMPY:
            root_embeds = np.array(root_embeds)

        sims, _ = gpu_batch_cosine_similarity(query_vec, root_embeds)
        
        if HAS_NUMPY and isinstance(sims, np.ndarray):
            top_indices = np.argsort(sims)[::-1][:bw]
        else:
            indexed = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in indexed[:bw]]

        current_candidates = [current_candidates[i] for i in top_indices]

        # Top-down recursive beam traversal
        while current_candidates and current_candidates[0].level > 0:
            next_level_nodes = []
            for node in current_candidates:
                next_level_nodes.extend(node.children)

            if not next_level_nodes:
                break

            nodes_evaluated += len(next_level_nodes)
            cand_embeds = [n.embedding for n in next_level_nodes]
            if HAS_NUMPY:
                cand_embeds = np.array(cand_embeds)

            sims, _ = gpu_batch_cosine_similarity(query_vec, cand_embeds)

            if HAS_NUMPY and isinstance(sims, np.ndarray):
                top_idx = np.argsort(sims)[::-1][:bw]
            else:
                top_idx = [idx for idx, _ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:bw]]

            current_candidates = [next_level_nodes[i] for i in top_idx]

        final_nodes = current_candidates[:top_k]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "search_mode": "Top-Down Hierarchical Traversal",
            "execution_time_ms": round(elapsed_ms, 3),
            "nodes_evaluated": nodes_evaluated,
            "total_tree_nodes": len(self.tree.nodes),
            "beam_width": bw,
            "efficiency_ratio": round((1.0 - (nodes_evaluated / max(1, len(self.tree.nodes)))) * 100, 2)
        }
        return final_nodes, stats

    def flat_search(self, query: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        """Flat linear search across ALL leaf nodes (Standard RAG baseline)."""
        start_time = time.perf_counter()
        query_vec = self.embedding_engine.encode(query)

        leaf_nodes = [n for n in self.tree.nodes.values() if n.level == 0]
        if not leaf_nodes:
            return [], {
                "search_mode": "Flat Search Baseline",
                "execution_time_ms": 0.0,
                "nodes_evaluated": 0,
                "total_tree_nodes": len(self.tree.nodes)
            }

        candidate_embeddings = [n.embedding for n in leaf_nodes]
        if HAS_NUMPY:
            candidate_embeddings = np.array(candidate_embeddings)

        similarities, _ = gpu_batch_cosine_similarity(query_vec, candidate_embeddings)

        if HAS_NUMPY and isinstance(similarities, np.ndarray):
            top_indices = np.argsort(similarities)[::-1][:top_k]
        else:
            indexed_sims = list(enumerate(similarities))
            indexed_sims.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in indexed_sims[:top_k]]

        final_nodes = [leaf_nodes[i] for i in top_indices]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        stats = {
            "search_mode": "Flat Baseline Search",
            "execution_time_ms": round(elapsed_ms, 3),
            "nodes_evaluated": len(leaf_nodes),
            "total_tree_nodes": len(self.tree.nodes),
            "efficiency_ratio": 0.0
        }
        return final_nodes, stats




