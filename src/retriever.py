import math
import random
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from hat_rag.src.cuda_utils import gpu_batch_cosine_similarity
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree, TreeNode

class HierarchicalRetriever:
    """
    Retriever for Hierarchical Abstract Trees (HAT).
    Utilizes CUDA GPU computing (with CPU/Native fallback) to accelerate top-down and multi-level vector similarity search.
    """
    
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree

    def _get_embedding(self, query: str):
        """Returns normalized embedding vector for query."""
        seed_val = abs(hash(query)) % (2**31)
        rng = random.Random(seed_val)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(64)]
        norm_val = math.sqrt(sum(x*x for x in vec)) + 1e-9
        vec = [x / norm_val for x in vec]
        if HAS_NUMPY:
            return np.array(vec)
        return vec

    def top_down_search(self, query: str, top_k: int = 3) -> List[TreeNode]:
        """
        Traverses tree from top root abstracts down to leaf nodes based on similarity scoring.
        """
        query_vec = self._get_embedding(query)
        current_candidates = self.tree.root_nodes
        
        while current_candidates:
            # Check if current level is leaf nodes
            if current_candidates[0].level == 0:
                break
                
            # Score children of current candidates
            next_level_nodes = []
            for node in current_candidates:
                next_level_nodes.extend(node.children)
                
            if not next_level_nodes:
                break
                
            # Compute vector similarities via GPU / CPU engine
            candidate_embeddings = [n.embedding for n in next_level_nodes]
            if HAS_NUMPY:
                candidate_embeddings = np.array(candidate_embeddings)
                
            similarities = gpu_batch_cosine_similarity(query_vec, candidate_embeddings)
            
            # Select top matching branch nodes
            if isinstance(similarities, list):
                indexed_sims = list(enumerate(similarities))
                indexed_sims.sort(key=lambda x: x[1], reverse=True)
                top_indices = [idx for idx, _ in indexed_sims[:top_k]]
            else:
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
            current_candidates = [next_level_nodes[i] for i in top_indices]
            
        return current_candidates[:top_k]


