import math
import random
from typing import List, Dict, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class TreeNode:
    """Represents a single node in the Hierarchical Abstract Tree."""
    
    def __init__(self, node_id: str, level: int, text: str, embedding=None):
        self.node_id = node_id
        self.level = level  # 0 = leaf chunk, >0 = abstract summary node
        self.text = text
        self.embedding = embedding
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

class HierarchicalAbstractTree:
    """
    Constructs and manages the multi-level Hierarchical Abstract Tree (HAT)
    for Cross-Document Retrieval. Uses GPU-compatible tensor operations with pure Python fallback.
    """
    
    def __init__(self, max_levels: int = 3, clusters_per_level: int = 2):
        self.max_levels = max_levels
        self.clusters_per_level = clusters_per_level
        self.nodes: Dict[str, TreeNode] = {}
        self.root_nodes: List[TreeNode] = []

    def _mock_embedding(self, text: str):
        """Generates a deterministic 64-dim vector representation based on text hash."""
        seed_val = abs(hash(text)) % (2**31)
        rng = random.Random(seed_val)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(64)]
        norm_val = math.sqrt(sum(x*x for x in vec)) + 1e-9
        vec = [x / norm_val for x in vec]
        
        if HAS_NUMPY:
            return np.array(vec)
        return vec

    def _summarize_cluster(self, child_texts: List[str]) -> str:
        """Creates an abstract summary for a cluster of child nodes."""
        prefix = "ABSTRACT SUMMARY: "
        key_phrases = [t[:40] for t in child_texts]
        return prefix + " | ".join(key_phrases)

    def build_tree(self, leaf_chunks: List[Dict]) -> List[TreeNode]:
        """Recursively builds tree levels from bottom (leaf chunks) to top (root abstracts)."""
        current_nodes = []
        
        # Level 0: Leaf Nodes
        for chunk in leaf_chunks:
            node = TreeNode(
                node_id=chunk["chunk_id"],
                level=0,
                text=chunk["text"],
                embedding=self._mock_embedding(chunk["text"])
            )
            self.nodes[node.node_id] = node
            current_nodes.append(node)
            
        current_level = 0
        
        # Build Higher Levels recursively
        while current_level < self.max_levels and len(current_nodes) > 1:
            next_level_nodes = []
            n_clusters = min(self.clusters_per_level, len(current_nodes))
            
            if HAS_SKLEARN and HAS_NUMPY:
                embeddings = np.array([n.embedding for n in current_nodes])
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
                labels = kmeans.fit_predict(embeddings)
            else:
                # Naive round-robin partitioning fallback
                labels = [i % n_clusters for i in range(len(current_nodes))]
            
            for cluster_idx in range(n_clusters):
                cluster_children = [current_nodes[i] for i in range(len(current_nodes)) if labels[i] == cluster_idx]
                if not cluster_children:
                    continue
                
                # Abstract summarization across cluster
                summary_text = self._summarize_cluster([c.text for c in cluster_children])
                parent_node = TreeNode(
                    node_id=f"level_{current_level+1}_cluster_{cluster_idx}",
                    level=current_level + 1,
                    text=summary_text,
                    embedding=self._mock_embedding(summary_text)
                )
                
                for child in cluster_children:
                    parent_node.add_child(child)
                    
                self.nodes[parent_node.node_id] = parent_node
                next_level_nodes.append(parent_node)
                
            current_nodes = next_level_nodes
            current_level += 1
            
        self.root_nodes = current_nodes
        return self.root_nodes


