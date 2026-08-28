import json
import math
import random
import logging
from typing import List, Dict, Optional, Any

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

from hat_rag.src.embeddings import EmbeddingEngine
from hat_rag.src.summarizer import AbstractSummarizer

logger = logging.getLogger(__name__)

class TreeNode:
    """Represents a single node in the Hierarchical Abstract Tree."""
    
    def __init__(
        self,
        node_id: str,
        level: int,
        text: str,
        embedding=None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.level = level  # 0 = leaf chunk, >0 = abstract summary node
        self.text = text
        self.embedding = embedding
        self.doc_id = doc_id
        self.metadata = metadata or {}
        self.children: List['TreeNode'] = []
        self.parent: Optional['TreeNode'] = None

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node to dictionary."""
        embed_list = None
        if self.embedding is not None:
            embed_list = self.embedding.tolist() if HAS_NUMPY and isinstance(self.embedding, np.ndarray) else list(self.embedding)
            
        return {
            "node_id": self.node_id,
            "level": self.level,
            "text": self.text,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
            "children_ids": [c.node_id for c in self.children],
            "embedding": embed_list
        }


class HierarchicalAbstractTree:
    """
    Constructs, manages, and serializes the multi-level Hierarchical Abstract Tree (HAT)
    for Cross-Document Retrieval.
    """
    
    def __init__(
        self,
        max_levels: int = 3,
        clusters_per_level: int = 2,
        embedding_engine: Optional[EmbeddingEngine] = None,
        summarizer: Optional[AbstractSummarizer] = None
    ):
        self.max_levels = max_levels
        self.clusters_per_level = clusters_per_level
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.summarizer = summarizer or AbstractSummarizer()
        self.nodes: Dict[str, TreeNode] = {}
        self.root_nodes: List[TreeNode] = []

    def build_tree(self, leaf_chunks: List[Dict]) -> List[TreeNode]:
        """Recursively builds tree levels from bottom (leaf chunks) to top (root abstracts)."""
        current_nodes = []
        
        # Level 0: Leaf Nodes
        for chunk in leaf_chunks:
            chunk_text = chunk["text"]
            embedding = self.embedding_engine.encode(chunk_text)
            
            node = TreeNode(
                node_id=chunk["chunk_id"],
                level=0,
                text=chunk_text,
                embedding=embedding,
                doc_id=chunk.get("doc_id"),
                metadata={"token_count": chunk.get("token_count", len(chunk_text.split()))}
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
                labels = [i % n_clusters for i in range(len(current_nodes))]
            
            for cluster_idx in range(n_clusters):
                cluster_children = [current_nodes[i] for i in range(len(current_nodes)) if labels[i] == cluster_idx]
                if not cluster_children:
                    continue
                
                child_texts = [c.text for c in cluster_children]
                summary_text = self.summarizer.summarize_cluster(child_texts)
                parent_embedding = self.embedding_engine.encode(summary_text)
                
                parent_node = TreeNode(
                    node_id=f"level_{current_level+1}_cluster_{cluster_idx}",
                    level=current_level + 1,
                    text=summary_text,
                    embedding=parent_embedding,
                    metadata={"child_count": len(cluster_children)}
                )
                
                for child in cluster_children:
                    parent_node.add_child(child)
                    
                self.nodes[parent_node.node_id] = parent_node
                next_level_nodes.append(parent_node)
                
            current_nodes = next_level_nodes
            current_level += 1
            
        self.root_nodes = current_nodes
        return self.root_nodes

    def save_tree(self, filepath: str):
        """Saves tree structure to JSON file."""
        serialized = {
            "max_levels": self.max_levels,
            "clusters_per_level": self.clusters_per_level,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_ids": [r.node_id for r in self.root_nodes]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        logger.info(f"Saved tree to {filepath}")

    def load_tree(self, filepath: str):
        """Loads tree structure from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.max_levels = data["max_levels"]
        self.clusters_per_level = data["clusters_per_level"]
        self.nodes = {}
        
        for node_id, d in data["nodes"].items():
            embed = np.array(d["embedding"]) if HAS_NUMPY and d["embedding"] else d["embedding"]
            node = TreeNode(
                node_id=d["node_id"],
                level=d["level"],
                text=d["text"],
                embedding=embed,
                doc_id=d.get("doc_id"),
                metadata=d.get("metadata", {})
            )
            self.nodes[node_id] = node
            
        for node_id, d in data["nodes"].items():
            parent_node = self.nodes[node_id]
            for child_id in d.get("children_ids", []):
                if child_id in self.nodes:
                    parent_node.add_child(self.nodes[child_id])
                    
        self.root_nodes = [self.nodes[rid] for rid in data["root_ids"] if rid in self.nodes]
        logger.info(f"Loaded tree from {filepath}")



