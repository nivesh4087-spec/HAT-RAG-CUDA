import time
import logging
from typing import List, Dict, Tuple, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from hat_rag.src.cuda_utils import gpu_batch_cosine_similarity
from hat_rag.src.embeddings import EmbeddingEngine
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree, TreeNode

logger = logging.getLogger(__name__)

class Approach1_HAT_RAG:
    """Approach 1: Hierarchical Abstract Tree RAG (Top-Down Logarithmic Traversal)"""
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.engine = tree.embedding_engine

    def query(self, query_text: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        start = time.perf_counter()
        q_vec = self.engine.encode(query_text)
        current = self.tree.root_nodes
        evaluated = len(current)
        
        while current and current[0].level > 0:
            next_nodes = []
            for n in current:
                next_nodes.extend(n.children)
            if not next_nodes:
                break
            evaluated += len(next_nodes)
            embeds = np.array([n.embedding for n in next_nodes]) if HAS_NUMPY else [n.embedding for n in next_nodes]
            sims, _ = gpu_batch_cosine_similarity(q_vec, embeds)
            top_idx = np.argsort(sims)[::-1][:top_k] if HAS_NUMPY else [i for i, _ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]]
            current = [next_nodes[i] for i in top_idx]
            
        nodes = current[:top_k]
        elapsed = (time.perf_counter() - start) * 1000
        return nodes, {
            "approach": "Approach 1: HAT-RAG (Top-Down Logarithmic Pruning)",
            "execution_time_ms": round(elapsed, 3),
            "nodes_evaluated": evaluated,
            "complexity": "O(k log N)"
        }

class Approach2_Flat_RAG:
    """Approach 2: Standard Flat / Dense Vector RAG (Global Brute Force Scan)"""
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.engine = tree.embedding_engine

    def query(self, query_text: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        start = time.perf_counter()
        q_vec = self.engine.encode(query_text)
        leafs = [n for n in self.tree.nodes.values() if n.level == 0]
        embeds = np.array([n.embedding for n in leafs]) if HAS_NUMPY else [n.embedding for n in leafs]
        sims, _ = gpu_batch_cosine_similarity(q_vec, embeds)
        top_idx = np.argsort(sims)[::-1][:top_k] if HAS_NUMPY else [i for i, _ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]]
        nodes = [leafs[i] for i in top_idx]
        elapsed = (time.perf_counter() - start) * 1000
        return nodes, {
            "approach": "Approach 2: Standard Flat Dense Vector RAG",
            "execution_time_ms": round(elapsed, 3),
            "nodes_evaluated": len(leafs),
            "complexity": "O(N)"
        }

class Approach3_Graph_RAG:

    """Approach 3: Knowledge Graph / Entity-Relation Traversal RAG"""
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.engine = tree.embedding_engine
        self._build_graph()

    def _build_graph(self):
        self.graph = {}
        nodes = list(self.tree.nodes.values())
        for i, n1 in enumerate(nodes):
            self.graph[n1.node_id] = []
            for j, n2 in enumerate(nodes):
                if i != j:
                    sim, _ = gpu_batch_cosine_similarity(n1.embedding, [n2.embedding])
                    if float(sim[0] if HAS_NUMPY else sim[0]) > 0.3:
                        self.graph[n1.node_id].append(n2.node_id)

    def query(self, query_text: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        start = time.perf_counter()
        q_vec = self.engine.encode(query_text)
        all_nodes = list(self.tree.nodes.values())
        embeds = np.array([n.embedding for n in all_nodes]) if HAS_NUMPY else [n.embedding for n in all_nodes]
        sims, _ = gpu_batch_cosine_similarity(q_vec, embeds)
        seed_idx = int(np.argmax(sims)) if HAS_NUMPY else sims.index(max(sims))
        seed_node = all_nodes[seed_idx]
        
        visited = {seed_node.node_id}
        result_nodes = [seed_node]
        neighbors = self.graph.get(seed_node.node_id, [])
        
        for n_id in neighbors:
            if n_id not in visited and len(result_nodes) < top_k:
                visited.add(n_id)
                result_nodes.append(self.tree.nodes[n_id])
                
        elapsed = (time.perf_counter() - start) * 1000
        return result_nodes, {
            "approach": "Approach 3: Graph-RAG (Entity Relation Multi-Hop Traversal)",
            "execution_time_ms": round(elapsed, 3),
            "nodes_evaluated": len(all_nodes) + len(neighbors),
            "complexity": "O(V + E)"
        }

class Approach4_RAPTOR_RAG:
    """Approach 4: RAPTOR-style Recursive Summarization Tree RAG (Collapsed Multi-Level Search)"""
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.engine = tree.embedding_engine

    def query(self, query_text: str, top_k: int = 3) -> Tuple[List[TreeNode], Dict[str, Any]]:
        start = time.perf_counter()
        q_vec = self.engine.encode(query_text)
        all_nodes = list(self.tree.nodes.values())
        embeds = np.array([n.embedding for n in all_nodes]) if HAS_NUMPY else [n.embedding for n in all_nodes]
        sims, _ = gpu_batch_cosine_similarity(q_vec, embeds)
        top_idx = np.argsort(sims)[::-1][:top_k] if HAS_NUMPY else [i for i, _ in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]]
        nodes = [all_nodes[i] for i in top_idx]
        elapsed = (time.perf_counter() - start) * 1000
        return nodes, {
            "approach": "Approach 4: RAPTOR-style Multi-Level Collapsed RAG",
            "execution_time_ms": round(elapsed, 3),
            "nodes_evaluated": len(all_nodes),
            "complexity": "O(N_all_levels)"
        }

class MultiApproachEngine:
    """Unified Orchestrator comparing all 4 RAG Architectural Approaches."""
    def __init__(self, tree: HierarchicalAbstractTree):
        self.tree = tree
        self.a1 = Approach1_HAT_RAG(tree)
        self.a2 = Approach2_Flat_RAG(tree)
        self.a3 = Approach3_Graph_RAG(tree)
        self.a4 = Approach4_RAPTOR_RAG(tree)

    def compare_all(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        n1, s1 = self.a1.query(query_text, top_k)
        n2, s2 = self.a2.query(query_text, top_k)
        n3, s3 = self.a3.query(query_text, top_k)
        n4, s4 = self.a4.query(query_text, top_k)
        
        return {
            "query": query_text,
            "approaches": [
                {"stats": s1, "results_snippets": [n.text[:80] + "..." for n in n1]},
                {"stats": s2, "results_snippets": [n.text[:80] + "..." for n in n2]},
                {"stats": s3, "results_snippets": [n.text[:80] + "..." for n in n3]},
                {"stats": s4, "results_snippets": [n.text[:80] + "..." for n in n4]}
            ]
        }

        nodes = [leafs[i] for i in top_idx]
        elapsed = (time.perf_counter() - start) * 1000
        return nodes, {
            "approach": "Approach 2: Standard Flat Dense Vector RAG",
            "execution_time_ms": round(elapsed, 3),
            "nodes_evaluated": len(leafs),
            "complexity": "O(N)"
        }
