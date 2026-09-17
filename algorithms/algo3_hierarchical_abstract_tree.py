import json
import math
import random
import re
from typing import List, Dict, Any, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class TreeNode:
    def __init__(
        self,
        node_id: str,
        level: int,
        text: str,
        embedding: Optional[List[float]] = None,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.level = level
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


class SimpleVectorEncoder:
    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        seed_val = abs(hash(text)) % (2**31)
        rng = random.Random(seed_val)
        words = text.lower().split()
        vec = [0.0] * self.dim
        for w in words:
            w_rng = random.Random(abs(hash(w)) % (2**31))
            for idx in range(self.dim):
                vec[idx] += w_rng.uniform(-1.0, 1.0)
        norm_val = math.sqrt(sum(x * x for x in vec)) + 1e-9
        return [x / norm_val for x in vec]


class ClusterSummarizer:
    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[A-Za-z0-9_-]{4,}\b', text)
        stopwords = {"this", "that", "with", "from", "were", "have", "been", "which", "under", "across", "during"}
        filtered = [w for w in words if w.lower() not in stopwords]
        freq = {}
        for w in filtered:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:5]

    def summarize(self, child_texts: List[str], level: int) -> str:
        combined = " ".join(child_texts)
        keywords = self.extract_keywords(combined)
        kw_str = ", ".join(keywords)

        sentences = [s.strip() for s in combined.split('.') if len(s.strip()) > 15]
        top_sentence = sentences[0] if sentences else combined[:120]
        
        if level == 1:
            return f"CLUSTER ABSTRACT (L1): Focus on [{kw_str}]. Key insight: {top_sentence}."
        else:
            return f"GLOBAL ROOT ABSTRACT (L2): Cross-domain synthesis of [{kw_str}]. Core concept: {top_sentence}."


class HierarchicalAbstractTreeBuilder:
    def __init__(self, max_levels: int = 2, branching_factor: int = 2):
        self.max_levels = max_levels
        self.branching_factor = branching_factor
        self.encoder = SimpleVectorEncoder(dim=64)
        self.summarizer = ClusterSummarizer()
        self.nodes: Dict[str, TreeNode] = {}
        self.root_nodes: List[TreeNode] = []

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))

    def _cluster_nodes(self, nodes: List[TreeNode], k: int) -> List[List[TreeNode]]:
        if len(nodes) <= k:
            return [[n] for n in nodes]
        
        clusters = [[] for _ in range(k)]
        for idx, node in enumerate(nodes):
            clusters[idx % k].append(node)
        return clusters

    def build_tree(self, leaf_chunks: List[Dict[str, Any]]) -> List[TreeNode]:
        current_level_nodes: List[TreeNode] = []

        # Level 0: Leaf Nodes
        for chunk in leaf_chunks:
            text = chunk["text"]
            embedding = self.encoder.encode(text)
            node = TreeNode(
                node_id=chunk["chunk_id"],
                level=0,
                text=text,
                embedding=embedding,
                doc_id=chunk.get("doc_id"),
                metadata={
                    "token_count": chunk.get("token_count", len(text.split())),
                    "section": chunk.get("section", "General")
                }
            )
            self.nodes[node.node_id] = node
            current_level_nodes.append(node)

        current_level = 0

        # Recursive Tree Construction: Level 0 -> Level 1 -> Level 2
        while current_level < self.max_levels and len(current_level_nodes) > 1:
            next_level_nodes: List[TreeNode] = []
            k = min(self.branching_factor, len(current_level_nodes))
            clusters = self._cluster_nodes(current_level_nodes, k)

            for c_idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                child_texts = [child.text for child in cluster]
                summary_text = self.summarizer.summarize(child_texts, level=current_level + 1)
                parent_embedding = self.encoder.encode(summary_text)

                parent_id = f"L{current_level + 1}_Cluster_{c_idx}"
                parent_node = TreeNode(
                    node_id=parent_id,
                    level=current_level + 1,
                    text=summary_text,
                    embedding=parent_embedding,
                    metadata={"child_count": len(cluster), "cluster_id": c_idx}
                )

                for child in cluster:
                    parent_node.add_child(child)

                self.nodes[parent_node.node_id] = parent_node
                next_level_nodes.append(parent_node)

            current_level_nodes = next_level_nodes
            current_level += 1

        self.root_nodes = current_level_nodes
        return self.root_nodes

    def save_tree_json(self, filepath: str):
        data = {
            "max_levels": self.max_levels,
            "branching_factor": self.branching_factor,
            "total_nodes": len(self.nodes),
            "root_ids": [r.node_id for r in self.root_nodes],
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_tree_json(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.nodes = {}
        for nid, d in data["nodes"].items():
            node = TreeNode(
                node_id=d["node_id"],
                level=d["level"],
                text=d["text"],
                embedding=d.get("embedding"),
                doc_id=d.get("doc_id"),
                metadata=d.get("metadata", {})
            )
            self.nodes[nid] = node

        for nid, d in data["nodes"].items():
            parent = self.nodes[nid]
            for cid in d.get("children_ids", []):
                if cid in self.nodes:
                    parent.add_child(self.nodes[cid])

        self.root_nodes = [self.nodes[rid] for rid in data["root_ids"] if rid in self.nodes]

    def display_tree(self):
        print("\n" + "=" * 85)
        print(" HIERARCHICAL ABSTRACT TREE (HAT) TOPOLOGY HIERARCHY")
        print("=" * 85)

        def _print_subtree(node: TreeNode, indent: str = "", is_last: bool = True):
            marker = "+-- " if is_last else "|-- "
            badge = f"[L{node.level}]"
            doc_badge = f" (Doc: {node.doc_id})" if node.doc_id else ""
            print(f"{indent}{marker}{badge} Node: {node.node_id}{doc_badge}")
            text_preview = node.text if len(node.text) < 95 else node.text[:95] + "..."
            child_indent = indent + ("    " if is_last else "|   ")
            print(f"{child_indent}Content: \"{text_preview}\"")
            if node.children:
                for idx, child in enumerate(node.children):
                    _print_subtree(child, child_indent, idx == len(node.children) - 1)

        for i, root in enumerate(self.root_nodes):
            _print_subtree(root, is_last=(i == len(self.root_nodes) - 1))
        print("=" * 85)


def run_hat_tree_demo():
    print("=" * 85)
    print(" ALGORITHM 3: HIERARCHICAL ABSTRACT TREE (HAT) CONSTRUCTION ALGORITHM")
    print("=" * 85)

    sample_leaf_chunks = [
        {
            "chunk_id": "Doc1_BladePitch_c0",
            "doc_id": "Paper_01_ThermalDynamics",
            "section": "Blade Design",
            "text": "High-speed ceiling fan blade profiles are engineered with optimal pitch angles between 12 to 15 degrees to maximize downward volumetric air displacement.",
            "token_count": 23
        },
        {
            "chunk_id": "Doc1_ThermalImaging_c1",
            "doc_id": "Paper_01_ThermalDynamics",
            "section": "Thermal Analysis",
            "text": "Empirical thermal imaging demonstrates a uniform 3.8 degree Celsius decrease in surface temperature across a 40 square meter test chamber.",
            "token_count": 22
        },
        {
            "chunk_id": "Doc2_ToolWear_c0",
            "doc_id": "Paper_02_MotorDiagnostics",
            "section": "Vibration Telemetry",
            "text": "Progressive tool wear during sheet metal stamping induces structural micro-asymmetries and high-frequency harmonic vibrations exceeding 120 Hz.",
            "token_count": 21
        },
        {
            "chunk_id": "Doc2_FaultTelemetry_c1",
            "doc_id": "Paper_02_MotorDiagnostics",
            "section": "Sensor Ingestion",
            "text": "Triaxial accelerometer telemetry combined with edge microcontroller inference enables real-time bearing fault classification and reduces downtime by 41%.",
            "token_count": 20
        },
        {
            "chunk_id": "Doc3_BLDCEfficiency_c0",
            "doc_id": "Paper_03_EnergyOptimization",
            "section": "Inverter Topology",
            "text": "Permanent magnet Brushless DC (BLDC) motor architectures achieve electrical efficiency ratings exceeding 88 percent with sensorless field-oriented control.",
            "token_count": 21
        },
        {
            "chunk_id": "Doc3_SpeedScheduling_c1",
            "doc_id": "Paper_03_EnergyOptimization",
            "section": "Speed Scheduling",
            "text": "Ambient temperature-adaptive pulse-width modulation dynamically modulates rotor RPM, reducing annual power consumption by 62%.",
            "token_count": 18
        }
    ]

    print(f"[Input Data] Ingesting {len(sample_leaf_chunks)} normalized leaf chunks from 3 documents...")
    
    # 1. Build Hierarchical Tree
    tree_builder = HierarchicalAbstractTreeBuilder(max_levels=2, branching_factor=2)
    root_nodes = tree_builder.build_tree(sample_leaf_chunks)

    # 2. Display Tree Topology
    tree_builder.display_tree()

    # 3. Test JSON Serialization & Deserialization
    json_path = "algorithms/hat_tree_structure.json"
    tree_builder.save_tree_json(json_path)
    print(f"\n[Tree Persistence] Successfully exported HAT structure to '{json_path}'")

    reload_test = HierarchicalAbstractTreeBuilder()
    reload_test.load_tree_json(json_path)
    print(f"[Tree Persistence] Successfully reloaded tree from JSON! Verified {len(reload_test.nodes)} nodes.\n")

    # 4. Print Tree Metrics
    print("=" * 85)
    print(" HAT TREE CONSTRUCTION METRICS")
    print("=" * 85)
    leaf_count = sum(1 for n in tree_builder.nodes.values() if n.level == 0)
    l1_count = sum(1 for n in tree_builder.nodes.values() if n.level == 1)
    l2_count = sum(1 for n in tree_builder.nodes.values() if n.level == 2)
    print(f"Total Tree Nodes Ingested / Created : {len(tree_builder.nodes)}")
    print(f"Level 0 (Leaf Passage Chunks)       : {leaf_count} nodes")
    print(f"Level 1 (Cluster Abstract Summaries): {l1_count} nodes")
    print(f"Level 2 (Global Root Abstract Nodes): {l2_count} root nodes")
    print(f"Hierarchical Depth Ratio            : {len(tree_builder.nodes) / max(1, leaf_count):.2f}x")
    print("=" * 85)

    return tree_builder


if __name__ == "__main__":
    run_hat_tree_demo()
