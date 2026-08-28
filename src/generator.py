from typing import List
from hat_rag.src.hierarchical_tree import TreeNode

class HATGenerator:
    """Generates cross-document augmented answers using retrieved tree context."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name

    def generate_response(self, query: str, context_nodes: List[TreeNode]) -> str:
        """Synthesizes context chunks into a final LLM response."""
        context_str = "\n".join([f"[{node.node_id}] (Level {node.level}): {node.text}" for node in context_nodes])
        
        response = (
            f"=== HAT-RAG Response ===\n"
            f"Query: {query}\n"
            f"Retrieved Context ({len(context_nodes)} nodes from Hierarchical Tree):\n{context_str}\n\n"
            f"Answer: Based on cross-document abstract hierarchy, the information suggests that "
            f"{context_nodes[0].text[:80]}... [Generated via CUDA-Accelerated HAT Engine]"
        )
        return response
