"""
HAT-RAG Demo: Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation
Accelerated with NVIDIA CUDA GPU Computing
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hat_rag.src.cuda_utils import check_cuda_availability
from hat_rag.src.document_processor import DocumentProcessor
from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
from hat_rag.src.retriever import HierarchicalRetriever
from hat_rag.src.generator import HATGenerator

def run_demo():
    print("==========================================================================")
    print("Hierarchical Abstract Tree (HAT) RAG - CUDA Accelerated Engine Demo")
    print("==========================================================================")
    
    # 1. GPU / Hardware Check
    gpu_info = check_cuda_availability()
    print(f"[Hardware Setup] Active Device: {gpu_info['device_name']}")
    print(f"[Hardware Setup] CUDA Status: {'Available' if gpu_info['cuda_available'] else 'CPU Execution Fallback Mode'}")
    print("-" * 74)

    # 2. Sample Cross-Document Data
    documents = {
        "doc1_cooling.txt": "Ceiling fans function by creating a wind chill factor. High speed rotation of blades produces downward airflow, reducing effective temperature in large open halls by up to 4 degrees Celsius.",
        "doc2_maintenance.txt": "Tool wear in stamping fan blades causes motor alignment errors. Daily torque inspections prevent excessive vibration, bearing wear, and thermal degradation in high-speed ceiling fans.",
        "doc3_energy.txt": "Brushless DC motor (BLDC) ceiling fans consume up to 65% less power compared to standard induction motors, integrating smart microcontrollers for temperature adaptive speed control."
    }

    # 3. Document Chunking
    processor = DocumentProcessor(chunk_size=20, chunk_overlap=5)
    leaf_chunks = processor.process_documents(documents)
    print(f"[Doc Processing] Processed {len(documents)} documents into {len(leaf_chunks)} leaf chunks.")

    # 4. Tree Construction
    print("[HAT Engine] Constructing Hierarchical Abstract Tree (Clustering & Summarization)...")
    hat_tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
    root_nodes = hat_tree.build_tree(leaf_chunks)
    print(f"[HAT Engine] Tree Built Successfully! Total nodes: {len(hat_tree.nodes)}, Root clusters: {len(root_nodes)}")

    # 5. Hierarchical Retrieval via CUDA Similarity
    query = "How to prevent motor failure and optimize ceiling fan cooling performance?"
    print(f"\n[Retrieval] Query: '{query}'")
    print("[Retrieval] Executing CUDA-Accelerated Top-Down Tree Traversal...")
    retriever = HierarchicalRetriever(hat_tree)
    retrieved_nodes, stats = retriever.top_down_search(query, top_k=2)

    # 6. Response Generation
    generator = HATGenerator()
    final_output = generator.generate_response(query, retrieved_nodes, search_stats=stats)
    
    print("\n=== HAT-RAG Response ===")
    print(f"Query: {query}")
    print(f"Answer: {final_output['answer']}")
    print(f"Context Nodes: {len(final_output['citations'])}")
    print(f"Traversal Execution Time: {stats['execution_time_ms']} ms")
    print("==========================================================================")
    print("[SUCCESS] 100% Project Pipeline Completed Successfully.")


def main():
    run_demo()

if __name__ == "__main__":
    run_demo()


