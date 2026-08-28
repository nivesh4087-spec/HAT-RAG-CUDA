"""
HAT-RAG Production Enterprise Web Dashboard
Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation
Accelerated via NVIDIA CUDA GPU Computing
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

if HAS_STREAMLIT:
    import json
    from hat_rag.src.cuda_utils import check_cuda_availability, benchmark_cuda_vs_cpu
    from hat_rag.src.document_processor import DocumentProcessor
    from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
    from hat_rag.src.retriever import HierarchicalRetriever
    from hat_rag.src.generator import HATGenerator
    from hat_rag.src.evaluator import RAGEvaluator

    st.set_page_config(
        page_title="HAT-RAG | CUDA Accelerated AI Platform",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    @st.cache_resource
    def get_default_tree():
        processor = DocumentProcessor(chunk_size=30, chunk_overlap=5)
        docs = {
            "cooling_spec.txt": "Ceiling fans function by creating a wind chill factor. High speed rotation produces downward airflow, reducing temperature in halls by up to 4 degrees Celsius.",
            "maintenance_manual.txt": "Tool wear in stamping fan blades causes motor alignment errors. Daily torque inspections prevent excessive vibration, bearing wear, and thermal failure.",
            "energy_guide.txt": "Brushless DC motor (BLDC) ceiling fans consume up to 65% less power compared to standard induction motors with smart microcontroller adaptive speed control."
    def main():
        st.title("⚡ HAT-RAG: Hierarchical Abstract Tree Engine")
        st.caption("Cross-Document Retrieval-Augmented Generation Accelerated with NVIDIA CUDA GPU Computing")
        st.divider()

        gpu_info = check_cuda_availability()
        st.sidebar.title("🎮 Hardware Command Center")
        st.sidebar.info(f"**Backend**: {gpu_info['backend']}")
        st.sidebar.text(f"Device: {gpu_info['device_name']}")
        st.sidebar.text(f"CUDA Available: {gpu_info['cuda_available']}")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Executive Overview",
            "📁 Ingest & Tree Builder",
            "🔍 RAG Search Engine",
            "📈 Baseline Benchmark"
        ])

        tree, docs = get_default_tree()

        with tab1:
            st.subheader("System Performance & Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Indexed Documents", len(docs))
            col2.metric("Total Tree Nodes", len(tree.nodes))
            col3.metric("Root Clusters", len(tree.root_nodes))

        with tab2:
            st.subheader("Document Ingestion & Tree Inspection")
            st.json({node_id: {"level": n.level, "text": n.text[:80] + "...", "doc_id": n.doc_id} for node_id, n in list(tree.nodes.items())[:6]})

        with tab3:
            st.subheader("Interactive Hierarchical RAG Query Engine")
            query = st.text_input("Enter Query:", "How to prevent motor failure and reduce power consumption in fans?")
            top_k = st.slider("Top K Results", 1, 5, 2)

            if st.button("Run HAT-RAG Query", type="primary"):
                retriever = HierarchicalRetriever(tree)
                nodes, stats = retriever.top_down_search(query, top_k=top_k)
                
                generator = HATGenerator()
                response = generator.generate_response(query, nodes, search_stats=stats)

                st.success(f"Execution Completed in {stats['execution_time_ms']} ms")
                st.markdown("### 💡 Answer:")
                st.write(response["answer"])

                st.markdown("### 🔍 Context Nodes:")
                for citation in response["citations"]:
                    st.markdown(f"**{citation['citation_id']} Node `{citation['node_id']}` (Level {citation['level']})**")
                    st.caption(citation["snippet"])

        with tab4:
            st.subheader("HAT-RAG vs Standard Flat RAG Benchmark")
            if st.button("Run Speed Benchmark"):
                evaluator = RAGEvaluator(tree)
                res = evaluator.evaluate_query("How to prevent motor alignment errors?")
                
                b1, b2, b3 = st.columns(3)
                b1.metric("HAT Traversal Time", f"{res['hat_rag']['execution_time_ms']} ms")
                b2.metric("Flat Search Time", f"{res['flat_rag']['execution_time_ms']} ms")
                b3.metric("Node Reduction", f"{res['comparison']['node_eval_reduction_percent']}% Saved")

                st.json(res)

    if __name__ == "__main__":
        main()
else:
    def main():
        print("Streamlit not installed. Launch demo via 'python hat_rag/demo_hat_rag.py'")

    if __name__ == "__main__":
        main()

        }
        chunks = processor.process_documents(docs)
        tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
        tree.build_tree(chunks)
        return tree, docs
