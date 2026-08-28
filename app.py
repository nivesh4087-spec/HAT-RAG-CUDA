"""
HAT-RAG Production Enterprise Web Dashboard
Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation
Accelerated via NVIDIA CUDA GPU Computing
Supports 4 Architectural Approaches & 29 Research Papers Repository
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

if HAS_STREAMLIT:
    from hat_rag.src.cuda_utils import check_cuda_availability, benchmark_cuda_vs_cpu
    from hat_rag.src.document_processor import DocumentProcessor
    from hat_rag.src.hierarchical_tree import HierarchicalAbstractTree
    from hat_rag.src.retriever import HierarchicalRetriever
    from hat_rag.src.generator import HATGenerator
    from hat_rag.src.evaluator import RAGEvaluator
    from hat_rag.src.multi_approach import MultiApproachEngine

    st.set_page_config(
        page_title="HAT-RAG | 4 Approaches & 29 Research Papers Platform",
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
        }
        chunks = processor.process_documents(docs)
        tree = HierarchicalAbstractTree(max_levels=2, clusters_per_level=2)
        tree.build_tree(chunks)
        return tree, docs

    def main():
        st.title("⚡ HAT-RAG: 4 Approaches & 29 Research Papers Platform")
        st.caption("Cross-Document Retrieval-Augmented Generation Accelerated with NVIDIA CUDA GPU Computing")
        st.divider()

        gpu_info = check_cuda_availability()
        st.sidebar.title("🎮 Hardware Command Center")
        st.sidebar.info(f"**Backend**: {gpu_info['backend']}")
        st.sidebar.text(f"Device: {gpu_info['device_name']}")
        st.sidebar.text(f"CUDA Available: {gpu_info['cuda_available']}")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Executive Overview",
            "📁 Ingest & Tree Builder",
            "🔍 RAG Search Engine",
            "🔀 4 Architectural Approaches",
            "📚 29 Research Papers",
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
            st.subheader("🔀 Comparison of 4 RAG Architectural Approaches")
            multi_query = st.text_input("Benchmark Query Across 4 Approaches:", "How to prevent motor failure and reduce power consumption?")
            if st.button("Compare All 4 Approaches", type="primary"):
                multi = MultiApproachEngine(tree)
                res = multi.compare_all(multi_query)
                
                cols = st.columns(4)
                for i, app in enumerate(res["approaches"]):
                    with cols[i]:
                        st.markdown(f"### {app['stats']['approach']}")
                        st.metric("Latency (ms)", app["stats"]["execution_time_ms"])
                        st.metric("Nodes Evaluated", app["stats"]["nodes_evaluated"])
                        st.caption(f"Complexity: `{app['stats']['complexity']}`")
                        st.markdown("**Top Result Snippet:**")
                        if app["results_snippets"]:
                            st.write(app["results_snippets"][0])

        with tab5:
            st.subheader("📚 29 Curated Research Papers Repository")
            st.caption("Seminal research papers covering Tree-RAG, Graph-RAG, RAPTOR, and CUDA Acceleration.")
            papers_path = Path(__file__).resolve().parent / "papers" / "papers_index.json"
            if papers_path.exists():
                with open(papers_path, "r", encoding="utf-8") as f:
                    papers = json.load(f)
                st.dataframe(
                    [{"Title": p["title"], "Category": p["topic"], "ArXiv ID": p["id"], "URL": p["pdf_url"]} for p in papers],
                    use_container_width=True
                )
            else:
                st.info("Papers index not generated. Run `python download_papers.py` to generate index.")

        with tab6:
            st.subheader("HAT-RAG vs Standard Flat RAG Benchmark")
            if st.button("Run Speed Benchmark"):
                evaluator = RAGEvaluator(tree)
                res = evaluator.evaluate_query("How to prevent motor alignment errors?")
                st.json(res)

    if __name__ == "__main__":
        main()

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
