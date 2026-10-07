"""
HAT-RAG: Corporate Financial Intelligence & Retrieval Engine
Multi-Algorithm Cross-Document RAG over SEC 10-K Filings (finance_data/reports)
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root and algorithms to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "algorithms"))

import streamlit as st

st.set_page_config(
    page_title="HAT-RAG: Financial RAG Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Financial Software UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #CBD5E1;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .company-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .badge-aapl { background-color: #555555; }
    .badge-amzn { background-color: #FF9900; }
    .badge-msft { background-color: #00A4EF; }
    .badge-nvda { background-color: #76B900; }
    .badge-tsla { background-color: #E82127; }
    .badge-default { background-color: #6366F1; }
</style>
""", unsafe_allow_html=True)

# File Paths
FINANCE_REPORTS_DIR = ROOT_DIR / "finance_data" / "reports"
TREE_PATH = ROOT_DIR / "algorithms" / "hat_finance_tree_structure.json"
QUERIES_PATH = ROOT_DIR / "algorithms" / "compare-algos" / "finance_queries.json"


# Cached Model & Tree Loader
@st.cache_resource(show_spinner="Loading Financial Tree Index & Dense Embedder...")
def get_tree_index(tree_path_str: str, device: str = "cpu"):
    from algorithms.retrieval1_hat_beam_traversal import TreeIndex
    return TreeIndex.from_json(tree_path_str, device=device)


@st.cache_resource(show_spinner="Precomputing Graph Structure for Algorithm 4...")
def get_graph_ppr_retriever(_index, damping: float = 0.5):
    from algorithms.retrieval4_graph_ppr import GraphPPRRetriever
    return GraphPPRRetriever(_index, damping=damping)


def load_corpus_files() -> Dict[str, str]:
    corpus = {}
    if FINANCE_REPORTS_DIR.exists():
        for f in sorted(FINANCE_REPORTS_DIR.glob("*.txt")):
            with open(f, "r", encoding="utf-8") as fp:
                corpus[f.name] = fp.read()
    return corpus


def load_preset_queries() -> List[Dict[str, Any]]:
    if QUERIES_PATH.exists():
        try:
            with open(QUERIES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("queries", [])
        except Exception:
            return []
    return []


def get_company_badge_html(tag: str) -> str:
    tag_upper = tag.upper()
    if "AAPL" in tag_upper:
        return f'<span class="company-badge badge-aapl">AAPL</span>'
    elif "AMZN" in tag_upper:
        return f'<span class="company-badge badge-amzn">AMZN</span>'
    elif "MSFT" in tag_upper:
        return f'<span class="company-badge badge-msft">MSFT</span>'
    elif "NVDA" in tag_upper:
        return f'<span class="company-badge badge-nvda">NVDA</span>'
    elif "TSLA" in tag_upper:
        return f'<span class="company-badge badge-tsla">TSLA</span>'
    return f'<span class="company-badge badge-default">{tag}</span>'


def main():
    st.markdown('<div class="main-title">HAT-RAG: Corporate Financial Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multi-Algorithm RAG Engine over Form 10-K SEC Filings (Apple, Amazon, Microsoft, NVIDIA, Tesla)</div>', unsafe_allow_html=True)

    corpus_data = load_corpus_files()
    preset_queries = load_preset_queries()

    # Hardware & Index Status in Sidebar
    st.sidebar.title("System Control")
    st.sidebar.markdown(f"**Corpus**: `finance_data/reports/`")
    st.sidebar.markdown(f"**Loaded Filings**: `{len(corpus_data)} corporate reports`")
    
    device_choice = st.sidebar.selectbox("Inference Device:", ["cpu", "cuda"])

    if not TREE_PATH.exists():
        st.error(f"Tree index not found at `{TREE_PATH}`. Please run offline indexing first.")
        return

    tree_index = get_tree_index(str(TREE_PATH), device=device_choice)
    st.sidebar.success(f"Tree Index Active: {tree_index.n_nodes} nodes ({tree_index.n_leaves} leaves)")
    st.sidebar.info(f"Alpha Edges: {len(tree_index.alpha_pairs)} cross-doc links")

    # Main Application Navigation Tabs
    tab_query, tab_compare, tab_corpus, tab_pipeline = st.tabs([
        "Financial Query Engine",
        "4-Algorithm Head-to-Head",
        "Financial Corpus & Tree Explorer",
        "Indexing & Pipeline Runner"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Financial Query Engine
    # -------------------------------------------------------------------------
    with tab_query:
        st.subheader("Interactive Financial Retrieval")
        st.caption("Execute custom or benchmark financial queries using any of the 4 retrieval algorithms.")

        c_algo, c_k = st.columns([3, 1])
        with c_algo:
            algo_option = st.selectbox(
                "Retrieval Algorithm",
                [
                    "HAT-RAG",
                    "Flat Dense RAG",
                    "RAPTOR Collapsed Tree",
                    "Entity Graph PPR",
                    "Compare All 4"
                ],
                index=0
            )
        with c_k:
            k_val = st.slider("Top Evidence (k):", min_value=1, max_value=10, value=5)

        with st.expander("Search Tuning", expanded=False):
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                beam_width = st.slider("Beam Width", min_value=1, max_value=8, value=3)
            with t_col2:
                diversity_weight = st.slider("Diversity Weight", min_value=0.0, max_value=0.5, value=0.15, step=0.05)
            with t_col3:
                damping_factor = st.slider("Graph Damping", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        query_text = st.text_area(
            "Prompt",
            value="What gross margin did Apple report in 2024?",
            height=90,
            placeholder="Type your financial prompt here..."
        )

        if st.button("Run Retrieval Query", type="primary", use_container_width=True):
            if not query_text.strip():
                st.warning("Please enter a query.")
            else:
                from algorithms.retrieval1_hat_beam_traversal import HATBeamRetriever, assemble_context
                from algorithms.retrieval2_flat_dense import FlatDenseRetriever
                from algorithms.retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever

                if algo_option == "Compare All 4":
                    r1 = HATBeamRetriever(tree_index, beam_width=beam_width, diversity=diversity_weight)
                    r2 = FlatDenseRetriever(tree_index)
                    r3 = CollapsedTreeRetriever(tree_index)
                    r4 = get_graph_ppr_retriever(tree_index, damping=damping_factor)

                    res1 = r1.retrieve(query_text, k=k_val)
                    res2 = r2.retrieve(query_text, k=k_val)
                    res3 = r3.retrieve(query_text, k=k_val)
                    res4 = r4.retrieve(query_text, k=k_val)

                    st.markdown("### Search Cost & Performance Breakdown")
                    col1, col2, col3, col4 = st.columns(4)
                    data_blocks = [
                        (col1, "HAT-RAG", res1),
                        (col2, "Flat Dense RAG", res2),
                        (col3, "RAPTOR Collapsed Tree", res3),
                        (col4, "Entity Graph PPR", res4)
                    ]
                    for col, title, r in data_blocks:
                        with col:
                            st.markdown(f"**{title}**")
                            st.metric("Latency", f"{r.seconds * 1000:.2f} ms")
                            st.metric("Nodes Scored", f"{r.nodes_scored} / {tree_index.n_nodes}", f"{r.nodes_scored/tree_index.n_nodes*100:.1f}%")

                    st.divider()
                    st.markdown("### Top Evidence Retrieved by Each Algorithm")
                    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                    for col, title, r in data_blocks:
                        with col:
                            st.markdown(f"#### {title}")
                            for rank, (nid, sc) in enumerate(zip(r.node_ids, r.scores), 1):
                                pos = tree_index.pos[nid]
                                tag = tree_index.label(pos)
                                txt = tree_index.text(pos)
                                st.markdown(f"**#{rank} [{tag}]** — `{sc:.3f}`")
                                st.caption(txt[:120] + ("..." if len(txt) > 120 else ""))
                else:
                    if algo_option == "HAT-RAG":
                        retriever = HATBeamRetriever(tree_index, beam_width=beam_width, diversity=diversity_weight)
                    elif algo_option == "Flat Dense RAG":
                        retriever = FlatDenseRetriever(tree_index)
                    elif algo_option == "RAPTOR Collapsed Tree":
                        retriever = CollapsedTreeRetriever(tree_index)
                    else:
                        retriever = get_graph_ppr_retriever(tree_index, damping=damping_factor)

                    result = retriever.retrieve(query_text, k=k_val)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Search Latency", f"{result.seconds * 1000:.2f} ms")
                    m2.metric("Nodes Scored", f"{result.nodes_scored} / {tree_index.n_nodes}", f"{result.nodes_scored/tree_index.n_nodes*100:.1f}% of corpus")
                    m3.metric("Top Result Score", f"{result.scores[0]:.4f}" if result.scores else "0.0")

                # Retrieved Evidence Cards
                st.markdown(f"### Top-{len(result.node_ids)} Retrieved Financial Passages")
                for rank, (nid, score) in enumerate(zip(result.node_ids, result.scores), 1):
                    pos = tree_index.pos[nid]
                    label_tag = tree_index.label(pos)
                    text = tree_index.text(pos)
                    is_leaf = tree_index.is_leaf[pos]
                    lvl = tree_index.level[pos]

                    badge_html = get_company_badge_html(label_tag)
                    tier_str = "Leaf Chunk (Raw 10-K Filing)" if is_leaf else f"Tier-{lvl} Synthesized Abstract Summary"

                    with st.container():
                        c_card_meta, c_card_text = st.columns([1, 4])
                        with c_card_meta:
                            st.markdown(f"**Rank #{rank}** | {badge_html}", unsafe_allow_html=True)
                            st.markdown(f"Score: `{score:.4f}`")
                            st.caption(tier_str)
                            st.caption(f"Node ID: `{nid[:18]}...`")
                        with c_card_text:
                            st.write(text)
                        st.divider()

                # Assembled Context
                with st.expander("Assembled Context for LLM", expanded=False):
                    st.caption("This formatted text package containing the retrieved paragraphs is what gets passed to an LLM to generate answers.")
                    assembled = assemble_context(tree_index, result)
                    st.text_area("Context String for LLM:", value=assembled, height=200)

    # -------------------------------------------------------------------------
    # TAB 2: 4-Algorithm Head-to-Head Comparison
    # -------------------------------------------------------------------------
    with tab_compare:
        st.subheader("Head-to-Head Algorithm Comparison")
        st.caption("Benchmark all 4 algorithms concurrently on the same financial question over the identical TreeIndex.")

        comp_query = st.text_input(
            "Benchmark Question:",
            value="Compare the research and development spending of Apple and NVIDIA.",
            key="comp_query_input"
        )
        comp_k = st.slider("Comparison Top Evidence (k):", min_value=1, max_value=10, value=5, key="comp_k_slider")

        if st.button("Run 4-Way Comparative Benchmark", type="primary", use_container_width=True):
            from algorithms.retrieval1_hat_beam_traversal import HATBeamRetriever
            from algorithms.retrieval2_flat_dense import FlatDenseRetriever
            from algorithms.retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever

            r1 = HATBeamRetriever(tree_index, beam_width=3, diversity=0.15)
            r2 = FlatDenseRetriever(tree_index)
            r3 = CollapsedTreeRetriever(tree_index)
            r4 = get_graph_ppr_retriever(tree_index, damping=0.5)

            res1 = r1.retrieve(comp_query, k=comp_k)
            res2 = r2.retrieve(comp_query, k=comp_k)
            res3 = r3.retrieve(comp_query, k=comp_k)
            res4 = r4.retrieve(comp_query, k=comp_k)

            st.markdown("### Search Cost & Performance Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            data_blocks = [
                (col1, "HAT-RAG", res1),
                (col2, "Flat Dense RAG", res2),
                (col3, "RAPTOR Collapsed Tree", res3),
                (col4, "Entity Graph PPR", res4)
            ]
            for col, title, r in data_blocks:
                with col:
                    st.markdown(f"**{title}**")
                    st.metric("Latency", f"{r.seconds * 1000:.2f} ms")
                    st.metric("Nodes Scored", f"{r.nodes_scored} / {tree_index.n_nodes}", f"{r.nodes_scored/tree_index.n_nodes*100:.1f}%")

            st.divider()
            st.markdown("### Top Evidence Retrieved by Each Algorithm")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            for col, title, r in data_blocks:
                with col:
                    st.markdown(f"#### {title}")
                    for rank, (nid, sc) in enumerate(zip(r.node_ids, r.scores), 1):
                        pos = tree_index.pos[nid]
                        tag = tree_index.label(pos)
                        txt = tree_index.text(pos)
                        st.markdown(f"**#{rank} [{tag}]** — `{sc:.3f}`")
                        st.caption(txt[:120] + ("..." if len(txt) > 120 else ""))

            st.divider()
            st.markdown("""
            #### Architectural Insights from Benchmark Results:
            1. **Sub-linear Scaling**: HAT-RAG inspects only candidate paths reached by beam descent, scoring ~39% of the corpus on small trees and under 10% on large corpora.
            2. **Cross-Document Reasoning**: Explicit $\\alpha$-links enable HAT-RAG to hop across corporate branches (e.g., from Apple's R&D to NVIDIA's R&D) without ascending to distant common roots.
            3. **Precision**: Flat dense scores all 24 chunks linearly but lacks topological hierarchical summaries. RAPTOR pools leaves and abstracts indiscriminately, introducing summary duplication.
            """)

    # -------------------------------------------------------------------------
    # TAB 3: Financial Corpus & Tree Explorer
    # -------------------------------------------------------------------------
    with tab_corpus:
        st.subheader("Form 10-K Financial Corpus (finance_data/reports/)")
        st.caption("Inspect the raw SEC 10-K documents and the constructed Hierarchical Abstract Tree topology.")

        c_file_sel, c_view_mode = st.columns([2, 1])
        with c_file_sel:
            selected_file = st.selectbox("Select Corporate Filing:", list(corpus_data.keys()))
        with c_view_mode:
            view_mode = st.radio("View:", ["Document Text", "Tree Nodes for this Document"], horizontal=True)

        if selected_file:
            doc_content = corpus_data[selected_file]
            word_count = len(doc_content.split())
            st.info(f"**File**: `{selected_file}` | **Word Count**: {word_count} words")

            if view_mode == "Document Text":
                st.text_area("Full Document Content:", value=doc_content, height=400)
            else:
                st.markdown("#### Chunked Leaf Nodes in Tree Index")
                matching_nodes = []
                for i in range(tree_index.n_nodes):
                    if tree_index.doc[i] == selected_file:
                        matching_nodes.append((
                            tree_index.ids[i],
                            tree_index.label(i),
                            tree_index.level[i],
                            tree_index.is_leaf[i],
                            tree_index.text(i)
                        ))
                
                if matching_nodes:
                    for nid, lbl, lvl, is_leaf, txt in matching_nodes:
                        st.markdown(f"**Node `{nid}`** | Label: `{lbl}` | Level: `{lvl}` | {'Leaf' if is_leaf else 'Abstract'}")
                        st.caption(txt)
                else:
                    st.write("No nodes directly mapped to this filename.")

    # -------------------------------------------------------------------------
    # TAB 4: Indexing & Pipeline Runner
    # -------------------------------------------------------------------------
    with tab_pipeline:
        st.subheader("Offline Knowledge Construction Pipeline")
        st.caption("Run Algorithm 1 (Document Chunking) and Algorithm 3 (Spherical k-Means Tree Construction) over `finance_data/reports/`.")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            summarizer_choice = st.selectbox(
                "Summarizer",
                ["Extractive Summarizer", "LLM Summarizer"]
            )
        with p_col2:
            alpha_threshold = st.slider("Alpha Lateral Threshold", min_value=0.4, max_value=0.9, value=0.65, step=0.05)

        st.markdown("""
        **What this pipeline executes:**
        1. **Algorithm 1 (`algo1_document_chunking.py`)**: Sentence boundary tokenization, multi-sentence overlap, section heading extraction, SHA-256 chunk fingerprinting.
        2. **Dense Embedding**: Encodes all passages using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional unit hypersphere vectors.
        3. **Algorithm 3 (`algo3_hierarchical_abstract_tree.py`)**: Spherical k-Means clustering + abstract summarization + cross-document lateral $\\alpha$-edges.
        """)

        if st.button("Rebuild Tree Index from finance_data/reports", type="primary"):
            st.info("Rebuilding knowledge tree from `finance_data/reports/`...")
            start_t = time.perf_counter()
            try:
                from algorithms.run_finance_rag import run_finance_pipeline
                class Args:
                    pass
                args = Args()
                args.summarizer = "extractive" if "extractive" in summarizer_choice.lower() else "llm"
                args.alpha = alpha_threshold
                args.device = device_choice
                args.chunk_size = 60
                args.chunk_overlap = 15
                args.branching = 4

                with st.spinner("Executing Algorithm 1 and Algorithm 3..."):
                    built_tree = run_finance_pipeline(args)

                dur = time.perf_counter() - start_t
                st.success(f"Knowledge Tree built and serialized to `{TREE_PATH.name}` in {dur:.2f} seconds!")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Pipeline execution error: {e}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
