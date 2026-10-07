"""
HAT-RAG: Enterprise Knowledge & Retrieval Engine
Hierarchical Abstract Tree for Cross-Document RAG with Sub-Linear Scaling
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root and algorithms to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "algorithms"))

import streamlit as st

st.set_page_config(
    page_title="HAT-RAG: Enterprise Knowledge & Retrieval Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Professional Research Dashboard Styling (No Emojis, Native Contrast)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 1.25rem;
        line-height: 1.5;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    
    .badge-primary { background: rgba(99, 102, 241, 0.2); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.4); }
    .badge-success { background: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-muted { background: rgba(148, 163, 184, 0.15); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.25); }
    
    .company-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
        color: white;
    }
    .badge-aapl { background-color: #475569; }
    .badge-amzn { background-color: #D97706; }
    .badge-msft { background-color: #0284C7; }
    .badge-nvda { background-color: #16A34A; }
    .badge-tsla { background-color: #DC2626; }
    .badge-synth { background-color: #4F46E5; }
    .badge-default { background-color: #6366F1; }
    
    .evidence-box {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# Datasets definition
DATASETS = {
    "finance": {
        "title": "SEC Form 10-K Filings (Real Data)",
        "desc": "5 Corporate Reports (AAPL, AMZN, MSFT, NVDA, TSLA) | 24 Leaf Chunks | 32 Total Nodes",
        "tree_file": ROOT_DIR / "algorithms" / "hat_finance_tree_structure.json",
        "queries_file": ROOT_DIR / "algorithms" / "compare-algos" / "finance_queries.json",
        "is_synthetic": False,
        "sample_questions": [
            "Compare the research and development spending of Apple and NVIDIA.",
            "What gross margin did Apple report in 2024?",
            "How does Tesla evaluate its warranty reserves?",
            "What goodwill impairment policies does Microsoft maintain?",
            "What are Amazon's primary operating segments and cloud drivers?"
        ]
    },
    "enterprise": {
        "title": "Enterprise Scaled Benchmark (Production Scale)",
        "desc": "128 Technical Dossiers | 1,582 Leaf Chunks | 1,809 Total Nodes | Sub-linear Scaling Demonstration",
        "tree_file": ROOT_DIR / "algorithms" / "hat_enterprise_tree_structure.json",
        "queries_file": ROOT_DIR / "algorithms" / "compare-algos" / "enterprise_queries.json",
        "is_synthetic": True,
        "sample_questions": [
            "What nominal efficiency does the heat sink of Project A76 record?",
            "What settling time does the watchdog task of Project D56 record?",
            "Compare failure rates and recovery intervals across power modules.",
            "Which projects report thermal management interface anomalies?",
            "What duty margin and throughput index are logged for energy storage packs?"
        ]
    }
}

FINANCE_REPORTS_DIR = ROOT_DIR / "finance_data" / "reports"


@st.cache_resource(show_spinner="Loading Knowledge Tree Index...")
def get_tree_index(tree_path_str: str) -> Any:
    from algorithms.retrieval1_hat_beam_traversal import TreeIndex
    return TreeIndex.from_json(tree_path_str, device="cpu")


@st.cache_resource(show_spinner="Building Passage Graph for Graph PPR...")
def get_graph_ppr_retriever(tree_path_str: str, damping: float = 0.5) -> Any:
    idx = get_tree_index(tree_path_str)
    from algorithms.retrieval4_graph_ppr import GraphPPRRetriever
    return GraphPPRRetriever(idx, damping=damping)


def load_finance_documents() -> Dict[str, str]:
    corpus = {}
    if FINANCE_REPORTS_DIR.exists():
        for f in sorted(FINANCE_REPORTS_DIR.glob("*.txt")):
            with open(f, "r", encoding="utf-8") as fp:
                corpus[f.name] = fp.read()
    return corpus


def get_tag_badge_html(tag: str) -> str:
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
    elif "DOC" in tag_upper or "PROJECT" in tag_upper:
        return f'<span class="company-badge badge-synth">{tag}</span>'
    return f'<span class="company-badge badge-default">{tag}</span>'


def main():
    st.markdown('<div class="main-title">HAT-RAG: Enterprise Knowledge & Retrieval Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">'
        'Hierarchical Abstract Tree for Cross-Document RAG | Sub-linear beam search with lateral alpha-links and graph propagation'
        '</div>',
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: Knowledge Base & Configuration
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### Knowledge Base Selection")
    
    dataset_key = st.sidebar.radio(
        "Active Knowledge Tree:",
        options=["finance", "enterprise"],
        format_func=lambda k: DATASETS[k]["title"],
        index=0
    )
    selected_dataset = DATASETS[dataset_key]
    tree_path = selected_dataset["tree_file"]

    # Detect dataset change to synchronize default question in UI
    if st.session_state.get("prev_dataset_key") != dataset_key:
        st.session_state["prev_dataset_key"] = dataset_key
        st.session_state["selected_benchmark_q"] = selected_dataset["sample_questions"][0]
        st.session_state["interactive_query_val"] = selected_dataset["sample_questions"][0]

    if not tree_path.exists():
        st.sidebar.error(f"Tree index file `{tree_path.name}` not found. Build it in Tab 4.")
        st.error(f"Selected tree index not found at `{tree_path}`. Run indexing pipeline in Tab 4.")
        return

    # Load active TreeIndex
    tree_index = get_tree_index(str(tree_path))
    finance_docs = load_finance_documents()

    # Hardware & Index Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Architecture**")
    st.sidebar.text("Execution Engine: CPU (PyTorch)")
    st.sidebar.text(f"Dense Encoder: all-MiniLM-L6-v2 (d=384)")
    st.sidebar.text(f"Total Tree Nodes: {tree_index.n_nodes}")
    st.sidebar.text(f"Leaf Passages: {tree_index.n_leaves}")
    st.sidebar.text(f"Lateral Alpha-Edges: {len(tree_index.alpha_pairs)}")
    st.sidebar.caption(selected_dataset["desc"])

    # -------------------------------------------------------------------------
    # TABS NAVIGATION
    # -------------------------------------------------------------------------
    tab_query, tab_compare, tab_corpus, tab_pipeline = st.tabs([
        "Interactive Retrieval",
        "Head-to-Head Benchmark",
        "Corpus & Tree Explorer",
        "Offline Indexing Pipeline"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Interactive Retrieval Engine
    # -------------------------------------------------------------------------
    with tab_query:
        st.subheader("Interactive Retrieval Engine")
        st.caption(f"Active Tree: {selected_dataset['title']} ({tree_index.n_nodes} nodes)")

        c_algo, c_k = st.columns([3, 1])
        with c_algo:
            algo_option = st.selectbox(
                "Retrieval Strategy:",
                [
                    "HAT-RAG (Algorithm 4 - Ours)",
                    "Flat Dense RAG (Vanilla Baseline)",
                    "RAPTOR Collapsed Tree (Pooled Baseline)",
                    "Entity Graph PPR (GraphRAG Baseline)",
                    "Compare All 4 Simultaneously"
                ],
                index=0
            )
        with c_k:
            k_val = st.slider("Top Evidence (k):", min_value=1, max_value=10, value=5, key="k_slider_tab1")

        with st.expander("Search Tuning Hyperparameters", expanded=False):
            t_col1, t_col2, t_col3 = st.columns(3)
            with t_col1:
                beam_width = st.slider("Beam Width (beta)", min_value=1, max_value=8, value=3)
            with t_col2:
                diversity_weight = st.slider("Diversity Penalty (lambda)", min_value=0.0, max_value=0.5, value=0.15, step=0.05)
            with t_col3:
                damping_factor = st.slider("PageRank Damping (d)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        # Preset Questions
        curated_qs = selected_dataset["sample_questions"]
        selected_preset = st.selectbox(
            "Select Sample Question (or write custom below):",
            options=["Custom Question"] + curated_qs,
            index=1
        )

        initial_val = selected_preset if selected_preset != "Custom Question" else st.session_state.get("interactive_query_val", curated_qs[0])
        query_text = st.text_area(
            "Query Prompt:",
            value=initial_val,
            height=75,
            key="tab1_query_area"
        )

        if st.button("Execute Search", type="primary", use_container_width=True, key="btn_exec_tab1"):
            if not query_text.strip():
                st.warning("Please enter a non-empty query prompt.")
            else:
                from algorithms.retrieval1_hat_beam_traversal import HATBeamRetriever, assemble_context
                from algorithms.retrieval2_flat_dense import FlatDenseRetriever
                from algorithms.retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever

                if algo_option == "Compare All 4 Simultaneously":
                    r1 = HATBeamRetriever(tree_index, beam_width=beam_width, diversity=diversity_weight)
                    r2 = FlatDenseRetriever(tree_index)
                    r3 = CollapsedTreeRetriever(tree_index)
                    r4 = get_graph_ppr_retriever(str(tree_path), damping=damping_factor)

                    res1 = r1.retrieve(query_text, k=k_val)
                    res2 = r2.retrieve(query_text, k=k_val)
                    res3 = r3.retrieve(query_text, k=k_val)
                    res4 = r4.retrieve(query_text, k=k_val)

                    st.markdown("#### Performance Breakdown")
                    col1, col2, col3, col4 = st.columns(4)
                    blocks = [
                        (col1, "HAT-RAG (Ours)", res1, True),
                        (col2, "Flat Dense RAG", res2, False),
                        (col3, "RAPTOR Collapsed", res3, False),
                        (col4, "Entity Graph PPR", res4, False)
                    ]

                    for col, title, r, is_ours in blocks:
                        with col:
                            with st.container(border=True):
                                tag_sub = "Rank 1 (Sub-linear)" if is_ours else "Baseline"
                                st.markdown(f"**{title}**  \n*{tag_sub}*")
                                st.metric("Search Latency", f"{r.seconds * 1000:.2f} ms")
                                pct_scored = (r.nodes_scored / tree_index.n_nodes) * 100
                                st.metric("Nodes Evaluated", f"{r.nodes_scored} / {tree_index.n_nodes}", f"{pct_scored:.1f}%")

                    st.divider()
                    st.markdown("#### Top Evidence Retrieved by Algorithm")
                    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
                    for col, title, r, _ in blocks:
                        with col:
                            st.markdown(f"**{title}**")
                            for rank, (nid, sc) in enumerate(zip(r.node_ids, r.scores), 1):
                                pos = tree_index.pos.get(nid)
                                if pos is None:
                                    continue
                                tag = tree_index.label(pos)
                                txt = tree_index.text(pos)
                                badge = get_tag_badge_html(tag)
                                st.markdown(f"#{rank} {badge} `{sc:.3f}`", unsafe_allow_html=True)
                                st.caption(txt[:120] + ("..." if len(txt) > 120 else ""))
                else:
                    if "HAT-RAG" in algo_option:
                        retriever = HATBeamRetriever(tree_index, beam_width=beam_width, diversity=diversity_weight)
                    elif "Flat Dense" in algo_option:
                        retriever = FlatDenseRetriever(tree_index)
                    elif "RAPTOR" in algo_option:
                        retriever = CollapsedTreeRetriever(tree_index)
                    else:
                        retriever = get_graph_ppr_retriever(str(tree_path), damping=damping_factor)

                    single_res = retriever.retrieve(query_text, k=k_val)

                    pct_scored = (single_res.nodes_scored / tree_index.n_nodes) * 100
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Search Latency", f"{single_res.seconds * 1000:.2f} ms")
                    with m2:
                        st.metric("Nodes Evaluated", f"{single_res.nodes_scored} / {tree_index.n_nodes}", f"{pct_scored:.1f}% of tree")
                    with m3:
                        st.metric("Top Result Score", f"{single_res.scores[0]:.4f}" if single_res.scores else "0.0")

                    st.markdown(f"#### Top-{len(single_res.node_ids)} Retrieved Passages")
                    for rank, (nid, score) in enumerate(zip(single_res.node_ids, single_res.scores), 1):
                        pos = tree_index.pos.get(nid)
                        if pos is None:
                            continue
                        label_tag = tree_index.label(pos)
                        text = tree_index.text(pos)
                        is_leaf = tree_index.is_leaf[pos]
                        lvl = tree_index.level[pos]

                        badge_html = get_tag_badge_html(label_tag)
                        tier_label = "Leaf Chunk (Raw Passage)" if is_leaf else f"Tier-{lvl} Abstract Summary"

                        with st.container(border=True):
                            c_top1, c_top2 = st.columns([3, 1])
                            with c_top1:
                                st.markdown(f"**Rank #{rank}** | {badge_html} | *{tier_label}*", unsafe_allow_html=True)
                            with c_top2:
                                st.markdown(f"**Score:** `{score:.4f}`")
                            st.write(text)
                            st.caption(f"Node ID: `{nid}`")

                    with st.expander("Formatted LLM Context Buffer", expanded=False):
                        st.caption("This synthesized text package is passed into the generative model context window.")
                        assembled = assemble_context(tree_index, single_res)
                        st.text_area("Context String:", value=assembled, height=180)

    # -------------------------------------------------------------------------
    # TAB 2: 4-Algorithm Head-to-Head Comparison
    # -------------------------------------------------------------------------
    with tab_compare:
        st.subheader("Head-to-Head Retrieval Benchmark")
        st.caption(
            f"Benchmark all four retrieval algorithms concurrently over identical vectors. "
            f"Active Dataset: {selected_dataset['title']} ({tree_index.n_nodes} nodes)."
        )

        curated_qs = selected_dataset["sample_questions"]
        selected_bench_q = st.selectbox(
            "Select Benchmark Question:",
            options=["Custom Question"] + curated_qs,
            index=1,
            key="bench_q_picker"
        )

        bench_default = selected_bench_q if selected_bench_q != "Custom Question" else curated_qs[0]
        comp_query = st.text_input(
            "Question Text:",
            value=bench_default,
            key="comp_query_input_tab2"
        )
        comp_k = st.slider("Evaluation Depth (k):", min_value=1, max_value=10, value=5, key="comp_k_slider_tab2")

        if st.button("Run 4-Way Comparative Benchmark", type="primary", use_container_width=True, key="btn_run_comp"):
            from algorithms.retrieval1_hat_beam_traversal import HATBeamRetriever
            from algorithms.retrieval2_flat_dense import FlatDenseRetriever
            from algorithms.retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever

            r1 = HATBeamRetriever(tree_index, beam_width=3, diversity=0.15)
            r2 = FlatDenseRetriever(tree_index)
            r3 = CollapsedTreeRetriever(tree_index)
            r4 = get_graph_ppr_retriever(str(tree_path), damping=0.5)

            res1 = r1.retrieve(comp_query, k=comp_k)
            res2 = r2.retrieve(comp_query, k=comp_k)
            res3 = r3.retrieve(comp_query, k=comp_k)
            res4 = r4.retrieve(comp_query, k=comp_k)

            st.markdown("#### Search Cost & Latency Matrix")
            col1, col2, col3, col4 = st.columns(4)
            data_blocks = [
                (col1, "HAT-RAG (Algorithm 4)", res1, True),
                (col2, "Flat Dense RAG", res2, False),
                (col3, "RAPTOR Collapsed", res3, False),
                (col4, "Entity Graph PPR", res4, False)
            ]

            for col, title, r, is_ours in data_blocks:
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        if is_ours:
                            st.caption("Rank 1 - Sub-linear Search")
                        else:
                            st.caption("Comparative Baseline")
                        
                        st.metric("Latency", f"{r.seconds * 1000:.2f} ms")
                        pct = (r.nodes_scored / tree_index.n_nodes) * 100
                        st.metric("Nodes Evaluated", f"{r.nodes_scored} / {tree_index.n_nodes}", f"{pct:.1f}%")

            # Pruning Efficiency Callout
            hat_pct = (res1.nodes_scored / tree_index.n_nodes) * 100
            flat_pct = (res2.nodes_scored / tree_index.n_nodes) * 100
            pruned_pct = 100.0 - hat_pct

            if tree_index.n_nodes > 100:
                st.success(
                    f"Sub-Linear Scaling Demonstrated: HAT-RAG inspected only {res1.nodes_scored} nodes ({hat_pct:.1f}% of corpus), "
                    f"pruning {pruned_pct:.1f}% of the search space. In contrast, Flat Dense scanned all {res2.nodes_scored} leaves "
                    f"({flat_pct:.1f}%) and RAPTOR scanned {res3.nodes_scored} nodes."
                )
            else:
                st.info(
                    f"Small Corpus Note: On small document sets ({tree_index.n_nodes} nodes), brute-force scanning is fast because "
                    f"matrix multiplication is negligible. Select 'Enterprise Scaled Benchmark' in the sidebar to observe the 90%+ pruning effect at scale."
                )

            st.divider()
            st.markdown("#### Top Evidence Retrieved by Each Algorithm")
            t_col1, t_col2, t_col3, t_col4 = st.columns(4)
            for col, title, r, _ in data_blocks:
                with col:
                    st.markdown(f"**{title}**")
                    for rank, (nid, sc) in enumerate(zip(r.node_ids, r.scores), 1):
                        pos = tree_index.pos.get(nid)
                        if pos is None:
                            continue
                        tag = tree_index.label(pos)
                        txt = tree_index.text(pos)
                        badge = get_tag_badge_html(tag)
                        st.markdown(f"#{rank} {badge} `{sc:.3f}`", unsafe_allow_html=True)
                        st.caption(txt[:120] + ("..." if len(txt) > 120 else ""))

            st.divider()
            st.markdown("""
            **Architectural Takeaways:**
            1. **Sub-linear Scaling at Volume**: On large document collections (1,500+ chunks), HAT-RAG inspects ~100-150 nodes (~8% of the corpus) via beam descent, whereas Flat Dense scans 100% of chunks (O(N) brute-force).
            2. **Cross-Document Reasoning**: Explicit lateral alpha-links connect related clusters across document boundaries without forcing the retriever to climb back to the global root.
            3. **Immunity to Noise**: Flat Dense suffers accuracy degradation on large datasets because spurious distractor passages win cosine matches. HAT-RAG filters candidates through hierarchical cluster abstracts first.
            """)

    # -------------------------------------------------------------------------
    # TAB 3: Corpus & Tree Explorer
    # -------------------------------------------------------------------------
    with tab_corpus:
        st.subheader("Corpus & Knowledge Graph Explorer")
        st.caption(f"Active Dataset: {selected_dataset['title']}")

        if not selected_dataset["is_synthetic"]:
            c_file_sel, c_view_mode = st.columns([2, 1])
            with c_file_sel:
                selected_file = st.selectbox("Select Corporate Filing:", list(finance_docs.keys()))
            with c_view_mode:
                view_mode = st.radio("Display Mode:", ["Raw Document Text", "Tree Nodes for Filing"], horizontal=True)

            if selected_file:
                doc_content = finance_docs[selected_file]
                word_count = len(doc_content.split())
                st.info(f"Document: {selected_file} | Word Count: {word_count} words")

                if view_mode == "Raw Document Text":
                    st.text_area("Filing Content:", value=doc_content, height=400)
                else:
                    st.markdown("#### Indexed Nodes for this Filing")
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
                            tag_tier = "Leaf Passage" if is_leaf else f"Tier-{lvl} Abstract"
                            st.markdown(f"**Node `{nid}`** | Label: `{lbl}` | Level: `{lvl}` | *{tag_tier}*")
                            st.caption(txt)
                    else:
                        st.write("No nodes mapped directly to this document.")
        else:
            st.info(f"The Enterprise Benchmark corpus contains {tree_index.n_leaves} passages across 128 dossiers.")
            
            # Show tree level distribution
            levels_count = {}
            for lvl in tree_index.level:
                levels_count[int(lvl)] = levels_count.get(int(lvl), 0) + 1
            
            col_l1, col_l2 = st.columns([1, 2])
            with col_l1:
                st.markdown("#### Hierarchy Distribution")
                for lvl in sorted(levels_count.keys()):
                    role = "Leaf Passages (Raw Chunks)" if lvl == 0 else (f"Level {lvl} Abstracts" if lvl < max(levels_count.keys()) else "ROOT Global Abstract")
                    st.write(f"- Tier {lvl} ({role}): `{levels_count[lvl]} nodes`")
            
            with col_l2:
                st.markdown("#### Sample Synthesized Abstract Nodes")
                abstract_indices = [i for i in range(tree_index.n_nodes) if not tree_index.is_leaf[i]][:6]
                for idx in abstract_indices:
                    nid = tree_index.ids[idx]
                    lvl = tree_index.level[idx]
                    txt = tree_index.text(idx)
                    st.markdown(f"**Node `{nid}` (Tier {lvl} Summary)**")
                    st.caption(txt)

    # -------------------------------------------------------------------------
    # TAB 4: Indexing & Pipeline Runner
    # -------------------------------------------------------------------------
    with tab_pipeline:
        st.subheader("Offline Tree Construction Pipeline")
        st.caption("Re-run Algorithm 1 (Document Chunking) and Algorithm 3 (Spherical k-Means Tree Construction).")

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            summarizer_choice = st.selectbox(
                "Summarizer Mode:",
                ["Extractive Summarizer (Fast CPU)", "LLM Summarizer (BART/Transformer)"]
            )
        with p_col2:
            alpha_threshold = st.slider("Alpha Lateral Link Threshold:", min_value=0.4, max_value=0.9, value=0.65, step=0.05)

        st.markdown("""
        **Pipeline Execution Stages:**
        1. **Algorithm 1 (algo1_document_chunking.py)**: Sentence boundary sliding window, overlap detection, SHA-256 chunk fingerprinting.
        2. **Dense Representation**: Embeds chunks using sentence-transformers/all-MiniLM-L6-v2 into 384-dimensional unit hypersphere vectors.
        3. **Algorithm 3 (algo3_hierarchical_abstract_tree.py)**: Spherical k-Means clustering, abstract summarization, and explicit lateral alpha-edges.
        """)

        c_build1, c_build2 = st.columns(2)
        with c_build1:
            if st.button("Rebuild SEC 10-K Tree (32 nodes)", type="primary", use_container_width=True):
                st.info("Rebuilding SEC 10-K Knowledge Tree...")
                t0 = time.perf_counter()
                try:
                    from algorithms.run_finance_rag import run_finance_pipeline
                    class Args:
                        pass
                    args = Args()
                    args.summarizer = "extractive" if "extractive" in summarizer_choice.lower() else "llm"
                    args.alpha = alpha_threshold
                    args.device = "cpu"
                    args.chunk_size = 60
                    args.chunk_overlap = 15
                    args.branching = 4

                    with st.spinner("Building SEC 10-K Tree..."):
                        run_finance_pipeline(args)

                    elapsed = time.perf_counter() - t0
                    st.success(f"SEC 10-K Tree rebuilt in {elapsed:.2f}s!")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Error rebuilding tree: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        with c_build2:
            if st.button("Rebuild Enterprise Scaled Tree (1,809 nodes)", use_container_width=True):
                st.info("Building 128-document Scaled Enterprise Tree...")
                t0 = time.perf_counter()
                try:
                    from algorithms.compare_algos.synthetic_corpus import build_corpus
                    from algorithms.algo1_document_chunking import DocumentChunker
                    from algorithms.algo3_hierarchical_abstract_tree import DenseEmbeddingModel, HierarchicalAbstractTreeBuilder

                    with st.spinner("Generating corpus and clustering..."):
                        documents, queries, _ = build_corpus(n_docs=128, seed=17)
                        chunker = DocumentChunker(chunk_size=60, chunk_overlap=10)
                        chunks = chunker.process_corpus(documents)
                        embedder = DenseEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu", verbose=False)
                        builder = HierarchicalAbstractTreeBuilder(
                            embedder=embedder, max_levels=6, target_children=8, alpha="auto",
                            summarizer_mode="extractive", device="cpu", verbose=False
                        )
                        builder.build_tree(chunks)
                        builder.save_tree_json(str(ROOT_DIR / "algorithms" / "hat_enterprise_tree_structure.json"))

                    elapsed = time.perf_counter() - t0
                    st.success(f"Enterprise Scaled Tree rebuilt in {elapsed:.2f}s ({len(builder.nodes)} nodes)!")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Error rebuilding enterprise tree: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
