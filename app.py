"""
HAT-RAG: Hierarchical Abstract Tree retrieval for large document estates.

    streamlit run app.py

Tabs
    Ask         chat with the knowledge base: any question -> retrieval (any of
                the four methods) -> grounded, cited answer from Groq / Gemini
    Compare     one question, all four retrievers side by side (+ their answers)
    Benchmarks  the evaluation results, interactive
    Corpus      browse documents and the tree
    Build       rebuild the indexes
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "algorithms", ROOT / "algorithms" / "compare-algos"):
    sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(page_title="HAT-RAG", page_icon=":material/account_tree:", layout="wide",
                   initial_sidebar_state="expanded")

from webapp import charts
from webapp.style import CSS, METHOD_COLORS, METHOD_FAMILY, METHOD_ORDER, dot, esc, hero, stats

st.markdown(CSS, unsafe_allow_html=True)

ALGO = ROOT / "algorithms"
QDIR = ALGO / "compare-algos"
MULTIHOP_TREE = ROOT / "data" / "multihop_rag" / "trees" / "hat_multihop_full.json"

KBS: Dict[str, Dict[str, Any]] = {
    "multihop": {
        "name": "MultiHop-RAG news archive",
        "short": "MultiHop-RAG",
        "tree": MULTIHOP_TREE,
        "blurb": "609 real articles from 49 publishers (TechCrunch, The Verge, BBC, Fortune ...), 1.06M words. "
                 "The large-scale benchmark: questions need evidence from 2-4 different articles.",
        "placeholder": "Ask anything about the late-2023 tech, business, science or sports news ...",
        "color": "#2a78d6",
    },
    "enterprise": {
        "name": "Engineering dossiers (synthetic)",
        "short": "Synthetic, 128 docs",
        "tree": ALGO / "hat_enterprise_tree_structure.json",
        "blurb": "128 generated technical dossiers with planted facts and cross-document bridges, "
                 "built so that surface similarity alone fails.",
        "placeholder": "e.g. What settling time does the watchdog task of Project D56 record?",
        "color": "#1baf7a",
    },
    "finance": {
        "name": "SEC 10-K summaries",
        "short": "10-K, 5 filings",
        "tree": ALGO / "hat_finance_tree_structure.json",
        "blurb": "Summaries of the 2024 10-K filings of Apple, Amazon, Microsoft, NVIDIA and Tesla. "
                 "Small: useful for reading every chunk, too small for cost differences to show.",
        "placeholder": "e.g. Compare the R&D spending of Apple and NVIDIA.",
        "color": "#eb6834",
    },
}
METHOD_LABEL = {"HAT-RAG": "HAT-RAG (ours)", "Flat dense": "Flat dense",
                "RAPTOR collapsed": "RAPTOR", "Graph PPR": "Graph PPR"}


# ============================================================================
# Cached resources
# ============================================================================
@st.cache_resource(show_spinner="Loading the encoder ...")
def get_embedder():
    from algo3_hierarchical_abstract_tree import DenseEmbeddingModel
    return DenseEmbeddingModel(device="cpu", verbose=False)


@st.cache_resource(show_spinner="Loading the knowledge tree ...")
def get_index(tree: str):
    from retrieval1_hat_beam_traversal import TreeIndex
    return TreeIndex.from_json(tree, embedder=get_embedder(), device="cpu")


@st.cache_resource(show_spinner="Preparing the four retrievers (one-off) ...")
def get_retrievers(tree: str, beam: int, lam: float, damping: float, n_seeds: int = 3,
                   max_link: Optional[int] = None):
    from retrieval1_hat_beam_traversal import HATBeamRetriever
    from retrieval2_flat_dense import FlatDenseRetriever
    from retrieval3_raptor_collapsed_tree import CollapsedTreeRetriever
    index = get_index(tree)
    return {"HAT-RAG": HATBeamRetriever(index, beam_width=beam, diversity=lam, n_seeds=n_seeds,
                                        max_link=max_link),
            "Flat dense": FlatDenseRetriever(index),
            "RAPTOR collapsed": CollapsedTreeRetriever(index),
            "Graph PPR": get_graph(tree, damping)}


@st.cache_resource(show_spinner="Building the passage graph for Graph PPR (one-off) ...")
def get_graph(tree: str, damping: float):
    from retrieval4_graph_ppr import GraphPPRRetriever
    return GraphPPRRetriever(get_index(tree), damping=damping)


@st.cache_data(show_spinner=False)
def get_doc_meta(kb: str) -> Dict[str, Dict[str, Any]]:
    if kb == "multihop":
        from multihop_corpus import article_meta, load_articles
        return article_meta(load_articles())
    if kb == "finance":
        names = {"AAPL": "Apple", "AMZN": "Amazon", "MSFT": "Microsoft", "NVDA": "NVIDIA", "TSLA": "Tesla"}
        out = {}
        for f in (ROOT / "finance_data" / "reports").glob("*.txt"):
            t = f.name.split("_")[0]
            out[f.name] = {"title": f"{names.get(t, t)} 10-K 2024", "source": t}
        return out
    from synthetic_corpus import build_corpus
    _, _, aliases = build_corpus(n_docs=128, seed=17)
    return {d: {"title": f"{a[0]} dossier", "source": d.split("_")[1].title()} for d, a in aliases.items()}


@st.cache_data(show_spinner=False)
def get_benchmark(kb: str) -> List[Dict[str, Any]]:
    import json
    if kb == "multihop":
        from multihop_corpus import load_articles, load_queries
        qs, _ = load_queries(load_articles())
        return qs
    name = "finance_queries.json" if kb == "finance" else "enterprise_queries.json"
    return json.loads((QDIR / name).read_text(encoding="utf-8"))["queries"]


@st.cache_data(show_spinner=False)
def get_doc_text(kb: str, doc_id: str) -> str:
    if kb == "multihop":
        from multihop_corpus import article_meta, load_articles
        arts = load_articles()
        return dict(zip(article_meta(arts), arts))[doc_id]["body"]
    if kb == "finance":
        return (ROOT / "finance_data" / "reports" / doc_id).read_text(encoding="utf-8")
    from synthetic_corpus import build_corpus
    docs, _, _ = build_corpus(n_docs=128, seed=17)
    return docs.get(doc_id, "")


@st.cache_resource(show_spinner=False)
def get_client(provider: str, model: Optional[str]):
    from llm_answer import LLMError, make_client
    try:
        return make_client(provider, model or None), None
    except LLMError as e:
        return None, str(e)


def suggestions(kb: str) -> List[str]:
    if kb == "multihop":
        picked, seen = [], set()
        for q in get_benchmark(kb):
            if q["split"] == "test" and len(q["query"]) < 170 and q["type"] not in seen:
                picked.append(q["query"])
                seen.add(q["type"])
        return ["What happened at OpenAI's board in November 2023, and how did Microsoft react?",
                "What are the main antitrust arguments the US government made against Google?"] + picked
    qs = get_benchmark(kb)
    return [q["query"] for q in qs if q["type"] == "cross"][:2] + [q["query"] for q in qs if q["type"] == "single"][:2]


# ============================================================================
# Small rendering helpers
# ============================================================================
def source_title(index, pos: int, meta: Dict[str, Dict[str, Any]]) -> str:
    node = index.nodes[pos]
    if node.level > 0:
        return f"Tier-{node.level} abstract over {len(index.docs[pos])} document(s)"
    m = meta.get(node.doc_id or "", {})
    title = m.get("title") or index.label(pos)
    return f"{m['source']} · {title}" if m.get("source") and m["source"] not in title else title


def strip_header(kb: str, text: str) -> str:
    """MultiHop chunks start with '[source, month] title.'; the card shows that already."""
    if kb == "multihop" and text.startswith("[") and "] " in text:
        head, _, rest = text.partition("] ")
        title_end = rest.find(". ")
        return rest[title_end + 2:] if 0 < title_end < 220 else rest
    return text


def gold_hits(index, result, item: Optional[Dict[str, Any]]):
    """(set of retrieved positions that hold gold evidence, recall) for a benchmark question."""
    if not item:
        return set(), None
    from run_comparison import covers
    pos = [index.pos[n] for n in result.node_ids]
    hit_pos = {i for i in pos if any(covers(index, i, g) for g in item["gold"])}
    found = sum(any(covers(index, i, g) for i in pos) for g in item["gold"])
    return hit_pos, found / max(1, len(item["gold"]))


def render_sources(kb, index, sources, cited, meta, gold_pos=frozenset(), limit_chars: int = 520):
    for s in sources:
        pos = index.pos[s.node_id]
        node = index.nodes[pos]
        tags = ""
        if node.level > 0:
            tags += '<span class="tag abs">abstract</span>'
        if pos in gold_pos:
            tags += '<span class="tag gold">gold evidence</span>'
        m = meta.get(node.doc_id or "", {})
        when = f" · {m['month']}" if m.get("month") else ""
        body = strip_header(kb, s.text)
        body = body[:limit_chars] + ("…" if len(body) > limit_chars else "")
        cls = "src" + (" cited" if s.n in cited else "") + (" gold" if pos in gold_pos else "")
        st.markdown(
            f'<div class="{cls}"><div class="h"><span class="n">[{s.n}]</span>'
            f'<span class="ti">{esc(source_title(index, pos, meta))}</span>{tags}'
            f'<span class="sc">{esc(when.strip(" ·"))} · score {s.score:.3f}</span></div>'
            f'<div class="b">{esc(body)}</div></div>', unsafe_allow_html=True)


def method_card(name: str, result, n_nodes: int, recall: Optional[float]) -> str:
    pct = 100.0 * result.nodes_scored / max(1, n_nodes)
    color = METHOD_COLORS[name]
    rec = (f'<div class="mrow"><span>Gold evidence</span><b>{recall * 100:.0f}%</b></div>'
           if recall is not None else "")
    return (f'<div class="mcard{" ours" if name == "HAT-RAG" else ""}"><div class="t">{dot(color)}{esc(METHOD_LABEL[name])}</div>'
            f'<div class="f">{esc(METHOD_FAMILY[name])}</div>'
            f'<div class="mrow"><span>Scored</span><b>{result.nodes_scored:,} / {n_nodes:,}</b></div>'
            f'<div class="bar"><i style="width:{min(100.0, pct):.1f}%;background:{color}"></i></div>'
            f'<div class="mrow"><span>Of the index</span><b>{pct:.1f}%</b></div>'
            f'<div class="mrow"><span>Latency</span><b>{result.seconds * 1000:.2f} ms</b></div>{rec}</div>')


def evidence_html(kb, index, result, meta, gold_pos) -> str:
    rows = []
    for r, (nid, sc) in enumerate(zip(result.node_ids, result.scores), 1):
        pos = index.pos[nid]
        txt = strip_header(kb, index.text(pos))
        g = '<span class="tag gold">gold</span>' if pos in gold_pos else ""
        rows.append(f'<div class="ev"><span class="r">{r}.</span><span class="who">{esc(source_title(index, pos, meta))}</span>'
                    f'{g}<br>{esc(txt[:170])}{"…" if len(txt) > 170 else ""}</div>')
    return "".join(rows)


def no_llm_note(err: Optional[str]):
    st.markdown(
        '<div class="note"><b>Retrieval only: no LLM key found.</b> Add <code>GROQ_API_KEY=...</code> '
        '(or <code>GEMINI_API_KEY</code>) to <code>.env</code> in the project root, then reload. '
        'The evidence below is what the model would receive.</div>', unsafe_allow_html=True)
    if err:
        st.caption(err)


# ============================================================================
# Sidebar
# ============================================================================
def sidebar() -> Dict[str, Any]:
    from llm_answer import PROVIDERS, available_providers
    with st.sidebar:
        st.markdown("### Knowledge base")
        built = [k for k in KBS if KBS[k]["tree"].exists()]
        kb = st.radio("Knowledge base", built, format_func=lambda k: KBS[k]["name"],
                      label_visibility="collapsed", key="kb")
        st.caption(KBS[kb]["blurb"])
        missing = [k for k in KBS if k not in built]
        if missing:
            st.caption("Not built yet: " + ", ".join(KBS[k]["name"] for k in missing) + " (see the Build tab).")

        st.markdown("### Language model")
        have = available_providers()
        options = ["auto"] + list(PROVIDERS)
        provider = st.selectbox("Provider", options, format_func=lambda p: {
            "auto": "Auto (Groq, then Gemini)", "groq": "Groq", "gemini": "Gemini"}[p], key="provider")
        client, err = get_client(provider, None)
        model = None
        if client:
            models = client.list_models() or [client.model]
            idx = models.index(client.model) if client.model in models else 0
            model = st.selectbox("Model", models, index=idx, key=f"model_{client.provider}")
            if model != client.model:
                client, err = get_client(client.provider, model)
            st.markdown(f'<div class="meta-line">{dot("#0f7b0f")} Connected: {esc(client.label)}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="meta-line">{dot("#d03b3b")} No key for '
                        f'{"any provider" if provider == "auto" else PROVIDERS[provider]["label"]}</div>',
                        unsafe_allow_html=True)
            st.caption("Put GROQ_API_KEY / GEMINI_API_KEY in .env and reload.")
        mode = st.radio("Answer style", ["grounded", "open"], horizontal=True, key="answer_mode",
                        format_func=lambda m: "Sources only" if m == "grounded" else "Sources + general knowledge")

        st.markdown("### Retrieval")
        k = st.slider("Evidence passages (k)", 3, 12, 8, key="k")
        large = kb == "multihop"
        with st.expander("HAT-RAG tuning"):
            fast = st.toggle("Fast mode: cap broad entity links", value=False, key=f"fast_{kb}",
                             help="Entities named in more than 200 passages (a publisher, say) only mark "
                                  "candidates instead of pulling every passage that names them into the pool. "
                                  "On MultiHop-RAG: ~10x faster, about half a point of recall lower.")
            beam = st.slider("Beam width β", 1, 12, 3 if (fast or not large) else 5, key=f"beam_{kb}_{fast}",
                             help="Large-corpus default (5) was chosen on the MultiHop-RAG dev split.")
            lam = st.slider("Diversity λ", 0.0, 0.5, 0.15, 0.05, key="lam")
            damping = st.slider("PageRank damping", 0.1, 0.9, 0.5, 0.05, key="damping")
        n_seeds = 6 if large and not fast else 3
        max_link = 200 if fast else None
        if have:
            st.caption("Keys found: " + ", ".join(PROVIDERS[p]["label"] for p in have))
    return dict(kb=kb, client=client, llm_error=err, mode=mode, k=k, beam=beam, lam=lam, damping=damping,
                n_seeds=n_seeds, max_link=max_link)


# ============================================================================
# Tabs
# ============================================================================
def answer_turn(cfg, index, retriever, method, meta, prompt, history) -> List[Dict[str, Any]]:
    """Render one question + streamed answer; return the two turns for the history."""
    from llm_answer import LLMError, build_messages, build_sources, cited, normalize_citations
    kb = cfg["kb"]
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/account_tree:"):
        result = retriever.retrieve(prompt, k=cfg["k"])
        sources = build_sources(index, result, meta)
        stat_line = (f'{dot(METHOD_COLORS[method])} {esc(METHOD_LABEL[method])} scored '
                     f'{result.nodes_scored:,} of {index.n_nodes:,} nodes '
                     f'({100 * result.nodes_scored / index.n_nodes:.1f}%) in {result.seconds * 1000:.1f} ms')
        client = cfg["client"]
        text = ""
        if client is None:
            no_llm_note(cfg["llm_error"])
            text = "_Retrieval only (no LLM configured)._"
        else:
            msgs = build_messages(prompt, sources, cfg["mode"],
                                  [{"role": t["role"], "content": t["content"]} for t in history
                                   if not t["content"].startswith("_")])
            t0 = time.perf_counter()
            try:
                text = st.write_stream(client.stream(msgs))
            except LLMError as e:
                try:                                       # one non-streaming retry (handles 429 backoff)
                    text = client.chat(msgs)["text"]
                    st.markdown(text)
                except LLMError as e2:
                    st.error(f"The model call failed: {e2}")
                    text = f"_Model call failed: {e}_"
            stat_line += f" · answered by {esc(client.label)} in {time.perf_counter() - t0:.1f} s"
        st.markdown(f'<div class="meta-line">{stat_line}</div>', unsafe_allow_html=True)
        if text and text != normalize_citations(text):
            text = normalize_citations(text)        # store the [n] form for re-rendering
        used = cited(text or "", len(sources))
        with st.expander(f"Sources ({len(sources)})", expanded=client is None):
            render_sources(kb, index, sources, used, meta)
    return [{"role": "user", "content": prompt},
            {"role": "assistant", "content": text or "", "sources": sources, "cited": used, "stats": stat_line}]


def tab_ask(cfg, index, retrievers, meta):
    kb = cfg["kb"]
    c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
    with c1:
        method = st.segmented_control("Retriever", METHOD_ORDER, default="HAT-RAG", key="ask_method",
                                      format_func=lambda m: METHOD_LABEL[m])
    with c2:
        if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
            st.session_state[f"chat_{kb}"] = []
    method = method or "HAT-RAG"
    history = st.session_state.setdefault(f"chat_{kb}", [])

    pending = st.session_state.pop(f"pending_{kb}", None)
    conv = st.container()
    with conv:
        for turn in history:
            with st.chat_message(turn["role"], avatar=":material/person:" if turn["role"] == "user" else ":material/account_tree:"):
                st.markdown(turn["content"])
                if turn.get("stats"):
                    st.markdown(f'<div class="meta-line">{turn["stats"]}</div>', unsafe_allow_html=True)
                if turn.get("sources"):
                    with st.expander(f"Sources ({len(turn['sources'])})"):
                        render_sources(kb, index, turn["sources"], turn.get("cited", []), meta)

    if not history and not pending and not st.session_state.get(f"chat_input_{kb}"):
        st.markdown('<div class="section-lead">Ask any question in your own words. The retriever finds evidence in '
                    f'{index.n_leaves:,} passages, and the model answers from it with numbered citations. '
                    'Some starting points:</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(suggestions(kb)[:6]):
            if cols[i % 2].button(q, key=f"sugg_{kb}_{i}", width="stretch"):
                st.session_state[f"pending_{kb}"] = q
                st.rerun()

    prompt = st.chat_input(KBS[kb]["placeholder"], key=f"chat_input_{kb}") or pending
    if not prompt:
        return
    with conv:
        history += answer_turn(cfg, index, retrievers[method], method, meta, prompt, history)


def tab_compare(cfg, index, retrievers, meta):
    from llm_answer import LLMError, build_messages, build_sources, normalize_citations
    kb = cfg["kb"]
    st.markdown('<div class="section-lead">Run one question through all four retrievers over the same vectors. '
                'Pick a benchmark question to see which method actually finds the labelled evidence.</div>',
                unsafe_allow_html=True)
    bench = get_benchmark(kb)
    pool = [q for q in bench if q.get("split", "test") == "test"][:400]
    c1, c2 = st.columns([1, 2])
    with c1:
        source = st.radio("Question", ["Benchmark question", "My own question"], key="cmp_src", horizontal=True)
    with c2:
        if source == "Benchmark question":
            types = sorted({q["type"] for q in pool})
            qtype = st.segmented_control("Type", types, default=types[0], key=f"cmp_type_{kb}") or types[0]
    if source == "Benchmark question":
        subset = [q for q in pool if q["type"] == qtype]
        item = st.selectbox("Benchmark question", subset, format_func=lambda q: q["query"],
                            key=f"cmp_q_{kb}_{qtype}")
        query = item["query"]
    else:
        item = None
        query = st.text_input("Your question", key=f"cmp_free_{kb}", placeholder=KBS[kb]["placeholder"])
    g1, g2 = st.columns([1, 1])
    k = g1.slider("k (evidence per method)", 3, 10, 5, key="cmp_k")
    with_answers = g2.toggle("Also generate an answer from each method's evidence", value=False,
                             disabled=cfg["client"] is None, key="cmp_ans")
    if not st.button("Compare the four retrievers", type="primary", icon=":material/compare_arrows:",
                     width="stretch") or not query:
        return

    q = index.encode_query(query)
    results = {m: retrievers[m].search(q, k=k, query=query) for m in METHOD_ORDER}
    golds = {m: gold_hits(index, results[m], item) for m in METHOD_ORDER}
    cols = st.columns(4)
    for col, m in zip(cols, METHOD_ORDER):
        col.markdown(method_card(m, results[m], index.n_nodes, golds[m][1]), unsafe_allow_html=True)

    if item:
        ans = f" · reference answer: <b>{esc(item['answer'])}</b>" if item.get("answer") else ""
        st.markdown(f'<div class="meta-line" style="margin-top:.6rem">This question needs {len(item["gold"])} piece(s) '
                    f'of evidence from {len({g["doc"] for g in item["gold"]})} document(s){ans}</div>',
                    unsafe_allow_html=True)

    st.markdown("#### Evidence retrieved")
    cols = st.columns(4)
    for col, m in zip(cols, METHOD_ORDER):
        col.markdown(f'<div class="mcard">{evidence_html(kb, index, results[m], meta, golds[m][0])}</div>',
                     unsafe_allow_html=True)

    if with_answers and cfg["client"]:
        st.markdown("#### Answers from each method's evidence")
        cols = st.columns(4)
        for col, m in zip(cols, METHOD_ORDER):
            with col, st.container(border=True):
                st.markdown(f'{dot(METHOD_COLORS[m])} **{METHOD_LABEL[m]}**', unsafe_allow_html=True)
                try:
                    with st.spinner("Generating ..."):
                        out = cfg["client"].chat(build_messages(query, build_sources(index, results[m], meta), "grounded"))
                    st.markdown(normalize_citations(out["text"]))
                except LLMError as e:
                    st.error(str(e))

    with st.expander("HAT-RAG search trace"):
        for line in results["HAT-RAG"].trace:
            st.text(line)


def tab_benchmarks():
    verdict = charts.load("verdict")
    multihop = charts.load("multihop")
    st.markdown('<div class="section-lead">All four retrievers read the same tree, chunks and vectors; '
                'every number here comes from <code>algorithms/compare-algos/results/</code>.</div>',
                unsafe_allow_html=True)

    if multihop:
        full = multihop["full"]
        s5 = full["by_k"]["5"]["summary"]
        s10 = full["by_k"]["10"]["summary"]
        base = [m for m in METHOD_ORDER if m != "HAT-RAG"]
        best = max(base, key=lambda m: s5[m]["recall"])
        h, b = s5["HAT-RAG"], s5[best]
        flat = s5["Flat dense"]
        st.markdown(f"#### MultiHop-RAG at full scale: {full['index']['leaves']:,} passages, "
                    f"{h['n_queries']:,} held-out questions")
        st.markdown(stats([
            (f"{h['recall'] * 100:.1f}%", "HAT-RAG recall@5", f"{(h['recall'] - b['recall']) * 100:+.1f} pts vs {best}", True),
            (f"{b['recall'] * 100:.1f}%", f"Best baseline ({best})", None, False),
            (f"{s10['HAT-RAG']['recall'] * 100:.1f}%", "HAT-RAG recall@10",
             f"{(s10['HAT-RAG']['recall'] - max(s10[m]['recall'] for m in base)) * 100:+.1f} pts vs best", False),
            (f"{h['nodes_scored'] / full['index']['nodes'] * 100:.1f}%", "Of the index scored per query",
             f"{flat['nodes_scored'] / h['nodes_scored']:.0f}x fewer than flat", False),
            (f"{h['latency_ms']:.1f} ms", "HAT-RAG search latency",
             f"flat {flat['latency_ms']:.1f} ms · graph {s5['Graph PPR']['latency_ms']:.1f} ms", False),
        ]), unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Recall@5 by question type**")
            st.plotly_chart(charts.type_bars(s5, ["inference", "comparison", "temporal"],
                                             {"inference": "Inference", "comparison": "Comparison",
                                              "temporal": "Temporal"}),
                            width="stretch", config={"displayModeBar": False})
        with c2:
            st.markdown("**Accuracy against cost** (HAT-RAG beam width swept)")
            st.plotly_chart(charts.frontier(full["beam"], {m: s5[m] for m in base}),
                            width="stretch", config={"displayModeBar": False})
        rows = charts.haystack_rows()
        if rows:
            st.markdown(f"**Same {multihop['haystack']['queries']} questions, growing haystack** "
                        "(real distractor articles added)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts.size_lines(rows, "recall"), width="stretch",
                            config={"displayModeBar": False})
            c2.plotly_chart(charts.size_lines(rows, "nodes_scored"), width="stretch",
                            config={"displayModeBar": False})
    else:
        st.info("MultiHop-RAG results not computed yet: run `python algorithms/compare-algos/run_multihop.py`.")

    syn = charts.synthetic_rows()
    if syn:
        st.markdown("#### Synthetic scaling benchmark")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.size_lines(syn, "recall"), width="stretch", config={"displayModeBar": False})
        c2.plotly_chart(charts.size_lines(syn, "nodes_scored"), width="stretch",
                        config={"displayModeBar": False})

    st.markdown("#### Tables")
    import pandas as pd
    table = []
    for m in METHOD_ORDER:
        row = {"Method": m}
        if multihop:
            row["MultiHop-RAG R@5"] = f"{multihop['full']['by_k']['5']['summary'][m]['recall'] * 100:.1f}%"
            row["MultiHop-RAG R@10"] = f"{multihop['full']['by_k']['10']['summary'][m]['recall'] * 100:.1f}%"
            row["MultiHop nodes scored"] = f"{multihop['full']['by_k']['5']['summary'][m]['nodes_scored']:,.0f}"
        if verdict:
            t = verdict["table"][m]
            row["10-K R@5"] = f"{t['finance_recall'] * 100:.1f}%"
            row["Synthetic R@5 (mean)"] = f"{t['synthetic_recall'] * 100:.1f}%"
        table.append(row)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")


def tab_corpus(cfg, index, meta):
    kb = cfg["kb"]
    levels = {}
    for lv in index.level:
        levels[int(lv)] = levels.get(int(lv), 0) + 1
    docs = sorted({d for d in index.doc if d})
    st.markdown(stats([(f"{len(docs):,}", "Documents", None, False),
                       (f"{index.n_leaves:,}", "Leaf passages", None, False),
                       (f"{index.n_nodes - index.n_leaves:,}", "Abstract nodes", None, False),
                       (f"{index.depth() + 1}", "Tree levels", None, False),
                       (f"{len(index.alpha_pairs):,}", "Alpha cross-links", None, False)]), unsafe_allow_html=True)
    st.caption("Nodes per level, leaves to root: " + " → ".join(f"{levels[l]:,}" for l in sorted(levels)))

    c1, c2 = st.columns([1, 2])
    with c1:
        needle = st.text_input("Filter documents", key=f"doc_filter_{kb}", placeholder="title or publisher")
        shown = [d for d in docs if not needle or needle.lower() in
                 (meta.get(d, {}).get("title", "") + meta.get(d, {}).get("source", "") + d).lower()]
        doc = st.selectbox(f"Document ({len(shown):,})", shown, key=f"doc_sel_{kb}",
                           format_func=lambda d: meta.get(d, {}).get("title", d))
    if not doc:
        return
    with c2:
        m = meta.get(doc, {})
        st.markdown(f"**{esc(m.get('title', doc))}**")
        bits = [m.get("source"), m.get("month"), m.get("category")]
        st.markdown(f'<div class="meta-line">{esc(" · ".join(b for b in bits if b))}</div>', unsafe_allow_html=True)
        if m.get("url"):
            st.caption(m["url"])
        view = st.segmented_control("View", ["Passages in the tree", "Full text"], default="Passages in the tree",
                                    key=f"doc_view_{kb}")
        if view == "Full text":
            st.text_area("Full text", get_doc_text(kb, doc), height=420, label_visibility="collapsed")
        else:
            rows = sorted((index.nodes[i].metadata.get("chunk_index", 0), i)
                          for i in range(index.n_nodes) if index.doc[i] == doc)
            for ci, i in rows[:60]:
                parent = index.parent[i]
                ptxt = f" · parent {index.ids[parent]}" if parent >= 0 else ""
                st.markdown(f'<div class="src"><div class="h"><span class="n">#{ci}</span>'
                            f'<span class="sc">{len(index.alpha[i])} alpha links{ptxt}</span></div>'
                            f'<div class="b">{esc(strip_header(kb, index.text(i)))}</div></div>', unsafe_allow_html=True)
            if len(rows) > 60:
                st.caption(f"Showing the first 60 of {len(rows)} passages.")


def tab_build():
    st.markdown('<div class="section-lead">Offline half of HAT-RAG: Algorithm 1 chunks each document, the encoder '
                'embeds the chunks, Algorithm 3 clusters them level by level with spherical k-means, writes an '
                'abstract per cluster and links related nodes across branches (alpha edges).</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**SEC 10-K tree** · 32 nodes")
        st.caption("Abstractive summaries with BART (downloads ~1.6 GB on first use).")
        summ = st.selectbox("Summarizer", ["extractive", "llm"], key="b_sum")
        alpha = st.slider("Alpha link threshold", 0.4, 0.9, 0.65, 0.05, key="b_alpha")
        if st.button("Rebuild 10-K tree", width="stretch"):
            from algorithms.run_finance_rag import run_finance_pipeline
            args = type("A", (), dict(summarizer=summ, alpha=alpha, device="auto", chunk_size=60,
                                      chunk_overlap=15, branching=4))()
            with st.status("Building ...") as s:
                t0 = time.perf_counter()
                run_finance_pipeline(args)
                s.update(label=f"Rebuilt in {time.perf_counter() - t0:.1f}s", state="complete")
            st.cache_resource.clear()
    with c2, st.container(border=True):
        st.markdown("**Synthetic tree** · 128 dossiers, ~1.8k nodes")
        st.caption("Extractive abstracts, about a minute on CPU.")
        if st.button("Rebuild synthetic tree", width="stretch"):
            from synthetic_corpus import build_corpus
            from algo1_document_chunking import DocumentChunker
            from algo3_hierarchical_abstract_tree import HierarchicalAbstractTreeBuilder
            with st.status("Building ...") as s:
                t0 = time.perf_counter()
                documents, _, _ = build_corpus(n_docs=128, seed=17)
                chunks = DocumentChunker(chunk_size=60, chunk_overlap=10).process_corpus(documents)
                b = HierarchicalAbstractTreeBuilder(embedder=get_embedder(), max_levels=6, target_children=8,
                                                    alpha="auto", summarizer_mode="extractive", device="cpu",
                                                    verbose=False)
                b.build_tree(chunks)
                b.save_tree_json(str(KBS["enterprise"]["tree"]))
                s.update(label=f"{len(b.nodes)} nodes in {time.perf_counter() - t0:.1f}s", state="complete")
            st.cache_resource.clear()
    with c3, st.container(border=True):
        st.markdown("**MultiHop-RAG tree** · 609 articles, ~26k nodes")
        st.caption("Downloads the dataset (~12 MB) and builds on CPU in roughly 25 minutes; run it from a terminal:")
        st.code("python algorithms/compare-algos/run_multihop.py --build-only", language="bash")
        st.caption("Status: " + ("built" if MULTIHOP_TREE.exists() else "not built"))


# ============================================================================
def main():
    cfg = sidebar()
    kb = cfg["kb"]
    if kb is None:
        st.error("No knowledge tree found. Build one from the Build tab.")
        tab_build()
        return
    tree = str(KBS[kb]["tree"])
    index = get_index(tree)
    meta = get_doc_meta(kb)
    n_docs = len({d for d in index.doc if d})

    st.markdown(hero(
        "HAT<span>-RAG</span>",
        "Hierarchical Abstract Tree retrieval: a beam search down a tree of cluster abstracts, so each question "
        "touches a small part of the index while the corpus grows.",
        [(f"<b>{esc(KBS[kb]['short'])}</b>", KBS[kb]["color"]),
         (f"<b>{n_docs:,}</b> documents", "#c3c2b7"),
         (f"<b>{index.n_leaves:,}</b> passages", "#c3c2b7"),
         (f"<b>{index.n_nodes:,}</b> tree nodes", "#c3c2b7")]), unsafe_allow_html=True)

    retrievers = get_retrievers(tree, cfg["beam"], cfg["lam"], cfg["damping"], cfg["n_seeds"], cfg["max_link"])
    t_ask, t_cmp, t_bench, t_corpus, t_build = st.tabs(
        ["Ask", "Compare retrievers", "Benchmarks", "Corpus & tree", "Build"])
    with t_ask:
        tab_ask(cfg, index, retrievers, meta)
    with t_cmp:
        tab_compare(cfg, index, retrievers, meta)
    with t_bench:
        tab_benchmarks()
    with t_corpus:
        tab_corpus(cfg, index, meta)
    with t_build:
        tab_build()


main()
