"""Tests for the large-corpus code paths and the LLM layer (no network, no model download)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms"))
sys.path.insert(0, str(ROOT / "algorithms" / "compare-algos"))

from retrieval1_hat_beam_traversal import EntityIndex, entity_weights, top_k  # noqa: E402
from algo3_hierarchical_abstract_tree import HierarchicalAbstractTreeBuilder, TreeNode  # noqa: E402

TEXTS = {
    0: "Apple reported revenue. Revenue at Apple grew while Microsoft held Azure steady.",
    1: "Microsoft said Azure grew. The Azure unit reported to Satya Nadella.",
    2: "NVIDIA sold GPUs to Microsoft and to Apple through the year.",
    3: "Tesla built cars. The cars of Tesla were sold in Europe, said Elon Musk.",
    4: "Elon Musk also runs SpaceX, which launched rockets for NASA.",
}


def test_incidence_matches_pair_weights():
    ents = EntityIndex(TEXTS)
    B, idf = ents.incidence(len(TEXTS))
    W = entity_weights(B, idf).toarray()
    expected = np.zeros_like(W)
    for (a, b), w in ents.pair_weights().items():
        expected[a, b] = expected[b, a] = w
    assert np.allclose(W, expected)
    pool = [1, 3, 4]
    assert np.allclose(entity_weights(B, idf, rows=pool).toarray(), expected[np.ix_(pool, pool)])


def test_top_k_orders_like_argsort():
    rng = np.random.default_rng(0)
    s = rng.random(1000)
    for k in (1, 5, 37, 1000, 2000):
        assert list(top_k(s, k)) == list(np.argsort(-s)[:k])


class _Stub:
    """Embedder stand-in: similarity is a plain matmul."""
    def similarity(self, a, b=None):
        return a @ (a if b is None else b).T


def _links(limit):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 16)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    nodes = [TreeNode(f"n{i}", 0, "t", x[i], doc_id=f"d{i % 7}") for i in range(300)]
    parents = [TreeNode(f"p{j}", 1, "p") for j in range(30)]
    for i, n in enumerate(nodes):
        parents[i % 30].add_child(n)
    b = HierarchicalAbstractTreeBuilder.__new__(HierarchicalAbstractTreeBuilder)
    b.embedder, b.alpha, b.max_cross_links, b.verbose = _Stub(), "auto", 3, False
    b.resolved_alpha = {}
    from algo3_hierarchical_abstract_tree import BuildMetrics
    b.metrics = BuildMetrics()
    b.DENSE_LINK_LIMIT = limit
    b.build_cross_links({0: nodes})
    return {(n.node_id, l["target_id"]) for n in nodes for l in n.cross_links}, b.resolved_alpha[0]


def test_blockwise_cross_links_match_dense():
    dense, a1 = _links(10_000)
    block, a2 = _links(10)
    assert abs(a1 - a2) < 1e-3
    assert len(dense ^ block) <= 0.01 * len(dense)


def test_prompt_and_citations():
    from llm_answer import Source, build_messages, cited
    srcs = [Source(1, "a", "TechCrunch — FTX trial", "SBF went on trial.", 0.9),
            Source(2, "b", "The Verge — FTX", "Prosecutors allege fraud.", 0.8)]
    msgs = build_messages("Who is on trial?", srcs, "grounded")
    assert msgs[0]["role"] == "system" and "[1] TechCrunch" in msgs[-1]["content"]
    assert cited("He is on trial [1][2], see also [7].", 2) == [1, 2]


def test_fact_matching():
    from run_comparison import shingles
    fact = "The highly anticipated criminal trial for Sam Bankman-Fried started Tuesday"
    chunk = "[TechCrunch, Oct 2023] SBF. The highly anticipated criminal trial for Sam Bankman-Fried started Tuesday."
    other = "[TechCrunch, Oct 2023] SBF. Sam Bankman-Fried founded FTX in 2019 and moved to the Bahamas."
    f = shingles(fact)
    assert len(f & shingles(chunk)) / len(f) == 1.0
    assert len(f & shingles(other)) / len(f) < 0.5          # same article, wrong passage: no credit


def test_answer_scoring():
    from run_answer_eval import correct
    assert correct("Sam Bankman-Fried.", "Sam Bankman-Fried")
    assert correct("Yes", "Yes") and not correct("No, it did not", "Yes")
    assert not correct("Insufficient information.", "Google")


def test_missing_key_is_reported(monkeypatch):
    import llm_answer
    monkeypatch.setattr(llm_answer, "load_env", lambda path=None: None)
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(llm_answer.LLMError):
        llm_answer.make_client("auto")
