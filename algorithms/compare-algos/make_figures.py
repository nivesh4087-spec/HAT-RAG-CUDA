"""
Figures and tables for the retrieval comparison.

Reads results/finance.json, results/scaling.json and results/verdict.json
(written by run_comparison.py) and writes:

    figures/fig1_verdict.png          overall score per method (the headline)
    figures/fig2_finance_by_type.png  finance recall@5 by query type
    figures/fig3_scaling_recall.png   recall@5 as the corpus grows
    figures/fig4_scaling_cost.png     nodes scored and latency as the corpus grows
    figures/fig5_frontier.png         accuracy vs cost, beam width swept
    figures/fig6_ablation.png         what each HAT-RAG component contributes
    figures/fig7_head_to_head.png     per-query wins / ties / losses for HAT-RAG
    results/tables.md                 every number behind the figures

    python algorithms/compare-algos/make_figures.py
"""

import json
from pathlib import Path
from typing import List, Sequence

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# -- palette: validated categorical slots 1-4 (adjacent pairs) + chart chrome --
COLOR = {
    "HAT-RAG": "#2a78d6",
    "Flat dense": "#eb6834",
    "RAPTOR collapsed": "#1baf7a",
    "Graph PPR": "#eda100",
    "HAT-RAG (paper spec)": "#898781",
}
MAIN = ["HAT-RAG", "Flat dense", "RAPTOR collapsed", "Graph PPR"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DE_EMPH = "#c3c2b7"
WIN, TIE, LOSS = "#2a78d6", "#f0efec", "#e34948"       # diverging pair, neutral midpoint

PX = 72 / 100                                          # 1 px in points at dpi 100
LINE = 2 * PX
MARKER = 8 * PX

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": PX,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 9,
})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def title(fig, text: str, sub: str = ""):
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.12 / h, text, ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
    if sub:
        fig.text(0.012, 1 - 0.40 / h, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=PX)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def vgrid(ax):
    ax.xaxis.grid(True, color=GRID, linewidth=PX)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", length=0)


def _px_to_data(ax):
    """Data units per pixel along x and y (after limits are final)."""
    fig = ax.figure
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    scale = fig.dpi / 100
    return (x1 - x0) / bbox.width * scale, (y1 - y0) / bbox.height * scale


def column(ax, x: float, width: float, height: float, color: str, radius_px: float = 4):
    """Vertical bar: 4px rounded data-end, square at the baseline."""
    if height <= 0:
        return
    ux, uy = _px_to_data(ax)
    r = min(radius_px * ux, width / 2)
    ry = r * uy / ux
    ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=uy / ux, linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((x, 0), width, min(height, ry), linewidth=0, facecolor=color))


def bar(ax, y: float, height: float, length: float, color: str, radius_px: float = 4, left: float = 0.0):
    """Horizontal bar growing from `left`: rounded far end, square at the baseline."""
    if length <= 0:
        return
    ux, uy = _px_to_data(ax)
    r = min(radius_px * ux, length / 2, (height / 2) * ux / uy)
    ax.add_patch(FancyBboxPatch((left, y), length, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=uy / ux, linewidth=0, facecolor=color))
    ax.add_patch(Rectangle((left, y), min(length, r), height, linewidth=0, facecolor=color))


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def save(fig, name: str):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}")


def legend_row(fig, methods: Sequence[str], y: float = 0.86, marker: str = "line"):
    handles = []
    for m in methods:
        if marker == "line":
            h, = plt.plot([], [], color=COLOR[m], linewidth=LINE, marker="o", markersize=MARKER * 0.75,
                          markeredgecolor=SURFACE, markeredgewidth=LINE)
        else:
            h = Rectangle((0, 0), 1, 1, facecolor=COLOR[m], linewidth=0)
        handles.append(h)
    fig.legend(handles, methods, loc="upper left", bbox_to_anchor=(0.005, y), ncol=len(methods),
               handlelength=1.6, columnspacing=1.4, labelcolor=INK2)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_verdict(verdict):
    order = verdict["ranking"]
    table = verdict["table"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.15, right=0.97, top=0.72, bottom=0.15)
    title(fig, "HAT-RAG ranks first on accuracy while scoring the smallest share of the corpus",
          "Left: ½ finance recall@5 + ½ mean synthetic recall@5 (every synthetic size). "
          "Right: share of the corpus compared against each query, averaged over all six corpora.")
    ys = list(range(len(order)))[::-1]
    panels = [(a1, "score", "Combined recall@5 score", lambda v: f"{v * 100:.1f}"),
              (a2, "cost_fraction", "Corpus scored per query (lower is better)", lambda v: f"{v * 100:.0f}%")]
    for ax, key, label, fmt in panels:
        vmax = max(table[m][key] for m in order)
        ax.set_xlim(0, vmax * 1.2)
        ax.set_ylim(-0.6, len(order) - 0.4)
        vgrid(ax)
        for y, m in zip(ys, order):
            v = table[m][key]
            bar(ax, y - 0.17, 0.34, v, COLOR[m] if m == order[0] else DE_EMPH)
            ax.text(v + vmax * 0.02, y, fmt(v), va="center", fontsize=9.5,
                    color=INK if m == order[0] else INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels(order if ax is a1 else ["" for _ in order])
        ax.set_xlabel(label)
        ax.xaxis.set_major_formatter((lambda v, _: f"{v * 100:.0f}") if key == "score"
                                     else (lambda v, _: f"{v * 100:.0f}%"))
    for t in a1.get_yticklabels():
        t.set_color(INK if t.get_text() == order[0] else INK2)
    save(fig, "fig1_verdict.png")


def fig_finance_by_type(finance):
    s = finance["summary"]
    groups = [("Overall", None), ("Single-hop", "single"), ("Cross-document", "cross"), ("Thematic", "thematic")]
    counts = {t: s["HAT-RAG"]["by_type"][t]["n_queries"] for _, t in groups if t}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.74, bottom=0.12)
    title(fig, "Finance corpus: HAT-RAG leads overall and finds every single-hop fact; flat dense still edges cross-document",
          "52 hand-labelled questions over five 10-K filings (24 chunks). Cross-document and thematic "
          "questions need evidence from 2 to 5 filings.")
    legend_row(fig, MAIN, y=0.86, marker="box")
    width, gap = 0.1, 0.012
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 1.1)
    hgrid(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    for g, (label, t) in enumerate(groups):
        for j, m in enumerate(MAIN):
            v = s[m]["recall"] if t is None else s[m]["by_type"][t]["recall"]
            x = g - 2 * (width + gap) + j * (width + gap) + gap / 2 + 0.006
            column(ax, x, width, v, COLOR[m])
            if t is None:                               # label the headline group only
                ax.text(x + width / 2, v + 0.015, f"{v * 100:.1f}", ha="center", va="bottom",
                        fontsize=8.5, color=INK if m == "HAT-RAG" else INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab if t is None else f"{lab} (n={counts[t]})" for lab, t in groups])
    ax.tick_params(axis="x", length=0)
    save(fig, "fig2_finance_by_type.png")


def _series(scaling, metric: str, method: str):
    return [r["leaves"] for r in scaling["rows"]], [r["summary"][method][metric] for r in scaling["rows"]]


def _line(ax, xs, ys, method):
    ref = method == "HAT-RAG (paper spec)"
    ax.plot(xs, ys, color=COLOR[method], linewidth=LINE * (0.75 if ref else 1), marker="o",
            markersize=MARKER * (0.75 if ref else 1), markeredgecolor=SURFACE, markeredgewidth=LINE,
            solid_capstyle="round", solid_joinstyle="round", zorder=2 if ref else 3)


def _size_axis(ax, scaling):
    xs = [r["leaves"] for r in scaling["rows"]]
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(xs[0] * 0.8, xs[-1] * 1.25)


def fig_scaling_recall(scaling):
    methods = MAIN + ["HAT-RAG (paper spec)"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(left=0.07, right=0.78, top=0.74, bottom=0.13)
    title(fig, "As the corpus grows, HAT-RAG holds its recall while flat and collapsed-tree retrieval fall away",
          f"Synthetic cross-document corpus, {scaling['rows'][0]['docs']} to {scaling['rows'][-1]['docs']} documents "
          f"({scaling['rows'][0]['leaves']:,} to {scaling['rows'][-1]['leaves']:,} chunks). "
          "Evidence recall@5 over single-hop and two-document queries.")
    legend_row(fig, methods, y=0.86)
    hgrid(ax)
    for m in methods:
        _line(ax, *_series(scaling, "recall", m), m)
    _size_axis(ax, scaling)
    low = min(min(_series(scaling, "recall", m)[1]) for m in methods)
    floor = max(0.0, int(low * 10) / 10 - 0.05)
    ax.set_ylim(floor, 1.03)
    ax.set_xlabel("Leaf chunks in the corpus (log scale)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    # end labels in the right margin, spaced so they never overlap
    ends = sorted((_series(scaling, "recall", m)[1][-1], m) for m in methods)
    placed: List[float] = []
    for v, m in ends:
        y = v if not placed else max(v, placed[-1] + 0.045 * (1.03 - floor) / 0.63)
        placed.append(y)
        ax.annotate(f"{m}  {v * 100:.1f}%", xy=(scaling["rows"][-1]["leaves"], v),
                    xytext=(1.02, (y - floor) / (1.03 - floor)), textcoords="axes fraction", va="center", fontsize=8.5,
                    color=INK if m == "HAT-RAG" else INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=PX, shrinkA=0, shrinkB=4))
    save(fig, "fig3_scaling_recall.png")


def fig_scaling_cost(scaling):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.68, bottom=0.14)
    title(fig, "HAT-RAG compares a small, slowly growing slice of the corpus, so its cost grows sub-linearly",
          "Left: node embeddings compared per query. Right: median search time (query encoding excluded, same CPU).\n"
          "Flat dense is one matrix-vector product and stays fastest on CPU at these sizes; HAT-RAG pays a fixed "
          "per-query cost for descent, linking and PageRank.")
    legend_row(fig, MAIN, y=0.81)
    for ax, metric, ylabel in [(a1, "nodes_scored", "Nodes scored per query"),
                               (a2, "latency_ms", "Search latency per query (ms)")]:
        hgrid(ax)
        ax.set_yscale("log")
        for m in MAIN:
            _line(ax, *_series(scaling, metric, m), m)
        _size_axis(ax, scaling)
        ax.set_xlabel("Leaf chunks in the corpus")
        ax.set_ylabel(ylabel)
        ticks = ([50, 100, 200, 500, 1000, 2000, 5000] if metric == "nodes_scored"
                 else [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10])
        lo, hi = ax.get_ylim()
        ax.set_yticks([t for t in ticks if lo <= t <= hi])
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    last = scaling["rows"][-1]
    hat, flat = last["summary"]["HAT-RAG"]["nodes_scored"], last["summary"]["Flat dense"]["nodes_scored"]
    a1.text(0.03, 0.95, f"At {last['leaves']:,} chunks: HAT-RAG {hat:,.0f} nodes vs flat {flat:,.0f}\n"
            f"({(1 - hat / flat) * 100:.0f}% fewer)", transform=a1.transAxes, va="top", fontsize=8.5, color=INK)
    save(fig, "fig4_scaling_cost.png")


def fig_frontier(scaling):
    n_docs = scaling["frontier_docs"]
    row = next(r for r in scaling["rows"] if r["docs"] == n_docs)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.13)
    title(fig, "Accuracy against cost: every HAT-RAG beam width sits above and left of the baselines",
          f"Synthetic corpus, {n_docs} documents ({row['leaves']:,} chunks). Lines sweep the beam width β from 1 to 32; "
          f"grey points are the three baselines, which always score the whole corpus.")
    hgrid(ax)
    ax.set_xscale("log")
    for m in ["HAT-RAG (paper spec)", "HAT-RAG"]:
        pts = scaling["frontier"][m]
        xs = [p["nodes_scored"] for p in pts]
        ys = [p["recall"] for p in pts]
        _line(ax, xs, ys, m)
        for p in pts:
            if p["beam_width"] in (1, 3, 32):
                ax.annotate(f"β={p['beam_width']}", (p["nodes_scored"], p["recall"]),
                            xytext=(0, 9 if m == "HAT-RAG" else -14), textcoords="offset points",
                            ha="center", fontsize=8, color=MUTED)
        end = -1 if m == "HAT-RAG" else 0              # paper spec labelled at its low, empty end
        ax.annotate(m, (xs[end], ys[end]), xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=9, color=INK if m == "HAT-RAG" else INK2)
    for m, mk in [("Flat dense", "s"), ("RAPTOR collapsed", "D"), ("Graph PPR", "^")]:
        s = row["summary"][m]
        ax.plot([s["nodes_scored"]], [s["recall"]], marker=mk, markersize=MARKER * 1.1, color=MUTED,
                markeredgecolor=SURFACE, markeredgewidth=LINE, linestyle="none", zorder=4)
        ax.annotate(f"{m}  {s['recall'] * 100:.1f}%", (s["nodes_scored"], s["recall"]), xytext=(-10, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2)
    xs_all = [p["nodes_scored"] for p in scaling["frontier"]["HAT-RAG"]] + [row["nodes"]]
    ax.set_xlim(min(xs_all) * 0.6, max(xs_all) * 1.3)
    lows = [p["recall"] for p in scaling["frontier"]["HAT-RAG (paper spec)"]]
    ax.set_ylim(min(lows) - 0.06, 1.02)
    ax.set_xticks([t for t in [50, 100, 200, 500, 1000] if min(xs_all) * 0.6 <= t <= max(xs_all) * 1.3])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Nodes scored per query (log scale)")
    ax.set_ylabel("Evidence recall@5")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    save(fig, "fig5_frontier.png")


def fig_ablation(finance, scaling):
    panels = [("Finance corpus (24 chunks)", finance["ablations"]["components"]),
              (f"Synthetic, {scaling['frontier_docs']} documents", scaling["components"])]
    names = list(panels[0][1].keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.22, right=0.97, top=0.74, bottom=0.12)
    title(fig, "Entity linking and graph propagation carry HAT-RAG; the original Algorithm 4 trails far behind",
          "Evidence recall@5 with one HAT-RAG component switched off at a time. 'paper spec' is beam descent + "
          "alpha edges + an ungated provenance bonus, ranked by cosine.")
    ys = list(range(len(names)))[::-1]
    for ax, (label, abl) in zip(axes, panels):
        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.6, len(names) - 0.4)
        vgrid(ax)
        full = abl["full HAT-RAG"]["recall"]
        for y, n in zip(ys, names):
            v = abl[n]["recall"]
            bar(ax, y - 0.18, 0.36, v, COLOR["HAT-RAG"] if n == "full HAT-RAG" else DE_EMPH)
            delta = "" if n == "full HAT-RAG" else f"  ({(v - full) * 100:+.1f})"
            ax.text(v + 0.015, y, f"{v * 100:.1f}{delta}", va="center", fontsize=8.5,
                    color=INK if n == "full HAT-RAG" else INK2)
        ax.set_yticks(ys)
        ax.set_yticklabels(names if ax is axes[0] else ["" for _ in names])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
        ax.set_title(label, loc="left", fontsize=10, color=INK, pad=6)
    save(fig, "fig6_ablation.png")


def _paired_totals(finance, scaling):
    totals = {m: {"wins": 0, "ties": 0, "losses": 0} for m in MAIN[1:]}
    for block in [finance["paired"]] + [r["paired"] for r in scaling["rows"]]:
        for m in totals:
            for k in totals[m]:
                totals[m][k] += block[m][k]
    return totals


def fig_head_to_head(finance, scaling):
    totals = _paired_totals(finance, scaling)
    n = sum(totals["Flat dense"].values())
    fig, ax = plt.subplots(figsize=(10, 3.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.62, bottom=0.06)
    title(fig, "Query by query, HAT-RAG is better far more often than it is worse, against every baseline",
          f"All {n} queries across the finance corpus and the five synthetic corpora, compared on evidence recall@5.")
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, linewidth=0) for c in (WIN, TIE, LOSS)]
    fig.legend(handles, ["HAT-RAG better", "Same", "HAT-RAG worse"], loc="upper left",
               bbox_to_anchor=(0.005, 0.77), ncol=3, handlelength=1.2, labelcolor=INK2)
    others = MAIN[1:]
    ys = list(range(len(others)))[::-1]
    ax.set_xlim(0, n * 1.07)
    ax.set_ylim(-0.6, len(others) - 0.4)
    ax.axis("off")
    gap = n * 0.003                                     # surface gap between touching segments
    for y, m in zip(ys, others):
        t = totals[m]
        left = 0.0
        for key, color in [("wins", WIN), ("ties", TIE), ("losses", LOSS)]:
            w = t[key]
            if w:
                start = left + (gap if left else 0)
                ax.add_patch(Rectangle((start, y - 0.2), w - (gap if left else 0), 0.4, facecolor=color, linewidth=0))
                if w / n > 0.04:
                    ax.text(left + w / 2, y, str(w), ha="center", va="center", fontsize=9,
                            color="#ffffff" if color != TIE else INK2)
                elif key == "losses":
                    ax.text(n + n * 0.008, y, f"{w} worse", ha="left", va="center", fontsize=9, color=INK2)
            left += w
        ax.text(-n * 0.012, y, f"vs {m}", ha="right", va="center", fontsize=9.5, color=INK2)
    save(fig, "fig7_head_to_head.png")


# ----------------------------------------------------------------------------
# MultiHop-RAG (results/multihop.json, written by run_multihop.py)
# ----------------------------------------------------------------------------
TYPE_LABEL = {"inference": "Inference", "comparison": "Comparison", "temporal": "Temporal"}


def fig_multihop_headline(mh):
    full = mh["full"]
    s5, s10 = full["by_k"]["5"]["summary"], full["by_k"]["10"]["summary"]
    order = sorted(MAIN, key=lambda m: -s5[m]["recall"])
    n_q = s5["HAT-RAG"]["n_queries"]
    nodes = full["index"]["nodes"]
    leaves = full["index"]["leaves"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(left=0.15, right=0.97, top=0.70, bottom=0.15)
    best_base = max((m for m in MAIN if m != "HAT-RAG"), key=lambda m: s5[m]["recall"])
    lead = s5["HAT-RAG"]["recall"] - s5[best_base]["recall"]
    share = s5["HAT-RAG"]["nodes_scored"] / nodes * 100
    head = (f"On {leaves:,} real passages HAT-RAG finds the most evidence while reading {share:.1f}% of the index"
            if lead > 0 else
            f"MultiHop-RAG at full scale: evidence found against the share of the index each query reads")
    title(fig, head,
          f"MultiHop-RAG: 609 news articles, {leaves:,} passages, {n_q:,} held-out multi-document questions. "
          f"\nLeft: evidence recall@5 (recall@10 in brackets). Right: nodes compared per query.")
    ys = list(range(len(order)))[::-1]
    a1.set_xlim(0, 1.0)
    a1.set_ylim(-0.6, len(order) - 0.4)
    vgrid(a1)
    for y, m in zip(ys, order):
        v, v10 = s5[m]["recall"], s10[m]["recall"]
        bar(a1, y - 0.17, 0.34, v, COLOR[m] if m == "HAT-RAG" else DE_EMPH)
        a1.text(v + 0.012, y, f"{v * 100:.1f}%  ({v10 * 100:.1f}%)", va="center", fontsize=9.5,
                color=INK if m == "HAT-RAG" else INK2)
    a1.set_yticks(ys)
    a1.set_yticklabels(order)
    for t in a1.get_yticklabels():
        t.set_color(INK if t.get_text() == "HAT-RAG" else INK2)
    a1.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a1.set_xlabel("Evidence recall@5")

    a2.set_xscale("log")
    a2.set_xlim(50, nodes * 12)
    a2.set_ylim(-0.6, len(order) - 0.4)
    vgrid(a2)
    for y, m in zip(ys, order):
        v = s5[m]["nodes_scored"]
        a2.barh(y, v - 50, left=50, height=0.34, color=COLOR[m] if m == "HAT-RAG" else DE_EMPH, linewidth=0)
        a2.text(v * 1.12, y, f"{v:,.0f} ({v / nodes * 100:.0f}%) · {s5[m]['latency_ms']:.1f} ms", va="center",
                fontsize=9, color=INK if m == "HAT-RAG" else INK2)
    a2.set_yticks(ys)
    a2.set_yticklabels(["" for _ in order])
    a2.set_xticks([100, 1000, 10000])
    a2.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    a2.set_xlabel("Nodes scored per query, log (share of index · search latency)")
    save(fig, "fig8_multihop_headline.png")


def fig_multihop_by_type(mh):
    s = mh["full"]["by_k"]["5"]["summary"]
    types = ["inference", "comparison", "temporal"]
    groups = [("Overall", None)] + [(TYPE_LABEL[t], t) for t in types]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.74, bottom=0.12)
    title(fig, "MultiHop-RAG evidence recall@5 by question type",
          "Inference: one fact assembled from several articles. Comparison: two sources' claims side by side. "
          "Temporal: which report came first.")
    legend_row(fig, MAIN, y=0.86, marker="box")
    width, gap = 0.1, 0.012
    ax.set_xlim(-0.5, len(groups) - 0.5)
    top = max(s[m]["recall"] if t is None else s[m]["by_type"][t]["recall"] for _, t in groups for m in MAIN)
    ax.set_ylim(0, min(1.0, top * 1.25))
    hgrid(ax)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    for g, (label, t) in enumerate(groups):
        for j, m in enumerate(MAIN):
            v = s[m]["recall"] if t is None else s[m]["by_type"][t]["recall"]
            x = g - 2 * (width + gap) + j * (width + gap) + gap / 2 + 0.006
            column(ax, x, width, v, COLOR[m])
            ax.text(x + width / 2, v + 0.012, f"{v * 100:.0f}", ha="center", va="bottom", fontsize=8,
                    color=INK if m == "HAT-RAG" else INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lab if t is None else f"{lab} (n={s['HAT-RAG']['by_type'][t]['n_queries']})"
                        for lab, t in groups])
    ax.tick_params(axis="x", length=0)
    save(fig, "fig9_multihop_by_type.png")


def fig_multihop_haystack(mh):
    hs = mh["haystack"]
    rows = hs["rows"]
    xs = [r["leaves"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.30})
    fig.subplots_adjust(left=0.07, right=0.93, top=0.68, bottom=0.14)
    title(fig, "Same questions, bigger haystack: adding real distractor articles to the corpus",
          f"{hs['queries']} fixed MultiHop-RAG test questions; the corpus grows from {rows[0]['docs']} to "
          f"{rows[-1]['docs']} articles ({xs[0]:,} to {xs[-1]:,} passages). Left: recall@5. Right: nodes scored per query.")
    legend_row(fig, MAIN, y=0.81)
    for ax, metric in [(a1, "recall"), (a2, "nodes_scored")]:
        hgrid(ax)
        for m in MAIN:
            _line(ax, xs, [r["summary"][m][metric] for r in rows], m)
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:,}" for x in xs])
        ax.minorticks_off()
        ax.set_xlim(xs[0] * 0.8, xs[-1] * 1.3)
        ax.set_xlabel("Passages in the corpus (log scale)")
    a1.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    a1.set_ylabel("Evidence recall@5")
    a2.set_yscale("log")
    a2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    lo, hi = a2.get_ylim()
    a2.set_yticks([t for t in [100, 200, 500, 1000, 2000, 5000, 10000, 20000] if lo <= t <= hi])
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    a2.set_ylabel("Nodes scored per query")
    ends = sorted((rows[-1]["summary"][m]["recall"], m) for m in MAIN)
    placed: List[float] = []
    y0, y1 = a1.get_ylim()
    for v, m in ends:
        y = v if not placed else max(v, placed[-1] + (y1 - y0) * 0.07)
        placed.append(y)
        a1.annotate(f"{v * 100:.1f}%", xy=(xs[-1], v), xytext=(xs[-1] * 1.08, y), va="center", fontsize=8.5,
                    color=INK if m == "HAT-RAG" else INK2, annotation_clip=False)
    save(fig, "fig10_multihop_haystack.png")


def fig_cost_all(finance, scaling, mh):
    """Nodes scored against corpus size over every benchmark: the scaling law in one picture."""
    import math
    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.subplots_adjust(left=0.08, right=0.80, top=0.70, bottom=0.12)
    pts = {m: [] for m in MAIN}
    for m in MAIN:
        pts[m].append((finance["config"]["leaves"], finance["summary"][m]["nodes_scored"], "o"))
        for r in scaling["rows"]:
            pts[m].append((r["leaves"], r["summary"][m]["nodes_scored"], "o"))
        if mh:
            rows = mh.get("haystack", {}).get("rows", [])
            for r in rows:
                pts[m].append((r["leaves"], r["summary"][m]["nodes_scored"], "s"))
            if not rows and mh.get("full"):
                pts[m].append((mh["full"]["index"]["leaves"],
                               mh["full"]["by_k"]["5"]["summary"][m]["nodes_scored"], "s"))
    top = max(p[0] for p in pts["HAT-RAG"])
    title(fig, f"From 24 to {top:,} passages, HAT-RAG's per-query cost grows sub-linearly; the baselines track the corpus",
          "Node embeddings compared per query, every benchmark on one log-log axis. Circles: 10-K and synthetic "
          "corpora (solid). Squares: MultiHop-RAG, real articles (dashed).\nFlat dense lies under Graph PPR: both score every passage.")
    legend_row(fig, MAIN, y=0.80)
    hgrid(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for m in MAIN:
        p = sorted(pts[m])
        for mk in ("o", "s"):                        # one line per benchmark family
            sel = [(x, y) for x, y, k in p if k == mk]
            if sel:
                ax.plot([x for x, _ in sel], [y for _, y in sel], color=COLOR[m], linewidth=LINE * 0.6,
                        alpha=0.6, linestyle="-" if mk == "o" else (0, (4, 2)), zorder=2)
                ax.plot([x for x, _ in sel], [y for _, y in sel], linestyle="none", marker=mk, color=COLOR[m],
                        markersize=MARKER, markeredgecolor=SURFACE, markeredgewidth=LINE, zorder=3)
    ends = sorted((sorted(pts[m])[-1][1], m) for m in MAIN)
    placed: List[float] = []
    for v, m in ends:
        y = math.log10(v) if not placed else max(math.log10(v), placed[-1] + 0.11)
        placed.append(y)
        ax.annotate(f"{m}  {v:,.0f}", xy=(top, v), xytext=(top * 1.35, 10 ** y), va="center", fontsize=8.5,
                    color=INK if m == "HAT-RAG" else INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=PX, shrinkA=0, shrinkB=4))
    ax.set_xlim(15, top * 1.3)
    ax.minorticks_off()
    ticks = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
    ax.set_xticks([t for t in ticks if 15 <= t <= top * 1.3])
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("Passages in the corpus (log scale)")
    ax.set_ylabel("Nodes scored per query (log)")
    save(fig, "fig11_cost_all_corpora.png")


def multihop_tables(mh) -> str:
    out: List[str] = []
    full = mh["full"]
    idx = full["index"]
    methods = MAIN + ["HAT-RAG (paper spec)"]
    out.append(f"\n### MultiHop-RAG: full corpus ({idx['leaves']:,} passages, {idx['nodes']:,} nodes; "
               f"test split, {full['by_k']['5']['summary']['HAT-RAG']['n_queries']:,} questions)\n")
    out.append("| Method | Recall@5 | Recall@10 | Hit@5 | All evidence@10 | MRR@5 | Nodes scored | Share of index | Latency (ms) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        a, b = full["by_k"]["5"]["summary"][m], full["by_k"]["10"]["summary"][m]
        out.append(f"| {m} | {pct(a['recall'])} | {pct(b['recall'])} | {pct(a['hit'])} | {pct(b['complete'])} | "
                   f"{a['mrr']:.3f} | {a['nodes_scored']:,.0f} | {a['nodes_scored'] / idx['nodes'] * 100:.1f}% | "
                   f"{a['latency_ms']:.2f} |")
    out.append("\n| Method (recall@5) | " + " | ".join(TYPE_LABEL[t] for t in TYPE_LABEL) + " |")
    out.append("|---|" + "---:|" * len(TYPE_LABEL))
    for m in methods:
        bt = full["by_k"]["5"]["summary"][m]["by_type"]
        out.append(f"| {m} | " + " | ".join(pct(bt[t]["recall"]) for t in TYPE_LABEL) + " |")
    out.append("\n| HAT-RAG vs (recall@5, per question) | better | same | worse |")
    out.append("|---|---:|---:|---:|")
    for m, t in full["by_k"]["5"]["paired"].items():
        out.append(f"| {m} | {t['wins']} | {t['ties']} | {t['losses']} |")
    out.append("\n| Variant (MultiHop-RAG, test, k = 5) | Recall@5 | Nodes scored |")
    out.append("|---|---:|---:|")
    for n, a in full["ablations"].items():
        out.append(f"| {n} | {pct(a['recall'])} | {a['nodes_scored']:,.0f} |")
    out.append("\n| β | Recall@5 | Nodes scored |")
    out.append("|---:|---:|---:|")
    for b in full["beam"]:
        out.append(f"| {b['beam_width']} | {pct(b['recall'])} | {b['nodes_scored']:,.0f} |")
    hs = mh.get("haystack")
    if hs:
        out.append(f"\n### MultiHop-RAG growing haystack ({hs['queries']} fixed test questions): "
                   "recall@5 · nodes scored · latency\n")
        out.append("| Method | " + " | ".join(f"{r['docs']} articles, {r['leaves']:,} passages" for r in hs["rows"]) + " |")
        out.append("|---|" + "---:|" * len(hs["rows"]))
        for m in methods:
            out.append(f"| {m} | " + " | ".join(
                f"{pct(r['summary'][m]['recall'])} · {r['summary'][m]['nodes_scored']:,.0f} · "
                f"{r['summary'][m]['latency_ms']:.1f} ms" for r in hs["rows"]) + " |")
    answers = RESULTS / "answers.json"
    if answers.exists():
        ans = json.loads(answers.read_text(encoding="utf-8"))
        out.append(f"\n### End-to-end answer accuracy ({ans['questions']} test questions, {ans['model']}, k = {ans['k']})\n")
        out.append("| Method | Accuracy | " + " | ".join(TYPE_LABEL[t] for t in TYPE_LABEL) + " |")
        out.append("|---|---:|" + "---:|" * len(TYPE_LABEL))
        for m, v in ans["summary"].items():
            out.append(f"| {m} | {pct(v['accuracy'])} | " + " | ".join(pct(v["by_type"][t]) for t in TYPE_LABEL) + " |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def tables(finance, scaling, verdict) -> str:
    out: List[str] = []
    s = finance["summary"]
    methods = MAIN + ["HAT-RAG (paper spec)"]

    out.append("### Verdict\n")
    out.append("| Rank | Method | Finance recall@5 | Synthetic recall@5 (mean, 5 sizes) | Combined score | Corpus scored / query |")
    out.append("|---:|---|---:|---:|---:|---:|")
    for i, m in enumerate(verdict["ranking"], 1):
        t = verdict["table"][m]
        name = f"**{m}**" if i == 1 else m
        out.append(f"| {i} | {name} | {pct(t['finance_recall'])} | {pct(t['synthetic_recall'])} | "
                   f"{t['score'] * 100:.1f} | {t['cost_fraction'] * 100:.0f}% |")

    out.append("\n### Finance corpus: all metrics (52 queries, k = 5)\n")
    out.append("| Method | Recall@5 | Hit@5 | All gold found | MRR | Filing coverage | Precision | Nodes scored | Latency (ms) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m in methods:
        v = s[m]
        out.append(f"| {m} | {pct(v['recall'])} | {pct(v['hit'])} | {pct(v['complete'])} | {v['mrr']:.3f} | "
                   f"{pct(v['doc_coverage'])} | {pct(v['precision'])} | {v['nodes_scored']:.1f} | {v['latency_ms']:.3f} |")

    out.append("\n### Finance corpus: recall@5 by query type\n")
    out.append("| Method | Single-hop (31) | Cross-document (12) | Thematic (9) |")
    out.append("|---|---:|---:|---:|")
    for m in methods:
        bt = s[m]["by_type"]
        out.append(f"| {m} | {pct(bt['single']['recall'])} | {pct(bt['cross']['recall'])} | {pct(bt['thematic']['recall'])} |")

    out.append("\n### Synthetic scaling: recall@5 · nodes scored · latency\n")
    head = " | ".join(f"{r['docs']} docs, {r['leaves']:,} chunks" for r in scaling["rows"])
    out.append(f"| Method | {head} |")
    out.append("|---|" + "---:|" * len(scaling["rows"]))
    for m in methods:
        cells = [f"{pct(r['summary'][m]['recall'])} · {r['summary'][m]['nodes_scored']:,.0f} · "
                 f"{r['summary'][m]['latency_ms']:.2f} ms" for r in scaling["rows"]]
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    out.append("\n### Index sizes per synthetic corpus\n")
    out.append("| Documents | Chunks N | Tree nodes | Level sizes (leaf to root) | Tree build (s) | Graph build (s) |")
    out.append("|---:|---:|---:|---|---:|---:|")
    for r in scaling["rows"]:
        out.append(f"| {r['docs']} | {r['leaves']:,} | {r['nodes']:,} | {' / '.join(map(str, r['levels']))} | "
                   f"{r['build_seconds']:.1f} | {r['graph_build_seconds']:.2f} |")

    out.append("\n### HAT-RAG ablations: recall@5\n")
    out.append(f"| Variant | Finance | Synthetic, {scaling['frontier_docs']} docs | Nodes scored (synthetic) |")
    out.append("|---|---:|---:|---:|")
    for n, a in finance["ablations"]["components"].items():
        b = scaling["components"][n]
        out.append(f"| {n} | {pct(a['recall'])} | {pct(b['recall'])} | {b['nodes_scored']:.0f} |")

    out.append(f"\n### Beam width (synthetic, {scaling['frontier_docs']} docs)\n")
    out.append("| β | HAT-RAG recall@5 | nodes scored | paper spec recall@5 | nodes scored |")
    out.append("|---:|---:|---:|---:|---:|")
    for a, b in zip(scaling["frontier"]["HAT-RAG"], scaling["frontier"]["HAT-RAG (paper spec)"]):
        out.append(f"| {a['beam_width']} | {pct(a['recall'])} | {a['nodes_scored']:.0f} | "
                   f"{pct(b['recall'])} | {b['nodes_scored']:.0f} |")

    out.append("\n### Beam width and diversity weight (finance)\n")
    out.append("| β | recall@5 | nodes scored |")
    out.append("|---:|---:|---:|")
    for a in finance["ablations"]["beam"]:
        out.append(f"| {a['config']['beam_width']} | {pct(a['recall'])} | {a['nodes_scored']:.1f} |")
    out.append("\n| λ | recall@5 | filing coverage | precision |")
    out.append("|---:|---:|---:|---:|")
    for a in finance["ablations"]["lambda"]:
        out.append(f"| {a['config']['diversity']} | {pct(a['recall'])} | {pct(a['doc_coverage'])} | {pct(a['precision'])} |")

    totals = _paired_totals(finance, scaling)
    out.append("\n### Per-query head-to-head (all corpora)\n")
    out.append("| HAT-RAG vs | better | same | worse |")
    out.append("|---|---:|---:|---:|")
    for m, t in totals.items():
        out.append(f"| {m} | {t['wins']} | {t['ties']} | {t['losses']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    finance = load("finance.json")
    scaling = load("scaling.json")
    verdict = load("verdict.json")
    print("[figures]")
    fig_verdict(verdict)
    fig_finance_by_type(finance)
    fig_scaling_recall(scaling)
    fig_scaling_cost(scaling)
    fig_frontier(scaling)
    fig_ablation(finance, scaling)
    fig_head_to_head(finance, scaling)
    mh = load("multihop.json") if (RESULTS / "multihop.json").exists() else None
    text = tables(finance, scaling, verdict)
    if mh:
        fig_multihop_headline(mh)
        fig_multihop_by_type(mh)
        if mh.get("haystack"):
            fig_multihop_haystack(mh)
        text += multihop_tables(mh)
    fig_cost_all(finance, scaling, mh)
    (RESULTS / "tables.md").write_text(text, encoding="utf-8")
    print("  wrote results/tables.md")
