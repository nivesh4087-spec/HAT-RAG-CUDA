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
          "Left: ½ finance recall@5 + ½ mean synthetic recall@5 (8 to 128 documents). "
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
          "Synthetic cross-document corpus, 8 to 128 documents (99 to 1,582 chunks). "
          "Evidence recall@5 over single-hop and two-document queries.")
    legend_row(fig, methods, y=0.86)
    hgrid(ax)
    for m in methods:
        _line(ax, *_series(scaling, "recall", m), m)
    _size_axis(ax, scaling)
    ax.set_ylim(0.4, 1.03)
    ax.set_xlabel("Leaf chunks in the corpus (log scale)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
    # end labels in the right margin, spaced so they never overlap
    ends = sorted((_series(scaling, "recall", m)[1][-1], m) for m in methods)
    placed: List[float] = []
    for v, m in ends:
        y = v if not placed else max(v, placed[-1] + 0.045)
        placed.append(y)
        ax.annotate(f"{m}  {v * 100:.1f}%", xy=(scaling["rows"][-1]["leaves"], v),
                    xytext=(1.02, (y - 0.4) / 0.63), textcoords="axes fraction", va="center", fontsize=8.5,
                    color=INK if m == "HAT-RAG" else INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=PX, shrinkA=0, shrinkB=4))
    save(fig, "fig3_scaling_recall.png")


def fig_scaling_cost(scaling):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.68, bottom=0.14)
    title(fig, "HAT-RAG compares a small, slowly growing slice of the corpus, so its cost grows sub-linearly",
          "Left: node embeddings compared per query. Right: median search time (query encoding excluded, same CPU).\n"
          "Below ~800 chunks HAT-RAG's fixed per-query work makes it slower than flat; at 1,582 chunks it is the fastest.")
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
        ticks = [50, 100, 200, 500, 1000, 2000] if metric == "nodes_scored" else [0.05, 0.1, 0.2, 0.5, 1, 2, 5]
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
    (RESULTS / "tables.md").write_text(tables(finance, scaling, verdict), encoding="utf-8")
    print("  wrote results/tables.md")
