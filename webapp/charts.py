"""Plotly figures for the Benchmarks tab, drawn straight from results/*.json."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import plotly.graph_objects as go

from webapp.style import HAIRLINE, INK, INK_2, METHOD_COLORS, METHOD_ORDER, MUTED

RESULTS = Path(__file__).resolve().parents[1] / "algorithms" / "compare-algos" / "results"
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"


def load(name: str) -> Optional[Dict[str, Any]]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _layout(fig: go.Figure, x_title: str, y_title: str, height: int = 340, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=110, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#fcfcfb",
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=HAIRLINE, font=dict(family=FONT, color=INK, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
    )
    axis = dict(showline=True, linecolor="#c3c2b7", linewidth=1, gridcolor="#ecebe6", gridwidth=1,
                zeroline=False, ticks="", tickfont=dict(color=MUTED, size=11), title_font=dict(color=INK_2, size=12))
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig


def _end_labels(fig: go.Figure, series: Dict[str, List[tuple]], fmt: str, min_gap: float, log_y: bool = False):
    """Direct labels at each line's right end, nudged apart so they never collide."""
    import math
    ends = []
    for m, pts in series.items():
        if pts:
            x, y = pts[-1]
            ends.append([m, x, y, math.log10(y) if log_y else y])
    ends.sort(key=lambda e: e[3])
    for a, b in zip(ends, ends[1:]):
        if b[3] - a[3] < min_gap:
            b[3] = a[3] + min_gap
    for m, x, y, ypos in ends:
        fig.add_annotation(x=x, y=10 ** ypos if log_y else ypos, xref="x", yref="y", text=f"{m} {fmt.format(y)}",
                           showarrow=False, xanchor="left", xshift=8, font=dict(size=11, color=INK, family=FONT))


def size_lines(rows: Sequence[Dict[str, Any]], metric: str = "recall", methods: Sequence[str] = METHOD_ORDER,
               x_key: str = "leaves", log_y: bool = False, height: int = 340) -> go.Figure:
    """One line per method over corpus size (log x)."""
    fig = go.Figure()
    series = {}
    for m in methods:
        pts = [(r[x_key], r["summary"][m][metric] * (100 if metric == "recall" else 1))
               for r in rows if m in r["summary"]]
        series[m] = pts
        ours = m == "HAT-RAG"
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], name=m, mode="lines+markers",
            line=dict(color=METHOD_COLORS[m], width=2.5 if ours else 2),
            marker=dict(size=9 if ours else 8, color=METHOD_COLORS[m], line=dict(color="#fcfcfb", width=2)),
            hovertemplate=(f"<b>{m}</b><br>%{{x:,}} passages<br>" +
                           ("recall@5 %{y:.1f}%" if metric == "recall" else
                            "%{y:,.0f} nodes scored" if metric == "nodes_scored" else "%{y:.2f} ms") +
                           "<extra></extra>")))
    if metric == "recall":
        _end_labels(fig, series, "{:.0f}%", 3.2)
        _layout(fig, "Passages in the corpus (log scale)", "Recall@5 (%)", height)
    elif metric == "nodes_scored":
        _end_labels(fig, series, "{:,.0f}", 0.09, log_y=True)
        _layout(fig, "Passages in the corpus (log scale)", "Nodes scored per query (log)", height)
        fig.update_yaxes(type="log")
    else:
        _end_labels(fig, series, "{:.1f} ms", 0.09, log_y=True)
        _layout(fig, "Passages in the corpus (log scale)", "Search latency per query, ms (log)", height)
        fig.update_yaxes(type="log")
    fig.update_xaxes(type="log")
    fig.update_layout(hovermode="closest")
    return fig


def type_bars(summary: Dict[str, Any], types: Sequence[str], labels: Dict[str, str],
              methods: Sequence[str] = METHOD_ORDER, height: int = 320) -> go.Figure:
    fig = go.Figure()
    for m in methods:
        vals = [summary[m]["by_type"].get(t, {}).get("recall", 0) * 100 for t in types]
        fig.add_trace(go.Bar(
            x=[labels.get(t, t) for t in types], y=vals, name=m, marker_color=METHOD_COLORS[m],
            marker_line=dict(color="#fcfcfb", width=2), marker_cornerradius=4,
            hovertemplate=f"<b>{m}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>"))
    _layout(fig, "", "Recall@5 (%)", height, legend=True)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08, margin=dict(l=8, r=8, t=40, b=8))
    return fig


def frontier(beam: Sequence[Dict[str, Any]], baselines: Dict[str, Dict[str, Any]], height: int = 340) -> go.Figure:
    """Recall against nodes scored: HAT-RAG traces a curve (beam width), baselines are points."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[b["nodes_scored"] for b in beam], y=[b["recall"] * 100 for b in beam], mode="lines+markers+text",
        name="HAT-RAG", line=dict(color=METHOD_COLORS["HAT-RAG"], width=2.5),
        marker=dict(size=9, color=METHOD_COLORS["HAT-RAG"], line=dict(color="#fcfcfb", width=2)),
        text=[f"β={b['beam_width']}" for b in beam], textposition="top center",
        textfont=dict(size=10, color=INK_2),
        hovertemplate="<b>HAT-RAG</b> %{text}<br>%{x:,.0f} nodes scored<br>recall@5 %{y:.1f}%<extra></extra>"))
    for m, s in baselines.items():
        fig.add_trace(go.Scatter(
            x=[s["nodes_scored"]], y=[s["recall"] * 100], mode="markers+text", name=m,
            marker=dict(size=11, color=METHOD_COLORS[m], line=dict(color="#fcfcfb", width=2)),
            text=[m], textposition="middle left", textfont=dict(size=11, color=INK),
            hovertemplate=f"<b>{m}</b><br>%{{x:,.0f}} nodes scored<br>recall@5 %{{y:.1f}}%<extra></extra>"))
    _layout(fig, "Nodes scored per query (log scale, lower is cheaper)", "Recall@5 (%)", height)
    fig.update_xaxes(type="log")
    fig.update_layout(margin=dict(l=8, r=20, t=12, b=8))
    return fig


def synthetic_rows() -> Optional[List[Dict[str, Any]]]:
    data = load("scaling")
    return data["rows"] if data else None


def haystack_rows() -> Optional[List[Dict[str, Any]]]:
    data = load("multihop")
    return data.get("haystack", {}).get("rows") if data else None
