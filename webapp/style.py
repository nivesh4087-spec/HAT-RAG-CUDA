"""Visual language for the HAT-RAG app: one palette, a few HTML components."""

import html
from typing import Iterable, Optional, Sequence, Tuple

# Categorical slots validated for CVD separation (light surface #fcfcfb).
# Identity follows the method everywhere -- app, figures, docs.
METHOD_COLORS = {
    "HAT-RAG": "#2a78d6",
    "Flat dense": "#eb6834",
    "RAPTOR collapsed": "#1baf7a",
    "Graph PPR": "#eda100",
    "HAT-RAG (paper spec)": "#898781",
}
METHOD_ORDER = ["HAT-RAG", "Flat dense", "RAPTOR collapsed", "Graph PPR"]
METHOD_FAMILY = {
    "HAT-RAG": "Ours: beam descent of the tree + entity links + PageRank on the visited subgraph",
    "Flat dense": "Vanilla RAG / DPR: cosine against every chunk",
    "RAPTOR collapsed": "RAPTOR: every level of the tree pooled, cosine top-k",
    "Graph PPR": "GraphRAG / HippoRAG: entity graph + PageRank over the whole corpus",
}

INK = "#1d1d1b"
INK_2 = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e3e2dc"
SURFACE = "#ffffff"

CSS = """
<style>
:root {
  --ink: #1d1d1b; --ink-2: #52514e; --muted: #898781;
  --hair: #e3e2dc; --surface: #ffffff; --plane: #f9f9f7; --wash: #f3f2ee;
  --accent: #2a6fd0; --accent-wash: #eaf1fb;
  --good: #0f7b0f; --good-wash: #e8f4e8; --warn-wash: #fdf3e1;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }
h1, h2, h3, h4 { letter-spacing: -0.015em; color: var(--ink); }
[data-testid="stSidebar"] h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 600; margin: 0.6rem 0 0.2rem 0; }

.hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 1.5rem;
  flex-wrap: wrap; margin-bottom: 0.4rem; }
.hero-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.025em; color: var(--ink); line-height: 1.15; }
.hero-title span { color: var(--accent); }
.hero p { color: var(--ink-2); margin: 0.35rem 0 0 0; font-size: 0.98rem; max-width: 760px; line-height: 1.5; }
.kb-chip { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.35rem 0.75rem;
  border: 1px solid var(--hair); border-radius: 999px; background: var(--surface);
  font-size: 0.82rem; color: var(--ink-2); white-space: nowrap; }
.kb-chip b { color: var(--ink); font-weight: 600; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin: 0.9rem 0 0.4rem 0; }
.stat { background: var(--surface); border: 1px solid var(--hair); border-radius: 12px; padding: 0.8rem 1rem; }
.stat .v { font-size: 1.45rem; font-weight: 650; color: var(--ink); line-height: 1.2; }
.stat .l { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }
.stat .d { font-size: 0.78rem; color: var(--good); margin-top: 0.25rem; font-weight: 550; }
.stat .d.neg { color: var(--ink-2); }
.stat.hl { border-color: #c9dbf4; background: linear-gradient(180deg, #f5f9fe 0%, #ffffff 100%); }

.mcard { background: var(--surface); border: 1px solid var(--hair); border-radius: 12px; padding: 0.85rem 1rem 0.9rem 1rem; height: 100%; }
.mcard.ours { border-color: #bcd3f2; box-shadow: 0 1px 0 #bcd3f2 inset; }
.mcard .t { display: flex; align-items: center; gap: 0.5rem; font-weight: 650; color: var(--ink); font-size: 0.95rem; }
.mcard .f { color: var(--muted); font-size: 0.76rem; margin: 0.2rem 0 0.6rem 0; line-height: 1.35; min-height: 2.1em; }
.mrow { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.82rem; color: var(--ink-2); padding: 0.18rem 0; }
.mrow span { white-space: nowrap; }
.mrow b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bar { height: 6px; background: var(--wash); border-radius: 3px; overflow: hidden; margin: 0.15rem 0 0.35rem 0; }
.bar > i { display: block; height: 100%; border-radius: 3px; }

.src { border: 1px solid var(--hair); border-radius: 10px; background: var(--surface); padding: 0.6rem 0.8rem; margin-bottom: 0.5rem; }
.src .h { display: flex; gap: 0.5rem; align-items: baseline; font-size: 0.82rem; color: var(--ink-2); }
.src .n { font-weight: 700; color: var(--accent); font-variant-numeric: tabular-nums; }
.src .ti { font-weight: 600; color: var(--ink); overflow: hidden; text-overflow: ellipsis; }
.src .sc { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.src .b { font-size: 0.86rem; color: var(--ink-2); margin-top: 0.3rem; line-height: 1.5; }
.src.cited { border-color: #bcd3f2; }
.src.gold { border-left: 3px solid var(--good); }
.tag { display: inline-block; font-size: 0.7rem; font-weight: 600; padding: 0.05rem 0.4rem; border-radius: 4px;
  background: var(--wash); color: var(--ink-2); margin-left: 0.25rem; }
.tag.gold { background: var(--good-wash); color: var(--good); }
.tag.abs { background: var(--accent-wash); color: var(--accent); }

.ev { font-size: 0.82rem; padding: 0.45rem 0; border-top: 1px solid var(--hair); color: var(--ink-2); line-height: 1.45; }
.ev:first-of-type { border-top: none; }
.ev .r { font-weight: 650; color: var(--ink); margin-right: 0.3rem; font-variant-numeric: tabular-nums; }
.ev .who { font-weight: 600; color: var(--ink); }

.meta-line { font-size: 0.8rem; color: var(--muted); margin: 0.1rem 0 0.4rem 0; }
.note { border: 1px solid var(--hair); background: var(--wash); border-radius: 10px; padding: 0.75rem 0.95rem;
  font-size: 0.88rem; color: var(--ink-2); line-height: 1.5; }
.note b { color: var(--ink); }
.section-lead { color: var(--ink-2); font-size: 0.92rem; margin: -0.3rem 0 0.9rem 0; line-height: 1.5; max-width: 900px; }
.legend { display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.8rem; color: var(--ink-2); margin: 0.2rem 0 0.4rem 0; }
.legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
</style>
"""


def esc(text) -> str:
    return html.escape(str(text), quote=True)


def dot(color: str) -> str:
    return f'<span class="dot" style="background:{color}"></span>'


def hero(title_html: str, subtitle: str, chips: Iterable[Tuple[str, str]]) -> str:
    chip_html = "".join(f'<span class="kb-chip">{dot(c)}{t}</span>' for t, c in chips)
    return (f'<div class="hero"><div><div class="hero-title">{title_html}</div><p>{esc(subtitle)}</p></div>'
            f'<div style="display:flex;gap:.5rem;flex-wrap:wrap">{chip_html}</div></div>')


def stats(items: Sequence[Tuple[str, str, Optional[str], bool]]) -> str:
    """items: (value, label, delta-or-None, highlight)."""
    cells = []
    for value, label, delta, hl in items:
        neg = bool(delta) and delta.lstrip().startswith(("-", "−"))
        d = f'<div class="d{" neg" if neg else ""}">{esc(delta)}</div>' if delta else ""
        cells.append(f'<div class="stat{" hl" if hl else ""}"><div class="v">{esc(value)}</div>'
                     f'<div class="l">{esc(label)}</div>{d}</div>')
    return f'<div class="stats">{"".join(cells)}</div>'


def legend(methods: Iterable[str]) -> str:
    return '<div class="legend">' + "".join(
        f"<span>{dot(METHOD_COLORS[m])}{esc(m)}</span>" for m in methods) + "</div>"
