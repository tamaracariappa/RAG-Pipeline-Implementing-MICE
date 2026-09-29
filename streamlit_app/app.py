"""
app.py - MICE Retrieval Evaluation Dashboard

Single-file review dashboard for the RAG pipeline evaluation: one tab per
embedding model, Strategies A / B / C side by side as cards, the metric
comparisons, and the PCA projection of that model's vector space.

Run with:
    streamlit run app.py          (from streamlit_app/)

Reads, per model key, from <project_root>/data/embeddings/<model_key>/:
    eval_results.json        aggregate metrics per strategy
    per_query_analysis.csv   one row per (query, strategy)
    text.index  / text_metadata.pkl    FAISS text track      (PCA section)
    mice.index  / mice_metadata.pkl    FAISS MICE track      (PCA section)

Every read is cached and guarded: a missing or malformed file degrades that
section to a warning instead of taking the app down. The index/pkl artifacts
are build outputs and are absent from a fresh checkout, so the PCA section is
expected to report "not built" until the ingestion pipeline has been run.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Paths and fixed vocabulary
# ─────────────────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
EMBED_ROOT = PROJECT_ROOT / "data" / "embeddings"

MODELS = {
    "bge_base": "BGE Base",
    "bge_m3": "BGE-M3",
    "jina_v3": "Jina v3",
}

STRATEGIES = ("A", "B", "C")

# Definitions taken from retrieval.py so the dashboard cannot drift from the code.
STRATEGY_NAME = {"A": "Baseline", "B": "Post-filter", "C": "MICE"}
STRATEGY_NOTE = {
    "A": "Dense search over the text index. No metadata.",
    "B": "Strategy A over-fetched, then filtered on metadata.",
    "C": "Metadata injected into the embedded text, searched on the MICE index.",
}

CSV_REQUIRED = [
    "model", "query_type", "equipment", "facility_type", "n_relevant",
    "query_text", "strategy", "recall", "mrr", "ndcg", "first_hit_rank",
    "metadata_gain", "delta_mice",
]

# PCA tracks: (index stem, display name, what it demonstrates).
PCA_TRACKS = (
    ("text", "Text track", "Raw work-order text, no metadata."),
    ("mice", "MICE track", "Metadata injected into the embedded text."),
)

# Dimensions the PCA scatter can be coloured by, mapped to metadata keys.
PCA_COLOR_BY = {
    "Equipment": "equipment",
    "Work-order type": "Type",
    "Building": "BuildingName",
}

# ─────────────────────────────────────────────────────────────────────────────
# Palette - the brand palette, used verbatim
#
# These four values are the specified brand palette and are applied exactly as
# given, for chrome and for the data series alike.
#
# Consequence worth knowing when reading a chart: slots A (#123F36) and B
# (#2A6B5C) are both dark desaturated greens and sit close together, so hue
# alone does not reliably separate those two series. Every bar therefore
# carries its own printed value and the legend is always present, so identity
# and magnitude are both readable without depending on the colour difference.
# ─────────────────────────────────────────────────────────────────────────────

BRAND = {
    "primary": "#123F36",
    "secondary": "#2A6B5C",
    "accent": "#C49A45",
    "cream": "#E8DCC4",
}

SERIES = {
    "A": BRAND["primary"],
    "B": BRAND["secondary"],
    "C": BRAND["accent"],
}

# Page neutrals, all tinted off the cream accent so nothing off-palette appears.
SURFACE = {
    "page": "#FBFAF6",
    "card": "#F4EEE1",
    "line": BRAND["cream"],
    "ink": "#14322B",
    "muted": "#5B6B64",
    "grid": "#E4DAC6",
}

FONT_SANS = "'Fira Sans', 'Segoe UI', system-ui, -apple-system, sans-serif"
FONT_MONO = "'Fira Code', 'Cascadia Mono', Consolas, monospace"

st.set_page_config(
    page_title="MICE Retrieval Evaluation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --rm-primary: %(primary)s;
  --rm-secondary: %(secondary)s;
  --rm-accent: %(accent)s;
  --rm-cream: %(cream)s;
  --rm-page: %(page)s;
  --rm-card: %(card)s;
  --rm-line: %(line)s;
  --rm-ink: %(ink)s;
  --rm-muted: %(muted)s;
  /* Dense dashboard spacing scale: 8 / 12 / 16 / 24 / 32. */
  --rm-s1: 8px; --rm-s2: 12px; --rm-s3: 16px; --rm-s4: 24px; --rm-s5: 32px;
}

.stApp { background: var(--rm-page); }
[data-testid="stSidebarNav"] { display: none; }
[data-testid="stHeader"] { background: transparent; height: 0; }
/* No sidebar is used, so its expand toggle is dead chrome. */
[data-testid="stSidebar"], [data-testid="stExpandSidebarButton"] { display: none; }

.block-container {
  max-width: 1420px;
  padding: var(--rm-s4) var(--rm-s4) var(--rm-s5);
}
/* Dense: collapse Streamlit's default inter-block gap. */
[data-testid="stVerticalBlock"] { gap: var(--rm-s2); }
[data-testid="stElementContainer"]:has(> .stMarkdown > [data-testid="stMarkdownContainer"] > .rm-sec) { margin-top: var(--rm-s3); }

html, body, [class*="css"], .stApp {
  font-family: %(sans)s;
  color: var(--rm-ink);
  font-size: 16px;
}

/* ── Masthead ─────────────────────────────────────────────── */
.rm-masthead {
  border-bottom: 3px solid var(--rm-primary);
  padding-bottom: var(--rm-s2);
  margin-bottom: var(--rm-s1);
}
.rm-title {
  font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--rm-primary); margin: 0 0 4px; line-height: 1.1;
}
.rm-sub { font-size: .95rem; color: var(--rm-muted); margin: 0; line-height: 1.45; }
.rm-sub b { color: var(--rm-secondary); font-weight: 600; }

/* ── Section headings ─────────────────────────────────────── */
.rm-sec {
  font-size: .92rem; font-weight: 700; letter-spacing: .08em;
  text-transform: uppercase; color: var(--rm-primary);
  border-bottom: 2px solid var(--rm-cream);
  padding-bottom: 6px; margin: var(--rm-s3) 0 6px;
}
.rm-note { font-size: .875rem; color: var(--rm-muted); margin: 0 0 var(--rm-s1); line-height: 1.45; }

/* ── Model meta strip ─────────────────────────────────────── */
.rm-meta {
  display: flex; flex-wrap: wrap; gap: var(--rm-s4);
  background: var(--rm-card); border: 1px solid var(--rm-line);
  border-radius: 6px; padding: var(--rm-s2) var(--rm-s3); margin-top: var(--rm-s2);
}
.rm-meta div { display: flex; flex-direction: column; gap: 2px; }
.rm-meta .k {
  font-size: .72rem; letter-spacing: .07em; text-transform: uppercase;
  color: var(--rm-muted); font-weight: 600;
}
.rm-meta .v {
  font-family: %(mono)s; font-size: 1rem; font-weight: 600; color: var(--rm-primary);
}

/* ── Strategy card ────────────────────────────────────────── */
.rm-card {
  background: var(--rm-card);
  border: 1px solid var(--rm-cream);
  border-left: 5px solid var(--rm-primary);
  border-radius: 8px;
  padding: var(--rm-s2) var(--rm-s3) var(--rm-s1);
  height: 100%%;
}
.rm-card .hd {
  font-size: 1.18rem; font-weight: 700; color: var(--rm-primary);
  letter-spacing: -.01em; line-height: 1.2;
}
.rm-card .hd span { color: var(--rm-muted); font-weight: 500; font-size: .95rem; }
.rm-card .note {
  font-size: .83rem; color: var(--rm-muted); line-height: 1.4;
  margin: 4px 0 var(--rm-s2); min-height: 2.8em;
}
.rm-stat {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: var(--rm-s2); padding: 7px 0;
  border-top: 1px solid rgba(18,63,54,.13);
}
.rm-stat .k {
  font-size: .78rem; letter-spacing: .05em; text-transform: uppercase;
  color: var(--rm-muted); font-weight: 600; white-space: nowrap;
}
.rm-stat .v {
  font-family: %(mono)s; font-size: 1.42rem; font-weight: 600;
  font-variant-numeric: tabular-nums; color: var(--rm-ink); line-height: 1.15;
}
.rm-stat .v.best { color: var(--rm-primary); }
.rm-stat .v.best::after {
  content: "best"; font-family: %(sans)s; font-size: .62rem; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase; color: var(--rm-cream);
  background: var(--rm-primary); border-radius: 3px; padding: 2px 5px;
  margin-left: 7px; vertical-align: .18em;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 2px solid var(--rm-cream); }
.stTabs [data-baseweb="tab"] {
  height: 2.7rem; padding: 0 var(--rm-s3); font-size: .98rem; font-weight: 600;
  color: var(--rm-muted); background: transparent;
  transition: color 180ms ease, background 180ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--rm-primary); background: rgba(232,220,196,.45); }
.stTabs [aria-selected="true"] {
  color: var(--rm-primary); background: var(--rm-card);
  border-bottom: 3px solid var(--rm-accent);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent; }

/* ── Widgets and table ────────────────────────────────────── */
[data-testid="stExpander"] details {
  border: 1px solid var(--rm-line); border-radius: 6px; background: var(--rm-card);
}
[data-testid="stExpander"] summary {
  font-size: .88rem; font-weight: 600; color: var(--rm-primary);
}
div[data-baseweb="select"] > div, .stTextInput input {
  border-color: var(--rm-line); border-radius: 5px; background: #FFFFFF; font-size: .92rem;
}
div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus {
  border-color: var(--rm-secondary); box-shadow: 0 0 0 2px rgba(42,107,92,.25);
}
.stMultiSelect [data-baseweb="tag"] {
  background: var(--rm-primary); color: var(--rm-cream); border-radius: 3px; font-weight: 500;
}
label[data-testid="stWidgetLabel"] p {
  font-size: .8rem !important; letter-spacing: .05em; text-transform: uppercase;
  color: var(--rm-primary); font-weight: 700;
}
.stRadio [role="radiogroup"] { gap: var(--rm-s3); }
[data-testid="stDataFrame"] { border: 1px solid var(--rm-line); border-radius: 6px; }
[data-testid="stAlert"] { border-radius: 6px; font-size: .9rem; }

@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0ms !important; animation-duration: 0ms !important; }
}
</style>
""" % {**BRAND, **SURFACE, "sans": FONT_SANS, "mono": FONT_MONO}

st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loading - cached, and every failure mode returns a message instead of raising
# ─────────────────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    """Path relative to the project root, for operator-facing messages."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@st.cache_data(show_spinner=False)
def load_eval(model_key: str) -> tuple[dict | None, str | None]:
    """Aggregate metrics for one model. Returns (payload, error_message)."""
    path = EMBED_ROOT / model_key / "eval_results.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, (f"`{_rel(path)}` not found. Run the evaluation for "
                      f"`{model_key}` to populate it.")
    except json.JSONDecodeError as exc:
        return None, f"`{_rel(path)}` is not valid JSON (line {exc.lineno}): {exc.msg}"
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"Could not read `{_rel(path)}`: {exc}"

    if not isinstance(payload, dict) or not payload.get("strategies"):
        return None, (f"`{_rel(path)}` has no `strategies` block - the evaluation "
                      f"may have been interrupted.")
    return payload, None


@st.cache_data(show_spinner=False)
def load_queries(model_key: str) -> tuple[pd.DataFrame | None, str | None]:
    """Per-query rows for one model, with derived columns. Returns (df, error)."""
    path = EMBED_ROOT / model_key / "per_query_analysis.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, f"`{_rel(path)}` not found."
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return None, f"`{_rel(path)}` could not be parsed: {exc}"
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"Could not read `{_rel(path)}`: {exc}"

    missing = [c for c in CSV_REQUIRED if c not in df.columns]
    if missing:
        return None, f"`{_rel(path)}` is missing column(s): {', '.join(missing)}"
    if df.empty:
        return None, f"`{_rel(path)}` has no rows."

    return _derive(df), None


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add the two columns the explorer filters on: outcome and winner."""
    df = df.copy()

    rank = pd.to_numeric(df["first_hit_rank"], errors="coerce")
    outcome = pd.Series("Miss - nothing relevant @10", index=df.index, dtype=object)
    outcome[rank.eq(1)] = "Hit @1"
    outcome[rank.between(2, 5)] = "Hit @2-5"
    outcome[rank.between(6, 10)] = "Hit @6-10"
    outcome[rank > 10] = "Hit beyond @10"
    df["outcome"] = outcome

    # Winner = the sole strategy with the highest recall on that query.
    recall = pd.to_numeric(df["recall"], errors="coerce")
    keys = [df["model"], df["query_text"]]
    best = recall.groupby(keys, sort=False).transform("max")
    is_best = recall.eq(best)
    n_best = is_best.groupby(keys, sort=False).transform("sum")
    sole = df["strategy"].astype(object).where(is_best & n_best.eq(1))
    df["winner"] = sole.groupby(keys, sort=False).transform("first").fillna("Tie")
    df.loc[best.eq(0), "winner"] = "None - all missed"

    return df


@st.cache_data(show_spinner=False)
def load_all_queries() -> tuple[pd.DataFrame, list[str]]:
    """Every model's per-query rows, concatenated. Missing models are reported."""
    frames, problems = [], []
    for key, label in MODELS.items():
        df, err = load_queries(key)
        if err:
            problems.append(f"{label} (`{key}`): {err}")
        else:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), problems
    return pd.concat(frames, ignore_index=True), problems


# ─────────────────────────────────────────────────────────────────────────────
# PCA projection
#
# Ported from the former pages/embedding_viz.py and charts/plotly_charts.py.
# One change from the original: it reads each model's FAISS index straight from
# that model's directory instead of going through faiss_store, which binds to a
# single ACTIVE_MODEL_KEY and so could only ever serve one of the three tabs.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=1800)
def pca_projection(model_key: str, track: str, color_by: str, n_samples: int = 1500):
    """
    2-D PCA of a sample of one model's vectors.

    Returns (payload, error_message); payload is
    (coords, labels, hover_texts, explained_variance, n_total, dim).
    """
    index_path = EMBED_ROOT / model_key / f"{track}.index"
    meta_path = EMBED_ROOT / model_key / f"{track}_metadata.pkl"

    missing = [p for p in (index_path, meta_path) if not p.exists()]
    if missing:
        return None, ("Vector index not built for this model: "
                      + ", ".join(f"`{_rel(p)}`" for p in missing)
                      + ". Run the ingestion pipeline to populate it.")

    try:
        import faiss
    except ImportError:
        return None, "`faiss` is not installed, so vectors cannot be read."

    try:
        index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as fh:
            metadata = pickle.load(fh)
    except (OSError, RuntimeError, pickle.UnpicklingError, EOFError) as exc:
        return None, f"Could not read `{_rel(index_path)}`: {exc}"

    total = int(index.ntotal)
    if total == 0 or not metadata:
        return None, f"`{_rel(index_path)}` is empty."

    usable = min(total, len(metadata))
    take = min(n_samples, usable)
    rng = np.random.default_rng(42)
    picks = sorted(rng.choice(usable, size=take, replace=False).tolist())

    try:
        vectors = np.zeros((take, index.d), dtype=np.float32)
        for row, idx in enumerate(picks):
            index.reconstruct(int(idx), vectors[row])
    except RuntimeError as exc:
        # Non-flat indexes need a direct map before reconstruct() works.
        return None, (f"Vectors cannot be reconstructed from "
                      f"`{_rel(index_path)}` ({exc}). A flat index is required.")

    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return None, "`scikit-learn` is not installed, so PCA cannot be computed."

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    sample_meta = [metadata[i] for i in picks]
    labels = [str(m.get(color_by) or "unknown") for m in sample_meta]
    hover = [
        f"<b>{m.get('WOID', '')}</b><br>"
        f"Equipment: {m.get('equipment', '')}<br>"
        f"Type: {m.get('Type', '')}<br>"
        f"<i>{str(m.get('WODescription', ''))[:80]}…</i>"
        for m in sample_meta
    ]
    payload = (coords, labels, hover,
               pca.explained_variance_ratio_.tolist(), total, int(index.d))
    return payload, None


def _brand_ramp(n: int) -> list[str]:
    """
    n steps interpolated across primary -> secondary -> accent.

    The PCA scatter colours by a metadata dimension with more classes than the
    four brand colours, so the intermediate steps are mixed from the brand
    anchors rather than introducing any hue from outside the palette.
    """
    anchors = [BRAND["primary"], BRAND["secondary"], BRAND["accent"]]
    rgb = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in anchors]
    if n <= 1:
        return [anchors[0]]
    out = []
    for step in range(n):
        pos = step / (n - 1) * (len(rgb) - 1)
        lo = min(int(pos), len(rgb) - 2)
        frac = pos - lo
        mixed = tuple(round(rgb[lo][c] + (rgb[lo + 1][c] - rgb[lo][c]) * frac)
                      for c in range(3))
        out.append("#%02X%02X%02X" % mixed)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style(fig: go.Figure, *, height: int, y_title: str = "", x_title: str = "",
           legend: bool = True) -> go.Figure:
    """Shared chart chrome: transparent surface, solid hairline grid, no clutter."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=30 if legend else 8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, size=13.5, color=SURFACE["ink"]),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
            font=dict(size=13, color=SURFACE["ink"]),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        bargap=0.3,
        bargroupgap=0.05,
        barcornerradius=3,
        hoverlabel=dict(
            bgcolor="#FFFFFF", bordercolor=BRAND["primary"],
            font=dict(family=FONT_SANS, size=13, color=SURFACE["ink"]),
        ),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(size=12.5, color=SURFACE["muted"])),
        showgrid=False, zeroline=False,
        showline=True, linecolor=SURFACE["grid"], linewidth=1,
        ticks="outside", ticklen=4, tickcolor=SURFACE["grid"],
        tickfont=dict(size=13, color=SURFACE["ink"]),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(size=12.5, color=SURFACE["muted"])),
        showgrid=True, gridcolor=SURFACE["grid"], gridwidth=1,
        zeroline=False, showline=False,
        tickfont=dict(size=12.5, color=SURFACE["muted"]),
    )
    return fig


def metric_by_k(strategies: list[dict], metric: str,
                title: str) -> tuple[go.Figure, pd.DataFrame]:
    """
    Grouped bars: one group per cut-off k, one bar per strategy.

    Every bar is labelled with its own value. Slots A and B are close in hue,
    so the printed number is what tells the two apart at a glance - and it is
    the denser read a reviewer wants anyway.
    """
    by_strategy = {s.get("strategy"): s for s in strategies}
    ks = sorted({int(k) for s in strategies for k in (s.get(metric) or {})})
    if not ks:
        return go.Figure(), pd.DataFrame()

    table = pd.DataFrame(
        {
            f"Strategy {code}": [
                (by_strategy[code].get(metric) or {}).get(str(k)) for k in ks
            ]
            for code in STRATEGIES if code in by_strategy
        },
        index=pd.Index([f"@{k}" for k in ks], name="Cut-off"),
    ).apply(pd.to_numeric, errors="coerce")

    fig = go.Figure()
    for code in STRATEGIES:
        column = f"Strategy {code}"
        if column not in table.columns:
            continue
        values = table[column].tolist()
        fig.add_bar(
            name=f"{code} · {STRATEGY_NAME[code]}",
            x=table.index.tolist(),
            y=values,
            marker=dict(color=SERIES[code], line=dict(width=0)),
            width=0.24,
            text=[f"{v:.4f}" if pd.notna(v) else "" for v in values],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=12, color=SURFACE["ink"]),
            cliponaxis=False,
            hovertemplate=(f"<b>Strategy {code} · {STRATEGY_NAME[code]}</b><br>"
                           f"{title} %{{x}}: %{{y:.4f}}<extra></extra>"),
        )

    ceiling = table.max(numeric_only=True).max()
    ceiling = float(ceiling) if pd.notna(ceiling) and ceiling > 0 else 1.0
    fig.update_yaxes(range=[0, ceiling * 1.26])
    return _style(fig, height=300, y_title=title), table


def delta_mice_chart(delta: dict) -> tuple[go.Figure, pd.DataFrame]:
    """
    Signed horizontal bars for delta-MICE (Strategy C minus Strategy A).

    Positive uses the brand primary, negative the brand accent. The side of the
    zero baseline and a signed label on every bar carry the polarity too, so it
    reads correctly without relying on the hue difference.
    """
    rows: list[tuple[str, float]] = []
    if isinstance(delta.get("mrr"), (int, float)):
        rows.append(("MRR", float(delta["mrr"])))
    for metric, label in (("recall", "Recall"), ("ndcg", "nDCG")):
        block = delta.get(metric) or {}
        for k in sorted(block, key=int):
            if isinstance(block[k], (int, float)):
                rows.append((f"{label}@{k}", float(block[k])))
    if not rows:
        return go.Figure(), pd.DataFrame()

    table = pd.DataFrame(rows, columns=["Metric", "ΔMICE (C − A)"]).set_index("Metric")
    values = table["ΔMICE (C − A)"].tolist()
    names = table.index.tolist()

    fig = go.Figure()
    for positive in (True, False):
        mask = [(v >= 0) == positive for v in values]
        if not any(mask):
            continue
        fig.add_bar(
            name="MICE helps (Δ ≥ 0)" if positive else "MICE hurts (Δ < 0)",
            orientation="h",
            y=[n for n, keep in zip(names, mask) if keep],
            x=[v for v, keep in zip(values, mask) if keep],
            marker=dict(color=BRAND["primary"] if positive else BRAND["accent"],
                        line=dict(width=0)),
            width=0.55,
            text=[f"{v:+.4f}" for v, keep in zip(values, mask) if keep],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=12, color=SURFACE["ink"]),
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Δ = %{x:+.4f}<extra></extra>",
        )

    # Zero stays on the axis as the reference, but the range is not mirrored:
    # when every value falls on one side, a symmetric range would leave half
    # the plot empty. Each side gets label headroom only if it carries bars.
    low = min(min(values), 0.0)
    high = max(max(values), 0.0)
    span = (high - low) or 1e-6
    low -= span * (0.30 if min(values) < 0 else 0.08)
    high += span * (0.30 if max(values) > 0 else 0.08)

    fig = _style(fig, height=max(200, 30 * len(values) + 62),
                 x_title="Strategy C minus Strategy A")
    fig.update_layout(barmode="relative")
    fig.update_xaxes(range=[low, high], showgrid=True,
                     gridcolor=SURFACE["grid"], zeroline=True,
                     zerolinecolor=BRAND["primary"], zerolinewidth=1)
    fig.update_yaxes(showgrid=False, autorange="reversed",
                     tickfont=dict(family=FONT_MONO, size=12.5,
                                   color=SURFACE["ink"]))
    return fig, table


def pca_scatter(coords, labels, hover, explained, title: str,
                max_classes: int = 6) -> go.Figure:
    """
    2-D PCA scatter, coloured by a metadata class.

    Classes past the largest `max_classes` fold into "Other" rather than being
    given generated hues, so the legend stays readable and every colour still
    comes from the brand ramp.
    """
    counts = pd.Series(labels).value_counts()
    keep = list(counts.index[:max_classes])
    shown = [lab if lab in keep else "Other" for lab in labels]
    order = keep + (["Other"] if "Other" in shown else [])
    ramp = _brand_ramp(len(order))

    fig = go.Figure()
    for colour, name in zip(ramp, order):
        mask = np.array([s == name for s in shown])
        if not mask.any():
            continue
        fig.add_scattergl(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=f"{name[:26]} ({int(mask.sum())})",
            marker=dict(color=colour, size=6, opacity=0.82,
                        line=dict(width=0.5, color=SURFACE["page"])),
            text=[t for t, m in zip(hover, mask) if m],
            hovertemplate="%{text}<extra></extra>",
        )

    fig = _style(
        fig, height=430,
        x_title=f"PC1 · {explained[0]:.1%} of variance",
        y_title=f"PC2 · {explained[1]:.1%} of variance",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=BRAND["primary"])),
        margin=dict(l=4, r=12, t=62, b=4),
        legend=dict(orientation="h", y=1.0, x=0, font=dict(size=11.5)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=SURFACE["grid"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _section(label: str, note: str = "") -> None:
    st.markdown(f'<div class="rm-sec">{label}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="rm-note">{note}</div>', unsafe_allow_html=True)


def _fmt(value, spec: str = ".4f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def _leaders(by_strategy: dict[str, dict]) -> dict[str, str | None]:
    """
    Winning strategy code per headline stat.

    Latency is deliberately absent: the p95 spread across strategies is a few
    milliseconds on ~320 ms, well inside one standard deviation of a single
    strategy's own latency, so a "best" badge there would claim a difference the
    measurement does not support. The number is still shown.
    """
    def pick(getter):
        scored = [(code, getter(record)) for code, record in by_strategy.items()]
        scored = [(c, v) for c, v in scored if isinstance(v, (int, float))]
        return max(scored, key=lambda pair: pair[1])[0] if scored else None

    return {
        "mrr": pick(lambda r: r.get("mrr")),
        "recall": pick(lambda r: (r.get("recall") or {}).get("10")),
        "ndcg": pick(lambda r: (r.get("ndcg") or {}).get("10")),
    }


def _strategy_card(record: dict, code: str, leaders: dict[str, str | None]) -> None:
    """One column of the A / B / C row: a tinted card with the headline stats."""
    latency = (record.get("latency") or {}).get("p95_ms")
    stats = [
        ("MRR", _fmt(record.get("mrr")), "mrr"),
        ("Recall@10", _fmt((record.get("recall") or {}).get("10")), "recall"),
        ("nDCG@10", _fmt((record.get("ndcg") or {}).get("10")), "ndcg"),
        ("Latency p95", _fmt(latency, ".0f") + (" ms" if latency is not None else ""),
         "latency"),
    ]
    rows = "".join(
        f'<div class="rm-stat"><span class="k">{key}</span>'
        f'<span class="v{" best" if leaders.get(tag) == code else ""}">{value}</span></div>'
        for key, value, tag in stats
    )
    st.markdown(
        f'<div class="rm-card" style="border-left-color:{SERIES[code]}">'
        f'<div class="hd">Strategy {code} <span>· {STRATEGY_NAME[code]}</span></div>'
        f'<div class="note">{STRATEGY_NOTE[code]}</div>{rows}</div>',
        unsafe_allow_html=True,
    )


def _table_view(label: str, table: pd.DataFrame, spec: str = "{:.4f}") -> None:
    """The plain-text twin of a chart: every plotted value, readable without colour."""
    if table.empty:
        return
    with st.expander(label):
        st.dataframe(table.style.format(spec, na_rep="—"), width="stretch")


def _meta_strip(payload: dict) -> None:
    cells = [
        ("Checkpoint", payload.get("model_id", "—")),
        ("Dimensions", f'{payload.get("embedding_dim", "—")}'),
        ("Queries", f'{payload.get("n_queries", "—")}'),
        ("Seed", f'{payload.get("seed", "—")}'),
        ("Cut-offs", ", ".join(str(k) for k in payload.get("top_k", [])) or "—"),
    ]
    st.markdown(
        '<div class="rm-meta">'
        + "".join(f'<div><span class="k">{k}</span><span class="v">{v}</span></div>'
                  for k, v in cells)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_pca(model_key: str) -> None:
    """The vector-space section: text track and MICE track, side by side."""
    _section("Vector space · PCA projection",
             "A fixed random sample of this model's indexed vectors, reduced to two "
             "principal components. The two tracks are the same work orders embedded "
             "without and with metadata injection, so the difference in cluster shape "
             "is the geometric footprint of MICE.")

    choice = st.radio("Colour points by", list(PCA_COLOR_BY),
                      horizontal=True, key=f"{model_key}-pca-colour")
    color_by = PCA_COLOR_BY[choice]

    for column, (track, track_label, track_note) in zip(
        st.columns(2, gap="medium"), PCA_TRACKS
    ):
        with column:
            payload, err = pca_projection(model_key, track, color_by)
            if err:
                st.warning(f"**{track_label}.** {err}", icon="⚠")
                continue
            coords, labels, hover, explained, total, dim = payload
            fig = pca_scatter(coords, labels, hover, explained,
                              f"{track_label} · {dim}-d → 2-d")
            st.plotly_chart(fig, width="stretch", key=f"{model_key}-pca-{track}",
                            config={"displayModeBar": False})
            st.markdown(
                f'<div class="rm-note">{track_note} '
                f'{len(labels):,} of {total:,} vectors sampled · '
                f'{sum(explained):.1%} of variance retained in two components.</div>',
                unsafe_allow_html=True,
            )


def render_model(model_key: str) -> None:
    payload, err = load_eval(model_key)
    if err:
        st.warning(f"**{MODELS[model_key]} unavailable.** {err}", icon="⚠")
        return

    strategies = [s for s in payload["strategies"] if s.get("strategy") in STRATEGIES]
    if not strategies:
        st.warning(f"**{MODELS[model_key]}** has no recognised strategies "
                   f"(expected A, B, C).", icon="⚠")
        return

    by_strategy = {s["strategy"]: s for s in strategies}
    _meta_strip(payload)

    _section("Strategies at a glance",
             "Headline metrics at k = 10. Latency is p95 over the query set.")
    leaders = _leaders(by_strategy)
    for column, code in zip(st.columns(3, gap="medium"), STRATEGIES):
        with column:
            if code in by_strategy:
                _strategy_card(by_strategy[code], code, leaders)
            else:
                st.markdown(
                    f'<div class="rm-card"><div class="hd">Strategy {code}</div>'
                    f'<div class="note">Not present in this run.</div></div>',
                    unsafe_allow_html=True,
                )

    _section("Retrieval quality by cut-off",
             "Recall and nDCG are plotted separately: they differ by an order of "
             "magnitude here, and forcing them onto one axis would invent a comparison.")
    recall_fig, recall_table = metric_by_k(strategies, "recall", "Recall")
    ndcg_fig, ndcg_table = metric_by_k(strategies, "ndcg", "nDCG")
    left, right = st.columns(2, gap="medium")
    with left:
        st.plotly_chart(recall_fig, width="stretch", key=f"{model_key}-recall",
                        config={"displayModeBar": False})
        _table_view("Recall — table view", recall_table)
    with right:
        st.plotly_chart(ndcg_fig, width="stretch", key=f"{model_key}-ndcg",
                        config={"displayModeBar": False})
        _table_view("nDCG — table view", ndcg_table)

    _section("ΔMICE — the isolated effect of metadata injection",
             "Strategy C minus Strategy A. Above zero, embedding the metadata helped; "
             "below zero, it cost accuracy.")
    delta = payload.get("delta_mice") or {}
    if not delta:
        st.info("This run has no `delta_mice` block — it needs both Strategy A "
                "and Strategy C.")
    else:
        delta_fig, delta_table = delta_mice_chart(delta)
        st.plotly_chart(delta_fig, width="stretch", key=f"{model_key}-delta",
                        config={"displayModeBar": False})
        _table_view("ΔMICE — table view", delta_table, "{:+.4f}")

    render_pca(model_key)


def _options(df: pd.DataFrame, column: str) -> list:
    return sorted(df[column].dropna().unique().tolist())


def render_explorer() -> None:
    df, problems = load_all_queries()
    for problem in problems:
        st.warning(f"**Per-query data unavailable.** {problem}", icon="⚠")
    if df.empty:
        st.info("No per-query data could be loaded, so the explorer has nothing to show.")
        return

    _section("Filters", "One filter row, scoping the whole table. "
                        "Leave a filter empty to include everything.")
    c1, c2, c3, c4 = st.columns([1.1, 0.9, 1.3, 1.3], gap="small")
    models = c1.multiselect("Model", list(MODELS), format_func=MODELS.get)
    strategies = c2.multiselect("Strategy", list(STRATEGIES),
                                format_func=lambda c: f"{c} · {STRATEGY_NAME[c]}")
    outcomes = c3.multiselect("Failure mode / rank of first hit", _options(df, "outcome"))
    winners = c4.multiselect("Strategy that won the query", _options(df, "winner"))

    c5, c6, c7 = st.columns([1.2, 1.4, 1.6], gap="small")
    equipment = c5.multiselect("Equipment", _options(df, "equipment"))
    facilities = c6.multiselect("Facility type", _options(df, "facility_type"))
    search = c7.text_input("Search query text", placeholder="e.g. roof leak")

    view = df
    for column, chosen in (
        ("model", models), ("strategy", strategies), ("outcome", outcomes),
        ("winner", winners), ("equipment", equipment), ("facility_type", facilities),
    ):
        if chosen:
            view = view[view[column].isin(chosen)]
    if search.strip():
        view = view[view["query_text"].str.contains(search.strip(), case=False,
                                                    na=False, regex=False)]

    _section("Per-query results",
             f"{len(view):,} of {len(df):,} rows · "
             f"{view['query_text'].nunique():,} distinct queries. Sort by any column. "
             "Metadata gain and ΔMICE are query-level, so they repeat across strategies.")
    st.dataframe(
        view,
        width="stretch",
        height=520,
        hide_index=True,
        column_order=[
            "model", "strategy", "winner", "outcome", "query_text", "equipment",
            "facility_type", "n_relevant", "recall", "mrr", "ndcg",
            "first_hit_rank", "metadata_gain", "delta_mice",
        ],
        column_config={
            "model": st.column_config.TextColumn("Model", width="small"),
            "strategy": st.column_config.TextColumn("Strat.", width="small"),
            "winner": st.column_config.TextColumn("Winner", width="small"),
            "outcome": st.column_config.TextColumn("Outcome", width="medium"),
            # Medium, not large: the metric columns are what a reviewer is here
            # for, and they must be on screen without a horizontal scroll.
            "query_text": st.column_config.TextColumn("Query", width="medium"),
            "equipment": st.column_config.TextColumn("Equipment", width="medium"),
            "facility_type": st.column_config.TextColumn("Facility type", width="medium"),
            "n_relevant": st.column_config.NumberColumn("Relevant", width="small",
                                                        format="%d"),
            "recall": st.column_config.ProgressColumn("Recall", min_value=0.0,
                                                      max_value=1.0, format="%.3f"),
            "mrr": st.column_config.NumberColumn("MRR", format="%.3f", width="small"),
            "ndcg": st.column_config.NumberColumn("nDCG", format="%.3f", width="small"),
            "first_hit_rank": st.column_config.NumberColumn("1st hit", format="%d",
                                                            width="small"),
            "metadata_gain": st.column_config.NumberColumn("Metadata gain", format="%+.3f"),
            "delta_mice": st.column_config.NumberColumn("ΔMICE", format="%+.3f"),
        },
    )
    st.markdown(
        '<div class="rm-note">Metadata gain = best of Strategies B and C minus '
        'Strategy A, on recall. ΔMICE = Strategy C minus Strategy A. Winner is the '
        'single highest-recall strategy for that query; ties and all-miss queries are '
        'labelled as such.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="rm-masthead">'
    '<div class="rm-title">MICE Retrieval Evaluation</div>'
    '<div class="rm-sub">Metadata-injected chunk embeddings for facilities-management '
    'RAG · Strategy <b>A</b> baseline, <b>B</b> post-filter, <b>C</b> MICE · '
    'three embedding backbones, one query set, one seed.</div></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs([*MODELS.values(), "Query Explorer"])
for tab, model_key in zip(tabs, MODELS):
    with tab:
        render_model(model_key)
with tabs[-1]:
    render_explorer()
