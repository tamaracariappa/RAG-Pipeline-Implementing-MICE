"""
app.py - MICE Retrieval Evaluation Dashboard

Standalone entry point. One tab per embedding model, Strategies A / B / C as
compact metric cards, ECharts metric comparisons, the precomputed vector
projection, and an AgGrid query explorer.

Run with:
    streamlit run app.py          (from streamlit_app/)

Reads, per model key, from <project_root>/data/embeddings/<model_key>/:
    eval_results.json        aggregate metrics per strategy
    per_query_analysis.csv   one row per (query, strategy)
    pca_cache.json           static 2-D projection, written offline

No FAISS index is ever opened here. The projection is precomputed by
scripts/precompute_pca.py so the app stays inside a small memory budget and
starts instantly; `pca_cache.json` is a few hundred KB of plain JSON.

Every read is cached and guarded: a missing or malformed file degrades that
section to a warning instead of taking the app down.
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import pandas as pd
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
    "C": "Metadata injected into the embedded text, MICE index.",
}

CSV_REQUIRED = [
    "model", "query_type", "equipment", "facility_type", "n_relevant",
    "query_text", "strategy", "recall", "mrr", "ndcg", "first_hit_rank",
    "metadata_gain", "delta_mice",
]

PCA_TRACKS = (("text", "Text track"), ("mice", "MICE track"))

PCA_COLOR_BY = {
    "Equipment": "equipment",
    "Work-order type": "Type",
    "Building": "BuildingName",
}

# The MICE template's field labels, bolded in the comparison pane so the
# injected structure is visible rather than buried in the sentence.
MICE_LABELS = re.compile(
    r"(building id|building name|facility type|equipment system|"
    r"work period|work order description):")

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens - primitive -> semantic -> component
#
# Primitives are the four brand values, verbatim. Nothing downstream uses a raw
# hex; every rule references a semantic or component token, so a palette change
# is a one-line edit here.
# ─────────────────────────────────────────────────────────────────────────────

PRIMITIVE = {
    "forest_900": "#123F36",   # brand primary
    "forest_600": "#2A6B5C",   # brand secondary
    "gold_500": "#C49A45",     # brand accent
    "cream_200": "#E8DCC4",    # brand panel tint
    "canvas_50": "#FAF7F0",    # app surface
}

SEMANTIC = {
    "primary": PRIMITIVE["forest_900"],
    "secondary": PRIMITIVE["forest_600"],
    "accent": PRIMITIVE["gold_500"],
    "canvas": PRIMITIVE["canvas_50"],
    # The panel tint at 15%, as specified - a wash over the canvas, not a block.
    "panel": "rgba(232, 220, 196, 0.15)",
    "panel_solid": "#F6F1E6",   # opaque equivalent, for chart surfaces
    "rule": PRIMITIVE["forest_600"],
    "grid": "rgba(42, 107, 92, 0.22)",
    "ink": PRIMITIVE["forest_900"],
    "ink_muted": "#5A6B64",
}

# Series colours, injected into ECharts verbatim as `color`.
SERIES_COLORS = [SEMANTIC["primary"], SEMANTIC["secondary"], SEMANTIC["accent"]]
SERIES = dict(zip(STRATEGIES, SERIES_COLORS))

# Vector Inspector: (cache key, legend label, colour). Text and MICE sit on the
# two ends of the palette so the traces stay apart where they overlap.
VECTOR_TRACKS = (
    ("text", "Text · Strategy A", SERIES["A"]),
    ("mice", "MICE · Strategy C", SERIES["C"]),
)

# Both display faces ship a single 400 weight. Nothing that uses them may ask
# for bold: the browser would synthesise one, and a faux-bold display face
# reads as a rendering fault rather than as emphasis.
#
# Limelight is loud, so it is spent once, on the masthead. The section rules
# repeat down every tab; at that frequency a decorative face stops being a
# signal and turns into texture, so they keep the quieter BBH Hegarty.
FONT_TITLE = "'Limelight', sans-serif"
FONT_DISPLAY = "'BBH Hegarty', sans-serif"
# One request carries both families.
FONT_IMPORT = ("@import url('https://fonts.googleapis.com/css2"
               "?family=BBH+Hegarty&family=Limelight&display=swap');")
FONT_SANS = ('system-ui, -apple-system, "Segoe UI", Roboto, '
             '"Helvetica Neue", Arial, sans-serif')
FONT_MONO = ('ui-monospace, "Cascadia Mono", "SF Mono", Menlo, Consolas, '
             '"Liberation Mono", monospace')

st.set_page_config(
    page_title="MICE Retrieval Evaluation",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The token block is generated from the dicts above rather than substituted into
# the stylesheet. The stylesheet body is a plain literal with no formatting
# applied, so a bare `%` in a rule like `height: 100%` cannot break it.
TOKENS = {
    # Primitive - the raw brand values, referenced by nothing but the semantics.
    "--c-forest-900": PRIMITIVE["forest_900"],
    "--c-forest-600": PRIMITIVE["forest_600"],
    "--c-gold-500": PRIMITIVE["gold_500"],
    "--c-cream-200": PRIMITIVE["cream_200"],
    "--c-canvas-50": PRIMITIVE["canvas_50"],

    # Semantic - purpose aliases; every rule below uses these, never a raw hex.
    "--color-primary": "var(--c-forest-900)",
    "--color-secondary": "var(--c-forest-600)",
    "--color-accent": "var(--c-gold-500)",
    "--surface-canvas": "var(--c-canvas-50)",
    "--surface-panel": SEMANTIC["panel"],
    "--border-rule": "var(--color-secondary)",
    "--text-primary": "var(--c-forest-900)",
    "--text-muted": SEMANTIC["ink_muted"],
    "--text-on-primary": "var(--c-cream-200)",

    # Scale - density 8/10, a tight dashboard rhythm.
    "--s-1": "4px", "--s-2": "8px", "--s-3": "12px",
    "--s-4": "16px", "--s-5": "24px", "--s-6": "32px",
    "--radius": "4px",
    "--font-title": FONT_TITLE,
    "--font-display": FONT_DISPLAY,
    "--font-sans": FONT_SANS,
    "--font-mono": FONT_MONO,

    # Component - the only layer a rule is allowed to tune per component.
    "--card-bg": "var(--surface-panel)",
    "--card-border": "1px solid var(--border-rule)",
    "--card-pad": "var(--s-3)",
    "--metric-size": "1.85rem",
    "--label-size": ".82rem",
    "--section-size": "1.05rem",
    # Body copy floor. Every explanatory line - subtitle, section note, card
    # note - sits at or above this, so the dashboard reads from a metre away.
    "--body-size": "1.05rem",
}

STYLESHEET = """
.stApp { background: var(--surface-canvas); }
/* This is a single-page dashboard, but `pages/` sits next to this file and
   Streamlit builds a multipage nav from that directory whatever this script
   does. Four surfaces carry it - the nav, the panel, the collapsed chevron and
   the expand button - and Streamlit's own rules are specific enough to need
   !important on each. Deleting or renaming `pages/` is the root fix; until
   then this is what keeps the chrome off the screen. */
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; height: 0; }

.block-container { max-width: 1440px; padding: var(--s-5) var(--s-5) var(--s-6); }
/* Density 8: collapse Streamlit's generous default block rhythm. */
[data-testid="stVerticalBlock"] { gap: var(--s-2); }

html, body, [class*="css"], .stApp {
  font-family: var(--font-sans);
  color: var(--text-primary);
  font-size: 16px;
}
/* Every numeral in the app aligns in a column. */
.rm-metric .v, .rm-meta .v, .ag-cell-value {
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}

/* ── Masthead ─────────────────────────────────────────────── */
.rm-masthead {
  border-bottom: 2px solid var(--color-primary);
  padding-bottom: var(--s-2); margin-bottom: var(--s-1);
}
.rm-title {
  font-family: var(--font-title);
  font-size: 1.9rem; letter-spacing: .01em;
  color: var(--color-primary); margin: 0 0 3px; line-height: 1.25;
}
.rm-sub { font-size: var(--body-size); color: var(--text-muted); margin: 0; line-height: 1.55; }
.rm-sub b { color: var(--color-secondary); font-weight: 600; }

/* ── Section rule ─────────────────────────────────────────── */
/* Tracking eases off as the size goes up: .09em on a 1.05rem uppercase run
   reads as spaced-out rather than as a heading. */
.rm-sec {
  font-family: var(--font-display);
  font-size: var(--section-size); letter-spacing: .06em;
  text-transform: uppercase; color: var(--color-primary); line-height: 1.3;
  border-bottom: 1px solid var(--border-rule);
  padding-bottom: var(--s-2); margin: var(--s-5) 0 var(--s-3);
}
/* No measure cap. What survives here is a data caption - a row count, a
   variance figure - never a paragraph, so it spans its container and stops. */
.rm-note {
  font-size: var(--body-size); color: var(--text-muted);
  margin: 0 0 var(--s-3); line-height: 1.6;
}

/* ── Meta strip ───────────────────────────────────────────── */
.rm-meta {
  display: flex; flex-wrap: wrap; gap: var(--s-5);
  background: var(--card-bg); border: var(--card-border);
  border-radius: var(--radius); padding: var(--s-2) var(--s-3); margin-top: var(--s-2);
}
.rm-meta div { display: flex; flex-direction: column; gap: 1px; }
.rm-meta .k {
  font-size: .76rem; letter-spacing: .08em; text-transform: uppercase;
  color: var(--text-muted); font-weight: 700;
}
.rm-meta .v { font-size: 1.08rem; font-weight: 600; color: var(--color-primary); }

/* ── Metric card ──────────────────────────────────────────── */
.rm-card {
  background: var(--card-bg); border: var(--card-border);
  border-radius: var(--radius); padding: var(--card-pad); height: 100%;
}
.rm-card .hd {
  display: flex; align-items: baseline; gap: var(--s-2);
  font-size: 1.18rem; font-weight: 700; color: var(--color-primary); line-height: 1.25;
}
.rm-card .hd i {
  width: 10px; height: 10px; border-radius: 2px; flex: none; font-style: normal;
}
.rm-card .hd span { color: var(--text-muted); font-weight: 500; font-size: .95rem; }
/* min-height holds the three cards' metric rows on one baseline when the notes
   wrap to different line counts; it tracks the line-height above it. */
.rm-card .note {
  font-size: .95rem; color: var(--text-muted); line-height: 1.5;
  margin: var(--s-1) 0 var(--s-2); min-height: 3.1em;
}
.rm-metric {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: var(--s-2); padding: 7px 0;
  border-top: 1px solid rgba(42, 107, 92, .2);
}
.rm-metric .k {
  font-size: var(--label-size); letter-spacing: .05em; text-transform: uppercase;
  color: var(--text-muted); font-weight: 700; white-space: nowrap;
}
.rm-metric .v {
  font-size: var(--metric-size); font-weight: 600;
  color: var(--text-primary); line-height: 1.1;
}
.rm-metric .v.lead { color: var(--color-primary); }
.rm-metric .v.lead::after {
  content: "lead"; font-family: var(--font-sans); font-size: .66rem; font-weight: 700;
  letter-spacing: .09em; text-transform: uppercase; color: var(--color-primary);
  background: var(--c-gold-500); border-radius: 2px; padding: 1px 4px;
  margin-left: 6px; vertical-align: .2em;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 1px; border-bottom: 1px solid var(--border-rule); }
.stTabs [data-baseweb="tab"] {
  height: 2.7rem; padding: 0 var(--s-4); font-size: 1rem; font-weight: 600;
  color: var(--text-muted); background: transparent; transition: all 180ms ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--color-primary); }
.stTabs [aria-selected="true"] {
  color: var(--color-primary); background: var(--surface-panel);
  border-bottom: 2px solid var(--color-accent);
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent; }

/* ── Widgets ──────────────────────────────────────────────── */
[data-testid="stExpander"] details {
  border: var(--card-border); border-radius: var(--radius); background: var(--card-bg);
}
[data-testid="stExpander"] summary {
  font-size: .95rem; font-weight: 600; color: var(--color-primary);
}
div[data-baseweb="select"] > div, .stTextInput input {
  border-color: var(--border-rule); border-radius: var(--radius);
  background: #FFFFFF; font-size: .95rem;
}
div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus {
  border-color: var(--color-primary); box-shadow: 0 0 0 2px rgba(18, 63, 54, .2);
}
.stMultiSelect [data-baseweb="tag"] {
  background: var(--color-primary); color: var(--text-on-primary);
  border-radius: 2px; font-weight: 500;
}
label[data-testid="stWidgetLabel"] p {
  font-size: var(--label-size) !important; letter-spacing: .06em;
  text-transform: uppercase; color: var(--color-primary); font-weight: 700;
}
.stRadio [role="radiogroup"] { gap: var(--s-4); }
.stRadio [role="radiogroup"] label p { font-size: .95rem; }
[data-testid="stAlert"] { border-radius: var(--radius); font-size: 1rem; }

/* ── Vector Inspector ─────────────────────────────────────── */
.rm-pane {
  background: var(--card-bg); border: var(--card-border);
  border-radius: var(--radius); padding: var(--s-3); height: 100%;
}
.rm-pane .hd {
  display: flex; align-items: baseline; gap: var(--s-2);
  font-size: 1.05rem; font-weight: 700; color: var(--color-primary);
  margin-bottom: var(--s-2);
}
.rm-pane .hd i {
  width: 10px; height: 10px; border-radius: 2px; flex: none; font-style: normal;
}
.rm-pane .hd span { color: var(--text-muted); font-weight: 500; font-size: .9rem; }
/* The embedded string, shown as embedded: monospace, wrapped, nothing elided. */
.rm-pane .body {
  font-family: var(--font-mono); font-size: .95rem; line-height: 1.65;
  color: var(--text-primary); white-space: pre-wrap; word-break: break-word;
  margin: 0; min-height: 7.5em;
}
.rm-pane .body em { color: var(--color-secondary); font-style: normal; font-weight: 700; }

@media (prefers-reduced-motion: reduce) {
  * { transition-duration: 0ms !important; animation-duration: 0ms !important; }
}
"""

CSS = ("<style>\n" + FONT_IMPORT + "\n:root {\n"
       + "".join(f"  {name}: {value};\n" for name, value in TOKENS.items())
       + "}\n" + STYLESHEET + "</style>")

st.markdown(CSS, unsafe_allow_html=True)

# AgGrid, re-skinned to the brand.
#
# These target AG Grid's element classes directly rather than setting `--ag-*`
# custom properties on `.ag-theme-alpine`. st_aggrid 1.2 ships an AG Grid built
# on the Theming API: the grid root carries generated classes
# (`ag-theme-params-1 ag-theme-columnDropStyle-2 …`) and never `.ag-theme-alpine`,
# so a `.ag-theme-alpine { --ag-header-background-color: … }` rule matches
# nothing and the header keeps its stock near-white. Element selectors match in
# both the legacy and the Theming API build.
GRID_CSS = {
    ".ag-root-wrapper": {
        "border": f"1px solid {SEMANTIC['rule']} !important",
        "border-radius": "4px",
    },
    ".ag-header": {
        "background-color": f"{SEMANTIC['primary']} !important",
        "border-bottom": f"1px solid {SEMANTIC['primary']} !important",
    },
    ".ag-header-cell, .ag-header-group-cell": {
        "color": f"{PRIMITIVE['cream_200']} !important",
    },
    ".ag-header-cell-text": {
        "font-family": FONT_SANS,
        "font-weight": "700",
        "font-size": "12.5px",
        "letter-spacing": "0.05em",
        "text-transform": "uppercase",
        "color": f"{PRIMITIVE['cream_200']} !important",
    },
    ".ag-header .ag-icon, .ag-header-cell-menu-button, .ag-header-icon": {
        "color": f"{PRIMITIVE['cream_200']} !important",
        "opacity": "0.85",
    },
    ".ag-header-cell-resize::after": {
        "background-color": "rgba(232, 220, 196, 0.35) !important",
    },
    ".ag-cell": {
        "font-family": FONT_MONO,
        "font-variant-numeric": "tabular-nums",
        "font-size": "14px",
        "color": f"{SEMANTIC['ink']} !important",
    },
    ".ag-row-odd": {"background-color": f"{SEMANTIC['canvas']} !important"},
    ".ag-row-hover": {"background-color": "rgba(232, 220, 196, 0.5) !important"},
    ".ag-row": {"border-color": "rgba(42, 107, 92, 0.18) !important"},
    ".ag-paging-panel": {
        "font-family": FONT_SANS,
        "font-size": "13px",
        "color": f"{SEMANTIC['primary']} !important",
        "border-top": f"1px solid {SEMANTIC['rule']} !important",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Loading - cached, and every failure mode returns a message instead of raising
# ─────────────────────────────────────────────────────────────────────────────

def _rel(path: Path) -> str:
    """Path relative to the project root, for operator-facing messages."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"`{_rel(path)}` not found."
    except json.JSONDecodeError as exc:
        return None, f"`{_rel(path)}` is not valid JSON (line {exc.lineno}): {exc.msg}"
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"Could not read `{_rel(path)}`: {exc}"


@st.cache_data(show_spinner=False)
def load_eval(model_key: str) -> tuple[dict | None, str | None]:
    """Aggregate metrics for one model. Returns (payload, error_message)."""
    path = EMBED_ROOT / model_key / "eval_results.json"
    payload, err = _read_json(path)
    if err:
        if "not found" in err:
            err += f" Run the evaluation for `{model_key}` to populate it."
        return None, err
    if not isinstance(payload, dict) or not payload.get("strategies"):
        return None, (f"`{_rel(path)}` has no `strategies` block - the evaluation "
                      f"may have been interrupted.")
    return payload, None


@st.cache_data(show_spinner=False)
def load_pca_cache(model_key: str) -> tuple[dict | None, str | None]:
    """
    Static projection for one model, written by scripts/precompute_pca.py.

    This is the only vector-space input the app has. It never opens a FAISS
    index, which is what keeps the memory footprint flat.
    """
    path = EMBED_ROOT / model_key / "pca_cache.json"
    payload, err = _read_json(path)
    if err:
        if "not found" in err:
            err += (" Build it once with "
                    "`python scripts/precompute_pca.py --models " + model_key + "`.")
        return None, err
    if not isinstance(payload, dict) or not payload.get("tracks"):
        return None, f"`{_rel(path)}` has no `tracks` block - rebuild the cache."
    return payload, None


@st.cache_data(show_spinner=False)
def load_vector_sample(model_key: str) -> tuple[dict | None, str | None]:
    """
    The 50-row inspector sample: text, MICE text and both raw vectors.

    Under a megabyte per model, and the same offline script that writes the
    projection writes this, from the same draw. The app still never opens an
    index.
    """
    path = EMBED_ROOT / model_key / "vector_sample_50.json"
    payload, err = _read_json(path)
    if err:
        if "not found" in err:
            err += (" Build it once with `python scripts/precompute_pca.py "
                    f"--models {model_key} --force`.")
        return None, err
    if not isinstance(payload, dict) or not payload.get("rows"):
        return None, f"`{_rel(path)}` has no `rows` block - rebuild the cache."
    if not (payload.get("vectors") or {}):
        return None, f"`{_rel(path)}` carries no vectors - rebuild the cache."
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
# ECharts option builders
#
# These are pure functions returning plain dicts. No Streamlit and no component
# import, so the chart configuration is unit-testable on its own.
# ─────────────────────────────────────────────────────────────────────────────

_AXIS_LABEL = {"color": SEMANTIC["ink_muted"], "fontSize": 13,
               "fontFamily": FONT_MONO}
_AXIS_LINE = {"lineStyle": {"color": SEMANTIC["grid"]}}
# Dashed split lines, as specified - kept faint so they stay behind the data.
_SPLIT_LINE = {"show": True, "lineStyle": {"color": SEMANTIC["grid"],
                                           "type": "dashed", "width": 1}}
_TOOLTIP = {
    "backgroundColor": SEMANTIC["panel_solid"],
    "borderColor": SEMANTIC["rule"],
    "borderWidth": 1,
    "padding": [6, 10],
    "textStyle": {"color": SEMANTIC["ink"], "fontSize": 13,
                  "fontFamily": FONT_SANS},
    "extraCssText": "box-shadow: 0 2px 8px rgba(18,63,54,.14); border-radius: 4px;",
}
_LEGEND = {
    "top": 0, "left": 0, "itemWidth": 12, "itemHeight": 12, "itemGap": 16,
    "icon": "roundRect",
    "textStyle": {"color": SEMANTIC["ink"], "fontSize": 13,
                  "fontFamily": FONT_SANS},
}
# Heatmap fold: 768 -> 32x24, 1024 -> 32x32. Both land on whole rows.
HEATMAP_COLS = 32


def _legend_right(**over) -> dict:
    """
    `_LEGEND` re-anchored to the top-right corner.

    Setting `right` while `_LEGEND`'s `left: 0` is still in place is what leaves
    the gap: ECharts honours the left anchor, draws the legend there, and still
    reserves the right-hand strip for it. Dropping `left` is the fix; tuning
    `right` only moves the empty band around.
    """
    kept = {k: v for k, v in _LEGEND.items() if k != "left"}
    return {**kept, "right": 0, "top": 0, **over}


_PEAK_LABEL = {
    "show": True, "position": "top", "distance": 4,
    "color": SEMANTIC["ink"], "fontSize": 12.5,
    "fontFamily": FONT_MONO, "fontWeight": 600,
}


def metric_table(strategies: list[dict], metric: str) -> pd.DataFrame:
    """Rows = cut-offs, columns = strategies. The shared source for chart + table."""
    by_strategy = {s.get("strategy"): s for s in strategies}
    ks = sorted({int(k) for s in strategies for k in (s.get(metric) or {})})
    if not ks:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            f"Strategy {code}": [
                (by_strategy[code].get(metric) or {}).get(str(k)) for k in ks
            ]
            for code in STRATEGIES if code in by_strategy
        },
        index=pd.Index([f"@{k}" for k in ks], name="Cut-off"),
    ).apply(pd.to_numeric, errors="coerce")


def grouped_bar_option(table: pd.DataFrame, axis_name: str) -> dict:
    """
    Grouped bars, one group per cut-off, one series per strategy.

    Only the peak bar in each group carries a printed value; the rest are read
    off the axis or the shared-axis tooltip.
    """
    if table.empty:
        return {}

    peaks = {
        row: table.loc[row].idxmax() if table.loc[row].notna().any() else None
        for row in table.index
    }

    series = []
    for code in STRATEGIES:
        column = f"Strategy {code}"
        if column not in table.columns:
            continue
        data = []
        for row in table.index:
            value = table.at[row, column]
            item = {"value": None if pd.isna(value) else round(float(value), 6)}
            if peaks[row] == column and pd.notna(value):
                # Literal formatter: ECharts would otherwise print the stored
                # precision, which is six decimals.
                item["label"] = {**_PEAK_LABEL, "formatter": f"{float(value):.4f}"}
            data.append(item)
        series.append({
            "name": f"{code} · {STRATEGY_NAME[code]}",
            "type": "bar",
            "barMaxWidth": 26,
            "itemStyle": {"borderRadius": [3, 3, 0, 0]},
            "emphasis": {"focus": "series"},
            "data": data,
        })

    ceiling = table.max(numeric_only=True).max()
    ceiling = float(ceiling) * 1.22 if pd.notna(ceiling) and ceiling > 0 else 1.0

    return {
        "color": SERIES_COLORS,
        # The measure is named by the title at top-left, not by a y-axis name:
        # ECharts draws an axis name in that same corner, on top of the legend.
        "title": {"text": axis_name, "left": 0, "top": 0,
                  "textStyle": {"color": SEMANTIC["primary"], "fontSize": 14.5,
                                "fontFamily": FONT_SANS, "fontWeight": 600}},
        "grid": {"top": 38, "left": 0, "right": 0, "bottom": 4,
                 "containLabel": True},
        "legend": _legend_right(),
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, **_TOOLTIP},
        "xAxis": {
            "type": "category",
            "data": list(table.index),
            "axisLabel": {**_AXIS_LABEL, "color": SEMANTIC["ink"], "fontSize": 13.5},
            "axisLine": _AXIS_LINE,
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "max": round(ceiling, 6),
            "axisLabel": _AXIS_LABEL,
            "axisLine": {"show": False},
            "splitLine": _SPLIT_LINE,
        },
        "series": series,
    }


def delta_rows(delta: dict) -> list[tuple[str, float]]:
    """Flatten the delta_mice block into ordered (metric, value) pairs."""
    rows: list[tuple[str, float]] = []
    if isinstance(delta.get("mrr"), (int, float)):
        rows.append(("MRR", float(delta["mrr"])))
    for metric, label in (("recall", "Recall"), ("ndcg", "nDCG")):
        block = delta.get(metric) or {}
        for k in sorted(block, key=int):
            if isinstance(block[k], (int, float)):
                rows.append((f"{label}@{k}", float(block[k])))
    return rows


def diverging_bar_option(rows: list[tuple[str, float]]) -> dict:
    """
    Diverging horizontal bars for ΔMICE (Strategy C minus Strategy A).

    Sign is carried by the side of the zero baseline, by the pole colour, and by
    a signed label on every bar - three channels, so it survives colour-vision
    deficiency and greyscale printing alike.
    """
    if not rows:
        return {}

    names = [name for name, _ in rows]
    values = [value for _, value in rows]

    data = []
    for value in values:
        positive = value >= 0
        data.append({
            "value": round(value, 6),
            "itemStyle": {
                "color": SEMANTIC["primary"] if positive else SEMANTIC["accent"],
                "borderRadius": [0, 3, 3, 0] if positive else [3, 0, 0, 3],
            },
            "label": {
                "show": True,
                "position": "right" if positive else "left",
                "distance": 5,
                "formatter": f"{value:+.4f}",
                "color": SEMANTIC["ink"],
                "fontSize": 12.5,
                "fontFamily": FONT_MONO,
                "fontWeight": 600,
            },
        })

    # Zero stays on the axis as the reference, but the range is not mirrored:
    # when every value falls on one side, a symmetric range wastes half the plot.
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    span = (high - low) or 1e-6
    low -= span * (0.28 if min(values) < 0 else 0.06)
    high += span * (0.28 if max(values) > 0 else 0.06)

    return {
        "grid": {"top": 8, "left": 4, "right": 10, "bottom": 22,
                 "containLabel": True},
        "tooltip": {"trigger": "item",
                    "valueFormatter": "__DELTA_FMT__", **_TOOLTIP},
        "xAxis": {
            "type": "value",
            "name": "Strategy C minus Strategy A",
            "nameLocation": "middle",
            "nameGap": 26,
            "nameTextStyle": {"color": SEMANTIC["ink_muted"], "fontSize": 12.5,
                              "fontFamily": FONT_SANS},
            # 4 dp, matching the bar labels - 6 would put "-0.042321" on a tick.
            "min": round(low, 4),
            "max": round(high, 4),
            "axisLabel": _AXIS_LABEL,
            "axisLine": {"show": False},
            "splitLine": _SPLIT_LINE,
        },
        "yAxis": {
            "type": "category",
            "data": names,
            "inverse": True,
            "axisLabel": {**_AXIS_LABEL, "color": SEMANTIC["ink"]},
            "axisLine": {"lineStyle": {"color": SEMANTIC["rule"]}},
            "axisTick": {"show": False},
        },
        "series": [{
            "type": "bar",
            "barMaxWidth": 18,
            "data": data,
            "markLine": {
                "silent": True, "symbol": "none",
                "lineStyle": {"color": SEMANTIC["rule"], "width": 1, "type": "solid"},
                "label": {"show": False},
                "data": [{"xAxis": 0}],
            },
        }],
    }


def scatter_option(track: dict, color_key: str, title: str,
                   max_classes: int = 6) -> dict:
    """
    Projection scatter from the precomputed cache.

    Classes past the largest `max_classes` fold into "Other" rather than being
    given generated hues, so the legend stays short and every colour is mixed
    from the three brand anchors.
    """
    xs, ys = track.get("x") or [], track.get("y") or []
    encoded = (track.get("labels") or {}).get(color_key) or {}
    cats, codes = encoded.get("cats") or [], encoded.get("codes") or []
    if not xs or len(xs) != len(ys) or len(codes) != len(xs):
        return {}

    counts = pd.Series(codes).value_counts()
    keep = list(counts.index[:max_classes])
    rank = {code: i for i, code in enumerate(keep)}
    names = [cats[c] if c < len(cats) else "unknown" for c in keep]

    woid, desc = track.get("woid") or [], track.get("desc") or []
    buckets: dict[int, list] = {i: [] for i in range(len(keep) + 1)}
    for i, code in enumerate(codes):
        slot = rank.get(code, len(keep))
        buckets[slot].append([
            xs[i], ys[i],
            woid[i] if i < len(woid) else "",
            desc[i] if i < len(desc) else "",
            cats[code] if code < len(cats) else "unknown",
        ])

    labels = names + ["Other"]
    colors = _brand_ramp(len(labels))
    series = []
    for slot, (name, color) in enumerate(zip(labels, colors)):
        points = buckets.get(slot) or []
        if not points:
            continue
        series.append({
            "name": f"{name[:24]} ({len(points)})",
            "type": "scatter",
            "symbolSize": 6,
            "large": True,
            "largeThreshold": 400,
            "itemStyle": {"color": color, "opacity": 0.8},
            "data": points,
        })

    explained = track.get("explained") or [0.0, 0.0]
    return {
        "title": {"text": title, "left": 0, "top": 0,
                  "textStyle": {"color": SEMANTIC["primary"], "fontSize": 14.5,
                                "fontFamily": FONT_SANS, "fontWeight": 600}},
        # The legend wraps to two rows at the class cap, so the grid starts
        # below both rows rather than under the first.
        "grid": {"top": 86, "left": 4, "right": 10, "bottom": 32,
                 "containLabel": True},
        "legend": {**_LEGEND, "top": 24, "itemGap": 12,
                   "textStyle": {**_LEGEND["textStyle"], "fontSize": 12}},
        "tooltip": {"trigger": "item", "formatter": "__SCATTER_FMT__", **_TOOLTIP},
        "xAxis": {
            "type": "value", "scale": True,
            # Both components are named here; a y-axis name would be drawn in
            # the same corner as the legend and collide with it.
            "name": (f"PC1 {explained[0]:.1%} · PC2 {explained[1]:.1%} "
                     f"of variance"),
            "nameLocation": "middle", "nameGap": 26,
            "nameTextStyle": {"color": SEMANTIC["ink_muted"], "fontSize": 12.5,
                              "fontFamily": FONT_SANS},
            "axisLabel": _AXIS_LABEL, "axisLine": {"show": False},
            "splitLine": _SPLIT_LINE,
        },
        "yAxis": {
            "type": "value", "scale": True,
            "axisLabel": _AXIS_LABEL, "axisLine": {"show": False},
            "splitLine": _SPLIT_LINE,
        },
        "series": series,
    }


def vector_trace_option(traces: dict[str, list[float]]) -> dict:
    """
    The two embeddings of one work order, plotted dimension by dimension.

    Both vectors are unit-normalised, so the two traces share a scale and can be
    read against each other directly: where MICE departs from the text trace is
    a dimension the injected metadata moved. The zoom band is the point of the
    figure - at 768 or 1024 dimensions the full sweep shows the envelope, and
    only a zoomed window shows individual components.
    """
    series, present = [], [t for t, _, _ in VECTOR_TRACKS if traces.get(t)]
    if not present:
        return {}
    length = max(len(traces[t]) for t in present)

    for track, label, colour in VECTOR_TRACKS:
        values = traces.get(track)
        if not values:
            continue
        series.append({
            "name": label,
            "type": "line",
            "data": [round(float(v), 5) for v in values],
            "showSymbol": False,
            "symbol": "none",
            "lineStyle": {"width": 1.1, "color": colour},
            "itemStyle": {"color": colour},
            "areaStyle": {"color": colour, "opacity": 0.10},
            # Down-sample for drawing only; the tooltip still reports the
            # stored value for whichever dimension the pointer is on.
            "sampling": "lttb",
            "emphasis": {"focus": "series"},
        })

    # Only the first series carries the baseline, or the mark line is drawn twice.
    series[0]["markLine"] = {
        "silent": True, "symbol": "none",
        "lineStyle": {"color": SEMANTIC["rule"], "width": 1, "type": "solid"},
        "label": {"show": False},
        "data": [{"yAxis": 0}],
    }

    return {
        "color": SERIES_COLORS,
        # left/right at 0 with containLabel: the axis labels still get
        # their room, so the plot spans the container and nothing else does.
        "grid": {"top": 36, "left": 0, "right": 12, "bottom": 58,
                 "containLabel": True},
        "legend": _legend_right(),
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"},
                    "formatter": "__VECTOR_FMT__", **_TOOLTIP},
        "xAxis": {
            "type": "category",
            "data": list(range(length)),
            "name": "dimension",
            "nameLocation": "middle",
            "nameGap": 28,
            "nameTextStyle": {"color": SEMANTIC["ink_muted"], "fontSize": 12.5,
                              "fontFamily": FONT_SANS},
            "boundaryGap": False,
            "axisLabel": _AXIS_LABEL,
            "axisLine": _AXIS_LINE,
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "axisLabel": {**_AXIS_LABEL, "formatter": "{value}"},
            "axisLine": {"show": False},
            "splitLine": _SPLIT_LINE,
        },
        "dataZoom": [
            {"type": "inside", "start": 0, "end": 100},
            {"type": "slider", "start": 0, "end": 100, "height": 18, "bottom": 6,
             "borderColor": SEMANTIC["rule"],
             "backgroundColor": SEMANTIC["panel_solid"],
             "fillerColor": "rgba(42, 107, 92, 0.18)",
             "handleStyle": {"color": SEMANTIC["primary"]},
             "moveHandleStyle": {"color": SEMANTIC["secondary"]},
             "textStyle": {"color": SEMANTIC["ink_muted"], "fontSize": 11,
                           "fontFamily": FONT_MONO}},
        ],
        "series": series,
    }


def robust_span(values: list[float], pct: float = 0.99) -> float:
    """
    The |component| at `pct` of the distribution - the heatmap's colour limit.

    An embedding carries a handful of components an order of magnitude past the
    rest. Scaling to the true maximum spends the whole ramp on those few and
    paints the other thousand the same cream, so the field shows nothing. The
    tail clips to the pole colour; the tooltip still reports its real value.
    """
    magnitudes = sorted(abs(float(v)) for v in values)
    if not magnitudes:
        return 0.0
    # Nearest-rank: the value at or below which `pct` of the components fall.
    # Truncating instead would land one past it and hand back the outlier.
    return magnitudes[max(0, math.ceil(pct * len(magnitudes)) - 1)]


def heatmap_option(values: list[float], title: str, span: float) -> dict:
    """
    One embedding folded into a 2-D field, `HEATMAP_COLS` components per row.

    The line chart answers *where* the two vectors differ; this answers *how the
    energy is spread* - a trace at 1,024 points reads as an envelope, a field
    reads as structure. Row-major from dimension 0 at the top-left.

    `span` is the symmetric limit, passed in rather than derived per panel: two
    diverging scales side by side, each normalised to its own extreme, look like
    a comparison and are not one. Symmetric because the palette diverges about
    zero - an off-centre midpoint would paint sign onto the wrong cells.
    """
    if not values or span <= 0:
        return {}

    cols = HEATMAP_COLS
    rows = -(-len(values) // cols)   # ceil; a short last row simply runs out
    data = [[i % cols, i // cols, round(float(v), 5)] for i, v in enumerate(values)]

    return {
        "title": {"text": title, "left": 0, "top": 0,
                  "textStyle": {"color": SEMANTIC["primary"], "fontSize": 14.5,
                                "fontFamily": FONT_SANS, "fontWeight": 600}},
        # The visual map lies flat under the plot: upright in the default right
        # gutter it would cost the field a fifth of its width.
        "grid": {"top": 34, "left": 0, "right": 0, "bottom": 46,
                 "containLabel": True},
        "tooltip": {"trigger": "item", "formatter": "__HEATMAP_FMT__", **_TOOLTIP},
        "xAxis": {
            # Unnamed: the y-axis already states that a row is a block of
            # dimensions, and a name here lands on the visual map's labels.
            "type": "category", "data": list(range(cols)),
            "splitArea": {"show": False},
            "axisLabel": {**_AXIS_LABEL, "interval": 7},
            "axisLine": _AXIS_LINE, "axisTick": {"show": False},
        },
        "yAxis": {
            # Labelled by the dimension each row starts at, so a cell's identity
            # is readable off the axes without the tooltip.
            "type": "category", "data": [r * cols for r in range(rows)],
            "inverse": True,
            "splitArea": {"show": False},
            "axisLabel": {**_AXIS_LABEL, "interval": 3},
            "axisLine": _AXIS_LINE, "axisTick": {"show": False},
        },
        "visualMap": {
            "type": "continuous",
            "min": -span, "max": span,
            "calculable": True,
            "orient": "horizontal", "left": "center", "bottom": 0,
            "itemWidth": 12, "itemHeight": 150,
            "precision": 3,
            "inRange": {"color": [SEMANTIC["primary"], SEMANTIC["canvas"],
                                  SEMANTIC["accent"]]},
            "outOfRange": {"color": SEMANTIC["ink_muted"]},
            "textStyle": {"color": SEMANTIC["ink_muted"], "fontSize": 11.5,
                          "fontFamily": FONT_MONO},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "progressive": 0,
            "itemStyle": {"borderWidth": 0},
            "emphasis": {"itemStyle": {"borderColor": SEMANTIC["ink"],
                                       "borderWidth": 1}},
        }],
    }


def vector_stats(a: list[float], b: list[float]) -> tuple[float, float] | None:
    """Cosine similarity and mean absolute component shift between two vectors."""
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if not norm:
        return None
    shift = sum(abs(x - y) for x, y in zip(a, b)) / len(a)
    return dot / norm, shift


def _brand_ramp(n: int) -> list[str]:
    """
    n steps interpolated across primary -> secondary -> accent.

    The scatter colours by a metadata dimension with more classes than the three
    brand anchors, so intermediate steps are mixed from those anchors rather
    than introducing any hue from outside the palette.
    """
    anchors = [SEMANTIC["primary"], SEMANTIC["secondary"], SEMANTIC["accent"]]
    rgb = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in anchors]
    if n <= 1:
        return [anchors[0]]
    out = []
    for step in range(n):
        pos = step / (n - 1) * (len(rgb) - 1)
        lo = min(int(pos), len(rgb) - 2)
        frac = pos - lo
        out.append("#%02X%02X%02X" % tuple(
            round(rgb[lo][c] + (rgb[lo + 1][c] - rgb[lo][c]) * frac) for c in range(3)
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

# Tooltip bodies, held here rather than inside the option builders so those
# stay plain JSON-serialisable dicts that a test can compare. A builder plants
# the sentinel; `_echart` swaps in the function on its way to the component.
_FORMATTERS = {
    "__DELTA_FMT__":
        "function (v) { return (v >= 0 ? '+' : '') + v.toFixed(4); }",
    "__SCATTER_FMT__":
        "function (p) { var d = p.value; return '<b>' + (d[2] || '') + '</b><br/>'"
        " + d[4] + '<br/><i>' + (d[3] || '') + '…</i>'; }",
    "__VECTOR_FMT__":
        "function (ps) { var s = 'dimension <b>' + ps[0].axisValue + '</b>';"
        " ps.forEach(function (p) { s += '<br/>' + p.marker + p.seriesName"
        " + '  <b>' + Number(p.value).toFixed(5) + '</b>'; });"
        " if (ps.length === 2) { s += '<br/>Δ  <b>' +"
        " (ps[1].value - ps[0].value).toFixed(5) + '</b>'; } return s; }",
    # Row-major, so the true dimension is the row offset plus the column.
    "__HEATMAP_FMT__":
        "function (p) { var d = p.data;"
        " return 'dimension <b>' + (d[1] * " + str(HEATMAP_COLS) + " + d[0])"
        " + '</b><br/><b>' + Number(d[2]).toFixed(5) + '</b>'; }",
}


def _echart(option: dict, height: str, key: str) -> None:
    """
    Render one ECharts figure.

    The component is imported here, not at module scope: streamlit-echarts
    resolves its bundled assets during server startup, so importing it outside a
    running server raises. A local import also keeps the module importable for
    the self-check in test_app.py.
    """
    if not option:
        st.info("Nothing to plot for this section.")
        return
    from streamlit_echarts import JsCode, st_echarts

    option = json.loads(json.dumps(option))  # detach from the cached source
    tooltip = option.get("tooltip") or {}
    for field in ("formatter", "valueFormatter"):
        source = _FORMATTERS.get(tooltip.get(field))
        if source:
            tooltip[field] = JsCode(source).js_code

    st_echarts(options=option, height=height, key=key, theme=None)


def _section(label: str, note: str = "") -> None:
    st.markdown(f'<div class="rm-sec">{label}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="rm-note">{note}</div>', unsafe_allow_html=True)


def _fmt(value, spec: str = ".4f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "—"


def _leaders(by_strategy: dict[str, dict]) -> dict[str, str | None]:
    """
    Leading strategy per headline metric.

    Latency is deliberately absent: the p95 spread across strategies is a few
    milliseconds on ~320 ms, well inside one standard deviation of a single
    strategy's own latency, so a badge there would claim a difference the
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


def _metric_card(record: dict, code: str, leaders: dict[str, str | None]) -> None:
    latency = (record.get("latency") or {}).get("p95_ms")
    stats = [
        ("MRR", _fmt(record.get("mrr")), "mrr"),
        ("Recall@10", _fmt((record.get("recall") or {}).get("10")), "recall"),
        ("nDCG@10", _fmt((record.get("ndcg") or {}).get("10")), "ndcg"),
        ("Latency p95", _fmt(latency, ".0f") + (" ms" if latency is not None else ""),
         "latency"),
    ]
    rows = "".join(
        f'<div class="rm-metric"><span class="k">{key}</span>'
        f'<span class="v{" lead" if leaders.get(tag) == code else ""}">{value}</span></div>'
        for key, value, tag in stats
    )
    st.markdown(
        f'<div class="rm-card">'
        f'<div class="hd"><i style="background:{SERIES[code]}"></i>'
        f'Strategy {code} <span>{STRATEGY_NAME[code]}</span></div>'
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
        ("Dimensions", payload.get("embedding_dim", "—")),
        ("Queries", payload.get("n_queries", "—")),
        ("Seed", payload.get("seed", "—")),
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
    _section("Vector space · precomputed projection")

    cache, err = load_pca_cache(model_key)
    if err:
        st.warning(f"**Projection unavailable.** {err}", icon="⚠")
        return

    choice = st.radio("Colour points by", list(PCA_COLOR_BY),
                      horizontal=True, key=f"{model_key}-pca-colour")
    color_key = PCA_COLOR_BY[choice]

    for column, (track_key, track_label) in zip(
        st.columns(2, gap="medium"), PCA_TRACKS
    ):
        with column:
            track = (cache.get("tracks") or {}).get(track_key)
            if not track:
                st.warning(f"**{track_label}** is not in the cache for this model.",
                           icon="⚠")
                continue
            option = scatter_option(
                track, color_key, f"{track_label} · {track.get('dim', '?')}-d → 2-d")
            _echart(option, "380px", f"{model_key}-pca-{track_key}")
            st.markdown(
                f'<div class="rm-note">'
                f'{track.get("n_sampled", 0):,} of {track.get("n_total", 0):,} vectors · '
                f'{sum(track.get("explained") or [0, 0]):.1%} of variance in two '
                f'components.</div>',
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

    _section("Strategies at a glance")
    leaders = _leaders(by_strategy)
    for column, code in zip(st.columns(3, gap="small"), STRATEGIES):
        with column:
            if code in by_strategy:
                _metric_card(by_strategy[code], code, leaders)
            else:
                st.markdown(
                    f'<div class="rm-card"><div class="hd">Strategy {code}</div>'
                    f'<div class="note">Not present in this run.</div></div>',
                    unsafe_allow_html=True,
                )

    _section("Retrieval quality by cut-off")
    recall_table = metric_table(strategies, "recall")
    ndcg_table = metric_table(strategies, "ndcg")
    left, right = st.columns(2, gap="medium")
    with left:
        _echart(grouped_bar_option(recall_table, "Recall"), "290px",
                f"{model_key}-recall")
        _table_view("Recall — table view", recall_table)
    with right:
        _echart(grouped_bar_option(ndcg_table, "nDCG"), "290px", f"{model_key}-ndcg")
        _table_view("nDCG — table view", ndcg_table)

    _section("ΔMICE — the isolated effect of metadata injection")
    rows = delta_rows(payload.get("delta_mice") or {})
    if not rows:
        st.info("This run has no `delta_mice` block — it needs both Strategy A "
                "and Strategy C.")
    else:
        _echart(diverging_bar_option(rows), f"{max(200, 30 * len(rows) + 60)}px",
                f"{model_key}-delta")
        _table_view("ΔMICE — table view",
                    pd.DataFrame(rows, columns=["Metric", "ΔMICE (C − A)"])
                    .set_index("Metric"), "{:+.4f}")

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

    _section("Filters")
    c1, c2, c3, c4 = st.columns([1.1, 0.9, 1.3, 1.3], gap="small")
    models = c1.multiselect("Model", list(MODELS), format_func=MODELS.get)
    strategies = c2.multiselect("Strategy", list(STRATEGIES),
                                format_func=lambda c: f"{c} · {STRATEGY_NAME[c]}")
    outcomes = c3.multiselect("Failure mode / rank of first hit", _options(df, "outcome"))
    winners = c4.multiselect("Strategy that won the query", _options(df, "winner"))

    c5, c6, c7 = st.columns([1.2, 1.4, 1.6], gap="small")
    equipment = c5.multiselect("Equipment", _options(df, "equipment"))
    facilities = c6.multiselect("Facility type", _options(df, "facility_type"))
    search = c7.text_input("Quick filter", placeholder="e.g. roof leak")

    view = df
    for column, chosen in (
        ("model", models), ("strategy", strategies), ("outcome", outcomes),
        ("winner", winners), ("equipment", equipment), ("facility_type", facilities),
    ):
        if chosen:
            view = view[view[column].isin(chosen)]

    columns = [
        "model", "strategy", "winner", "outcome", "query_text", "equipment",
        "facility_type", "n_relevant", "recall", "mrr", "ndcg",
        "first_hit_rank", "metadata_gain", "delta_mice",
    ]
    view = view[[c for c in columns if c in view.columns]]

    _section("Per-query results",
             f"{len(view):,} of {len(df):,} rows · "
             f"{df['query_text'].nunique():,} distinct queries.")
    _render_grid(view, search.strip())


def grid_options(view: pd.DataFrame, quick_filter: str,
                 single_select: bool = False) -> dict:
    """Build the AgGrid option dict. Pure, so the configuration is testable."""
    from st_aggrid import GridOptionsBuilder

    builder = GridOptionsBuilder.from_dataframe(view)
    builder.configure_default_column(
        sortable=True, filterable=True, resizable=True,
        minWidth=92, flex=1, suppressMovable=True,
    )
    builder.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=50)
    if single_select:
        builder.configure_selection("single")
    for column, decimals in (("recall", 3), ("mrr", 3), ("ndcg", 3),
                             ("metadata_gain", 3), ("delta_mice", 3)):
        if column in view.columns:
            builder.configure_column(column, type=["numericColumn"],
                                     valueFormatter=(
                                         f"value == null ? '' : "
                                         f"value.toFixed({decimals})"))
    if "query_text" in view.columns:
        builder.configure_column("query_text", header_name="query", minWidth=260, flex=3)
    if "first_hit_rank" in view.columns:
        builder.configure_column("first_hit_rank", header_name="1st hit", minWidth=84)
    if "row" in view.columns:
        builder.configure_column("row", header_name="#", minWidth=60, maxWidth=76,
                                 flex=0, pinned="left")
    for column, header in (("raw_text", "raw text"), ("mice_text", "MICE text")):
        if column in view.columns:
            builder.configure_column(column, header_name=header, flex=2)

    builder.configure_grid_options(
        quickFilterText=quick_filter,
        cacheQuickFilter=True,
        suppressFieldDotNotation=True,
        # Row metrics go through grid options, not CSS: AG Grid computes row
        # offsets in JS for virtualisation, and a CSS height override
        # desynchronises them from the rendered rows. These track the 14px cell
        # font set in GRID_CSS - shrink one and the other has to follow.
        rowHeight=36,
        headerHeight=40,
        # Columns size themselves to the available width on first render.
        autoSizeStrategy={"type": "fitGridWidth"},
    )
    return builder.build()


def _render_grid(view: pd.DataFrame, quick_filter: str, key: str = "query-grid",
                 height: int = 520, single_select: bool = False):
    from st_aggrid import AgGrid

    return AgGrid(
        view,
        gridOptions=grid_options(view, quick_filter, single_select),
        theme="alpine",
        custom_css=GRID_CSS,
        height=height,
        allow_unsafe_jscode=True,
        # A selecting grid has to round-trip; a read-only one must not, or every
        # sort and scroll costs a rerun.
        update_mode="SELECTION_CHANGED" if single_select else "NO_UPDATE",
        enable_enterprise_modules=False,
        key=key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vector Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _pane(label: str, note: str, colour: str, body: str, mark: bool) -> str:
    """One text pane. `body` is escaped here - it is raw work-order text."""
    body = html.escape(body or "—")
    if mark:
        body = MICE_LABELS.sub(r"<em>\1:</em>", body)
    return (f'<div class="rm-pane"><div class="hd">'
            f'<i style="background:{colour}"></i>{label} <span>{note}</span></div>'
            f'<p class="body">{body}</p></div>')


def _selected_row(response, total: int) -> int:
    """
    Zero-based index of the selected row, defaulting to the first.

    The grid can be sorted or filtered, so position on screen is not position in
    the sample; the `row` column carries the identity across.
    """
    selected = getattr(response, "selected_rows", None)
    if selected is None or getattr(selected, "empty", True):
        return 0
    try:
        index = int(selected.iloc[0]["row"]) - 1
    except (KeyError, IndexError, TypeError, ValueError):
        return 0
    return index if 0 <= index < total else 0


def render_inspector() -> None:
    _section("Vector Inspector · one work order, both embeddings")

    choice = st.radio("Embedding model", list(MODELS), format_func=MODELS.get,
                      horizontal=True, key="inspector-model")
    payload, err = load_vector_sample(choice)
    if err:
        st.warning(f"**Vector sample unavailable.** {err}", icon="⚠")
        return

    rows = payload["rows"]
    vectors = payload.get("vectors") or {}
    missing = [label for key, label, _ in VECTOR_TRACKS if key not in vectors]
    if missing:
        st.warning(f"**{', '.join(missing)}** is absent from this sample; only the "
                   f"other track is plotted.", icon="⚠")

    frame = pd.DataFrame({
        "row": range(1, len(rows) + 1),
        "WOID": [r.get("woid", "") for r in rows],
        "raw_text": [r.get("text", "") for r in rows],
        "mice_text": [r.get("mice", "") for r in rows],
    })
    response = _render_grid(frame, "", key=f"inspector-grid-{choice}",
                            height=300, single_select=True)
    index = _selected_row(response, len(rows))
    record = rows[index]

    dim = payload.get("dim", "?")
    st.markdown(
        f'<div class="rm-meta">'
        f'<div><span class="k">Row</span><span class="v">{index + 1} of '
        f'{len(rows)}</span></div>'
        f'<div><span class="k">WOID</span><span class="v">'
        f'{html.escape(str(record.get("woid") or "—"))}</span></div>'
        f'<div><span class="k">Dimensions</span><span class="v">{dim}</span></div>'
        f'<div><span class="k">Model</span><span class="v">{MODELS[choice]}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    _section("Embedded text · raw against MICE-injected")
    for column, (track, label, colour) in zip(st.columns(2, gap="medium"),
                                              VECTOR_TRACKS):
        with column:
            body = record.get("text" if track == "text" else "mice", "")
            note = f"{len(body):,} chars"
            column.markdown(_pane(label, note, colour, body, track == "mice"),
                            unsafe_allow_html=True)

    traces = {track: (vectors.get(track) or [None] * len(rows))[index]
              for track, _, _ in VECTOR_TRACKS if track in vectors}
    traces = {track: values for track, values in traces.items() if values}

    _section("Vector signature")
    stats = vector_stats(traces.get("text") or [], traces.get("mice") or [])
    if stats:
        cosine, shift = stats
        st.markdown(
            f'<div class="rm-meta">'
            f'<div><span class="k">Cosine similarity</span>'
            f'<span class="v">{cosine:.4f}</span></div>'
            f'<div><span class="k">Mean |Δ| per component</span>'
            f'<span class="v">{shift:.5f}</span></div>'
            f'<div><span class="k">Angle between</span>'
            f'<span class="v">{math.degrees(math.acos(max(-1.0, min(1.0, cosine)))):.1f}°'
            f'</span></div></div>',
            unsafe_allow_html=True,
        )
    _echart(vector_trace_option(traces), "360px", f"inspector-trace-{choice}")

    _section("Component field")
    # One limit for both panels, taken across both vectors pooled: per-panel
    # scaling would put the same colour on two different magnitudes.
    span = robust_span([v for values in traces.values() for v in values])
    for column, (track, label, _) in zip(st.columns(2, gap="medium"), VECTOR_TRACKS):
        with column:
            _echart(heatmap_option(traces.get(track) or [], label, span),
                    "360px", f"inspector-heat-{choice}-{track}")



# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

def render_page() -> None:
    st.markdown(
        '<div class="rm-masthead">'
        '<div class="rm-title">MICE Retrieval Evaluation</div>'
        '<div class="rm-sub">Metadata-injected chunk embeddings for '
        'facilities-management RAG · Strategy <b>A</b> baseline, <b>B</b> '
        'post-filter, <b>C</b> MICE · three embedding backbones, one query set, '
        'one seed.</div></div>',
        unsafe_allow_html=True,
    )
    tabs = st.tabs([*MODELS.values(), "Vector Inspector", "Query Explorer"])
    for tab, model_key in zip(tabs, MODELS):
        with tab:
            render_model(model_key)
    with tabs[-2]:
        render_inspector()
    with tabs[-1]:
        render_explorer()


# Render only under a live Streamlit server. Both st_echarts and st_aggrid
# resolve their bundled assets during server startup and raise if imported
# without one, so importing this module outside `streamlit run` - as the
# self-check does - must not walk the render path.
if st.runtime.exists():
    render_page()
