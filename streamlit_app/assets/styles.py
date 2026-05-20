"""
assets/styles.py - Global CSS injection for the FM-RAG platform.

Design direction: Industrial-precision.
Monospace accents, slate/stone palette, sharp borders, no gradients.
Feels like a well-engineered internal tool.
"""

import streamlit as st


CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Root palette ───────────────────────────────────────── */
:root {
    --bg:        #FFF2F2;
    --surface:   #FFFFFF;
    --border:    #A9B5DF;
    --accent:    #2D336B;
    --accent2:   #7886C7;
    --accent3:   #7886C7;
    --text:      #2D336B;
    --muted:     #7886C7;
    --mono:      'IBM Plex Mono', monospace;
}

/* ── Main container ─────────────────────────────────────── */
.main .block-container {
    padding: 2rem 2.5rem 4rem;
    max-width: 1400px;
}

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--border);
}

/* ── Metric cards ───────────────────────────────────────── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
    position: relative;
}
.metric-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
}
.metric-card .delta {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--accent2);
    margin-top: 0.2rem;
}

/* ── Section headers ────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2rem 0 1.2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
}
.section-header h2 {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--text);
    margin: 0;
}
.section-header .badge {
    font-family: var(--mono);
    font-size: 0.65rem;
    background: var(--accent);
    color: #FFF2F2;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.06em;
}

/* ── Pipeline step cards ────────────────────────────────── */
.pipeline-step {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.5rem;
}
.pipeline-step .step-num {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.pipeline-step .step-title {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text);
    margin: 0.15rem 0;
}
.pipeline-step .step-desc {
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.5;
}

/* ── Strategy cards ─────────────────────────────────────── */
.strategy-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem;
    height: 100%;
}
.strategy-card .strat-label {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 0.4rem;
}
.strategy-card .strat-title {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text);
    margin-bottom: 0.75rem;
}
.strategy-card ul {
    margin: 0;
    padding-left: 1.1rem;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.8;
}
.strategy-card .pro  { color: var(--accent2); font-size: 0.75rem; }
.strategy-card .con  { color: #7886C7;        font-size: 0.75rem; }

/* ── Code / repr blocks ─────────────────────────────────── */
.repr-block {
    background: #FFF2F2;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.85rem 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #2D336B;
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
}

/* ── Result rows ────────────────────────────────────────── */
.result-row {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.4rem;
}
.result-row .rank-badge {
    font-family: var(--mono);
    font-size: 0.7rem;
    background: var(--border);
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    color: var(--muted);
}
.result-row .score-badge {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 600;
}
.result-row .woid {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent3);
}
.result-row .desc {
    font-size: 0.82rem;
    color: var(--text);
    margin-top: 0.25rem;
    line-height: 1.4;
}
.result-row .meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── Info pill ──────────────────────────────────────────── */
.info-pill {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.68rem;
    background: var(--border);
    color: var(--muted);
    padding: 0.15rem 0.55rem;
    border-radius: 20px;
    margin-right: 0.3rem;
}

/* ── Latency badge ──────────────────────────────────────── */
.latency-badge {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--accent2);
}

/* ── Page title ─────────────────────────────────────────── */
.page-title {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 2rem;
}

/* ── Streamlit overrides ────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.9rem 1.1rem;
}
div[data-testid="stMetric"] label {
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Tabs ───────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
}

/* ── Scrollbar ──────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""


def inject_global_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def section_header(title: str, badge: str = "") -> None:
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-header"><h2>{title}</h2>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="delta">↑ {delta}</div>' if delta else ""
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>"""


def repr_block(text: str) -> None:
    st.markdown(f'<div class="repr-block">{text}</div>', unsafe_allow_html=True)


def result_row(rank: int, woid: str, score: float, desc: str,
               btype: str, equipment: str) -> None:
    st.markdown(f"""
    <div class="result-row">
        <span class="rank-badge">#{rank}</span>&nbsp;
        <span class="woid">{woid}</span>&nbsp;&nbsp;
        <span class="score-badge">{score:.4f}</span>
        <div class="desc">{desc[:180]}{'…' if len(desc) > 180 else ''}</div>
        <div class="meta">
            <span class="info-pill">{btype}</span>
            <span class="info-pill">{equipment}</span>
        </div>
    </div>""", unsafe_allow_html=True)


def strategy_card(label: str, title: str, bullets: list,
                  pros: list, cons: list, color: str = "#2D336B") -> None:
    bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
    pros_html = "".join(
        f'<div class="pro">✓ {p}</div>' for p in pros)
    cons_html = "".join(
        f'<div class="con">✗ {c}</div>' for c in cons)
    st.markdown(f"""
    <div class="strategy-card">
        <div class="strat-label" style="color:{color}">{label}</div>
        <div class="strat-title">{title}</div>
        <ul>{bullets_html}</ul>
        <br/>
        {pros_html}
        {cons_html}
    </div>""", unsafe_allow_html=True)
