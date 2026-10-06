"""
test_app.py - self-check for the MICE evaluation dashboard.

Run with:
    python test_app.py            (from streamlit_app/)

app.py renders only under a live Streamlit server, because st_echarts and
st_aggrid resolve their bundled assets at server startup. So importing it here
is a smoke test of module scope only - the tokens, the stylesheet and the grid
theme. Everything below it is reachable because the real work is done by pure
functions: the loaders return (payload, error) and the chart and grid
configurations are plain dicts, both checkable without a running component.
"""

import json
import math
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import app  # noqa: E402  - importing renders the dashboard headlessly

BRAND_SERIES = ["#123F36", "#2A6B5C", "#C49A45"]


def test_app_opens_no_vector_index():
    """Requirement 1: the app must not reach for FAISS, sklearn or Plotly."""
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    for banned in ("faiss", "faiss_store", "sklearn", "plotly", "pickle"):
        assert banned not in source, f"app.py still references {banned!r}"


def test_derive_outcome_and_winner():
    """B wins q1 outright; q2 is a tie; q3 is missed by every strategy."""
    frame = pd.DataFrame({
        "model": ["m"] * 9,
        "query_text": ["q1"] * 3 + ["q2"] * 3 + ["q3"] * 3,
        "strategy": list("ABC") * 3,
        "recall": [0.0, 0.5, 0.2, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0],
        "first_hit_rank": [None, 1, 7, 3, 11, 5, None, None, None],
    })
    out = app._derive(frame)

    assert out["outcome"].tolist() == [
        "Miss - nothing relevant @10", "Hit @1", "Hit @6-10",
        "Hit @2-5", "Hit beyond @10", "Hit @2-5",
        "Miss - nothing relevant @10", "Miss - nothing relevant @10",
        "Miss - nothing relevant @10",
    ], out["outcome"].tolist()

    assert out.loc[out.query_text == "q1", "winner"].unique().tolist() == ["B"]
    assert out.loc[out.query_text == "q2", "winner"].unique().tolist() == ["Tie"]
    assert out.loc[out.query_text == "q3", "winner"].unique().tolist() == ["None - all missed"]


def test_missing_files_warn_instead_of_raising():
    """Absent inputs must degrade to a message, never an exception."""
    for loader in (app.load_eval, app.load_queries):
        payload, err = loader("model_that_does_not_exist")
        assert payload is None and err and "not found" in err, (loader.__name__, err)

    for loader in (app.load_pca_cache, app.load_vector_sample):
        payload, err = loader("model_that_does_not_exist")
        assert payload is None and err and "precompute_pca.py" in err, (
            loader.__name__, err)


def test_body_type_meets_the_requested_floor():
    """Subtitle, section rule and note all sit at 1.05rem or above."""
    floor = 1.05
    for selector in (".rm-sub", ".rm-sec", ".rm-note"):
        rule = re.search(re.escape(selector) + r"\s*\{(.*?)\}",
                         app.STYLESHEET, re.S)
        assert rule, f"{selector} has no rule"
        size = re.search(r"font-size:\s*([^;]+);", rule.group(1))
        assert size, f"{selector} sets no font-size"
        value = size.group(1).strip()
        # Resolve one level of var(): these three read from the type tokens.
        token = re.fullmatch(r"var\((--[\w-]+)\)", value)
        if token:
            value = app.TOKENS[token.group(1)]
        assert value.endswith("rem"), (selector, value)
        assert float(value[:-3]) >= floor, (selector, value)

    # The numerals have to scale with the prose, not stay at the old size.
    assert float(app.TOKENS["--metric-size"][:-3]) >= 1.7
    assert float(app.GRID_CSS[".ag-cell"]["font-size"][:-2]) >= 14


def test_palette_is_exactly_the_brand_palette():
    """The four brand values are applied verbatim - no substituted hues."""
    assert app.PRIMITIVE == {
        "forest_900": "#123F36",
        "forest_600": "#2A6B5C",
        "gold_500": "#C49A45",
        "cream_200": "#E8DCC4",
        "canvas_50": "#FAF7F0",
    }
    assert app.SERIES_COLORS == BRAND_SERIES
    assert app.SERIES == {"A": "#123F36", "B": "#2A6B5C", "C": "#C49A45"}


def test_brand_ramp_stays_on_the_brand_anchors():
    """The scatter ramp interpolates between brand colours; it never leaves them."""
    assert app._brand_ramp(1) == ["#123F36"]
    ramp = app._brand_ramp(7)
    assert len(ramp) == 7
    assert ramp[0] == "#123F36" and ramp[-1] == "#C49A45"
    assert "#2A6B5C" in ramp, ramp  # the mid anchor is hit exactly
    assert all(len(c) == 7 and c.startswith("#") for c in ramp)


def test_grouped_bar_option_from_real_payload():
    """Palette injected verbatim, dashed gridlines, one peak label per group."""
    built = 0
    for key in app.MODELS:
        payload, err = app.load_eval(key)
        if err:
            continue
        built += 1
        for metric, axis in (("recall", "Recall"), ("ndcg", "nDCG")):
            table = app.metric_table(payload["strategies"], metric)
            assert table.shape == (len(payload["top_k"]), len(app.STRATEGIES))

            option = app.grouped_bar_option(table, axis)
            assert option["color"] == BRAND_SERIES, option["color"]
            assert len(option["series"]) == len(app.STRATEGIES)
            assert option["yAxis"]["splitLine"]["lineStyle"]["type"] == "dashed"

            # Direct labels on peak values only: exactly one per cut-off.
            labelled = sum(1 for s in option["series"]
                           for item in s["data"] if "label" in item)
            assert labelled == len(table.index), (key, metric, labelled)

            json.dumps(option)  # st_echarts ships this over the wire as JSON

    assert built, "no eval_results.json could be loaded - nothing was exercised"


def test_diverging_option_encodes_sign_three_ways():
    """Colour, side of the baseline, and a signed label all carry the sign."""
    rows = [("MRR", -0.0323), ("Recall@10", 0.0041)]
    option = app.diverging_bar_option(rows)
    items = option["series"][0]["data"]

    assert [i["itemStyle"]["color"] for i in items] == ["#C49A45", "#123F36"]
    assert [i["label"]["position"] for i in items] == ["left", "right"]
    assert [i["label"]["formatter"] for i in items] == ["-0.0323", "+0.0041"]
    assert option["xAxis"]["min"] < 0 < option["xAxis"]["max"]
    json.dumps(option)

    # All-negative input must not mirror the range and waste half the plot.
    option = app.diverging_bar_option([("MRR", -0.03), ("nDCG@10", -0.01)])
    assert option["xAxis"]["max"] < abs(option["xAxis"]["min"]) / 2

    assert app.diverging_bar_option([]) == {}


def test_scatter_option_folds_the_tail_into_other():
    """Classes past the cap collapse to 'Other' rather than gaining new hues."""
    n = 40
    track = {
        "dim": 768, "n_total": 999, "n_sampled": n, "explained": [0.21, 0.09],
        "x": [float(i) for i in range(n)], "y": [float(i % 5) for i in range(n)],
        "labels": {"equipment": {
            "cats": [f"cat{i}" for i in range(10)],
            # cat0 is the largest class, then cat1, ... so the tail is the cap.
            "codes": [min(i % 10, 9) for i in range(n)],
        }},
        "woid": [f"W{i:03d}" for i in range(n)],
        "desc": ["desc"] * n,
    }
    option = app.scatter_option(track, "equipment", "t", max_classes=6)
    assert len(option["series"]) == 7  # 6 named classes + Other
    assert option["series"][-1]["name"].startswith("Other")
    assert sum(len(s["data"]) for s in option["series"]) == n
    colours = [s["itemStyle"]["color"] for s in option["series"]]
    assert colours[0] == "#123F36" and colours[-1] == "#C49A45"
    json.dumps(option)

    assert app.scatter_option({}, "equipment", "t") == {}


def test_grid_options_paginate_and_quick_filter():
    """AgGrid: 50 rows a page, sorting and filtering on, quick filter wired."""
    frame = pd.DataFrame({
        "model": ["bge_base"] * 3,
        "query_text": ["roof leak", "chiller", "lift"],
        "recall": [0.1, 0.2, 0.0],
    })
    options = app.grid_options(frame, "roof")

    assert options["paginationPageSize"] == 50
    assert options["pagination"] is True
    assert options["quickFilterText"] == "roof"
    defaults = options["defaultColDef"]
    assert defaults["sortable"] is True and defaults["resizable"] is True
    assert {c["field"] for c in options["columnDefs"]} == set(frame.columns)

    # The read-only explorer must not turn selection on by accident.
    assert "rowSelection" not in options


def test_inspector_grid_selects_one_row():
    """The inspector grid is single-select and pins its identity column."""
    frame = pd.DataFrame({"row": [1, 2], "WOID": ["W1", "W2"],
                          "raw_text": ["a", "b"], "mice_text": ["c", "d"]})
    options = app.grid_options(frame, "", single_select=True)

    assert options["rowSelection"] == "single"
    identity = next(c for c in options["columnDefs"] if c["field"] == "row")
    assert identity["pinned"] == "left" and identity["headerName"] == "#"


def test_vector_stats_measures_the_shift():
    """Cosine and mean shift, with the degenerate inputs returning None."""
    cosine, shift = app.vector_stats([0.6, 0.8], [0.6, 0.8])
    assert math.isclose(cosine, 1.0, abs_tol=1e-12) and shift == 0.0

    cosine, shift = app.vector_stats([1.0, 0.0], [0.0, 1.0])
    assert math.isclose(cosine, 0.0, abs_tol=1e-12)
    assert math.isclose(shift, 1.0, abs_tol=1e-12)

    assert app.vector_stats([1.0], [1.0, 2.0]) is None   # length mismatch
    assert app.vector_stats([0.0, 0.0], [1.0, 0.0]) is None  # zero norm
    assert app.vector_stats([], []) is None


def test_vector_trace_option_plots_both_tracks_on_brand():
    """Two traces, the palette verbatim, one baseline, and a zoom band."""
    traces = {"text": [0.1, -0.2, 0.05], "mice": [0.12, -0.18, 0.09]}
    option = app.vector_trace_option(traces)

    assert option["color"] == BRAND_SERIES
    assert len(option["series"]) == 2
    assert [s["lineStyle"]["color"] for s in option["series"]] == ["#123F36", "#C49A45"]
    # The baseline belongs to exactly one series or it is drawn twice.
    assert sum("markLine" in s for s in option["series"]) == 1
    assert option["xAxis"]["data"] == [0, 1, 2]
    assert {z["type"] for z in option["dataZoom"]} == {"inside", "slider"}
    assert option["yAxis"]["splitLine"]["lineStyle"]["type"] == "dashed"
    json.dumps(option)

    # One track missing is a degraded plot, not a crash; none is an empty dict.
    assert len(app.vector_trace_option({"mice": [0.1, 0.2]})["series"]) == 1
    assert app.vector_trace_option({}) == {}
    assert app.vector_trace_option({"text": []}) == {}


def test_vector_sample_rows_align_with_their_vectors():
    """The real artifact: 50 rows, both tracks, same work order on each side."""
    built = 0
    for key in app.MODELS:
        payload, err = app.load_vector_sample(key)
        if err:
            continue
        built += 1
        rows, vectors = payload["rows"], payload["vectors"]
        dim = payload["dim"]

        assert len(rows) == payload["n"] == 50, (key, len(rows))
        assert set(vectors) == {"text", "mice"}, (key, list(vectors))
        for track, values in vectors.items():
            assert len(values) == len(rows), (key, track)
            assert all(len(v) == dim for v in values), (key, track)

        # Every row carries the string that produced its vector, and the two
        # representations of one work order are genuinely different text.
        for record in rows:
            assert record["woid"] and record["text"] and record["mice"]
            assert record["text"] != record["mice"]
            assert "work order description:" in record["mice"]

        # Vectors are unit-normalised, which is what lets the two traces share
        # a y-axis and the cosine be read as an angle.
        norm = math.sqrt(sum(v * v for v in vectors["text"][0]))
        assert math.isclose(norm, 1.0, abs_tol=2e-3), (key, norm)

        cosine, shift = app.vector_stats(vectors["text"][0], vectors["mice"][0])
        assert -1.0 <= cosine <= 1.0 and shift >= 0.0

    assert built, "no vector_sample_50.json could be loaded - nothing was exercised"


def _rule(selector: str) -> str:
    """The declaration block of one selector, without a regex to escape."""
    parts = app.STYLESHEET.split(selector + " {", 1)
    assert len(parts) == 2, f"{selector} has no rule"
    return parts[1].split("}", 1)[0]


def test_display_face_is_imported_first_and_never_faux_bold():
    """Limelight ships a 400 only - a bold request gets synthesised and smears."""
    # An @import after any rule is dropped by the browser, silently.
    assert app.CSS.index("@import") < app.CSS.index(":root"), "@import is not first"
    for family in ("family=BBH+Hegarty", "family=Limelight"):
        assert family in app.FONT_IMPORT, family
    assert app.FONT_IMPORT.count("@import") == 1, "both families ride one request"
    assert app.TOKENS["--font-title"] == app.FONT_TITLE
    assert app.TOKENS["--font-display"] == app.FONT_DISPLAY

    for selector, token in ((".rm-title", "var(--font-title)"),
                            (".rm-sec", "var(--font-display)")):
        body = _rule(selector)
        assert token in body, selector
        assert "font-weight" not in body, f"{selector} wants a weight the face lacks"

    # Limelight is the masthead's alone - the section rules repeat too often.
    assert "var(--font-title)" not in _rule(".rm-sec")
    # Display faces are for headings only; body copy stays on the sans.
    for selector in (".rm-note", ".rm-sub"):
        assert "var(--font-display)" not in _rule(selector), selector
        assert "var(--font-title)" not in _rule(selector), selector


def test_notes_are_captions_not_paragraphs():
    """The 92ch cap was the narrow left column - 768px inside a 1392px row."""
    assert "max-width" not in _rule(".rm-note")

    # Nothing left tells the reader how to operate the UI.
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    for phrase in ("Select a row", "Drag the", "Sort or filter",
                   "search box is a", "A fixed 50-row", "A fixed 1,000-vector"):
        assert phrase not in source, f"instructional copy survives: {phrase!r}"


def test_no_sidebar_surface_survives():
    """`pages/` next to this script makes Streamlit build a nav regardless."""
    rule = _rule('[data-testid="stExpandSidebarButton"]')
    for testid in ("stSidebar", "stSidebarNav", "stSidebarCollapsedControl",
                   "stExpandSidebarButton"):
        assert f'[data-testid="{testid}"]' in app.STYLESHEET, testid
    # Streamlit's own rules are specific enough to win without this.
    assert "display: none !important" in rule, rule

    # The standalone dashboard owns no sidebar code of its own.
    source = (APP_DIR / "app.py").read_text(encoding="utf-8")
    for banned in ("st.sidebar", "components.sidebar", "render_sidebar"):
        assert banned not in source, f"app.py still references {banned!r}"


def test_right_anchored_legend_drops_the_left_anchor():
    """Both anchors set at once is what reserved the empty right-hand band."""
    legend = app._legend_right()
    assert "left" not in legend
    assert legend["right"] == 0 and legend["top"] == 0

    table = pd.DataFrame({"Strategy A": [0.5], "Strategy C": [0.6]}, index=["@10"])
    for option in (app.vector_trace_option({"text": [0.1, 0.2], "mice": [0.1, 0.3]}),
                   app.grouped_bar_option(table, "Recall")):
        assert "left" not in option["legend"], option["legend"]
        # The plot spans the container. The right margin is only what the last
        # axis label overhangs by - not a band reserved for a phantom legend.
        assert option["grid"]["left"] == 0, option["grid"]
        assert option["grid"]["right"] <= 12, option["grid"]


def test_heatmap_option_folds_row_major_and_diverges_on_brand():
    """Row-major fold, a scale symmetric about zero, the three brand anchors."""
    values = [0.02 * (i % 7) - 0.06 for i in range(768)]
    option = app.heatmap_option(values, "Text", 0.1)
    cols = app.HEATMAP_COLS

    data = option["series"][0]["data"]
    assert len(data) == 768
    assert option["xAxis"]["data"] == list(range(cols))
    assert option["yAxis"]["data"] == [r * cols for r in range(768 // cols)]
    assert option["yAxis"]["inverse"] is True
    # Cell i sits at (column, row) = (i % cols, i // cols) and keeps its value.
    for i in (0, 1, cols, 767):
        assert data[i][:2] == [i % cols, i // cols], (i, data[i])
        assert data[i][2] == round(values[i], 5)

    vmap = option["visualMap"]
    assert (vmap["min"], vmap["max"]) == (-0.1, 0.1), "scale is not symmetric"
    assert vmap["inRange"]["color"] == ["#123F36", "#FAF7F0", "#C49A45"]
    json.dumps(option)

    # A short tail gets its own row instead of being dropped.
    assert len(app.heatmap_option([0.1] * (cols + 1), "t", 0.1)["yAxis"]["data"]) == 2
    assert app.heatmap_option([], "t", 0.1) == {}
    assert app.heatmap_option([0.1], "t", 0.0) == {}   # no span, no scale


def test_robust_span_ignores_the_outlier_tail():
    """Scaling to the true maximum would spend the whole ramp on a few cells."""
    values = [0.01] * 990 + [0.9] * 10          # 1% of the mass is 90x the rest
    span = app.robust_span(values)
    assert span == 0.01, span
    assert max(abs(v) for v in values) == 0.9   # the maximum is what it replaces

    # It is still a real value from the data, and symmetric in sign.
    assert app.robust_span([-0.4, -0.2, 0.1]) == 0.4
    assert app.robust_span([]) == 0.0
    assert app.robust_span([0.05]) == 0.05      # single value, no index overrun


def test_every_planted_sentinel_resolves_to_a_formatter():
    """An unregistered sentinel renders the literal '__X_FMT__' as the tooltip."""
    options = [
        app.vector_trace_option({"text": [0.1], "mice": [0.2]}),
        app.heatmap_option([0.1] * 64, "t", 0.1),
        app.diverging_bar_option([("MRR", 0.02)]),
    ]
    planted = {(option.get("tooltip") or {}).get(field)
               for option in options for field in ("formatter", "valueFormatter")}
    planted = {v for v in planted if isinstance(v, str) and v.startswith("__")}
    assert planted, "no sentinels found - the formatter table is untested"
    assert planted <= set(app._FORMATTERS), planted - set(app._FORMATTERS)
    assert "__SCATTER_FMT__" in app._FORMATTERS


def test_heatmap_fold_lands_on_whole_rows_for_the_real_dimensions():
    """768 -> 32x24 and 1024 -> 32x32; neither leaves a ragged last row."""
    for dim in (768, 1024):
        assert dim % app.HEATMAP_COLS == 0, dim
    for key in app.MODELS:
        payload, err = app.load_vector_sample(key)
        if err:
            continue
        assert payload["dim"] % app.HEATMAP_COLS == 0, (key, payload["dim"])


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok  {name}")
    print("\nall checks passed")
