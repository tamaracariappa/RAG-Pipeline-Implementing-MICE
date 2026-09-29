"""
test_app.py - self-check for the MICE evaluation dashboard.

Run with:
    python test_app.py            (from streamlit_app/)

Importing app.py executes the whole dashboard in Streamlit's bare mode, so the
import alone is a smoke test: every loader, chart and render path runs against
the real data files. The asserts below then pin the logic that is easy to get
quietly wrong - the derived outcome/winner columns, and the guarded I/O.
"""

import os
import sys
import warnings

os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402  - importing renders the dashboard headlessly


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
    """A model with no output directory must degrade, not explode."""
    for loader in (app.load_eval, app.load_queries):
        payload, err = loader("model_that_does_not_exist")
        assert payload is None and err and "not found" in err, (loader.__name__, err)


def test_charts_build_from_real_payload():
    """Every shipped model renders a full set of figures with table twins."""
    built = 0
    for key in app.MODELS:
        payload, err = app.load_eval(key)
        if err:
            continue
        built += 1
        strategies = payload["strategies"]

        for metric in ("recall", "ndcg"):
            fig, table = app.metric_by_k(strategies, metric, metric)
            assert len(fig.data) == len(strategies), (key, metric)
            assert table.shape == (len(payload["top_k"]), len(strategies))
            # Every bar is labelled: slots A and B are close in hue, so the
            # printed value is what separates them.
            labelled = sum(bool(t) for trace in fig.data for t in trace.text)
            assert labelled == table.size, (key, metric, labelled)

        fig, table = app.delta_mice_chart(payload["delta_mice"])
        values = table.iloc[:, 0]
        assert len(table) == 1 + 2 * len(payload["top_k"]), key
        # One trace per sign actually present, and no value lands in both.
        assert len(fig.data) == len({v >= 0 for v in values}), key

    assert built, "no eval_results.json could be loaded - nothing was exercised"


def test_palette_is_exactly_the_brand_palette():
    """The brand palette is applied verbatim - no substituted or derived hues."""
    assert app.BRAND == {
        "primary": "#123F36",
        "secondary": "#2A6B5C",
        "accent": "#C49A45",
        "cream": "#E8DCC4",
    }
    assert app.SERIES == {"A": "#123F36", "B": "#2A6B5C", "C": "#C49A45"}


def test_brand_ramp_stays_on_the_brand_anchors():
    """The PCA ramp interpolates between brand colours; it never leaves them."""
    assert app._brand_ramp(1) == ["#123F36"]
    ramp = app._brand_ramp(7)
    assert len(ramp) == 7
    assert ramp[0] == "#123F36" and ramp[-1] == "#C49A45"
    assert "#2A6B5C" in ramp, ramp  # the mid anchor is hit exactly
    assert all(len(c) == 7 and c.startswith("#") for c in ramp)


def test_pca_reports_missing_index_instead_of_raising():
    """The FAISS artifacts are build outputs; absence must degrade, not crash."""
    payload, err = app.pca_projection("model_that_does_not_exist", "text", "equipment")
    assert payload is None and err and "not built" in err, err

    # And for the real models, it either projects or explains why it cannot.
    for key in app.MODELS:
        for track, _, _ in app.PCA_TRACKS:
            payload, err = app.pca_projection(key, track, "equipment")
            assert (payload is None) != (err is None), (key, track)
            if payload is not None:
                coords, labels, hover, explained, total, dim = payload
                assert coords.shape == (len(labels), 2)
                assert len(hover) == len(labels) and len(explained) == 2
                assert 0 < len(labels) <= total and dim > 0


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_") and callable(case):
            case()
            print(f"ok  {name}")
    print("\nall checks passed")
