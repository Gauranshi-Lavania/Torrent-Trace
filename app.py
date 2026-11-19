"""
app.py - Flood Risk Studio (Streamlit)
======================================

Interactive web app on top of the models trained by ``train.py``.

Run:
    python train.py          # once, creates saved_models/
    streamlit run app.py

Tabs
----
1. Dashboard & Simulator  - risk gauge, risk tier banner, what-if simulator
2. Explainability (SHAP)  - why the model produced this risk, in plain language
3. Model Benchmarking     - metric cards, comparison table, ablation study
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# Same feature engineering function as training -> no train/serve skew.
from train import ENSEMBLE_NAME, FEATURE_GROUPS, MODEL_FILES, engineer_features

# --------------------------------------------------------------------------- #
# Page setup & design tokens
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Torrent Trace", page_icon="🌊", layout="wide",
                   initial_sidebar_state="expanded")

SAVE_DIR = Path(__file__).parent / "saved_models"
INK, RIVER, LINE = "#0D2B3E", "#0F7B8A", "#D3E0E6"
OK, WARN, BAD = "#2F9E6E", "#D9930D", "#D5453F"
FONT = "Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, label, p {{ font-family: {FONT}; }}
h1, h2, h3, .hero-title {{ font-family: 'Bricolage Grotesque', {FONT}; letter-spacing: -0.01em; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* Hero */
.hero {{ background: {INK}; border-radius: 18px; padding: 28px 34px; margin-bottom: 18px; text-align: center; }}
.hero .hero-title {{ color: #FFFFFF; font-size: clamp(2.2rem, 5vw, 3.2rem); font-weight: 800; margin: 0 0 6px 0; padding: 0; text-align: center; }}
.hero-sub {{ color: #B9D3DD; font-size: 1.02rem; margin: 0 auto; max-width: 760px; line-height: 1.5; }}
.hero [data-testid="stHeaderActionElements"] {{ display: none; }}

/* Cards */
.card {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 16px 18px; height: 100%; }}
.card.best {{ border: 2px solid {RIVER}; }}
.card .label {{ color: #5B7686; font-size: 0.88rem; font-weight: 500; margin-bottom: 2px; }}
.card .value {{ color: {INK}; font-family: 'Bricolage Grotesque', {FONT}; font-size: 1.75rem; font-weight: 700; line-height: 1.15; }}
.card .sub {{ color: #5B7686; font-size: 0.85rem; margin-top: 4px; }}
.card .tag {{ display: inline-block; background: {RIVER}; color: #fff; font-size: 0.72rem; font-weight: 600;
              padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }}

/* Risk badge + banner */
.badge {{ display: inline-block; color: #fff; font-weight: 700; font-size: 1.05rem; padding: 6px 18px; border-radius: 999px; }}
.banner {{ border-left: 6px solid; border-radius: 10px; padding: 14px 18px; margin-top: 12px; color: {INK}; line-height: 1.5; }}
.banner b {{ font-family: 'Bricolage Grotesque', {FONT}; }}

/* Streamlit native pieces */
[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 18px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: 10px 18px; border-radius: 10px 10px 0 0; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {LINE}; }}
.note {{ color: #5B7686; font-size: 0.88rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Loading artifacts
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained models ...")
def load_artifacts() -> Dict:
    """Load metadata, the five models, ensemble weights, explainer and background data."""
    meta = json.loads((SAVE_DIR / "metadata.json").read_text(encoding="utf-8"))
    models = {name: joblib.load(SAVE_DIR / f"{stem}.joblib") for name, stem in MODEL_FILES.items()}
    weights = joblib.load(SAVE_DIR / "ensemble.joblib")["weights"]
    explainer_model = joblib.load(SAVE_DIR / "explainer_lgbm_raw.joblib")
    background = pd.read_csv(SAVE_DIR / "background_sample.csv")
    return {"meta": meta, "models": models, "weights": weights,
            "explainer_model": explainer_model, "background": background}


@st.cache_resource
def get_explainer(_model) -> shap.TreeExplainer:
    """TreeExplainer for the raw-factor LightGBM model (underscore = not hashed)."""
    return shap.TreeExplainer(_model)


try:
    ART = load_artifacts()
except Exception as exc:  # missing files, version mismatch, ...
    st.markdown('<div class="hero"><p class="hero-title">Torrent Trace</p>'
                '<p class="hero-sub">Trained models were not found.</p></div>', unsafe_allow_html=True)
    st.error(f"Could not load models from `{SAVE_DIR.name}/`. Run `python train.py` first, "
             f"then restart the app.\n\nDetails: {exc}")
    st.stop()

META, MODELS, WEIGHTS = ART["meta"], ART["models"], ART["weights"]
FEATS: List[str] = META["features"]
STATS = META["feature_stats"]
LOW_MAX, HIGH_MIN = META["tiers"]["low_max"], META["tiers"]["high_min"]
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
MODEL_OPTIONS = [ENSEMBLE_NAME] + list(MODELS)


def pretty(name: str) -> str:
    """'TopographyDrainage' -> 'Topography Drainage'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)


def key(feature: str) -> str:
    return f"x_{feature}"


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict(model_name: str, raw: pd.DataFrame) -> np.ndarray:
    """Predict flood probability for each row of ``raw`` (columns = FEATS)."""
    X = engineer_features(raw, FEATS)
    if model_name == ENSEMBLE_NAME:
        out = sum(w * MODELS[n].predict(X) for n, w in WEIGHTS.items())
    else:
        out = MODELS[model_name].predict(X)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def tier_of(p: float):
    """Return (name, colour, message) for a probability using data-calibrated tiers."""
    if p < LOW_MAX:
        return "Low risk", OK, ("This profile sits in the safer third of regions in the training data. "
                                "Keep drainage, river management and planning in good shape to stay there.")
    if p < HIGH_MIN:
        return "Medium risk", WARN, ("This profile is in the middle of the pack. Check the what-if simulator "
                                     "below to see which single improvement lowers the risk the most.")
    return "High risk", BAD, ("This profile is among the most flood-prone third of regions in the training data. "
                              "Prioritise the highest-impact levers shown in the simulator.")


def make_gauge(prob: float, colour: str) -> go.Figure:
    pct = prob * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 56, "color": colour}},
        gauge={
            "axis": {"range": [0, 100], "tickvals": [0, 25, 50, 75, 100], "tickcolor": "#8AA3B0"},
            "bar": {"color": INK, "thickness": 0.22}, "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, LOW_MAX * 100], "color": "#8FD4AE"},
                {"range": [LOW_MAX * 100, HIGH_MIN * 100], "color": "#F4D27A"},
                {"range": [HIGH_MIN * 100, 100], "color": "#EE9089"},
            ],
            "threshold": {"line": {"color": colour, "width": 5}, "thickness": 0.85, "value": pct},
        },
    ))
    fig.update_layout(height=320, margin=dict(l=30, r=30, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT})
    return fig


def card(label: str, value: str, sub: str = "", best: bool = False) -> str:
    tag = '<span class="tag">best</span>' if best else ""
    return (f'<div class="card {"best" if best else ""}"><div class="label">{label}{tag}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


# --------------------------------------------------------------------------- #
# Sidebar: model choice, presets, 20 sliders
# --------------------------------------------------------------------------- #
for f in FEATS:  # defaults = dataset median (set once, before the sliders exist)
    st.session_state.setdefault(key(f), int(round(STATS[f]["q50"])))


def apply_preset(kind: str) -> None:
    """Callback for preset buttons: overwrite every slider value."""
    if kind == "random":
        row = ART["background"].sample(1).iloc[0]
        vals = {f: int(row[f]) for f in FEATS}
    else:
        q = {"low": "q25", "typical": "q50", "high": "q75"}[kind]
        vals = {f: int(round(STATS[f][q])) for f in FEATS}
    for f, v in vals.items():
        st.session_state[key(f)] = int(np.clip(v, STATS[f]["min"], STATS[f]["max"]))


with st.sidebar:
    st.markdown("### Region profile")
    model_name = st.selectbox("Prediction model", MODEL_OPTIONS, index=0,
                              help="The weighted ensemble blends all five models.")
    st.caption("Quick start")
    c1, c2 = st.columns(2)
    c1.button("Typical", on_click=apply_preset, args=("typical",))
    c2.button("Random real row", on_click=apply_preset, args=("random",))
    c3, c4 = st.columns(2)
    c3.button("Safer profile", on_click=apply_preset, args=("low",))
    c4.button("Riskier profile", on_click=apply_preset, args=("high",))
    st.markdown('<p class="note">Each factor is a score. In this dataset a higher score pushes flood '
                'probability up.</p>', unsafe_allow_html=True)

    grouped: Dict[str, List[str]] = {}
    for f in FEATS:
        grouped.setdefault(GROUP_OF.get(f, "Other factors"), []).append(f)
    for i, (group, fs) in enumerate(grouped.items()):
        with st.expander(group, expanded=(i == 0)):
            for f in fs:
                st.slider(pretty(f), int(STATS[f]["min"]), int(STATS[f]["max"]), key=key(f))

VALUES = {f: int(st.session_state[key(f)]) for f in FEATS}
ROW = pd.DataFrame([VALUES])[FEATS]
PRED = float(predict(model_name, ROW)[0])
TIER, TIER_COLOUR, TIER_MSG = tier_of(PRED)
PERCENTILE = float(np.interp(PRED, META["target_quantiles"], np.linspace(0, 100, len(META["target_quantiles"]))))

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="hero"><h1 class="hero-title">Torrent Trace</h1>'
    '<p class="hero-sub">Set 20 environmental and infrastructure scores, read the predicted flood probability, '
    'test what an improvement would change, and see which factors drive the result.</p></div>',
    unsafe_allow_html=True)

tab_dash, tab_shap, tab_bench = st.tabs(["📊 Dashboard & Simulator", "🔍 Explainability (SHAP)", "🧪 Model Benchmarking"])

# =========================================================================== #
# TAB 1 - Dashboard & what-if simulator
# =========================================================================== #
with tab_dash:
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(make_gauge(PRED, TIER_COLOUR), key="gauge")
    with right:
        st.markdown(f'<span class="badge" style="background:{TIER_COLOUR}">{TIER}</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="banner" style="border-color:{TIER_COLOUR}; background:{TIER_COLOUR}14">'
            f'<b>{TIER}.</b> {TIER_MSG}</div>', unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        k1.markdown(card("Riskier than", f"{PERCENTILE:.0f}%", "of regions in the training data"), unsafe_allow_html=True)
        k2.markdown(card("Total factor score", f"{sum(VALUES.values())}",
                         f"dataset average {sum(STATS[f]['mean'] for f in FEATS):.0f}"), unsafe_allow_html=True)
    with st.expander("See what each model predicts for this profile"):
        every = {m: float(predict(m, ROW)[0]) * 100 for m in MODEL_OPTIONS}
        st.dataframe(pd.DataFrame({"Model": list(every), "Flood probability (%)": [round(v, 2) for v in every.values()]}),
                     hide_index=True)

    st.markdown("---")
    st.subheader("What-if simulator")
    st.caption("Lower one factor's score (an improvement) and compare the predicted risk before and after.")

    default_idx = FEATS.index("TopographyDrainage") if "TopographyDrainage" in FEATS else 0
    s1, s2 = st.columns([1, 1])
    factor = s1.selectbox("Factor to improve", FEATS, index=default_idx, format_func=pretty)
    improve = s2.slider("Improvement (points the score is reduced)", -8, 8, 3,
                        help="Positive = improvement (score goes down). Negative = deterioration.")

    base_val = VALUES[factor]
    new_val = int(np.clip(base_val - improve, STATS[factor]["min"], STATS[factor]["max"]))
    sim_row = ROW.copy()
    sim_row[factor] = new_val
    SIM = float(predict(model_name, sim_row)[0])
    delta_pp = (SIM - PRED) * 100

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline risk", f"{PRED * 100:.1f}%", help=f"{pretty(factor)} = {base_val}")
    m2.metric("Simulated risk", f"{SIM * 100:.1f}%", delta=f"{delta_pp:+.2f} pp",
              delta_color="inverse", help=f"{pretty(factor)} = {new_val}")
    rel = (delta_pp / (PRED * 100) * 100) if PRED > 0 else 0.0
    m3.metric("Relative change", f"{rel:+.1f}%", delta="lower is better", delta_color="off")

    if new_val == base_val:
        st.info("The score is already at its limit, so nothing changes. Try another factor.")
    else:
        verb = "drops" if delta_pp < 0 else "rises"
        st.markdown(f"Moving **{pretty(factor)}** from **{base_val}** to **{new_val}** makes the predicted risk "
                    f"**{verb} by {abs(delta_pp):.2f} percentage points** ({PRED * 100:.1f}% to {SIM * 100:.1f}%).")

    curve_col, lever_col = st.columns(2)
    with curve_col:
        st.markdown("**Risk across the full range of this factor**")
        grid = np.arange(int(STATS[factor]["min"]), int(STATS[factor]["max"]) + 1)
        sweep = pd.concat([ROW] * len(grid), ignore_index=True)
        sweep[factor] = grid
        fig = go.Figure(go.Scatter(x=grid, y=predict(model_name, sweep) * 100, mode="lines+markers",
                                   line=dict(color=RIVER, width=3), name="Predicted risk"))
        fig.add_trace(go.Scatter(x=[base_val], y=[PRED * 100], mode="markers", name="Baseline",
                                 marker=dict(size=14, color=INK)))
        fig.add_trace(go.Scatter(x=[new_val], y=[SIM * 100], mode="markers", name="Simulated",
                                 marker=dict(size=14, color=TIER_COLOUR, symbol="diamond")))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=f"{pretty(factor)} score",
                          yaxis_title="Flood probability (%)", plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.12))
        fig.update_yaxes(gridcolor="#E6EEF2")
        st.plotly_chart(fig, key="curve")
    with lever_col:
        st.markdown(f"**Which factor helps most? (each reduced by {improve} pts)**")
        lever = pd.concat([ROW] * len(FEATS), ignore_index=True)
        for i, f in enumerate(FEATS):
            lever.loc[i, f] = int(np.clip(VALUES[f] - improve, STATS[f]["min"], STATS[f]["max"]))
        drops = (PRED - predict(model_name, lever)) * 100
        order = np.argsort(drops)  # biggest drop ends up at the top of the bar chart
        top = order[-10:]
        fig2 = go.Figure(go.Bar(
            x=drops[top], y=[pretty(FEATS[i]) for i in top], orientation="h",
            marker_color=[RIVER if FEATS[i] == factor else "#9DB8C4" for i in top],
            text=[f"{drops[i]:+.2f}" for i in top], textposition="outside"))
        fig2.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Risk reduction (percentage points)",
                           plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
        fig2.update_xaxes(gridcolor="#E6EEF2")
        st.plotly_chart(fig2, key="levers")

# =========================================================================== #
# TAB 2 - SHAP explainability
# =========================================================================== #
with tab_shap:
    st.subheader("Why this prediction?")
    st.caption("SHAP splits the prediction into one contribution per factor. The explainer is a LightGBM model "
               "trained on the 20 raw factors, so every bar maps to something you can change with a slider.")
    explainer = get_explainer(ART["explainer_model"])
    sv = np.asarray(explainer.shap_values(ROW)).reshape(-1)
    base = float(np.ravel(explainer.expected_value)[0])
    explainer_out = base + sv.sum()

    order = np.argsort(-np.abs(sv))
    top_n = 10
    top_idx, rest_idx = order[:top_n], order[top_n:]
    labels = ["Dataset average"] + [f"{pretty(FEATS[i])} = {VALUES[FEATS[i]]}" for i in top_idx]
    amounts = [base * 100] + [sv[i] * 100 for i in top_idx]
    measures = ["absolute"] + ["relative"] * len(top_idx)
    if len(rest_idx):
        labels.append(f"Other {len(rest_idx)} factors")
        amounts.append(sv[rest_idx].sum() * 100)
        measures.append("relative")
    labels.append("Explainer prediction")
    amounts.append(0)
    measures.append("total")
    texts = [f"{amounts[0]:.1f}%"] + [f"{a:+.2f}" for a in amounts[1:-1]] + [f"{explainer_out * 100:.1f}%"]

    wf = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=amounts, measure=measures, text=texts, textposition="outside",
        increasing={"marker": {"color": BAD}}, decreasing={"marker": {"color": OK}},
        totals={"marker": {"color": INK}}, connector={"line": {"color": LINE}}))
    cum = np.cumsum([amounts[0]] + amounts[1:-1])
    wf.update_layout(height=470, margin=dict(l=10, r=60, t=10, b=10), plot_bgcolor="white",
                     paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Flood probability (%)",
                     xaxis_range=[float(cum.min()) - 3, float(cum.max()) + 3], showlegend=False)
    wf.update_yaxes(autorange="reversed")
    wf.update_xaxes(gridcolor="#E6EEF2")

    w_col, t_col = st.columns([1.5, 1])
    with w_col:
        st.plotly_chart(wf, key="waterfall")
    with t_col:
        ups = [(FEATS[i], sv[i] * 100) for i in np.argsort(-sv) if sv[i] > 0][:3]
        downs = [(FEATS[i], sv[i] * 100) for i in np.argsort(sv) if sv[i] < 0][:3]
        st.markdown("#### In plain words")
        st.markdown(f"The typical region in the data has about **{base * 100:.1f}%** flood probability. "
                    f"For this profile the explainer model gets **{explainer_out * 100:.1f}%**"
                    f" (selected model: **{PRED * 100:.1f}%**).")
        if ups:
            st.markdown("**Pushing risk up**")
            for f, v in ups:
                st.markdown(f"- {pretty(f)} (score {VALUES[f]}): **{v:+.2f} pp**")
        if downs:
            st.markdown("**Holding risk down**")
            for f, v in downs:
                st.markdown(f"- {pretty(f)} (score {VALUES[f]}): **{v:+.2f} pp**")
        st.markdown('<p class="note">pp = percentage points. Red bars add risk, green bars reduce it.</p>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Which factors matter most overall?")

    @st.cache_data(show_spinner="Computing global importance ...")
    def global_importance(_explainer, _background: pd.DataFrame, cache_key: str) -> pd.Series:
        vals = np.asarray(_explainer.shap_values(_background[FEATS]))
        return pd.Series(np.abs(vals).mean(axis=0) * 100, index=FEATS).sort_values()

    imp = global_importance(explainer, ART["background"], str(META["created_at"]))
    gfig = go.Figure(go.Bar(x=imp.values, y=[pretty(f) for f in imp.index], orientation="h", marker_color=RIVER))
    gfig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Average absolute impact (percentage points)",
                       plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    gfig.update_xaxes(gridcolor="#E6EEF2")
    st.plotly_chart(gfig, key="global_imp")
    st.caption(f"Computed on {len(ART['background'])} random training rows. Top three drivers: "
               + ", ".join(pretty(f) for f in imp.index[::-1][:3]) + ".")

# =========================================================================== #
# TAB 3 - Model benchmarking
# =========================================================================== #
with tab_bench:
    metrics = pd.DataFrame(META["metrics"]).rename(columns={"R2": "R²"}).sort_values("R²", ascending=False).reset_index(drop=True)
    ds = META["dataset"]
    st.subheader("Model comparison")
    st.caption(f"Scores are measured on a held-out test split of {ds['test']:,} rows the models never saw "
               f"(trained on {ds['train']:,} rows).")

    best_model = metrics.loc[0, "Model"]
    for start in range(0, len(metrics), 3):
        cols = st.columns(3)
        for col, (_, r) in zip(cols, metrics.iloc[start:start + 3].iterrows()):
            col.markdown(card(r["Model"], f"R² {r['R²']:.4f}", f"RMSE {r['RMSE']:.5f}  |  MAE {r['MAE']:.5f}",
                              best=(r["Model"] == best_model)), unsafe_allow_html=True)
        st.write("")

    table = metrics.copy()
    table.insert(0, "Rank", range(1, len(table) + 1))
    st.dataframe(table.style.format({"R²": "{:.5f}", "RMSE": "{:.5f}", "MAE": "{:.5f}"})
                 .highlight_max(subset=["R²"], color="#CDEBDD").highlight_min(subset=["RMSE", "MAE"], color="#CDEBDD"),
                 hide_index=True)

    c_r2, c_rmse = st.columns(2)
    for col, metric, better in ((c_r2, "R²", "higher"), (c_rmse, "RMSE", "lower")):
        lo, hi = metrics[metric].min(), metrics[metric].max()
        pad = max((hi - lo) * 0.6, 1e-4)
        f = go.Figure(go.Bar(x=metrics["Model"], y=metrics[metric],
                             marker_color=[RIVER if m == best_model else "#9DB8C4" for m in metrics["Model"]],
                             text=[f"{v:.4f}" for v in metrics[metric]], textposition="outside"))
        f.update_layout(title=f"{metric} ({better} is better)", height=340, margin=dict(l=10, r=10, t=50, b=10),
                        plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", yaxis_range=[lo - pad, hi + pad])
        f.update_yaxes(gridcolor="#E6EEF2")
        col.plotly_chart(f, key=f"bar_{metric}")
    st.caption("The y-axis is zoomed in so small differences are visible. Check the numbers before drawing conclusions.")

    st.markdown("**Ensemble weights** (learned on the validation split)")
    wdf = pd.DataFrame({"Model": list(WEIGHTS), "Weight": [round(v, 3) for v in WEIGHTS.values()]})
    wfig = go.Figure(go.Bar(x=wdf["Weight"], y=wdf["Model"], orientation="h", marker_color=RIVER,
                            text=wdf["Weight"], textposition="outside"))
    wfig.update_layout(height=240, margin=dict(l=10, r=40, t=10, b=10), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(wfig, key="weights")

    st.markdown("---")
    st.subheader("Feature engineering ablation study")
    st.caption("Each row adds more row-wise statistics (computed across the 20 factors) to the raw inputs. "
               "Δ columns show the change in R² versus the raw-factor baseline.")
    abl = pd.DataFrame(META["ablation"]).rename(columns={
        "Ridge R2": "Ridge R²", "LightGBM R2": "LightGBM R²",
        "Ridge dR2 vs raw": "Ridge Δ R²", "LightGBM dR2 vs raw": "LightGBM Δ R²"})
    st.dataframe(abl.style.format({"Ridge R²": "{:.5f}", "Ridge RMSE": "{:.5f}", "LightGBM R²": "{:.5f}",
                                   "LightGBM RMSE": "{:.5f}", "Ridge Δ R²": "{:+.5f}", "LightGBM Δ R²": "{:+.5f}"})
                 .highlight_max(subset=["Ridge R²", "LightGBM R²"], color="#CDEBDD"), hide_index=True)

    afig = go.Figure()
    for name, colour in (("Ridge R²", INK), ("LightGBM R²", RIVER)):
        afig.add_trace(go.Scatter(x=abl["Feature set"], y=abl[name], mode="lines+markers", name=name,
                                  line=dict(color=colour, width=3), marker=dict(size=10)))
    afig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Test R²",
                       plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12))
    afig.update_yaxes(gridcolor="#E6EEF2")
    st.plotly_chart(afig, key="ablation")

    st.markdown("---")
    st.markdown(
        f'<p class="note">Data: Kaggle Playground Series S4E5 (synthetic, generated from a real flood-factors dataset). '
        f'{ds["rows_used"]:,} of {ds["rows_in_file"]:,} rows used. Models trained {META["created_at"].replace("T", " ")}. '
        'Risk tiers are relative to this dataset, and the app is an educational demo, not an early-warning system.</p>',
        unsafe_allow_html=True)