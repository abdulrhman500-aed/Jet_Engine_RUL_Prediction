from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Page configuration and visual system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroMind | Engine Health Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, #142943 0, #0b1422 38%, #080f1b 100%); color: #e8eef7; }
[data-testid="stSidebar"] { background: #0b1727; border-right: 1px solid #24364d; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
.hero { padding: 1.45rem 1.7rem; border: 1px solid #29415d; border-radius: 20px; background: linear-gradient(120deg, rgba(24,54,82,.96), rgba(14,27,45,.96)); margin-bottom: 1.1rem; }
.eyebrow { color:#65d6c5; font-size:.75rem; letter-spacing:.16em; font-weight:800; text-transform:uppercase; }
.hero h1 { color:#f4f8ff; font-size:2rem; margin:.35rem 0 .25rem 0; font-weight:800; }
.hero p { color:#aebed1; margin:0; }
.section-title { color:#f1f6ff; font-size:1.08rem; font-weight:700; margin:.6rem 0 .75rem 0; }
.metric-card { background:linear-gradient(145deg,#14263b,#101c2d); border:1px solid #2a4058; border-radius:16px; padding:16px 18px; min-height:112px; }
.metric-label { color:#9fb2c8; font-size:.78rem; font-weight:600; }
.metric-value { color:#f4f8ff; font-size:1.75rem; font-weight:800; margin-top:8px; }
.metric-sub { color:#8da3bb; font-size:.75rem; margin-top:2px; }
.panel { background:rgba(17,31,49,.82); border:1px solid #263c54; border-radius:16px; padding:16px 18px; }
.small-muted { color:#9db0c6; font-size:.83rem; }
.status-critical { color:#ff8f8f; font-weight:800; }
.status-warning { color:#ffd27a; font-weight:800; }
.status-healthy { color:#6fe0bd; font-weight:800; }
div[data-testid="stMetric"] { background:#14263b; border:1px solid #2a4058; padding:14px 16px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#a8bad0; }
div[data-testid="stMetricValue"] { color:#f4f8ff; }
.stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid #3b7890; background:linear-gradient(90deg,#176b78,#218b83); color:white; font-weight:700; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:#75d9cb; color:white; }
hr { border-color:#263c54; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "rul_model.joblib"
META_PATH = ROOT / "metadata.joblib"
PRED_PATH = ROOT / "test_predictions.csv"
IMPORTANCE_PATH = ROOT / "permutation_importance.csv"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Model files are missing. Keep rul_model.joblib and metadata.joblib beside app.py.")
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta

try:
    model, meta = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

BASE_FEATURES = list(meta["base_features"])
MODEL_FEATURES = list(meta["model_features"])
ALERT_THRESHOLD = float(meta.get("alert_threshold", 30))
DATASET = meta.get("dataset_subset", "FD001")

# The exact feature engineering used in notebook cell 17.
def make_features(history: pd.DataFrame) -> pd.DataFrame:
    required = ["time_cycles"] + BASE_FEATURES
    out = history[required].copy()
    out["time_cycles"] = pd.to_numeric(out["time_cycles"], errors="coerce")
    for col in BASE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("time_cycles").reset_index(drop=True)
    for col in BASE_FEATURES:
        out[f"{col}_diff1"] = out[col].diff(1)
        for window in (5, 10, 20):
            rolling = out[col].rolling(window=window, min_periods=1)
            out[f"{col}_rollmean_{window}"] = rolling.mean()
            out[f"{col}_rollstd_{window}"] = rolling.std()
    # Keep time_cycles: it is one of the 137 model inputs in the trained artifact.
    return out.reindex(columns=MODEL_FEATURES)


def predict_history(history: pd.DataFrame):
    required = ["time_cycles"] + BASE_FEATURES
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    clean = history[required].copy()
    for col in required:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    if clean["time_cycles"].isna().any():
        raise ValueError("time_cycles must contain valid numeric cycle numbers.")
    if clean[BASE_FEATURES].isna().any().any():
        bad = clean[BASE_FEATURES].columns[clean[BASE_FEATURES].isna().any()].tolist()
        raise ValueError("Please fill every sensor/setting value. Missing values in: " + ", ".join(bad))
    if len(clean) < 2:
        raise ValueError("Enter at least 2 consecutive engine readings. A history is needed to calculate changes and rolling features.")
    if clean["time_cycles"].duplicated().any():
        raise ValueError("Duplicate time_cycles detected. Each row must represent a distinct cycle.")
    clean = clean.sort_values("time_cycles").reset_index(drop=True)
    feature_frame = make_features(clean)
    latest = feature_frame.iloc[[-1]]
    prediction = max(0.0, float(model.predict(latest)[0]))
    return prediction, clean, feature_frame


def status_for(rul):
    if rul < ALERT_THRESHOLD:
        return "CRITICAL", "status-critical", "Maintenance review required now"
    if rul < ALERT_THRESHOLD * 2:
        return "WARNING", "status-warning", "Plan an inspection soon"
    return "HEALTHY", "status-healthy", "No threshold alert"


def metric_card(label, value, sub=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------
st.markdown('''<div class="hero"><div class="eyebrow">Predictive maintenance • Machine learning</div><h1>✈️ AeroMind | Engine Health Intelligence</h1><p>Remaining Useful Life prediction powered by the trained NASA C-MAPSS FD001 model.</p></div>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✈️ AeroMind")
    st.caption("ENGINE HEALTH CONSOLE")
    page = st.radio("Workspace", ["Live RUL Prediction", "Model Performance", "Feature Guide"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Model configuration**")
    st.write(f"Dataset: `{DATASET}`")
    st.write("Estimator: `Random Forest Regressor`")
    st.write(f"Engine alert: `< {ALERT_THRESHOLD:.0f} cycles`")
    st.write(f"Input variables: `{len(BASE_FEATURES)}`")
    st.write(f"Engineered features: `{len(MODEL_FEATURES)}`")
    st.divider()
    st.caption("Research prototype • Simulated benchmark data • Not certified for aviation maintenance")

if page == "Live RUL Prediction":
    st.markdown('<div class="section-title">01 / Select input method</div>', unsafe_allow_html=True)
    mode = st.radio("Input source", ["Manual sensor entry", "Upload engine history CSV"], horizontal=True)
    st.markdown('<div class="small-muted">The model needs the chronological history of an engine, not just one isolated reading. The app calculates cycle differences and rolling statistics exactly as in training.</div>', unsafe_allow_html=True)
    history = None

    if mode == "Upload engine history CSV":
        uploaded = st.file_uploader("Upload CSV containing one engine's chronological history", type=["csv"], help="Required columns: time_cycles plus all 17 setting/sensor columns listed below.")
        with st.expander("Required input columns"):
            st.code(", ".join(["time_cycles"] + BASE_FEATURES), language="text")
        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                if "unit_number" in raw.columns:
                    ids = sorted(raw["unit_number"].dropna().unique().tolist())
                    if len(ids) > 1:
                        chosen = st.selectbox("Choose engine ID", ids)
                        raw = raw[raw["unit_number"] == chosen].copy()
                    elif len(ids) == 1:
                        st.caption(f"Engine ID: {ids[0]}")
                history = raw
                st.success(f"Loaded {len(raw):,} readings from CSV.")
            except Exception as exc:
                st.error(f"Could not read CSV: {exc}")
    else:
        st.markdown('<div class="panel"><b>Manual engine telemetry</b><br><span class="small-muted">Enter one row per cycle. All 17 inputs are required because these are the exact non-constant variables retained during training.</span></div>', unsafe_allow_html=True)
        template = pd.DataFrame({"time_cycles": list(range(1, 6))})
        for col in BASE_FEATURES:
            template[col] = np.nan
        with st.expander("Show the 17 model input variables", expanded=True):
            st.write("**Operating settings:** " + ", ".join(c for c in BASE_FEATURES if c.startswith("setting_")))
            st.write("**Sensors:** " + ", ".join(c for c in BASE_FEATURES if c.startswith("sensor_")))
        st.caption("You can edit the sample cycle numbers, fill in sensor values, and add more rows from the table controls.")
        edited = st.data_editor(
            template,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={"time_cycles": st.column_config.NumberColumn("Cycle number", min_value=1, step=1, format="%d")},
            key="manual_telemetry_editor",
        )
        st.download_button("⬇ Download blank input template", template.to_csv(index=False).encode("utf-8"), "FD001_manual_input_template.csv", "text/csv")
        if st.button("Run RUL prediction", type="primary", use_container_width=True):
            st.session_state["manual_history"] = edited.copy()
            st.session_state["run_manual"] = True
        if st.session_state.get("run_manual"):
            history = st.session_state.get("manual_history")

    if history is not None:
        try:
            prediction, clean_history, engineered = predict_history(history)
            status, status_class, recommendation = status_for(prediction)
            st.divider()
            st.markdown('<div class="section-title">02 / Engine health assessment</div>', unsafe_allow_html=True)
            a, b, c, d = st.columns(4)
            with a: metric_card("PREDICTED REMAINING LIFE", f"{prediction:.1f}", "Operating cycles")
            with b: metric_card("LATEST OBSERVED CYCLE", f"{clean_history.time_cycles.iloc[-1]:.0f}", "Most recent telemetry")
            with c: metric_card("HISTORY LENGTH", f"{len(clean_history)}", "Recorded cycles")
            with d: metric_card("MODEL INPUTS", f"{len(MODEL_FEATURES)}", "Engineered features")
            status_color = "#ff8f8f" if status == "CRITICAL" else "#ffd27a" if status == "WARNING" else "#6fe0bd"
            st.markdown(f'<div class="panel" style="margin-top:14px;border-left:5px solid {status_color};"><div class="metric-label">ENGINE CONDITION</div><div style="font-size:1.55rem;font-weight:800;color:{status_color};margin:4px 0;">{status}</div><div class="small-muted">{recommendation} • Threshold configured from project metadata: {ALERT_THRESHOLD:.0f} cycles.</div></div>', unsafe_allow_html=True)
            if status == "CRITICAL":
                st.error("Predicted RUL is below the project's 30-cycle alert threshold. Escalate for qualified maintenance review.")
            elif status == "WARNING":
                st.warning("Predicted RUL is below 60 cycles. Schedule a closer inspection and monitor the next readings.")
            else:
                st.success("Prediction is currently above the configured alert range.")

            left, right = st.columns([1.25, 1])
            with left:
                st.markdown('<div class="section-title">Sensor trend explorer</div>', unsafe_allow_html=True)
                sensor_cols = [c for c in BASE_FEATURES if c.startswith("sensor_")]
                selected_sensor = st.selectbox("Choose sensor", sensor_cols)
                fig = px.line(clean_history, x="time_cycles", y=selected_sensor, markers=True, template="plotly_dark", labels={"time_cycles":"Operating cycle", selected_sensor:selected_sensor})
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=25,b=10), height=330)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.markdown('<div class="section-title">Latest telemetry snapshot</div>', unsafe_allow_html=True)
                latest_values = clean_history.iloc[-1][BASE_FEATURES].rename("Latest value").to_frame()
                latest_values.index.name = "Input variable"
                st.dataframe(latest_values, use_container_width=True, height=330)

            with st.expander("Inspect engineered features sent to the model"):
                st.caption("The last row is passed to the saved 137-feature model pipeline in the exact metadata order.")
                st.dataframe(engineered.tail(1).T.rename(columns={engineered.tail(1).index[0]:"Model input"}), use_container_width=True)
            st.download_button("⬇ Download prediction-ready engineered row", engineered.tail(1).to_csv(index=False).encode("utf-8"), "engineered_model_input.csv", "text/csv")
        except Exception as exc:
            st.error(str(exc))
            st.info("Check that every required input has a numeric value and that each row represents a different engine cycle.")

elif page == "Model Performance":
    st.markdown('<div class="section-title">02 / Official NASA FD001 test benchmark</div>', unsafe_allow_html=True)
    if PRED_PATH.exists():
        results = pd.read_csv(PRED_PATH)
        mae = float((results.actual_RUL - results.predicted_RUL).abs().mean())
        rmse = float(np.sqrt(((results.actual_RUL - results.predicted_RUL) ** 2).mean()))
        ss_res = float(((results.actual_RUL - results.predicted_RUL) ** 2).sum())
        ss_tot = float(((results.actual_RUL - results.actual_RUL.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        p1,p2,p3,p4 = st.columns(4)
        with p1: metric_card("TEST ENGINES", f"{len(results)}", "Official FD001 test set")
        with p2: metric_card("MAE", f"{mae:.2f}", "Cycles average absolute error")
        with p3: metric_card("RMSE", f"{rmse:.2f}", "Cycles; penalizes large errors")
        with p4: metric_card("R²", f"{r2:.3f}", "Coefficient of determination")
        st.markdown('<div class="section-title">Actual vs predicted RUL</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results.engine_id, y=results.actual_RUL, mode="lines+markers", name="Actual RUL"))
        fig.add_trace(go.Scatter(x=results.engine_id, y=results.predicted_RUL, mode="lines+markers", name="Predicted RUL"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Test engine", yaxis_title="RUL (cycles)", height=390)
        st.plotly_chart(fig, use_container_width=True)
        l,r = st.columns(2)
        with l:
            st.markdown('<div class="section-title">Prediction error distribution</div>', unsafe_allow_html=True)
            err = results.actual_RUL - results.predicted_RUL
            fig2 = px.histogram(x=err, nbins=20, template="plotly_dark", labels={"x":"Actual − predicted (cycles)"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig2, use_container_width=True)
        with r:
            st.markdown('<div class="section-title">Top validation permutation importance</div>', unsafe_allow_html=True)
            if IMPORTANCE_PATH.exists():
                imp = pd.read_csv(IMPORTANCE_PATH).head(12).sort_values("importance_mean")
                fig3 = px.bar(imp, x="importance_mean", y="feature", orientation="h", error_x="importance_std", template="plotly_dark", labels={"importance_mean":"Importance mean", "feature":"Feature"})
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
                st.plotly_chart(fig3, use_container_width=True)
        with st.expander("View all official test predictions"):
            st.dataframe(results, use_container_width=True)
            st.download_button("Download test predictions", results.to_csv(index=False).encode("utf-8"), "test_predictions.csv", "text/csv")
    else:
        st.warning("test_predictions.csv is not present in the app folder.")

else:
    st.markdown('<div class="section-title">03 / What the trained model expects</div>', unsafe_allow_html=True)
    st.markdown("""<div class="panel"><b>Raw telemetry → feature engineering → trained estimator</b><p class="small-muted">The notebook removes constant columns, retains the following 17 raw inputs, calculates previous-cycle differences and rolling statistics over windows of 5, 10 and 20 cycles, and preserves time_cycles. This produces the exact 137 model columns.</p></div>""", unsafe_allow_html=True)
    st.write("**Operating settings**")
    st.code(", ".join(c for c in BASE_FEATURES if c.startswith("setting_")))
    st.write("**Sensor measurements**")
    st.code(", ".join(c for c in BASE_FEATURES if c.startswith("sensor_")))
    st.write("**Feature engineering per raw variable**")
    st.code("current value + diff(1) + rolling mean/std (windows=5, 10, 20)")
    st.info("A single isolated row is insufficient for a faithful prediction: rolling statistics and cycle-to-cycle differences require chronological history.")
    st.warning("The model and benchmark use simulated NASA C-MAPSS FD001 data. This dashboard is an educational/research prototype, not a certified aviation maintenance system.")

st.divider()
st.caption("AeroMind • NASA C-MAPSS FD001 • RUL = Remaining Useful Life • Predictions are estimates, not maintenance authorization.")
