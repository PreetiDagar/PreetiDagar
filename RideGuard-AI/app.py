from __future__ import annotations

import io
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from rideguard import (
    FEATURES, TARGET_ORDER, get_permutation_importance, make_demo_data,
    prepare_dataset, train_evaluate,
)

st.set_page_config(
    page_title="RideGuard AI | Mobility Risk Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "bg": "#0A1020", "panel": "#111B2E", "panel2": "#17243B",
    "border": "#273751", "text": "#F3F6FC", "muted": "#9BAAC2",
    "teal": "#35D0BA", "blue": "#70A7FF", "amber": "#F5BE5B",
    "coral": "#FF7C83", "purple": "#B6A0FF", "green": "#67D9A5",
}
st.markdown(f"""
<style>
.stApp {{background:{PALETTE['bg']}; color:{PALETTE['text']};}}
[data-testid="stHeader"] {{background:{PALETTE['bg']};}}
.block-container {{max-width:1500px; padding-top:1.2rem; padding-bottom:2.5rem;}}
[data-testid="stSidebar"] {{background:{PALETTE['panel']}; border-right:1px solid {PALETTE['border']};}}
[data-testid="stMetric"] {{background:{PALETTE['panel']}; border:1px solid {PALETTE['border']}; padding:13px; border-radius:14px;}}
[data-testid="stMetricLabel"] p {{color:{PALETTE['muted']} !important;}}
[data-testid="stMetricValue"] {{color:{PALETTE['text']} !important;}}
[data-testid="stVerticalBlockBorderWrapper"] {{background:{PALETTE['panel']}; border:1px solid {PALETTE['border']}; border-radius:15px;}}
[data-testid="stTabs"] button {{color:{PALETTE['muted']} !important;}}
[data-testid="stTabs"] button[aria-selected="true"] {{color:{PALETTE['teal']} !important; border-bottom-color:{PALETTE['teal']} !important;}}
.stButton button, .stDownloadButton button {{border-radius:10px; border:1px solid {PALETTE['border']};}}
div[data-testid="stFileUploaderDropzone"] {{background:{PALETTE['panel2']}; border:1px dashed {PALETTE['border']}; border-radius:12px;}}
.hero {{padding:1.55rem 1.8rem; border:1px solid {PALETTE['border']}; border-radius:20px;
background:linear-gradient(120deg,#14223B,#111A2D 65%,#132A38); margin-bottom:1.15rem;}}
.eyebrow {{font-size:.72rem; letter-spacing:.16em; color:{PALETTE['teal']}; font-weight:750;}}
.hero h1 {{font-size:2.25rem; letter-spacing:-.04em; margin:.35rem 0 .25rem; color:#fff;}}
.hero p {{color:{PALETTE['muted']}; margin:0;}}
.pill {{display:inline-block; border:1px solid {PALETTE['border']}; border-radius:999px; padding:.22rem .65rem; color:{PALETTE['muted']}; font-size:.75rem;}}
.small-muted {{color:{PALETTE['muted']}; font-size:.84rem;}}
.risk-high {{color:{PALETTE['coral']}; font-weight:800;}}
.risk-mid {{color:{PALETTE['amber']}; font-weight:800;}}
.risk-low {{color:{PALETTE['green']}; font-weight:800;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def demo_data():
    return make_demo_data()

@st.cache_data(show_spinner=False)
def parse_uploaded(raw: bytes):
    return pd.read_csv(io.BytesIO(raw))

def chart_theme(fig, height=330):
    fig.update_layout(
        template="plotly_dark", height=height,
        margin=dict(l=8, r=8, t=35, b=8),
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
        yaxis=dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"]),
    )
    return fig

def outcome_flags(series: pd.Series) -> pd.Series:
    return series.astype(str).ne("Completed")

def show_header():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">MOBILITY OPERATIONS · PREDICTIVE INTELLIGENCE</div>
      <h1>🚕 RideGuard AI</h1>
      <p>Predict booking risk. Understand route patterns. Make smarter operational decisions.</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("## 🚕 RIDEGUARD AI")
st.sidebar.caption("MOBILITY RISK & OPERATIONS INTELLIGENCE")
source_choice = st.sidebar.radio("DATA SOURCE", ["Synthetic demo data", "Upload my ride data"])
uploaded = None
if source_choice == "Upload my ride data":
    uploaded = st.sidebar.file_uploader("Upload ride bookings CSV", type=["csv"])
    if uploaded is None:
        st.sidebar.info("Upload a CSV to activate analytics on your data.")

if source_choice == "Synthetic demo data":
    raw_df = demo_data()
    synthetic = True
elif uploaded is not None:
    try:
        raw_df = parse_uploaded(uploaded.getvalue())
        synthetic = False
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    st.markdown("## Welcome to RideGuard AI")
    st.info("Choose Synthetic demo data or upload your NCR ride-booking CSV in the sidebar.")
    st.stop()

try:
    X, y, source_df = prepare_dataset(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if synthetic:
    st.info("Demo mode: this dataset is synthetically generated for product demonstration. It is not real NCR ride-booking history and its model scores are not evidence of real-world performance.")

min_class = int(y.value_counts().min()) if not y.empty else 0
if len(y) < 30 or y.nunique() < 2 or min_class < 2:
    st.error("This dataset does not contain enough labelled booking outcomes to train a reliable demonstration model.")
    st.stop()

with st.sidebar:
    st.markdown("---")
    page = st.radio("NAVIGATION", [
        "Command Center", "Risk Lab", "Route Intelligence",
        "Model Diagnostics", "Data Explorer",
    ])
    st.markdown("---")
    st.caption(f"{len(source_df):,} usable labelled bookings")
    st.caption("Prediction features exclude final status, cancellation reasons, ratings and post-trip fields.")

@st.cache_resource(show_spinner="Training the selected model...")
def train_cached(data_hash: str, model_name: str, X_data: pd.DataFrame, y_data: pd.Series):
    return train_evaluate(X_data, y_data, model_name)

model_name = st.sidebar.selectbox("MODEL", ["Random Forest", "Logistic Regression"])
raw_hash = hashlib.sha256(raw_df.to_csv(index=False).encode("utf-8")).hexdigest()
try:
    result = train_cached(raw_hash, model_name, X, y)
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

predicted = pd.Series(result.predictions, index=result.y_test.index)
failed_actual = outcome_flags(result.y_test)
failure_rate = float(failed_actual.mean() * 100)
completion_rate = float((y == "Completed").mean() * 100)
routes_df = source_df.copy()
routes_df["_target"] = y.values
pickup_col = next((c for c in routes_df.columns if c.lower().strip() in {"pickup location", "pickup"}), None)
drop_col = next((c for c in routes_df.columns if c.lower().strip() in {"drop location", "destination"}), None)

show_header()
if page == "Command Center":
    st.subheader("Network performance snapshot")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Bookings analysed", f"{len(source_df):,}")
    c2.metric("Completion rate", f"{completion_rate:.1f}%")
    c3.metric("Unsuccessful outcomes", f"{failure_rate:.1f}%")
    c4.metric("Balanced accuracy", f"{result.balanced_accuracy:.1%}")
    c5.metric("Macro F1", f"{result.macro_f1:.1%}")
    st.markdown('<span class="pill">MODEL: '+model_name.upper()+'</span>  <span class="pill">LEAKAGE-SAFE FEATURE SET</span>', unsafe_allow_html=True)
    left,right = st.columns([1,1.35])
    with left:
        st.markdown("#### Booking outcome mix")
        counts = y.value_counts().reindex(TARGET_ORDER, fill_value=0).rename_axis("Outcome").reset_index(name="Bookings")
        fig = px.pie(counts, names="Outcome", values="Bookings", hole=.62,
                     color="Outcome", color_discrete_map={
                         "Completed":PALETTE["teal"], "Cancelled":PALETTE["coral"],
                         "Incomplete":PALETTE["amber"], "No Driver Found":PALETTE["purple"]})
        fig.update_traces(textposition="inside", textinfo="percent")
        st.plotly_chart(chart_theme(fig,350), use_container_width=True)
    with right:
        st.markdown("#### Prediction performance by outcome")
        report = pd.DataFrame(result.report).T.reindex([x for x in result.labels])
        report = report[["precision","recall","f1-score","support"]].reset_index(names="Outcome")
        fig = px.bar(report, x="Outcome", y=["precision","recall","f1-score"], barmode="group",
                     color_discrete_sequence=[PALETTE["blue"],PALETTE["teal"],PALETTE["purple"]])
        fig.update_yaxes(range=[0,1])
        st.plotly_chart(chart_theme(fig,350), use_container_width=True)
    st.markdown("#### What the model is allowed to know")
    st.caption("Booking hour, weekday, month, vehicle type, pickup location and drop location. Final status, cancellation reason, driver/customer ratings, arrival times and completion times are excluded to reduce target leakage.")

elif page == "Risk Lab":
    st.subheader("New booking risk simulator")
    st.write("Change the booking conditions to estimate the historical outcome pattern learned by the selected model.")
    vehicle_values = sorted(X["Vehicle Type"].dropna().astype(str).unique())
    pickup_values = sorted(X["Pickup Location"].dropna().astype(str).unique())
    drop_values = sorted(X["Drop Location"].dropna().astype(str).unique())
    with st.form("scenario_form"):
        a,b,c = st.columns(3)
        with a:
            booking_date = st.date_input("Booking date", value=pd.Timestamp.today().date())
            booking_time = st.time_input("Booking time", value=pd.Timestamp("2025-01-01 18:30").time())
        with b:
            vehicle = st.selectbox("Vehicle type", vehicle_values)
            pickup = st.selectbox("Pickup location", pickup_values)
        with c:
            drop = st.selectbox("Drop location", drop_values)
        submitted = st.form_submit_button("Estimate booking risk", type="primary", use_container_width=True)
    if submitted:
        scenario = pd.DataFrame([{
            "Hour": booking_time.hour, "Weekday": pd.Timestamp(booking_date).day_name(),
            "Month": pd.Timestamp(booking_date).month_name(), "Vehicle Type":vehicle,
            "Pickup Location":pickup, "Drop Location":drop,
        }], columns=FEATURES)
        probabilities = result.model.predict_proba(scenario)[0]
        classes = result.model.named_steps["classifier"].classes_
        prob = dict(zip(classes, probabilities))
        completion_prob = float(prob.get("Completed", 0))
        risk = max(0.0, min(1.0, 1 - completion_prob))
        label = "HIGH" if risk >= .55 else "MODERATE" if risk >= .30 else "LOW"
        color = PALETTE["coral"] if label == "HIGH" else PALETTE["amber"] if label == "MODERATE" else PALETTE["green"]
        st.markdown(f'<div class="panel"><div class="small-muted">ESTIMATED UNSUCCESSFUL-OUTCOME RISK</div><div style="font-size:2.7rem;font-weight:850;color:{color}">{risk:.1%}</div><div style="font-weight:750;color:{color}">{label} RISK</div><div class="small-muted">Estimated completion probability: {completion_prob:.1%}</div></div>', unsafe_allow_html=True)
        probs = pd.DataFrame({"Outcome":list(prob.keys()),"Estimated probability":list(prob.values())}).sort_values("Estimated probability",ascending=False)
        fig = px.bar(probs, x="Estimated probability", y="Outcome", orientation="h",
                     color="Outcome", color_discrete_map={
                         "Completed":PALETTE["teal"],"Cancelled":PALETTE["coral"],
                         "Incomplete":PALETTE["amber"],"No Driver Found":PALETTE["purple"]})
        fig.update_xaxes(tickformat=".0%", range=[0,1])
        st.plotly_chart(chart_theme(fig,300), use_container_width=True)
        st.caption("This is a model estimate based on historical patterns, not a guarantee or a causal explanation. Unsuccessful risk = 1 − estimated probability of Completed.")
    else:
        st.info("Submit a booking scenario to see the predicted outcome distribution.")

elif page == "Route Intelligence":
    st.subheader("Route-level operational signals")
    if pickup_col and drop_col:
        route_data = pd.DataFrame({
            "Pickup": routes_df[pickup_col].astype("string").fillna("Unknown"),
            "Drop": routes_df[drop_col].astype("string").fillna("Unknown"),
            "Outcome": y.values,
        })
        route_data["Route"] = route_data["Pickup"] + " → " + route_data["Drop"]
        route_summary = route_data.groupby("Route").agg(
            Bookings=("Outcome","size"),
            Completed=("Outcome",lambda s:(s=="Completed").sum()),
            Unsuccessful=("Outcome",lambda s:(s!="Completed").sum()),
        ).reset_index()
        route_summary["Unsuccessful rate"] = route_summary["Unsuccessful"] / route_summary["Bookings"]
        min_bookings = st.slider("Minimum bookings per route", 1, max(50,int(route_summary["Bookings"].max())), 5)
        view = route_summary[route_summary["Bookings"] >= min_bookings].sort_values(
            ["Unsuccessful rate","Bookings"], ascending=[False,False]
        ).head(20)
        if view.empty:
            st.warning("No routes meet the selected minimum booking count.")
        else:
            fig = px.bar(view.sort_values("Unsuccessful rate"), x="Unsuccessful rate", y="Route",
                         orientation="h", color="Unsuccessful rate", color_continuous_scale="Tealrose",
                         hover_data=["Bookings","Completed","Unsuccessful"])
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(chart_theme(fig,500), use_container_width=True)
            st.dataframe(view, use_container_width=True, hide_index=True,
                         column_config={"Unsuccessful rate":st.column_config.NumberColumn(format="%.1%%")})
            st.download_button("Download route risk summary", route_summary.to_csv(index=False).encode(),
                               "rideguard_route_summary.csv", "text/csv")
            st.caption("Rates from small samples are unstable. Use the minimum-booking filter before interpreting route comparisons.")
    else:
        st.warning("Pickup and drop columns were not detected.")

elif page == "Model Diagnostics":
    st.subheader("Model quality & explainability")
    m1,m2,m3 = st.columns(3)
    m1.metric("Accuracy", f"{result.accuracy:.1%}")
    m2.metric("Balanced accuracy", f"{result.balanced_accuracy:.1%}")
    m3.metric("Macro F1", f"{result.macro_f1:.1%}")
    st.caption("Metrics are calculated on a stratified held-out test set. Compare balanced accuracy and per-class recall when outcomes are imbalanced.")
    cm = go.Figure(data=go.Heatmap(
        z=result.matrix, x=result.labels, y=result.labels,
        colorscale="Teal", text=result.matrix, texttemplate="%{text}",
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
    ))
    cm.update_layout(title="Confusion matrix · actual vs predicted", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(chart_theme(cm,390), use_container_width=True)
    report = pd.DataFrame(result.report).T.reindex(result.labels)
    st.dataframe(report[["precision","recall","f1-score","support"]], use_container_width=True)
    with st.spinner("Calculating permutation importance on a test sample..."):
        importance = get_permutation_importance(result)
    fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                 error_x="Std", color="Importance", color_continuous_scale="Teal")
    st.markdown("#### Global feature importance")
    st.plotly_chart(chart_theme(fig,320), use_container_width=True)
    st.caption("Permutation importance measures the drop in held-out balanced accuracy when one feature is shuffled. It describes model reliance, not causality.")

elif page == "Data Explorer":
    st.subheader("Data quality & booking explorer")
    a,b,c = st.columns(3)
    a.metric("Rows loaded",f"{len(raw_df):,}")
    b.metric("Rows with recognized outcomes",f"{len(source_df):,}")
    c.metric("Unmapped / missing outcomes",f"{len(raw_df)-len(source_df):,}")
    st.markdown("#### Outcome distribution")
    st.dataframe(y.value_counts().reindex(TARGET_ORDER,fill_value=0).rename("Bookings").to_frame(),use_container_width=True)
    st.markdown("#### Preview")
    st.dataframe(source_df.head(100),use_container_width=True,hide_index=True)
    st.download_button("Download cleaned labelled rows",source_df.to_csv(index=False).encode(),
                       "rideguard_cleaned_bookings.csv","text/csv")

st.markdown("---")
st.markdown('<div class="small-muted">RideGuard AI · Built by Preeti Dagar · Portfolio demonstration. Predictions are estimates, not operational guarantees.</div>',unsafe_allow_html=True)
