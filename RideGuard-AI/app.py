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
    initial_sidebar_state="collapsed",
)

DARK_PALETTE = {
    "bg": "#0A1020", "panel": "#111B2E", "panel2": "#17243B",
    "border": "#273751", "text": "#F3F6FC", "muted": "#9BAAC2",
    "teal": "#35D0BA", "blue": "#70A7FF", "amber": "#F5BE5B",
    "coral": "#FF7C83", "purple": "#B6A0FF", "green": "#67D9A5",
}
LIGHT_PALETTE = {
    "bg": "#F3F6FB", "panel": "#FFFFFF", "panel2": "#EAF0F7",
    "border": "#D6DFEA", "text": "#172338", "muted": "#64748B",
    "teal": "#078F83", "blue": "#316FC5", "amber": "#A96B08",
    "coral": "#D94D5A", "purple": "#7454C8", "green": "#16845C",
}
PALETTE = DARK_PALETTE
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

def chart_theme(fig, height=285):
    axis_style = dict(
        gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"],
        tickfont=dict(color=PALETTE["text"], size=10),
        title_font=dict(color=PALETTE["text"], size=11),
        linecolor=PALETTE["border"],
    )
    fig.update_layout(
        template="plotly_dark" if theme_mode == "Dark" else "plotly_white",
        height=height, autosize=True,
        margin=dict(l=10, r=12, t=28, b=12),
        paper_bgcolor=PALETTE["panel"], plot_bgcolor=PALETTE["panel"],
        font=dict(color=PALETTE["text"], size=11),
        title_font=dict(color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["text"], size=10)),
        xaxis=axis_style, yaxis=axis_style,
        transition=dict(duration=650, easing="cubic-in-out"),
    )
    fig.update_annotations(font=dict(color=PALETTE["text"]))
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

theme_mode = st.sidebar.radio("APPEARANCE", ["Dark", "Light"], horizontal=True, index=0)
PALETTE = DARK_PALETTE if theme_mode == "Dark" else LIGHT_PALETTE

st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"] {{background:{PALETTE["bg"]};color:{PALETTE["text"]};}}
[data-testid="stSidebar"],[data-testid="stSidebar"] > div {{background:{PALETTE["panel"]};color:{PALETTE["text"]};}}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{color:{PALETTE["text"]} !important;}}
[data-testid="stHeader"] {{background:{PALETTE["bg"]};}}
[data-testid="stMetric"] {{background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};}}
[data-testid="stMetricLabel"] p {{color:{PALETTE["muted"]} !important;}}
[data-testid="stMetricValue"] {{color:{PALETTE["text"]} !important;}}
[data-testid="stDataFrame"] {{border:1px solid {PALETTE["border"]};border-radius:12px;}}
</style>
""", unsafe_allow_html=True)

st.sidebar.caption("Choose your preferred appearance.")
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
pickup_col = next((c for c in routes_df.columns if c.lower().strip() in {"pickup location", "pickup"}), next((c for c in routes_df.columns if "pickup" in c.lower()), None))
drop_col = next((c for c in routes_df.columns if c.lower().strip() in {"drop location", "destination"}), next((c for c in routes_df.columns if "drop" in c.lower() or "destination" in c.lower()), None))

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
        st.plotly_chart(chart_theme(fig,285), use_container_width=True)
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
        st.markdown(f'<div class="risk-card"><div class="section-kicker">ESTIMATED UNSUCCESSFUL-OUTCOME RISK</div><div style="font-size:clamp(2rem,4vw,3rem);font-weight:850;line-height:1.15;color:{color};margin:.35rem 0">{risk:.1%}</div><div style="font-weight:800;color:{color};letter-spacing:.08em">{label} RISK</div><div class="small-muted" style="margin-top:.45rem">Estimated completion probability: {completion_prob:.1%}</div></div>', unsafe_allow_html=True)
        probs = pd.DataFrame({"Outcome":list(prob.keys()),"Estimated probability":list(prob.values())}).sort_values("Estimated probability",ascending=True)
        left_result, right_result = st.columns([.8,1.4], gap="large")
        with left_result:
            st.markdown("#### Scenario summary")
            st.metric("Completion probability", f"{completion_prob:.1%}")
            st.metric("Unsuccessful-outcome risk", f"{risk:.1%}")
            st.caption("Risk = 1 − probability of Completed.")
        with right_result:
            st.markdown("#### Predicted outcome distribution")
            fig = px.bar(probs, x="Estimated probability", y="Outcome", orientation="h",
                         color="Outcome", color_discrete_map={
                             "Completed":PALETTE["teal"],"Cancelled":PALETTE["coral"],
                             "Incomplete":PALETTE["amber"],"No Driver Found":PALETTE["purple"]})
            fig.update_xaxes(tickformat=".0%", range=[0,1], title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(chart_theme(fig,235), use_container_width=True)
        st.caption("Model estimates reflect patterns in the selected dataset; they are not guarantees or causal explanations.")
    else:
        st.info("Submit a booking scenario to see the predicted outcome distribution.")

elif page == "Route Intelligence":
    st.subheader("Mobility network map")
    st.caption("Explore demand corridors and historical unsuccessful-outcome signals across your ride network.")

    # Approximate NCR area centroids for the built-in demo. Uploaded GPS columns take priority.
    ncr_coords = {
        "Connaught Place":(28.6315,77.2167),"Dwarka":(28.5921,77.0460),"Gurugram":(28.4595,77.0266),
        "Noida":(28.5355,77.3910),"Saket":(28.5245,77.2066),"Rohini":(28.7495,77.0565),
        "Karol Bagh":(28.6514,77.1907),"IGI Airport":(28.5562,77.1000),"Greater Noida":(28.4744,77.5040),
        "Lajpat Nagar":(28.5677,77.2433),"Faridabad":(28.4089,77.3178),"Ghaziabad":(28.6692,77.4538),
        "Indirapuram":(28.6415,77.3715),"Vaishali":(28.6450,77.3390),"Vasant Kunj":(28.5200,77.1580),
        "Janakpuri":(28.6219,77.0878),"Pitampura":(28.7033,77.1322),"Vikaspuri":(28.6380,77.0700),
        "South Delhi":(28.5350,77.2200),"Delhi":(28.6139,77.2090),
    }
    route_data = pd.DataFrame({
        "Pickup":routes_df[pickup_col].astype("string").fillna("Unknown"),
        "Drop":routes_df[drop_col].astype("string").fillna("Unknown"),
        "Outcome":y.values,
    })
    vehicle_col = next((c for c in routes_df.columns if c.lower().strip() in {"vehicle type","vehicle"}),None)
    if vehicle_col:
        route_data["Vehicle Type"] = routes_df[vehicle_col].astype("string").fillna("Unknown").values

    cols = {str(c).strip().lower().replace("_"," "):c for c in routes_df.columns}
    plat = next((cols[k] for k in ["pickup latitude","pickup lat","origin latitude","origin lat"] if k in cols),None)
    plon = next((cols[k] for k in ["pickup longitude","pickup lon","pickup lng","origin longitude","origin lon"] if k in cols),None)
    dlat = next((cols[k] for k in ["drop latitude","drop lat","destination latitude","destination lat"] if k in cols),None)
    dlon = next((cols[k] for k in ["drop longitude","drop lon","drop lng","destination longitude","destination lon"] if k in cols),None)
    gps = all([plat,plon,dlat,dlon])

    f1,f2 = st.columns(2)
    with f1:
        min_bookings = st.number_input("Minimum bookings per corridor",min_value=1,max_value=max(1,len(route_data)),value=min(5,len(route_data)),step=1)
    with f2:
        vehicle_options = ["All vehicles"] + sorted(route_data["Vehicle Type"].dropna().astype(str).unique().tolist()) if "Vehicle Type" in route_data else ["All vehicles"]
        vehicle_filter = st.selectbox("Vehicle type",vehicle_options)

    filtered = route_data.copy()
    if vehicle_filter != "All vehicles" and "Vehicle Type" in filtered:
        filtered = filtered[filtered["Vehicle Type"] == vehicle_filter]
    filtered["Route"] = filtered["Pickup"] + " → " + filtered["Drop"]
    summary = filtered.groupby(["Pickup","Drop","Route"]).agg(
        Bookings=("Outcome","size"),
        Completed=("Outcome",lambda v:(v=="Completed").sum()),
        Unsuccessful=("Outcome",lambda v:(v!="Completed").sum()),
    ).reset_index()
    summary["Unsuccessful rate"] = summary["Unsuccessful"] / summary["Bookings"]
    view = summary[summary["Bookings"] >= min_bookings].sort_values(["Bookings","Unsuccessful rate"],ascending=[False,False]).head(35)

    if view.empty:
        st.warning("No corridors meet this booking threshold. Reduce the minimum bookings filter.")
    else:
        node_rows = []
        if gps:
            for side,loc,latcol,loncol in [("Pickup","Pickup",plat,plon),("Drop","Drop",dlat,dlon)]:
                temp = pd.DataFrame({"Location":routes_df.loc[filtered.index,loc].astype(str),
                                     "Latitude":pd.to_numeric(routes_df.loc[filtered.index,latcol],errors="coerce"),
                                     "Longitude":pd.to_numeric(routes_df.loc[filtered.index,loncol],errors="coerce")}).dropna()
                node_rows.extend(temp.to_dict("records"))
        if node_rows:
            nodes = pd.DataFrame(node_rows).groupby("Location").agg(
                Latitude=("Latitude","mean"),Longitude=("Longitude","mean")).reset_index()
        else:
            names = pd.unique(pd.concat([view["Pickup"],view["Drop"]],ignore_index=True))
            nodes = pd.DataFrame([{"Location":str(n),"Latitude":ncr_coords[str(n)][0],"Longitude":ncr_coords[str(n)][1]}
                                  for n in names if str(n) in ncr_coords])

        if nodes.empty:
            st.info("No map coordinates are available for these areas. Add pickup/drop latitude and longitude columns to map this dataset.")
        else:
            nodes["Bookings"] = nodes["Location"].apply(lambda n:int(view.loc[(view["Pickup"]==n)|(view["Drop"]==n),"Bookings"].sum()))
            nodes["Unsuccessful rate"] = nodes["Location"].apply(
                lambda n: float(view.loc[(view["Pickup"]==n)|(view["Drop"]==n),"Unsuccessful"].sum() /
                                max(1,view.loc[(view["Pickup"]==n)|(view["Drop"]==n),"Bookings"].sum()))
            )
            map_fig = go.Figure()
            line_colors = {"LOW":PALETTE["teal"],"MODERATE":PALETTE["amber"],"HIGH":PALETTE["coral"]}
            for _,r in view.iterrows():
                a=nodes[nodes["Location"].astype(str)==str(r["Pickup"])]
                b=nodes[nodes["Location"].astype(str)==str(r["Drop"])]
                if a.empty or b.empty: continue
                rate=float(r["Unsuccessful rate"])
                risk="HIGH" if rate>=.35 else "MODERATE" if rate>=.20 else "LOW"
                map_fig.add_trace(go.Scattermapbox(
                    lat=[a.iloc[0]["Latitude"],b.iloc[0]["Latitude"]],
                    lon=[a.iloc[0]["Longitude"],b.iloc[0]["Longitude"]],
                    mode="lines",line=dict(width=float(np.clip(2+np.log1p(r["Bookings"]),2,7)),color=line_colors[risk]),
                    opacity=.68,hoverinfo="text",
                    text=[f'{r["Route"]}<br>{int(r["Bookings"]):,} bookings<br>{rate:.1%} unsuccessful']*2,
                    showlegend=False,
                ))
            node_colors=[PALETTE["coral"] if x>=.35 else PALETTE["amber"] if x>=.20 else PALETTE["teal"] for x in nodes["Unsuccessful rate"]]
            map_fig.add_trace(go.Scattermapbox(
                lat=nodes["Latitude"],lon=nodes["Longitude"],mode="markers+text",
                text=nodes["Location"],textposition="top right",
                marker=dict(size=np.clip(12+np.log1p(nodes["Bookings"])*2,13,25),color=node_colors,opacity=.98),
                customdata=np.column_stack([nodes["Bookings"],nodes["Unsuccessful rate"]]),
                hovertemplate="<b>%{text}</b><br>Booking volume: %{customdata[0]:,.0f}<br>Unsuccessful rate: %{customdata[1]:.1%}<extra></extra>",
                name="Pickup / drop areas",
            ))
            map_fig.update_layout(
                mapbox=dict(style="carto-darkmatter" if theme_mode=="Dark" else "carto-positron",
                            center={"lat":float(nodes["Latitude"].mean()),"lon":float(nodes["Longitude"].mean())},zoom=8.5),
                height=485,autosize=True,margin=dict(l=0,r=0,t=5,b=0),
                paper_bgcolor=PALETTE["panel"],font=dict(color=PALETTE["text"]),
            )
            st.plotly_chart(map_fig,use_container_width=True,config={"displayModeBar":False,"scrollZoom":True})
            st.markdown(
                f'<span class="pill">CORRIDORS: {len(view)}</span>  <span class="pill">MAPPED AREAS: {len(nodes)}</span>  <span class="pill">VEHICLE: {vehicle_filter}</span>',
                unsafe_allow_html=True,
            )
            st.caption("Area pins use approximate centroids when GPS is unavailable. Lines are schematic pickup-to-drop connections, not actual road paths or live vehicle locations.")

        st.markdown("#### Highest-risk corridors")
        left,right=st.columns([1.2,1])
        with left:
            fig=px.bar(view.sort_values("Unsuccessful rate").tail(12),x="Unsuccessful rate",y="Route",
                       orientation="h",color="Unsuccessful rate",color_continuous_scale="Tealrose",
                       hover_data=["Bookings","Completed","Unsuccessful"])
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(chart_theme(fig,300),use_container_width=True)
        with right:
            st.dataframe(view[["Route","Bookings","Unsuccessful rate"]].head(15),use_container_width=True,hide_index=True,
                         column_config={"Unsuccessful rate":st.column_config.NumberColumn(format="%.1f%%")})
        st.download_button("Download corridor risk summary",summary.to_csv(index=False).encode(),
                           "rideguard_corridor_summary.csv","text/csv")
    st.caption("Rates from small samples are unstable. Use the booking threshold before comparing corridors.")

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
    st.plotly_chart(chart_theme(cm,320), use_container_width=True)
    report = pd.DataFrame(result.report).T.reindex(result.labels)
    st.dataframe(report[["precision","recall","f1-score","support"]], use_container_width=True)
    with st.spinner("Calculating permutation importance on a test sample..."):
        importance = get_permutation_importance(result)
    fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                 error_x="Std", color="Importance", color_continuous_scale="Teal")
    st.markdown("#### Global feature importance")
    st.plotly_chart(chart_theme(fig,275), use_container_width=True)
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

# Responsive styling overrides
st.markdown(f"""
<style>
:root {{ color-scheme: {theme_mode.lower()}; }}
.stApp {{background:{PALETTE['bg']}; color:{PALETTE['text']};}}
[data-testid="stHeader"] {{background:{PALETTE["bg"]};}}
.block-container {{max-width:1600px; padding:clamp(.65rem,1.6vw,1.45rem) clamp(.8rem,2.1vw,2rem) 2rem;}}
[data-testid="stSidebar"] {{background:{PALETTE['panel']}; border-right:1px solid {PALETTE['border']};}}
[data-testid="stSidebar"] > div {{padding-top:1rem;}}
/* Theme-aware sidebar typography and controls */
[data-testid="stSidebar"], [data-testid="stSidebar"] * {{
  color:{PALETTE["text"]};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  color:{PALETTE["muted"]} !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] input {{
  color:{PALETTE["text"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {{
  gap:.45rem;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{
  color:{PALETTE["text"]} !important;
}}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{
  background:{PALETTE["panel2"]};
  border:1px dashed {PALETTE["border"]};
}}

[data-testid="stMetric"] {{background:linear-gradient(145deg,{PALETTE['panel']},{PALETTE['panel2']}); border:1px solid {PALETTE['border']}; padding:clamp(9px,1vw,15px); border-radius:14px; min-width:0;}}
[data-testid="stMetricLabel"] p {{color:{PALETTE['muted']} !important; font-size:.78rem;}}
[data-testid="stMetricValue"] {{color:{PALETTE['text']} !important; font-size:clamp(1.15rem,2vw,1.8rem);}}
[data-testid="stVerticalBlockBorderWrapper"] {{background:{PALETTE['panel']}; border:1px solid {PALETTE['border']}; border-radius:15px;}}
[data-testid="stTabs"] button {{color:{PALETTE['muted']} !important;}}
[data-testid="stTabs"] button[aria-selected="true"] {{color:{PALETTE['teal']} !important; border-bottom-color:{PALETTE['teal']} !important;}}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {{border-radius:11px; border:1px solid {PALETTE['border']}; min-height:2.65rem; font-weight:650;}}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {{background:linear-gradient(100deg,{PALETTE['teal']},#27B9B0); color:#061421; border:0;}}
div[data-testid="stFileUploaderDropzone"] {{background:{PALETTE['panel2']}; border:1px dashed {PALETTE['border']}; border-radius:12px;}}
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {{border-radius:10px; background:{PALETTE['panel2']};}}
[data-testid="stDataFrame"], [data-testid="stTable"] {{border:1px solid {PALETTE['border']}; border-radius:12px; overflow:hidden;}}
.hero {{position:relative; isolation:isolate; overflow:hidden; padding:clamp(1rem,2vw,1.65rem) clamp(1rem,2.3vw,2rem); border:1px solid {PALETTE['border']}; border-radius:20px;
background:radial-gradient(circle at 92% 10%,rgba(53,208,186,.13),transparent 30%),linear-gradient(120deg,{PALETTE["panel2"]},{PALETTE["panel"]} 65%,{PALETTE["bg"]}); margin-bottom:.7rem;}}
.hero::after {{content:"";position:absolute;z-index:0;width:190px;height:190px;right:4%;top:-92px;border-radius:50%;background:radial-gradient(circle,rgba(53,208,186,.24),rgba(53,208,186,0) 70%);animation:rgFloat 7s ease-in-out infinite alternate;pointer-events:none;}}
.hero::before {{content:"";position:absolute;z-index:0;width:9px;height:9px;right:20%;bottom:24px;border-radius:50%;background:{PALETTE["teal"]};box-shadow:0 0 18px {PALETTE["teal"]},-85px -36px 0 -2px {PALETTE["blue"]},-145px 12px 0 -3px {PALETTE["purple"]};animation:rgDrift 5s ease-in-out infinite alternate;pointer-events:none;}}
.hero .eyebrow,.hero h1,.hero p {{position:relative;z-index:1;}}
@keyframes rgDrift {{from {{transform:translate3d(0,0,0);}} to {{transform:translate3d(-16px,-10px,0);}}}}
.hero::after {{content:"";position:absolute;z-index:-1;width:210px;height:210px;right:-50px;top:-95px;border-radius:50%;background:radial-gradient(circle,rgba(53,208,186,.22),transparent 70%);animation:rgFloat 9s ease-in-out infinite alternate;}}
@keyframes rgFloat {{from {{transform:translate(0,0) scale(.9);}} to {{transform:translate(-16px,18px) scale(1.12);}}}}
@keyframes rgEnter {{from {{opacity:0;transform:translateY(8px);}} to {{opacity:1;transform:translateY(0);}}}}
.hero {{animation:rgEnter .6s ease-out both;}}
[data-testid="stMetric"],[data-testid="stVerticalBlockBorderWrapper"] {{transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;}}
[data-testid="stMetric"]:hover,[data-testid="stVerticalBlockBorderWrapper"]:hover {{transform:translateY(-2px);border-color:{PALETTE['teal']};box-shadow:0 8px 24px rgba(0,0,0,.08);}}
.stButton button:hover,.stDownloadButton button:hover {{transform:translateY(-1px);transition:all .18s ease;}}
.eyebrow {{font-size:.68rem; letter-spacing:.16em; color:{PALETTE['teal']}; font-weight:800;}}
.hero h1 {{font-size:clamp(1.55rem,3vw,2.2rem); letter-spacing:-.045em; margin:.35rem 0 .25rem; color:{PALETTE["text"]}; line-height:1.15;}}
.hero p {{color:{PALETTE['muted']}; margin:0; font-size:clamp(.88rem,1.2vw,1rem);}}
.pill {{display:inline-block; border:1px solid {PALETTE['border']}; border-radius:999px; padding:.22rem .65rem; color:{PALETTE['muted']}; font-size:.72rem; margin:.2rem .25rem .2rem 0;}}
.small-muted {{color:{PALETTE['muted']}; font-size:.82rem;}}
.risk-card {{padding:1.1rem 1.25rem; border:1px solid {PALETTE['border']}; border-radius:16px; background:linear-gradient(140deg,{PALETTE['panel2']},{PALETTE['panel']}); height:100%;}}
.section-kicker {{color:{PALETTE['teal']};font-size:.68rem;letter-spacing:.12em;font-weight:800;}}
div[data-testid="stAlert"] {{border-radius:12px;}}
h2, h3 {{letter-spacing:-.025em;}}
@media (max-width: 900px) {{
  .block-container {{padding-left:1rem;padding-right:1rem;}}
  [data-testid="stHorizontalBlock"] {{gap:.7rem;}}
}}
@media (max-width: 640px) {{
  .block-container {{padding:.6rem .7rem 1.25rem;}}
  .hero {{border-radius:15px;margin-bottom:.55rem;padding:1rem 1.1rem;}}
  .hero h1 {{font-size:1.55rem;}}\n  [data-testid="stPlotlyChart"] {{min-height:0 !important;}}
  [data-testid="stHorizontalBlock"] {{flex-direction:column;gap:.5rem;}}
  [data-testid="column"] {{width:100% !important;flex:1 1 100% !important;}}
  [data-testid="stMetric"] {{padding:9px 11px;}}
  [data-testid="stMetricValue"] {{font-size:1.25rem;}}
  .stButton button, .stDownloadButton button {{width:100%;}}
}}
@media (max-height: 780px) and (min-width: 641px) {{
  .block-container {{padding-top:.55rem;padding-bottom:1rem;}}
  .hero {{padding:1rem 1.35rem;margin-bottom:.7rem;}}
  .hero h1 {{font-size:1.8rem;}}
  h2 {{font-size:1.35rem !important;}}
}}
</style>
""", unsafe_allow_html=True)
