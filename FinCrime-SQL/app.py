from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FinCrime SQL | Transaction Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# -------------------------------------------------------------------
# Deterministic synthetic dataset. No real customer or financial data.
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def make_demo_data(seed: int = 42, n: int = 1800):
    rng = np.random.default_rng(seed)
    now = pd.Timestamp("2026-10-31 23:59:00")
    customer_ids = [f"C{i:04d}" for i in range(1, 181)]
    names = [f"Demo Customer {i:03d}" for i in range(1, 181)]
    countries = np.array(["IN", "GB", "US", "AE", "SG"])
    segments = np.array(["Retail", "SME", "Premium", "Micro-business"])
    tiers = np.array(["Low", "Medium", "High"])
    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_name": names,
        "customer_segment": rng.choice(segments, len(customer_ids), p=[.52,.18,.20,.10]),
        "risk_tier": rng.choice(tiers, len(customer_ids), p=[.60,.31,.09]),
        "home_country": rng.choice(countries, len(customer_ids), p=[.66,.08,.10,.10,.06]),
        "onboarding_date": (now - pd.to_timedelta(rng.integers(60, 2200, len(customer_ids)), unit="D")).date
    })
    accounts = pd.DataFrame({
        "account_id": [f"A{i:05d}" for i in range(1, len(customer_ids)+1)],
        "customer_id": customer_ids,
        "account_type": rng.choice(["Savings","Current","Business"],len(customer_ids),p=[.62,.26,.12]),
        "currency": "INR",
        "opened_date": customers["onboarding_date"],
        "account_status": rng.choice(["Active","Active","Active","Dormant"],len(customer_ids))
    })
    cust_idx = rng.integers(0,len(customer_ids),n)
    base_time = now - pd.to_timedelta(rng.integers(0,90*24*60,n),unit="m")
    amounts = np.round(np.exp(rng.normal(7.25,1.05,n)),2)
    tx = pd.DataFrame({
        "transaction_id":[f"T{i:07d}" for i in range(1,n+1)],
        "account_id":[f"A{i+1:05d}" for i in cust_idx],
        "customer_id":[customer_ids[i] for i in cust_idx],
        "transaction_ts":base_time,
        "amount":amounts,
        "currency":"INR",
        "transaction_type":rng.choice(["CARD_PAYMENT","UPI_TRANSFER","BANK_TRANSFER","CASH_DEPOSIT","CASH_WITHDRAWAL"],n,p=[.36,.30,.18,.09,.07]),
        "channel":rng.choice(["Mobile","Web","ATM","Branch","POS"],n,p=[.34,.20,.10,.08,.28]),
        "merchant_category":rng.choice(["Retail","Travel","Electronics","Grocery","Cash","Gaming","Professional Services"],n),
        "merchant_country":rng.choice(countries,n,p=[.70,.07,.09,.09,.05]),
        "ip_country":rng.choice(countries,n,p=[.72,.06,.08,.09,.05]),
        "device_id":[f"D{v:04d}" for v in rng.integers(1,520,n)],
        "counterparty_id":[f"CP{v:04d}" for v in rng.integers(1,430,n)],
        "synthetic_scenario":"Baseline"
    })
    # Plant inspectable synthetic patterns so each rule can be demonstrated.
    for j in range(5):
        idx = 10+j
        tx.loc[idx,["customer_id","account_id","transaction_ts","amount","transaction_type","synthetic_scenario"]] = [
            "C0007","A00007",pd.Timestamp("2026-10-30 11:00:00")+pd.Timedelta(minutes=2*j),float(4200+j*130),"UPI_TRANSFER","Velocity pattern"]
    for j in range(3):
        idx=30+j
        tx.loc[idx,["customer_id","account_id","transaction_ts","amount","transaction_type","synthetic_scenario"]] = [
            "C0012","A00012",pd.Timestamp("2026-10-29 10:00:00")+pd.Timedelta(hours=j*2),float(9200+j*150),"CASH_DEPOSIT","Near-threshold cash pattern"]
    tx.loc[45,["merchant_country","ip_country","synthetic_scenario"]] = ["SG","IN","Geographic mismatch"]
    tx.loc[46,["amount","synthetic_scenario"]] = [275000.0,"High-value transaction"]
    tx["transaction_ts"] = pd.to_datetime(tx["transaction_ts"])
    tx["amount"] = tx["amount"].astype(float)
    return customers, accounts, tx

customers, accounts, transactions = make_demo_data()
con = duckdb.connect(database=":memory:")
con.register("customers_df", customers)
con.register("accounts_df", accounts)
con.register("transactions_df", transactions)
con.execute("CREATE TABLE customers AS SELECT * FROM customers_df")
con.execute("CREATE TABLE accounts AS SELECT * FROM accounts_df")
con.execute("CREATE TABLE transactions AS SELECT * FROM transactions_df")

# -------------------------------------------------------------------
# SQL detection rules — transparent review signals, not proof of crime.
# -------------------------------------------------------------------
alerts = con.execute("""
WITH velocity AS (
 SELECT transaction_id, customer_id, transaction_ts, amount,
        COUNT(*) OVER (PARTITION BY customer_id ORDER BY transaction_ts
          RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW) AS n10
 FROM transactions
),
cash AS (
 SELECT customer_id, CAST(transaction_ts AS DATE) AS tx_date,
        COUNT(*) AS n_deposits, SUM(amount) AS total_amount
 FROM transactions
 WHERE transaction_type='CASH_DEPOSIT' AND amount BETWEEN 8000 AND 9999.99
 GROUP BY customer_id, CAST(transaction_ts AS DATE)
),
hits AS (
 SELECT transaction_id, customer_id, amount, 'Velocity burst' AS rule_name,
        78 AS risk_score, 'High' AS severity, transaction_ts AS created_at,
        '5 or more transactions within a rolling 10-minute window' AS rationale
 FROM velocity WHERE n10>=5
 UNION ALL
 SELECT t.transaction_id,t.customer_id,t.amount,'Near-threshold cash activity',82,'High',t.transaction_ts,
        'Customer made at least 3 cash deposits in the illustrative near-threshold band on one day'
 FROM transactions t JOIN cash c ON t.customer_id=c.customer_id
   AND CAST(t.transaction_ts AS DATE)=c.tx_date
 WHERE c.n_deposits>=3 AND c.total_amount>=20000
   AND t.transaction_type='CASH_DEPOSIT' AND t.amount BETWEEN 8000 AND 9999.99
 UNION ALL
 SELECT transaction_id,customer_id,amount,'Geographic inconsistency',68,'Medium',transaction_ts,
        'Merchant country and IP country differ; travel, VPNs and data quality can explain this'
 FROM transactions
 WHERE merchant_country<>ip_country AND merchant_country<>'UNKNOWN' AND ip_country<>'UNKNOWN'
 UNION ALL
 SELECT transaction_id,customer_id,amount,'High-value activity',72,'Medium',transaction_ts,
        'Transaction exceeds the demonstration review threshold of INR 100,000'
 FROM transactions WHERE amount>=100000
)
SELECT *, 'Open' AS status FROM hits ORDER BY created_at DESC
""").df()
if not alerts.empty:
    alerts.insert(0,"alert_id",[f"AL-{i:05d}" for i in range(1,len(alerts)+1)])
    alerts["created_at"]=pd.to_datetime(alerts["created_at"])
con.register("alerts_df",alerts)
con.execute("CREATE TABLE alerts AS SELECT * FROM alerts_df")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#08111e;--surface:#101d2d;--surface2:#14243a;--line:#263a52;--text:#e8eff8;--muted:#8fa3bb;--cyan:#39bde8;--blue:#3989d8;--amber:#f4a340;--red:#ef6470;--green:#43c6a5}
.stApp{background:radial-gradient(ellipse at 75% -15%,rgba(33,105,150,.22),transparent 38%),linear-gradient(135deg,#08111e,#0b1727 55%,#0b1b2b);color:var(--text);font-family:'Manrope',sans-serif}
[data-testid="stHeader"]{background:rgba(8,17,30,.94);height:2.3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1727,#0d1d30 70%,#101d2d);border-right:1px solid #243850}
[data-testid="stSidebar"] *{color:#dce8f5!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:6px 9px;border:1px solid #203750;border-radius:8px;margin:2px 0;background:rgba(16,35,55,.5)}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#162b42;border-color:#39bde8}
.block-container{max-width:1600px;padding:.65rem 1.15rem .65rem}
h1,h2,h3,h4{color:#edf4fc!important;font-family:'Manrope',sans-serif!important;letter-spacing:-.025em}
p,li,label,.stCaption{color:#aebed0}
.hero{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:12px;background:linear-gradient(105deg,rgba(19,39,61,.98),rgba(12,30,48,.96));border:1px solid #29425d;border-radius:13px;padding:11px 17px;margin:0 0 9px;box-shadow:0 10px 26px rgba(0,0,0,.2)}
.hero:after{content:"";position:absolute;right:5%;top:-70px;width:160px;height:160px;border:1px solid rgba(57,189,232,.24);border-radius:50%;box-shadow:0 0 0 16px rgba(57,189,232,.035),0 0 0 32px rgba(57,189,232,.025);pointer-events:none}
.eyebrow{font:600 9px 'IBM Plex Mono';letter-spacing:.17em;color:#5dc8ec;text-transform:uppercase}
.hero h1{font-size:24px;margin:2px 0 1px;color:#f0f5fb}
.hero p{font-size:10px;margin:0;color:#9db0c4}
.pill{display:inline-block;padding:3px 8px;border-radius:30px;background:#10243a;border:1px solid #2b4864;font:9px 'IBM Plex Mono';letter-spacing:.06em;color:#9fb5cc;margin:6px 5px 0 0}
.metric{background:linear-gradient(145deg,#14263c,#101e30);border:1px solid #2b435d;border-radius:11px;padding:9px 12px;box-shadow:0 7px 20px rgba(0,0,0,.15)}
.panel{background:linear-gradient(145deg,rgba(18,34,53,.96),rgba(13,27,43,.98));border:1px solid #263d56;border-radius:11px;padding:10px 12px;margin-bottom:8px}
h1,h2,h3{color:#edf4fc!important}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #2a4059;border-radius:8px;overflow:hidden;background:#0e1b2b}
.stButton button,.stDownloadButton button{background:#142942;color:#dceafa;border:1px solid #365571;border-radius:8px;font-weight:600;min-height:31px}
.stButton button[kind="primary"]{background:linear-gradient(100deg,#176f9c,#1686a4);border:1px solid #39bde8;color:#fff}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{background:#0d1c2d!important;color:#eaf2fb!important;border-color:#304861!important}
[data-testid="stPlotlyChart"]{background:linear-gradient(145deg,rgba(16,30,47,.9),rgba(12,25,40,.95));border:1px solid #263d56;border-radius:10px;padding:3px}
div[data-testid="stMetric"]{background:#13253a;border:1px solid #2b435d;border-radius:9px;padding:7px 10px}
div[data-testid="stMetricLabel"] p{color:#92a9c1!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.05em}
div[data-testid="stMetricValue"]{color:#f1f6fc!important;font-family:'IBM Plex Mono'}
div[data-testid="stAlert"]{border-radius:8px}
hr{border-color:#263a52}
footer{visibility:hidden}
@media(max-width:900px){.block-container{padding:.5rem}.hero h1{font-size:20px}}
/* FORENSIC NEON OVERRIDES */
:root{--fc-cyan:#55d9ff;--fc-amber:#ff9b3c}
html,body,[data-testid="stAppViewContainer"],.stApp{background:radial-gradient(ellipse at 55% 42%,rgba(12,55,85,.34),transparent 48%),linear-gradient(135deg,#030812,#071323 58%,#091827)!important;color:#eaf3ff!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#06111f,#0b1c2e)!important;border-right:1px solid #24445f!important}
.block-container{max-width:100%!important;padding:.35rem .85rem .4rem!important}
.hero{display:block;padding:7px 13px;margin-bottom:6px;border-radius:8px;background:linear-gradient(100deg,#0b1c30,#0d2940);border:1px solid #28516d;box-shadow:inset 0 0 22px #1b9ac010,0 8px 24px #0007}
.hero h1{font-size:20px!important;margin:0!important}.hero p{font-size:10px;margin:0}
.hero:after{content:"";position:absolute;right:6%;top:-70px;width:160px;height:160px;border:1px solid #55d9ff44;border-radius:50%;box-shadow:0 0 0 14px #55d9ff0c,0 0 0 28px #55d9ff09,0 0 28px #55d9ff22;animation:fcspin 12s linear infinite}
@keyframes fcspin{to{transform:rotate(360deg)}}
.eyebrow{font:600 9px 'IBM Plex Mono';letter-spacing:.18em;color:#65dfff}
.pill{background:#0b2338;border:1px solid #2c5875;color:#a7d9ec;font:8px 'IBM Plex Mono';padding:3px 7px;border-radius:20px}
.panel,.metric,[data-testid="stMetric"]{background:linear-gradient(145deg,#102238,#081727)!important;border:1px solid #294964!important;border-radius:8px!important;box-shadow:inset 0 0 18px #39bdf00a,0 7px 18px #0005}
[data-testid="stMetric"]{padding:5px 8px!important}
[data-testid="stMetricLabel"] p{font-size:9px!important;color:#91acc4!important;letter-spacing:.08em;text-transform:uppercase}
[data-testid="stMetricValue"]{font:600 18px 'IBM Plex Mono'!important;color:#f2f8ff!important}
[data-testid="stPlotlyChart"]{background:linear-gradient(145deg,#081727,#050d19);border:1px solid #254863;border-radius:9px;padding:2px;box-shadow:0 0 18px #1685b018}
[data-testid="stDataFrame"]{border:1px solid #28465f;border-radius:7px;overflow:hidden}
.stButton button,.stDownloadButton button{background:linear-gradient(100deg,#102e49,#10405a)!important;color:#effaff!important;border:1px solid #3b7796!important;border-radius:7px!important}
.stButton button[kind="primary"]{background:linear-gradient(100deg,#087ca5,#087fa5)!important;border-color:#55d9ff!important;box-shadow:0 0 14px #32bce833}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div,.stNumberInput input{background:#081a2b!important;color:#eff8ff!important;border-color:#31516c!important}
[data-testid="stSidebar"] [data-testid="stRadio"] label{background:#0b2033;border:1px solid #1d3d56;border-radius:7px;padding:5px 8px;margin:2px 0}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{border-color:#55d9ff;box-shadow:0 0 12px #55d9ff22}
h1,h2,h3,h4{color:#f0f7ff!important}p,label,.stCaption{color:#a9bdd1}
[data-testid="stVerticalBlock"]{gap:.45rem}hr{border-color:#20394f}
footer{visibility:hidden}

/* FUTURISTIC POLISH / MICRO-INTERACTIONS */
:root{--fc-cyan:#62e0ff;--fc-blue:#398dff;--fc-amber:#ffad55;--fc-glass:rgba(12,29,48,.82)}
.stApp{background-image:radial-gradient(ellipse at 12% 8%,rgba(42,112,176,.13),transparent 27%),radial-gradient(ellipse at 86% 72%,rgba(20,156,184,.10),transparent 30%),linear-gradient(135deg,#030812 0%,#071323 55%,#081a29 100%)!important}
[data-testid="stMainBlockContainer"]{animation:fc-enter .55s ease-out both}
@keyframes fc-enter{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
.hero{isolation:isolate;min-height:70px;background:linear-gradient(105deg,rgba(9,25,43,.98),rgba(12,39,60,.94) 62%,rgba(8,25,43,.98));border:1px solid rgba(91,203,244,.38);box-shadow:inset 0 1px 0 rgba(255,255,255,.06),inset 0 0 32px rgba(34,160,214,.09),0 8px 26px rgba(0,0,0,.28),0 0 20px rgba(34,170,225,.07)}
.hero:before{content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(110deg,transparent 20%,rgba(85,213,255,.06) 48%,transparent 70%);background-size:220% 100%;animation:fc-sheen 9s ease-in-out infinite}
@keyframes fc-sheen{0%,100%{background-position:120% 0}50%{background-position:-80% 0}}
.hero:after{border-color:rgba(98,224,255,.42);box-shadow:0 0 0 13px rgba(98,224,255,.055),0 0 0 27px rgba(98,224,255,.035),0 0 34px rgba(70,194,255,.22);animation:fc-orbit 18s linear infinite,fc-breathe 4s ease-in-out infinite}
@keyframes fc-orbit{to{transform:rotate(360deg)}}
@keyframes fc-breathe{0%,100%{opacity:.58}50%{opacity:1}}
.panel,.metric,[data-testid="stMetric"]{position:relative;transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}
.panel:hover,.metric:hover,[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:rgba(98,224,255,.68)!important;box-shadow:inset 0 0 22px rgba(57,189,240,.08),0 9px 25px rgba(0,0,0,.30),0 0 17px rgba(57,189,240,.10)!important}
[data-testid="stPlotlyChart"]{position:relative;overflow:hidden;border:1px solid rgba(69,151,194,.42);box-shadow:inset 0 0 26px rgba(45,161,215,.055),0 7px 22px rgba(0,0,0,.22),0 0 15px rgba(35,153,210,.07);transition:border-color .25s ease,box-shadow .25s ease}
[data-testid="stPlotlyChart"]:hover{border-color:rgba(98,224,255,.72);box-shadow:inset 0 0 28px rgba(45,180,235,.08),0 8px 26px rgba(0,0,0,.28),0 0 22px rgba(45,180,235,.13)}
[data-testid="stPlotlyChart"] svg .scatterlayer path.js-line{filter:drop-shadow(0 0 3px rgba(68,197,255,.55))}
[data-testid="stPlotlyChart"] svg .scatterlayer .point{filter:drop-shadow(0 0 4px rgba(74,201,255,.42))}
[data-testid="stSidebar"] [data-testid="stRadio"] label{transition:all .18s ease}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{transform:translateX(3px);background:linear-gradient(90deg,#102d45,#0b2033)}
.stButton button,.stDownloadButton button{transition:all .18s ease;box-shadow:0 3px 10px rgba(0,0,0,.16)}
.stButton button:hover,.stDownloadButton button:hover{transform:translateY(-1px);border-color:#62e0ff!important;box-shadow:0 0 15px rgba(76,205,255,.18)}
[data-testid="stDataFrame"]{box-shadow:0 6px 18px rgba(0,0,0,.16)}
[data-testid="stDataFrame"] th{font-size:11px!important;letter-spacing:.025em}
[data-testid="stDataFrame"] td{font-size:11px!important}
h1,h2,h3,h4{letter-spacing:-.035em}
h2{font-size:1.22rem!important}h3{font-size:1.02rem!important}
.stCaption{font-size:10px!important}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style>
""",unsafe_allow_html=True)

def money(v):
    return f"₹{v:,.0f}"

def chart_theme(fig,height=235):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope",color="#b8cbe0",size=10),margin=dict(l=14,r=14,t=32,b=24),height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=9)),hoverlabel=dict(bgcolor="#10263b",font_color="#f3f8ff"),
        transition=dict(duration=550,easing="cubic-in-out"),uirevision="fincrime-visual")
    fig.update_xaxes(gridcolor="#20354b",tickfont_color="#90a7bf",automargin=True)
    fig.update_yaxes(gridcolor="#20354b",tickfont_color="#90a7bf",automargin=True)
    return fig


def make_network_figure(selected, sample):
    nodes=[selected]
    for node in sample.counterparty_id.tolist()+sample.device_id.tolist():
        if node not in nodes: nodes.append(node)
    nodes=nodes[:22]
    angles=np.linspace(-np.pi/2,3*np.pi/2,len(nodes),endpoint=False)
    pos={node:(float(np.cos(a)),float(np.sin(a))) for node,a in zip(nodes,angles)}
    ex=[];ey=[];hx=[];hy=[]
    for _,r in sample.iterrows():
        target=r.counterparty_id if r.counterparty_id in pos else r.device_id
        if target not in pos: continue
        xx,yy=(hx,hy) if r.synthetic_scenario!="Baseline" else (ex,ey)
        xx.extend([pos[selected][0],pos[target][0],None]); yy.extend([pos[selected][1],pos[target][1],None])
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=8,color="rgba(48,166,235,.10)"),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=1.4,color="rgba(67,190,255,.78)"),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=hx,y=hy,mode="lines",line=dict(width=10,color="rgba(255,137,48,.14)"),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=hx,y=hy,mode="lines",line=dict(width=2.2,color="#ff963c"),hoverinfo="skip",showlegend=False))
    xs=[pos[n][0] for n in nodes];ys=[pos[n][1] for n in nodes]
    cs=["#ff9d3f" if n==selected else "#49d6ff" if n.startswith("CP") else "#91a9c2" for n in nodes]
    sizes=[32 if n==selected else 17 for n in nodes]
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers",marker=dict(size=[z+10 for z in sizes],color=cs,opacity=.12,line=dict(width=0)),hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(x=xs,y=ys,mode="markers+text",text=nodes,textposition="bottom center",textfont=dict(size=9,color="#dcecff"),marker=dict(size=sizes,color=cs,line=dict(width=1.4,color="#e9faff")),hovertemplate="<b>%{text}</b><extra>Linked entity</extra>",showlegend=False))
    fig.update_layout(title=dict(text="ENTITY RELATIONSHIP INTELLIGENCE",font=dict(size=11,color="#7bdcff"),x=.02),xaxis=dict(visible=False,range=[-1.35,1.35]),yaxis=dict(visible=False,range=[-1.35,1.35],scaleanchor="x",scaleratio=1),margin=dict(l=3,r=3,t=32,b=20),height=440,showlegend=False)
    return chart_theme(fig,440)

def run_readonly_sql(query: str):
    if not query.strip():
        raise ValueError("Enter a SQL query.")
    if len(query)>12000:
        raise ValueError("Query is too long (maximum 12,000 characters).")
    statements=duckdb.extract_statements(query)
    if len(statements)!=1:
        raise ValueError("Run one SQL statement at a time.")
    first=re.match(r"\s*(SELECT|WITH|EXPLAIN)\b",query,re.I)
    if not first:
        raise ValueError("The workbench is read-only. Start with SELECT, WITH or EXPLAIN.")
    blocked=r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|CALL|PRAGMA|SET|RESET|VACUUM|TRUNCATE|SEQUENCE|MACRO|READ_CSV|READ_PARQUET|READ_JSON|HTTPFS)\b"
    if re.search(blocked,query,re.I):
        raise ValueError("This read-only workbench blocks data changes, file access and extension commands.")
    return con.execute(query).df()

# -------------------------------------------------------------------
# Navigation and header
# -------------------------------------------------------------------
PAGES=["Command Center","Transaction Explorer","Detection Lab","Network Map","Investigation Desk","SQL Workbench","Data Quality"]
with st.sidebar:
    st.markdown("## 🛡️ FINCRIME SQL")
    st.caption("TRANSACTION INTELLIGENCE LAB")
    st.markdown("---")
    st.markdown("**NAVIGATION**")
    page=st.radio("Go to",PAGES,label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**DEMO ENVIRONMENT**")
    st.success(f"{len(transactions):,} synthetic transactions loaded")
    st.caption("All customers, accounts and activity are fictional.")
    st.markdown("---")
    st.caption("DuckDB · SQL analytics · Rule-based monitoring")

page_copy={
"Command Center":("FINANCIAL CRIME / COMMAND CENTRE","FinCrime SQL","Investigate transaction patterns, review explainable alerts and explore financial activity with SQL."),
"Transaction Explorer":("TRANSACTION MONITORING","Transaction Explorer","Filter synthetic transaction activity and inspect customer, channel and counterparty patterns."),
"Detection Lab":("RULE ENGINE / ALERT INTELLIGENCE","Detection Lab","Explore transparent SQL detection rules and understand why each transaction was flagged."),
"Network Map":("ENTITY RESOLUTION / CONNECTION ANALYSIS","Transaction Network","Explore connections between a selected entity, counterparties and devices."),
"Investigation Desk":("CASE MANAGEMENT / REVIEW WORKFLOW","Investigation Desk","Create and track demonstration investigations from generated alerts."),
"SQL Workbench":("ADVANCED SQL / READ-ONLY QUERY LAB","SQL Workbench","Write SQL against the live DuckDB demo schema and export query results."),
"Data Quality":("DATA GOVERNANCE / CONTROL CHECKS","Data Quality","Inspect key integrity, completeness, duplicate identifiers and synthetic scenario coverage.")
}
eyebrow,title,subtitle=page_copy[page]
st.markdown(f'<div class="hero"><div class="eyebrow">✦ {eyebrow}</div><h1>🛡️ {title}</h1><p>{subtitle}</p><span class="pill">DUCKDB SQL ENGINE</span><span class="pill">EXPLAINABLE DETECTION RULES</span><span class="pill">SYNTHETIC DATA</span></div>',unsafe_allow_html=True)

# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------
if page=="Command Center":
    total_value=transactions.amount.sum()
    k=st.columns(4)
    for col,label,value in zip(k,["Transactions monitored","Value monitored","Review signals","Entities in scope"],[f"{len(transactions):,}",money(total_value),f"{len(alerts):,}",f"{transactions.customer_id.nunique():,}"]):
        with col: st.metric(label,value)
    left,center,right=st.columns([.78,2.1,.92],gap="small")
    with left:
        st.markdown('<div class="panel"><div class="eyebrow">PRIORITY QUEUE</div><h4 style="margin:5px 0 8px">Suspicious activity</h4></div>',unsafe_allow_html=True)
        for _,ar in alerts.sort_values("risk_score",ascending=False).head(7).iterrows():
            color="#ff6878" if ar.severity=="High" else "#ffae50"
            st.markdown(f'<div class="panel" style="padding:7px 8px;border-left:3px solid {color};margin-bottom:4px"><div style="font:9px IBM Plex Mono;color:{color}">{ar.alert_id} · {ar.severity.upper()}</div><div style="font-size:10px;color:#edf6ff;margin:3px 0">{ar.rule_name}</div><div style="font-size:9px;color:#91a9c0">{ar.customer_id} · ₹{ar.amount:,.0f}</div></div>',unsafe_allow_html=True)
    with center:
        st.markdown('<div class="eyebrow" style="margin:2px 0 4px">NETWORK INTELLIGENCE / RELATIONSHIP VIEW</div>',unsafe_allow_html=True)
        focus="C0007" if "C0007" in transactions.customer_id.values else transactions.customer_id.iloc[0]
        focus_rows=transactions[transactions.customer_id==focus].sort_values("transaction_ts",ascending=False).head(18)
        st.plotly_chart(make_network_figure(focus,focus_rows),use_container_width=True,config={"displayModeBar":False})
        st.markdown('<div style="font:8px IBM Plex Mono;color:#819ab3">● FOCAL CUSTOMER &nbsp; ━ CYAN: ROUTINE LINK &nbsp; ━ AMBER: SEEDED PATTERN</div>',unsafe_allow_html=True)
    with right:
        entity=customers[customers.customer_id==focus].iloc[0]
        st.markdown(f'<div class="panel"><div class="eyebrow">FOCAL ENTITY</div><h3 style="margin:5px 0">{focus}</h3><div style="font-size:10px;color:#9ab1c8">{entity.customer_name}</div><div class="pill">{entity.customer_segment}</div><div class="pill">RISK {entity.risk_tier.upper()}</div></div>',unsafe_allow_html=True)
        st.metric("Linked transactions",f"{len(focus_rows):,}")
        st.metric("Value in view",money(focus_rows.amount.sum()))
        st.metric("Counterparties",f"{focus_rows.counterparty_id.nunique():,}")
        st.markdown('<div class="panel"><div class="eyebrow">SYSTEM STATUS</div><div style="color:#55d6ad;font-size:10px">● SQL ENGINE ONLINE</div><div style="color:#55d6ad;font-size:10px">● RULE MONITORING ACTIVE</div><div style="color:#55d6ad;font-size:10px">● ENTITY GRAPH READY</div></div>',unsafe_allow_html=True)
    st.caption("Synthetic demonstration data. A review signal is not a finding of misconduct.")

elif page=="Transaction Explorer":
    f1,f2,f3=st.columns(3)
    with f1: types=st.multiselect("Transaction type",sorted(transactions.transaction_type.unique()),default=sorted(transactions.transaction_type.unique()))
    with f2: channels=st.multiselect("Channel",sorted(transactions.channel.unique()),default=sorted(transactions.channel.unique()))
    with f3: min_amount=st.number_input("Minimum amount (INR)",min_value=0.0,value=0.0,step=1000.0)
    filtered=transactions[transactions.transaction_type.isin(types)&transactions.channel.isin(channels)&(transactions.amount>=min_amount)]
    st.metric("Matching transactions",f"{len(filtered):,}")
    st.dataframe(filtered.sort_values("transaction_ts",ascending=False).head(500),use_container_width=True,hide_index=True)
    st.download_button("Download filtered transactions",filtered.to_csv(index=False).encode(), "fincrime_transactions.csv","text/csv")

elif page=="Detection Lab":
    st.markdown("### Detection rules and explainability")
    rule_counts=alerts.groupby(["rule_name","severity"]).size().reset_index(name="Alert count")
    if len(rule_counts):
        fig=px.bar(rule_counts,x="rule_name",y="Alert count",color="severity",barmode="stack",title="Alert volume by rule",color_discrete_map={"High":"#d77c9d","Medium":"#e5b66c","Low":"#68b9b6"})
        st.plotly_chart(chart_theme(fig,350),use_container_width=True)
    st.dataframe(alerts.sort_values("risk_score",ascending=False),use_container_width=True,hide_index=True)
    with st.expander("View SQL rule examples"):
        st.code("""-- Velocity: rolling count over ten minutes
SELECT transaction_id, customer_id,
       COUNT(*) OVER (
         PARTITION BY customer_id ORDER BY transaction_ts
         RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW
       ) AS transactions_last_10m
FROM transactions;

-- Near-threshold cash deposits grouped by customer/day
SELECT customer_id, CAST(transaction_ts AS DATE) AS txn_day,
       COUNT(*) AS deposit_count, SUM(amount) AS daily_total
FROM transactions
WHERE transaction_type='CASH_DEPOSIT'
  AND amount BETWEEN 8000 AND 9999.99
GROUP BY customer_id, CAST(transaction_ts AS DATE)
HAVING COUNT(*) >= 3 AND SUM(amount) >= 20000;""",language="sql")
    st.caption("All thresholds are illustrative demo parameters, not regulatory thresholds.")

elif page=="Network Map":
    st.markdown('<div class="eyebrow">ENTITY RESOLUTION / CONNECTION ANALYSIS</div>',unsafe_allow_html=True)
    entity_ids=sorted(transactions.customer_id.unique())
    selected=st.selectbox("Focal customer entity",entity_ids,index=entity_ids.index("C0007") if "C0007" in entity_ids else 0)
    focal=transactions[transactions.customer_id==selected].sort_values("transaction_ts",ascending=False)
    sample=focal.head(18)
    left,right=st.columns([2.15,1],gap="small")
    with left:
        st.plotly_chart(make_network_figure(selected,sample),use_container_width=True,config={"displayModeBar":False})
        st.caption("Cyan links show transaction relationships; amber links highlight seeded demonstration scenarios. Positions are schematic, not geographic.")
    with right:
        entity=customers[customers.customer_id==selected].iloc[0]
        st.markdown(f'<div class="panel"><div class="eyebrow">ENTITY PROFILE</div><h3 style="margin:5px 0">{entity.customer_name}</h3><div style="font:9px IBM Plex Mono;color:#8fa9c3">{selected} · {entity.customer_segment}</div><div class="pill">RISK {entity.risk_tier.upper()}</div><div class="pill">{entity.home_country}</div></div>',unsafe_allow_html=True)
        st.metric("Transactions",f"{len(focal):,}")
        st.metric("Total value",money(focal.amount.sum()))
        st.metric("Counterparties",f"{focal.counterparty_id.nunique():,}")
        st.metric("Devices",f"{focal.device_id.nunique():,}")
        st.markdown('<div class="panel"><div class="eyebrow">LINK LEGEND</div><div style="color:#50cfff;font-size:10px">● Counterparty / routine link</div><div style="color:#ff9b3c;font-size:10px">● Seeded pattern / priority link</div><div style="color:#8ca7c3;font-size:10px">● Device node</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">RECENT LINKED ACTIVITY</div>',unsafe_allow_html=True)
    st.dataframe(sample[["transaction_id","transaction_ts","amount","counterparty_id","device_id","channel"]].head(5),use_container_width=True,hide_index=True,height=150)

elif page=="Investigation Desk":
    if "cases" not in st.session_state:
        st.session_state.cases=pd.DataFrame(columns=["case_id","alert_id","customer_id","assigned_to","case_status","disposition","opened_at","notes"])
    st.markdown("### Open a demonstration case")
    if len(alerts):
        with st.form("case_form"):
            alert_id=st.selectbox("Select alert",alerts.alert_id.tolist())
            analyst=st.selectbox("Assign to",["Analyst A","Analyst B","Analyst C"])
            note=st.text_area("Investigation notes",placeholder="Record what should be reviewed. Avoid real personal information.")
            submitted=st.form_submit_button("Create investigation",type="primary")
            if submitted:
                arow=alerts[alerts.alert_id==alert_id].iloc[0]
                new={"case_id":"CASE-"+uuid.uuid4().hex[:8].upper(),"alert_id":alert_id,"customer_id":arow.customer_id,"assigned_to":analyst,"case_status":"Open","disposition":"Pending","opened_at":datetime.now().isoformat(timespec="seconds"),"notes":note}
                st.session_state.cases=pd.concat([st.session_state.cases,pd.DataFrame([new])],ignore_index=True)
                st.success(f"Created {new['case_id']}")
    st.markdown("### Case register")
    if len(st.session_state.cases):
        st.dataframe(st.session_state.cases,use_container_width=True,hide_index=True)
        case_ids=st.session_state.cases.case_id.tolist()
        chosen=st.selectbox("Update case status",case_ids)
        status=st.selectbox("New status",["Open","In Review","Escalated","Closed"])
        disposition=st.selectbox("Disposition",["Pending","No issue identified","Further review required","Escalated for specialist review"])
        if st.button("Save case update"):
            ix=st.session_state.cases.index[st.session_state.cases.case_id==chosen][0]
            st.session_state.cases.loc[ix,["case_status","disposition"]]=[status,disposition]
            st.success("Case updated for this session.")
        st.download_button("Export case register",st.session_state.cases.to_csv(index=False).encode(),"fincrime_case_register.csv","text/csv")
    else:
        st.info("No cases have been created in this session. Choose an alert above to begin.")

elif page=="SQL Workbench":
    st.markdown("### Read-only SQL editor")
    st.caption("One SELECT, WITH or EXPLAIN statement per run. Data modification, extension loading and external file access are blocked.")
    examples={
        "Top customers by transaction value":"SELECT customer_id, COUNT(*) AS transaction_count, ROUND(SUM(amount),2) AS total_value FROM transactions GROUP BY customer_id ORDER BY total_value DESC LIMIT 15",
        "Daily transaction trend":"SELECT CAST(transaction_ts AS DATE) AS txn_day, COUNT(*) AS tx_count, ROUND(SUM(amount),2) AS total_value FROM transactions GROUP BY 1 ORDER BY 1",
        "Alert counts by rule":"SELECT rule_name, severity, COUNT(*) AS alert_count FROM alerts GROUP BY 1,2 ORDER BY alert_count DESC",
        "High velocity events":"WITH x AS (SELECT *, COUNT(*) OVER(PARTITION BY customer_id ORDER BY transaction_ts RANGE BETWEEN INTERVAL 10 MINUTES PRECEDING AND CURRENT ROW) AS n10 FROM transactions) SELECT * FROM x WHERE n10>=5 ORDER BY transaction_ts DESC LIMIT 100"
    }
    choice=st.selectbox("Load example query",["Write your own"]+list(examples))
    default=examples.get(choice,"SELECT customer_id, COUNT(*) AS transactions, ROUND(SUM(amount), 2) AS total_amount\nFROM transactions\nGROUP BY customer_id\nORDER BY total_amount DESC\nLIMIT 20;")
    query=st.text_area("SQL query",value=default,height=190)
    if st.button("▶ Run query",type="primary"):
        try:
            result=run_readonly_sql(query)
            st.session_state.sql_result=result
            st.session_state.sql_query=query
            st.success(f"Query completed · {len(result):,} rows returned")
        except Exception as exc:
            st.error(f"Query could not run: {exc}")
    if "sql_result" in st.session_state:
        st.dataframe(st.session_state.sql_result,use_container_width=True,hide_index=True)
        st.download_button("Export query result (CSV)",st.session_state.sql_result.to_csv(index=False).encode(),"sql_query_result.csv","text/csv")
        if st.session_state.sql_query.strip().upper().startswith("EXPLAIN"):
            st.caption("DuckDB execution-plan output is shown above.")

elif page=="Data Quality":
    checks=[
        ("Unique transaction IDs",transactions.transaction_id.is_unique),
        ("Unique customer IDs",customers.customer_id.is_unique),
        ("All transactions have customer keys",transactions.customer_id.isin(customers.customer_id).all()),
        ("All transactions have account keys",transactions.account_id.isin(accounts.account_id).all()),
        ("Non-negative transaction amounts",(transactions.amount>=0).all()),
        ("No missing transaction IDs",transactions.transaction_id.notna().all()),
        ("All timestamps parsed",transactions.transaction_ts.notna().all())
    ]
    st.markdown("### Data control checks")
    for label,passed in checks:
        (st.success if passed else st.error)(("✓ " if passed else "✕ ")+label)
    st.markdown("### Synthetic scenario coverage")
    sc=transactions.synthetic_scenario.value_counts().rename_axis("Scenario").reset_index(name="Records")
    st.dataframe(sc,use_container_width=True,hide_index=True)
    st.markdown("### Table inventory")
    inventory=pd.DataFrame({"Table":["customers","accounts","transactions","alerts"],"Rows":[len(customers),len(accounts),len(transactions),len(alerts)],"Purpose":["Customer profile","Account reference","Synthetic activity","Rule-based review signals"]})
    st.dataframe(inventory,use_container_width=True,hide_index=True)

st.markdown("---")
st.caption("FinCrime SQL · Portfolio demonstration · Synthetic data only · Alerts are review signals, not conclusions.")
