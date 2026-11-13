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
 SELECT transaction_id, customer_id, 'Velocity burst' AS rule_name,
        78 AS score, 'High' AS severity, transaction_ts AS created_at,
        '5 or more transactions within a rolling 10-minute window' AS rationale
 FROM velocity WHERE n10>=5
 UNION ALL
 SELECT t.transaction_id,t.customer_id,'Near-threshold cash activity',82,'High',t.transaction_ts,
        'Customer made at least 3 cash deposits in the illustrative near-threshold band on one day'
 FROM transactions t JOIN cash c ON t.customer_id=c.customer_id
   AND CAST(t.transaction_ts AS DATE)=c.tx_date
 WHERE c.n_deposits>=3 AND c.total_amount>=20000
   AND t.transaction_type='CASH_DEPOSIT' AND t.amount BETWEEN 8000 AND 9999.99
 UNION ALL
 SELECT transaction_id,customer_id,'Geographic inconsistency',68,'Medium',transaction_ts,
        'Merchant country and IP country differ; travel, VPNs and data quality can explain this'
 FROM transactions
 WHERE merchant_country<>ip_country AND merchant_country<>'UNKNOWN' AND ip_country<>'UNKNOWN'
 UNION ALL
 SELECT transaction_id,customer_id,'High-value activity',72,'Medium',transaction_ts,
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#262344;--muted:#6d6888;--line:#ded8ef;--violet:#7461b8;--aqua:#48b9c4;--rose:#d985ae;}
.stApp{background:radial-gradient(ellipse at 0% 0%,rgba(199,174,246,.34),transparent 35%),radial-gradient(ellipse at 100% 10%,rgba(133,218,226,.28),transparent 34%),linear-gradient(135deg,#f7f4ff,#eff8fc 58%,#fff3f8);font-family:'DM Sans',sans-serif;color:var(--ink);}
[data-testid="stSidebar"]{background:linear-gradient(165deg,#5e4e9a,#7668b5 60%,#4b9aa8);border-right:1px solid rgba(255,255,255,.4);}
[data-testid="stSidebar"] *{color:#fff!important;}
[data-testid="stSidebar"] [data-testid="stRadio"] label{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);border-radius:10px;padding:7px 10px;margin:3px 0;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:1.7rem;padding-bottom:2.5rem;max-width:1500px;}
.hero{position:relative;overflow:hidden;border:1px solid rgba(255,255,255,.9);border-radius:24px;padding:28px 32px;background:linear-gradient(110deg,rgba(180,151,232,.46),rgba(119,190,221,.32),rgba(236,174,211,.32));box-shadow:0 16px 40px rgba(70,61,125,.12);margin-bottom:22px;}
.hero:after{content:"";position:absolute;width:220px;height:220px;right:4%;top:-95px;border:1px solid rgba(116,97,184,.3);border-radius:50%;box-shadow:0 0 0 18px rgba(255,255,255,.09),0 0 0 38px rgba(255,255,255,.08);pointer-events:none;}
.eyebrow{font-size:.72rem;letter-spacing:.18em;font-weight:700;color:#62549b;text-transform:uppercase;}
.hero h1{font-family:'Space Grotesk',sans-serif;color:#292548;font-size:clamp(2rem,4vw,3rem);margin:.35rem 0;}
.hero p{color:#514c70;font-size:1rem;margin:0;max-width:850px;}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;background:rgba(255,255,255,.48);border:1px solid rgba(255,255,255,.75);font-size:.68rem;letter-spacing:.08em;color:#504878;margin:12px 6px 0 0;}
.metric{background:rgba(255,255,255,.83);border:1px solid #fff;border-radius:17px;padding:16px 18px;box-shadow:0 8px 24px rgba(71,63,120,.08);}
.metric-label{color:#6e6889;font-size:.8rem}.metric-value{color:#292548;font-family:'Space Grotesk';font-size:1.8rem;font-weight:700;}
.panel{background:rgba(255,255,255,.82);border:1px solid #fff;border-radius:18px;padding:16px 18px;box-shadow:0 8px 25px rgba(71,63,120,.07);margin-bottom:14px;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#292548!important;}
p,li,label,.stCaption{color:#555171;}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid #ded8ef;border-radius:12px;overflow:hidden;background:#fff;}
.stButton button,.stDownloadButton button{border-radius:10px;border:1px solid #d7cdeb;background:#fff;color:#423b69;font-weight:600;}
.stButton button[kind="primary"]{background:linear-gradient(110deg,#7461b8,#56b9c3);color:#fff;border:0;}
[data-testid="stPlotlyChart"]{background:rgba(255,255,255,.78);border:1px solid #fff;border-radius:16px;padding:8px;}
div[data-testid="stMetric"]{background:rgba(255,255,255,.84);border:1px solid #fff;border-radius:16px;padding:12px;}
div[data-testid="stMetricLabel"] p{color:#6e6889!important}div[data-testid="stMetricValue"]{color:#292548!important}
</style>
""",unsafe_allow_html=True)

def money(v):
    return f"₹{v:,.0f}"

def chart_theme(fig,height=330):
    fig.update_layout(template="plotly_white",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,.15)",
        font=dict(family="DM Sans",color="#3b365e",size=12),margin=dict(l=18,r=18,t=42,b=34),height=height,
        legend=dict(bgcolor="rgba(255,255,255,.4)"),hoverlabel=dict(bgcolor="#fff",font_color="#292548"))
    fig.update_xaxes(gridcolor="#e9e4f2",tickfont_color="#514c70",automargin=True)
    fig.update_yaxes(gridcolor="#e9e4f2",tickfont_color="#514c70",automargin=True)
    return fig

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
PAGES=["Command Center","Transaction Explorer","Detection Lab","Entity Intelligence","Investigation Desk","SQL Workbench","Data Quality"]
with st.sidebar:
    st.markdown("## 🛡️ FINCRIME SQL")
    st.caption("TRANSACTION INTELLIGENCE LAB")
    st.markdown("---")
    st.markdown("**APPEARANCE**")
    appearance=st.radio("Theme",["Light","Dark"],horizontal=True,label_visibility="collapsed",index=0)
    st.markdown("---")
    st.markdown("**NAVIGATION**")
    page=st.radio("Go to",PAGES,label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**DEMO ENVIRONMENT**")
    st.success(f"{len(transactions):,} synthetic transactions loaded")
    st.caption("All customers, accounts and activity are fictional.")
    st.markdown("---")
    st.caption("DuckDB · SQL analytics · Rule-based monitoring")

if appearance=="Dark":
    st.markdown("""<style>
    .stApp{background:radial-gradient(ellipse at 0% 0%,rgba(116,97,184,.24),transparent 35%),linear-gradient(135deg,#201d3b,#252849 55%,#263c4b);color:#f4f1ff}
    h1,h2,h3,p,li,label{color:#f2efff!important}.hero{background:linear-gradient(110deg,rgba(116,97,184,.48),rgba(62,135,157,.35));border-color:rgba(255,255,255,.18)}
    .hero h1,.hero p{color:#fff!important}.metric,.panel,[data-testid="stPlotlyChart"]{background:rgba(45,42,76,.88);border-color:#514b75}
    .metric-label{color:#c9c3e5}.metric-value{color:#fff}.stButton button,.stDownloadButton button{background:#38345b;color:#fff;border-color:#625a8b}
    div[data-testid="stMetric"]{background:#38345b;border-color:#514b75}div[data-testid="stMetricLabel"] p,div[data-testid="stMetricValue"]{color:#fff!important}
    </style>""",unsafe_allow_html=True)

page_copy={
"Command Center":("FINANCIAL CRIME / COMMAND CENTRE","FinCrime SQL","Investigate transaction patterns, review explainable alerts and explore financial activity with SQL."),
"Transaction Explorer":("TRANSACTION MONITORING","Transaction Explorer","Filter synthetic transaction activity and inspect customer, channel and counterparty patterns."),
"Detection Lab":("RULE ENGINE / ALERT INTELLIGENCE","Detection Lab","Explore transparent SQL detection rules and understand why each transaction was flagged."),
"Entity Intelligence":("CUSTOMER / NETWORK ANALYTICS","Entity Intelligence","Review customer exposure, transaction behaviour and connected counterparties."),
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
    total_value=transactions["amount"].sum()
    alert_customers=alerts["customer_id"].nunique() if len(alerts) else 0
    k=st.columns(4)
    for col,label,value in zip(k,["Transactions monitored","Transaction value","Alerts generated","Customers reviewed"],[f"{len(transactions):,}",money(total_value),f"{len(alerts):,}",f"{alert_customers:,}"]):
        with col: st.metric(label,value)
    left,right=st.columns([1.5,1])
    with left:
        daily=transactions.assign(day=transactions.transaction_ts.dt.date).groupby("day",as_index=False).agg(Transactions=("transaction_id","count"),Value=("amount","sum"))
        fig=px.area(daily,x="day",y="Value",title="Transaction value over time",color_discrete_sequence=["#7664b7"])
        fig.update_traces(line_width=2,fillcolor="rgba(118,100,183,.18)")
        st.plotly_chart(chart_theme(fig,350),use_container_width=True)
    with right:
        sev=alerts.groupby("severity").size().reset_index(name="Alerts") if len(alerts) else pd.DataFrame(columns=["severity","Alerts"])
        fig=px.pie(sev,names="severity",values="Alerts",hole=.58,title="Alert severity mix",color="severity",color_discrete_map={"High":"#d77c9d","Medium":"#e5b66c","Low":"#68b9b6"})
        st.plotly_chart(chart_theme(fig,350),use_container_width=True)
    st.subheader("Latest review signals")
    st.dataframe(alerts[["alert_id","customer_id","rule_name","amount","severity","created_at"]].head(12) if len(alerts) else alerts,use_container_width=True,hide_index=True)
    st.caption("A generated alert is a prompt for review, not a determination of misconduct.")

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

elif page=="Entity Intelligence":
    summary=transactions.groupby("customer_id",as_index=False).agg(Transactions=("transaction_id","count"),Total_value=("amount","sum"),Average_value=("amount","mean"),Distinct_devices=("device_id","nunique"))
    alert_summary=alerts.groupby("customer_id").size().rename("Alert_count") if len(alerts) else pd.Series(dtype=int)
    summary=summary.join(alert_summary,on="customer_id")
    summary["Alert_count"]=summary["Alert_count"].fillna(0).astype(int)
    summary=summary.merge(customers[["customer_id","customer_name","customer_segment","risk_tier","home_country"]],on="customer_id")
    summary["Review_score"]=np.minimum(100,summary["Alert_count"]*18+np.where(summary["risk_tier"]=="High",22,np.where(summary["risk_tier"]=="Medium",10,0))+np.where(summary["Total_value"]>500000,15,0))
    summary=summary.sort_values(["Review_score","Total_value"],ascending=False)
    st.dataframe(summary.head(50),use_container_width=True,hide_index=True)
    selected=st.selectbox("Inspect customer",summary.customer_id.tolist())
    row=summary[summary.customer_id==selected].iloc[0]
    a,b,c=st.columns(3)
    a.metric("Review score",int(row.Review_score))
    b.metric("Transactions",int(row.Transactions))
    c.metric("Alert count",int(row.Alert_count))
    st.dataframe(transactions[transactions.customer_id==selected].sort_values("transaction_ts",ascending=False),use_container_width=True,hide_index=True)

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
