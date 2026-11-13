import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="TryOn Intelligence | AR Shopping Analytics", page_icon="🪞", layout="wide", initial_sidebar_state="expanded")

CONSTRUCTS = {
    "Online Shopping Value": [1,2,3],
    "Physical Store Value": [4,5,6],
    "AR Experience Quality": [7,8,9,10,11],
    "AR-Enabled Risk Reduction": [12,13,14],
    "Omnichannel Behaviour": [15,16,17,18],
    "Purchase Intention": [19,20,21],
}
LABELS = {
    1:"Convenience",2:"Product comparison",3:"Information access",
    4:"Touch and feel",5:"Fit and quality checking",6:"Physical confidence",
    7:"AR realism",8:"AR accuracy",9:"Ease of use",10:"Product interaction",11:"AR trust",
    12:"Uncertainty reduction",13:"Perceived risk reduction",14:"Purchase confidence",
    15:"Webrooming",16:"Showrooming",17:"Store-visit substitution",18:"Channel continuity",
    19:"Purchase willingness",20:"Online purchase intention",21:"AR-assisted purchase intention",
}
DARK = {"bg":"#191735","panel":"#27234B","panel2":"#393263","border":"#5B5687","text":"#F7F4FF","muted":"#C9C3E5","accent":"#79DCE5","grid":"#49446C"}
LIGHT = {"bg":"#F4F1FC","panel":"#FFFFFF","panel2":"#EEE9FA","border":"#DCD4EE","text":"#302B50","muted":"#625C7D","accent":"#6654A5","grid":"#E5DFF0"}

def demo_data(n=70, seed=42):
    """Illustrative synthetic data only; not the original research responses."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(3.6, .55, (n,6))
    latent[:,2] = np.clip(.45*latent[:,0] + .30*latent[:,3] + rng.normal(1,.55,n),1,5)
    latent[:,4] = np.clip(.35*latent[:,2] + .25*latent[:,1] + rng.normal(1.5,.6,n),1,5)
    latent[:,5] = np.clip(.35*latent[:,2] + .30*latent[:,3] + rng.normal(1.2,.55,n),1,5)
    out = {"Respondent ID":[f"DEMO-{i+1:03d}" for i in range(n)]}
    for j, (name, items) in enumerate(CONSTRUCTS.items()):
        for q in items:
            out[f"Q{q}"] = np.clip(np.rint(latent[:,j] + rng.normal(0,.42,n)),1,5).astype(int)
    out["Shopping Journey"] = rng.choice(["Webrooming","Showrooming","Mixed","Online-first"],n,p=[.32,.28,.28,.12])
    out["Product Category"] = rng.choice(["Fashion","Beauty","Footwear"],n,p=[.48,.27,.25])
    out["AR Usage"] = rng.choice(["Used AR try-on","Not used AR"],n,p=[.62,.38])
    return pd.DataFrame(out)

def alpha(frame):
    x=frame.dropna().astype(float); k=x.shape[1]
    if k<2 or len(x)<2 or x.sum(axis=1).var(ddof=1)==0: return np.nan
    return (k/(k-1))*(1-x.var(axis=0,ddof=1).sum()/x.sum(axis=1).var(ddof=1))

def theme_chart(fig):
    """Apply one consistent high-contrast chart theme across every module."""
    template = "plotly_dark" if mode == "Dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"],family="Inter, Arial",size=12),
        title=dict(font=dict(color=C["text"],size=15)),
        legend=dict(font=dict(color=C["text"],size=11),bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=C["panel"],font=dict(color=C["text"],size=12)),
        margin=dict(l=24,r=20,t=52,b=52),
        height=370,
        autosize=True
    )
    fig.update_xaxes(
        gridcolor=C["grid"], zerolinecolor=C["grid"],
        tickfont=dict(color=C["text"],size=11),
        title_font=dict(color=C["text"],size=12),
        automargin=True
    )
    fig.update_yaxes(
        gridcolor=C["grid"], zerolinecolor=C["grid"],
        tickfont=dict(color=C["text"],size=11),
        title_font=dict(color=C["text"],size=12),
        automargin=True
    )
    if fig.data:
        for trace in fig.data:
            if getattr(trace,"type","")=="pie":
                trace.update(textfont=dict(color="#FFFFFF",size=12),insidetextorientation="auto")
    return fig

def score_data(frame, mapping):
    result=frame.copy()
    for name, nums in CONSTRUCTS.items():
        result[name]=result[[mapping[q] for q in nums]].apply(pd.to_numeric,errors="coerce").mean(axis=1)
    return result

if "theme" not in st.session_state: st.session_state.theme="Light"
with st.sidebar:
    st.markdown("## 🪞 TRYON INTELLIGENCE")
    st.caption("AR COMMERCE · CUSTOMER INSIGHTS")
    st.markdown("---")
    st.markdown("**APPEARANCE**")
    st.session_state.theme=st.radio("Theme",["Dark","Light"],horizontal=True,label_visibility="collapsed",index=0 if st.session_state.theme=="Dark" else 1)
    mode=st.session_state.theme; C=DARK if mode=="Dark" else LIGHT
    st.markdown("---")
    st.markdown("**DATA SOURCE**")
    upload=st.file_uploader("Upload survey data",type=["csv","xlsx"],help="Upload respondent-level responses with 21 Likert items.")
    if upload is None:
        raw=demo_data(); mapping={q:f"Q{q}" for q in range(1,22)}; demo=True
        st.info("Illustrative demo data active. Upload your actual survey file for real analysis.")
    else:
        try:
            raw=pd.read_csv(upload) if upload.name.lower().endswith(".csv") else pd.read_excel(upload)
            raw.columns=[str(c).strip() for c in raw.columns]
            normalized={re.sub(r"[^a-z0-9]","",str(c).lower()):c for c in raw.columns}
            defaults=[]
            for q in range(1,22):
                found=next((normalized[k] for k in [f"q{q}",f"question{q}",f"item{q}",f"likert{q}"] if k in normalized),None)
                if found is not None: defaults.append(found)
            # Google Forms exports often use full question text as headers.
            # If exactly 21 numeric columns exist, preserve their left-to-right order.
            if len(defaults)!=21:
                numeric_defaults=raw.select_dtypes(include="number").columns.tolist()
                if len(numeric_defaults)==21: defaults=numeric_defaults
            selected=st.multiselect("Select 21 item columns in questionnaire order",list(raw.columns),default=defaults,
                help="Order: Q1–3 Online Value; Q4–6 Physical Store Value; Q7–11 AR Quality; Q12–14 Risk Reduction; Q15–18 Omnichannel; Q19–21 Purchase Intention. Google Forms exports with 21 numeric Likert columns are auto-mapped.")
            mapping={q+1:selected[q] for q in range(min(21,len(selected)))}
            demo=False
            st.success(f"{len(raw):,} rows loaded")
        except Exception as exc:
            st.error(f"File could not be read: {exc}")
            raw=pd.DataFrame(); mapping={}; demo=False
    st.markdown("---")
    page=st.radio("NAVIGATION",["Executive Overview","AR Experience Lab","Omnichannel Journey","Customer Insights","Research Analytics","Customer Segmentation","Strategy Simulator","Data Studio"])
    st.markdown("---")
    st.caption("6 constructs · 21 Likert items · S-O-R + Dual-Factor lens")

st.markdown(f"""
<style>
.stApp {{background:{C['bg']};color:{C['text']};}}
[data-testid="stSidebar"] {{background:{C['panel']};border-right:1px solid {C['border']};}}
[data-testid="stSidebar"] * {{color:{C['text']};}}
[data-testid="stHeader"] {{background:{C['bg']};}}
.block-container {{padding-top:1.3rem;padding-bottom:2rem;max-width:1500px;}}
.hero {{position:relative;overflow:hidden;border:1px solid {C['border']};border-radius:22px;padding:26px 30px;margin-bottom:22px;background:linear-gradient(115deg,{C['panel2']},{C['panel']} 65%,{C['bg']});animation:enter .6s ease-out both;}}
.hero:after {{content:"";position:absolute;width:230px;height:230px;right:-50px;top:-110px;border-radius:50%;background:radial-gradient(circle,rgba(139,124,255,.28),transparent 70%);animation:float 8s ease-in-out infinite alternate;}}
@keyframes float {{from {{transform:translate(0,0) scale(.9);}}to {{transform:translate(-20px,22px) scale(1.12);}}}}
@keyframes enter {{from {{opacity:0;transform:translateY(8px);}}to {{opacity:1;transform:translateY(0);}}}}
.panel {{background:{C['panel']};border:1px solid {C['border']};border-radius:16px;padding:16px 18px;}}
.kicker {{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:{C['accent']};font-weight:800;}}
.muted {{color:{C['muted']};}}
div[data-testid="stMetric"] {{background:{C['panel']};border:1px solid {C['border']};padding:14px 16px;border-radius:14px;}}
div[data-testid="stMetricLabel"] p {{color:{C['muted']} !important;}}
div[data-testid="stMetricValue"] {{color:{C['text']} !important;}}
.stButton button,.stDownloadButton button {{border-radius:10px;border:1px solid {C['border']};}}
.stButton button:hover,.stDownloadButton button:hover {{border-color:{C['accent']};transform:translateY(-1px);}}
@media(max-width:900px) {{.block-container {{padding-left:1rem;padding-right:1rem;}}}}

/* ===== Inspired by the AR research visual system: aurora, holo-glass and motion ===== */
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700;800&family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"] {{font-family:'Instrument Sans',Inter,Arial,sans-serif;}}
.stApp {{background:
 radial-gradient(ellipse at 8% 3%,color-mix(in srgb,{C['accent']} 13%,transparent),transparent 34%),
 radial-gradient(ellipse at 92% 18%,rgba(139,108,255,.13),transparent 32%),
 radial-gradient(ellipse at 55% 100%,rgba(224,48,111,.08),transparent 36%),
 {C['bg']};}}
.stApp:before {{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.25;
 background-image:linear-gradient({C['grid']} 1px,transparent 1px),linear-gradient(90deg,{C['grid']} 1px,transparent 1px);
 background-size:42px 42px;mask-image:linear-gradient(to bottom,black,transparent 78%);}}
.block-container {{position:relative;z-index:1;max-width:1420px;padding-top:1.05rem;padding-bottom:1.4rem;}}
[data-testid="stSidebar"] {{background:linear-gradient(165deg,{C['panel']},color-mix(in srgb,{C['panel']} 86%,{C['accent']} 14%));}}
[data-testid="stSidebar"] > div:first-child {{border-right:1px solid {C['border']};}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{padding:5px 9px;border-radius:9px;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{background:color-mix(in srgb,{C['accent']} 13%,transparent);}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{border:1px dashed {C['accent']};border-radius:12px;background:color-mix(in srgb,{C['panel2']} 75%,transparent);}}
[data-testid="stHeader"] {{background:transparent;}}
h1,h2,h3,h4 {{font-family:'Bricolage Grotesque',Inter,sans-serif;letter-spacing:-.025em;}}
h2,h3 {{font-weight:700;}}
.hero {{min-height:210px;border-radius:22px;padding:30px 34px;margin:4px 0 24px;
 background:linear-gradient(112deg,color-mix(in srgb,{C['panel']} 95%,{C['accent']} 5%),{C['panel2']} 62%,color-mix(in srgb,{C['panel']} 80%,#8B7CFF 20%));
 box-shadow:0 18px 48px rgba(0,0,0,.18),inset 0 1px rgba(255,255,255,.10);}}
.hero:before {{content:"";position:absolute;inset:0;pointer-events:none;border-radius:inherit;
 background:linear-gradient(115deg,transparent 20%,rgba(255,255,255,.08) 48%,transparent 68%);
 transform:translateX(-110%);animation:heroSweep 9s ease-in-out infinite;}}
.hero:after {{width:300px;height:300px;right:-65px;top:-145px;
 background:radial-gradient(circle,rgba(49,224,245,.25),rgba(139,108,255,.14) 42%,transparent 70%);
 filter:blur(4px);animation:orbFloat 8s ease-in-out infinite alternate;}}
.hero h1 {{font-family:'Bricolage Grotesque',sans-serif;font-size:clamp(2rem,4vw,3.15rem);letter-spacing:-.045em;
 background:linear-gradient(100deg,{C['text']},#67dff0 55%,#b7a5ff);-webkit-background-clip:text;background-clip:text;color:transparent;}}
.hero p {{max-width:760px;line-height:1.55;}}
.hero .kicker {{font:600 .72rem 'JetBrains Mono',monospace;letter-spacing:.16em;color:{C['accent']};}}
.hero-tags {{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px;}}
.hero-tags span {{border:1px solid {C['border']};border-radius:99px;padding:5px 11px;font:500 .72rem 'JetBrains Mono',monospace;color:{C['muted']};background:rgba(255,255,255,.035);}}
@keyframes heroSweep {{0%,58%{{transform:translateX(-110%)}}82%,100%{{transform:translateX(110%)}}}}
@keyframes orbFloat {{from{{transform:translate(0,0) scale(.92)}}to{{transform:translate(-22px,24px) scale(1.12)}}}}
div[data-testid="stMetric"] {{position:relative;overflow:hidden;min-height:112px;background:linear-gradient(145deg,{C['panel']},color-mix(in srgb,{C['panel']} 84%,{C['accent']} 16%));
 border:1px solid {C['border']};border-radius:16px;padding:15px 17px;box-shadow:0 9px 28px rgba(0,0,0,.12),inset 0 1px rgba(255,255,255,.045);transition:transform .25s,border-color .25s,box-shadow .25s;}}
div[data-testid="stMetric"]:hover {{transform:translateY(-3px);border-color:{C['accent']};box-shadow:0 12px 30px color-mix(in srgb,{C['accent']} 15%,transparent);}}
div[data-testid="stMetric"] label p {{font-family:'JetBrains Mono',monospace;font-size:.68rem!important;letter-spacing:.08em;text-transform:uppercase;}}
div[data-testid="stMetricValue"] {{font-family:'Bricolage Grotesque',sans-serif;font-weight:700;}}
[data-testid="stPlotlyChart"] {{border:1px solid {C['border']};border-radius:16px;padding:7px;background:color-mix(in srgb,{C['panel']} 68%,transparent);box-shadow:0 10px 28px rgba(0,0,0,.10);}}
[data-testid="stDataFrame"],[data-testid="stTable"] {{border:1px solid {C['border']};border-radius:14px;overflow:hidden;}}
[data-testid="stTabs"] button {{border-radius:10px 10px 0 0;}}
.stButton button,.stDownloadButton button {{border-radius:10px;border:1px solid {C['border']};font-weight:600;transition:all .2s;}}
.stButton button[kind="primary"] {{background:linear-gradient(110deg,{C['accent']},#7564d9);color:#071321;border:0;}}
.stButton button:hover,.stDownloadButton button:hover {{border-color:{C['accent']};transform:translateY(-2px);box-shadow:0 7px 18px color-mix(in srgb,{C['accent']} 18%,transparent);}}
[data-testid="stExpander"] {{border:1px solid {C['border']};border-radius:13px;background:{C['panel']};}}
hr {{border-color:{C['border']};opacity:.8;}}
.small-note {{font:500 .7rem 'JetBrains Mono',monospace;letter-spacing:.08em;color:{C['muted']};}}
@media(max-width:900px) {{
 .block-container {{padding-left:1rem;padding-right:1rem;}}
 .hero {{padding:23px 21px;min-height:0;}}
 .hero:after {{width:190px;height:190px;right:-90px;}}
 [data-testid="stPlotlyChart"] {{padding:2px;}}
}}
@media(prefers-reduced-motion:reduce) {{
 *,*::before,*::after {{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important;}}
}}

/* Page-aware holographic interface refinements */
.hero {{isolation:isolate;}}
.hero-orbit {{position:absolute;right:5%;top:50%;width:150px;height:150px;transform:translateY(-50%);border:1px solid color-mix(in srgb,{C['accent']} 48%,transparent);border-radius:50%;opacity:.8;pointer-events:none;box-shadow:0 0 35px color-mix(in srgb,{C['accent']} 14%,transparent),inset 0 0 25px color-mix(in srgb,{C['accent']} 10%,transparent);}}
.hero-orbit:before,.hero-orbit:after {{content:"";position:absolute;inset:17px;border:1px dashed color-mix(in srgb,{C['accent']} 50%,transparent);border-radius:50%;animation:orbitSpin 18s linear infinite;}}
.hero-orbit:after {{inset:37px;border-style:solid;opacity:.55;animation-direction:reverse;animation-duration:12s;}}
.hero-orbit span {{position:absolute;left:50%;top:50%;width:14px;height:14px;border-radius:50%;background:{C['accent']};transform:translate(-50%,-50%);box-shadow:0 0 20px {C['accent']};}}
.hero-orbit i,.hero-orbit b {{position:absolute;width:7px;height:7px;border-radius:50%;background:#9c8cff;box-shadow:0 0 14px #9c8cff;left:12%;top:50%;animation:orbitDot 7s linear infinite;}}
.hero-orbit b {{left:auto;top:12%;right:22%;background:#ef6ca8;box-shadow:0 0 14px #ef6ca8;animation-duration:9s;animation-direction:reverse;}}
@keyframes orbitSpin {{to{{transform:rotate(360deg)}}}}
@keyframes orbitDot {{to{{transform:rotate(360deg) translateX(58px) rotate(-360deg)}}}}
[data-testid="stMainBlockContainer"] h2,[data-testid="stMainBlockContainer"] h3 {{letter-spacing:-.025em;}}
[data-testid="stMainBlockContainer"] h3 {{margin-top:.3rem;}}
[data-testid="stPlotlyChart"] iframe {{max-width:100%!important;}}
[data-testid="stDataFrame"] {{max-width:100%;font-size:.9rem;}}
.stAlert {{border-radius:12px;}}
@media(max-width:720px) {{.hero-orbit {{width:84px;height:84px;right:16px;top:22px;transform:none;opacity:.45}}.hero {{padding-right:90px}}.hero-tags span {{font-size:.62rem}}}}
@media(max-height:800px) and (min-width:900px) {{.hero {{min-height:0;padding:20px 26px;margin-bottom:16px}}.hero h1 {{font-size:2.2rem}}.hero-orbit {{width:112px;height:112px}}}}

/* Complementary lavender / aqua / rose glass interface — no pure-black panels */
.stApp {{background:
 radial-gradient(ellipse at 8% 4%,rgba(198,139,255,.46),transparent 36%),
 radial-gradient(ellipse at 92% 8%,rgba(87,214,231,.36),transparent 35%),
 radial-gradient(ellipse at 92% 88%,rgba(244,151,204,.28),transparent 32%),
 linear-gradient(135deg,#f3efff 0%,#eaf5ff 48%,#fff0f7 100%) !important;color:#29264d !important;}}
.stApp:before {{opacity:.07;background-size:54px 54px;}}
[data-testid="stSidebar"] {{background:linear-gradient(160deg,#7660b5 0%,#6258a5 55%,#438da9 100%)!important;backdrop-filter:blur(24px);border-right:1px solid rgba(255,255,255,.42)!important;box-shadow:12px 0 42px rgba(57,49,112,.14);}}
[data-testid="stSidebar"] > div:first-child {{border-right:0!important;}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{color:#fff!important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:10px;transition:all .2s;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{background:rgba(255,255,255,.18);border-color:rgba(255,255,255,.4);}}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{background:rgba(255,255,255,.14)!important;border:1px dashed rgba(255,255,255,.58)!important;backdrop-filter:blur(12px);}}
[data-testid="stHeader"] {{background:transparent!important;}}
[data-testid="stMainBlockContainer"] {{color:#29264d;}}
[data-testid="stMainBlockContainer"] h1,[data-testid="stMainBlockContainer"] h2,[data-testid="stMainBlockContainer"] h3,[data-testid="stMainBlockContainer"] h4 {{color:#29264d;}}
[data-testid="stMainBlockContainer"] p,[data-testid="stMainBlockContainer"] label,[data-testid="stMainBlockContainer"] li {{color:#514d70;}}
.hero {{background:linear-gradient(112deg,rgba(190,151,245,.34),rgba(130,184,236,.30) 54%,rgba(255,185,218,.34))!important;border:1px solid rgba(255,255,255,.82)!important;backdrop-filter:blur(28px);-webkit-backdrop-filter:blur(28px);box-shadow:0 18px 48px rgba(91,76,150,.14),inset 0 1px rgba(255,255,255,.8)!important;}}
.hero h1 {{background:linear-gradient(100deg,#39316f 5%,#496f9d 52%,#9361a8)!important;-webkit-background-clip:text!important;background-clip:text!important;color:transparent!important;}}
.hero p {{color:#575477!important;}}
.hero .kicker {{color:#6555a8!important;}}
.hero-tags span {{background:rgba(255,255,255,.48)!important;border-color:rgba(255,255,255,.8)!important;color:#514b7c!important;backdrop-filter:blur(10px);}}
.panel,div[data-testid="stMetric"],[data-testid="stPlotlyChart"],[data-testid="stExpander"] {{background:linear-gradient(135deg,rgba(255,255,255,.82),rgba(239,235,255,.70))!important;border:1px solid rgba(255,255,255,.95)!important;backdrop-filter:blur(22px);-webkit-backdrop-filter:blur(22px);box-shadow:0 12px 32px rgba(83,75,139,.10),inset 0 1px rgba(255,255,255,.9)!important;}}
div[data-testid="stMetric"] {{border-radius:18px!important;}}
div[data-testid="stMetricLabel"] p {{color:#696287!important;}}
div[data-testid="stMetricValue"] {{color:#302b59!important;}}
[data-testid="stMetricDelta"] {{color:#238e86!important;}}
[data-testid="stPlotlyChart"] {{border-radius:18px!important;padding:10px!important;}}
[data-testid="stDataFrame"],[data-testid="stTable"] {{background:rgba(255,255,255,.88)!important;border:1px solid #d9d1f1!important;border-radius:15px;overflow:hidden;box-shadow:0 8px 24px rgba(76,66,130,.08);}}
[data-testid="stDataFrame"] * {{color:#302b59;}}
.stButton button,.stDownloadButton button {{background:linear-gradient(110deg,#ffffff,#eee9ff)!important;color:#393263!important;border:1px solid #d7cdef!important;border-radius:11px;backdrop-filter:blur(12px);}}
.stButton button[kind="primary"] {{background:linear-gradient(110deg,#72d9dd,#b59af0)!important;color:#28244e!important;border:0!important;}}
.stButton button:hover,.stDownloadButton button:hover {{border-color:#8b79ca!important;transform:translateY(-2px);box-shadow:0 7px 18px rgba(110,91,174,.16);}}
[data-testid="stExpander"] {{background:rgba(255,255,255,.72)!important;border:1px solid #d9d1f1!important;border-radius:14px;}}
.stAlert {{background:rgba(255,255,255,.78)!important;border:1px solid #d9d1f1!important;color:#39345f!important;}}
hr {{border-color:#d7d0eb!important;opacity:.8;}}
@media(max-width:900px) {{.block-container {{padding-left:1rem;padding-right:1rem;}}.hero {{padding:22px;min-height:0;}}}}
@media(prefers-reduced-motion:reduce) {{*,*::before,*::after {{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;}}}}

/* Contrast and legibility corrections */
[data-testid="stMainBlockContainer"] {{color:{C['text']} !important;}}
[data-testid="stMainBlockContainer"] h1,[data-testid="stMainBlockContainer"] h2,[data-testid="stMainBlockContainer"] h3,[data-testid="stMainBlockContainer"] h4 {{color:{C['text']} !important;}}
[data-testid="stMainBlockContainer"] p,[data-testid="stMainBlockContainer"] label,[data-testid="stMainBlockContainer"] li {{color:{C['text']} !important;}}
[data-testid="stDataFrame"],[data-testid="stTable"] {{background:{C['panel']} !important;border:1px solid {C['border']} !important;border-radius:14px !important;}}
[data-testid="stDataFrame"] iframe {{color-scheme:${"dark" if mode == "Dark" else "light"};}}
[data-testid="stPlotlyChart"] {{background:{C['panel']} !important;border:1px solid {C['border']} !important;border-radius:16px !important;padding:8px !important;}}
.stMarkdown .panel h3 {{color:{C['text']} !important;}}
</style>
""",unsafe_allow_html=True)

PAGE_COPY = {
    "Executive Overview": ("RESEARCH COMMAND CENTER", "🪞 TryOn Intelligence", "A clear view of AR experience, shopper confidence and omnichannel purchase behaviour."),
    "AR Experience Lab": ("EXPERIENCE QUALITY / STIMULUS", "✦ AR Experience Lab", "Explore the five experience dimensions shaping how shoppers evaluate virtual try-on."),
    "Omnichannel Journey": ("CHANNEL INTELLIGENCE / BEHAVIOUR", "↗ Omnichannel Journey", "Understand webrooming, showrooming and the connections between digital and physical retail."),
    "Customer Insights": ("SEGMENTATION / CUSTOMER SIGNALS", "◎ Customer Insights", "Compare customer groups and discover relationships between the six research constructs."),
    "Research Analytics": ("INFERENCE / HYPOTHESIS TESTING", "⌘ Research Analytics", "Test the nine proposed relationships, estimate regression effects and inspect model assumptions."),
    "Customer Segmentation": ("UNSUPERVISED LEARNING / PERSONAS", "◈ Customer Segmentation", "Discover exploratory respondent profiles from the six measured constructs."),
    "Strategy Simulator": ("SCENARIO PLANNING / WHAT-IF", "⌁ Strategy Simulator", "Explore transparent improvement scenarios for AR experience and purchase intention."),
    "Data Studio": ("DATA QUALITY / MEASUREMENT", "▦ Data Studio", "Inspect response completeness, construct reliability and respondent-level scores.")
}
page_kicker, page_title, page_subtitle = PAGE_COPY[page]
st.markdown(f"""
<div class="hero">
 <div class="kicker">✦ {page_kicker}</div>
 <h1 style="margin:.45rem 0 .35rem 0;">{page_title}</h1>
 <p style="font-size:1.04rem;color:{C['muted']};margin:0;">{page_subtitle}</p>
 <div class="hero-tags"><span>21 SURVEY ITEMS</span><span>6 RESEARCH CONSTRUCTS</span><span>S-O-R + DUAL-FACTOR LENS</span><span>INTERACTIVE ANALYTICS</span></div>
 <div class="hero-orbit" aria-hidden="true"><span></span><i></i><b></b></div>
</div>
""",unsafe_allow_html=True)

if raw.empty: st.warning("Upload a valid dataset to continue."); st.stop()
if not demo and len(mapping)!=21:
    st.warning(f"Select exactly 21 Likert item columns in the sidebar. Selected: {len(mapping)}."); st.stop()
for q,col in mapping.items():
    raw[col]=pd.to_numeric(raw[col],errors="coerce")
    raw.loc[~raw[col].between(1,5),col]=np.nan
data=score_data(raw,mapping)
names=list(CONSTRUCTS)
means=data[names].mean()
journey_col=next((c for c in data.columns if str(c).strip().lower() in ["shopping journey","channel behaviour","channel behavior","journey type"] or "shopping method" in str(c).strip().lower()),None)
category_col=next((c for c in data.columns if "category" in str(c).lower()),None)
ar_col=next((c for c in data.columns if str(c).strip().lower() in ["ar usage","used ar","ar experience"]),None)
if demo: st.caption("DEMO DATA · Synthetic illustrative responses only; not original research respondents.")
else: st.caption(f"UPLOADED DATA · {len(data):,} respondent rows · Scores calculated from the selected 21 Likert items.")

if page=="Executive Overview":
    st.subheader("Executive overview")
    st.caption("A high-level view of AR experience quality, channel choices and purchase intention.")
    a,b,c,d=st.columns(4)
    a.metric("Respondents",f"{len(data):,}")
    b.metric("AR experience quality",f"{means['AR Experience Quality']:.2f}/5")
    c.metric("Purchase intention",f"{means['Purchase Intention']:.2f}/5")
    d.metric("Overall construct mean",f"{means.mean():.2f}/5")
    left,right=st.columns([1.15,1])
    with left:
        st.markdown("#### Construct performance")
        fig=px.bar(x=means.index,y=means.values,color=means.index,color_discrete_sequence=["#7866B1","#5A9EAD","#D58BAF","#8C9ACB","#B69A72","#6F8D83"],labels={"x":"Construct","y":"Mean score"})
        fig.update_yaxes(range=[0,5],dtick=1); fig.update_layout(showlegend=False)
        st.plotly_chart(theme_chart(fig),use_container_width=True)
    with right:
        st.markdown("#### Shopping journey mix")
        if journey_col:
            vc=data[journey_col].fillna("Not specified").value_counts()
            fig=px.pie(values=vc.values,names=vc.index,hole=.62,color_discrete_sequence=["#7866B1","#5A9EAD","#D58BAF","#B69A72"])
            fig.update_traces(textposition="inside",textinfo="percent+label")
            st.plotly_chart(theme_chart(fig),use_container_width=True)
        else: st.info("Add a Shopping Journey column to analyse webrooming and showrooming.")
    st.markdown("#### S-O-R research lens")
    x,y,z=st.columns(3)
    x.markdown('<div class="panel"><div class="kicker">STIMULUS</div><h3>AR Experience Quality</h3><p class="muted">Realism, accuracy, ease, product interaction and trust.</p></div>',unsafe_allow_html=True)
    y.markdown('<div class="panel"><div class="kicker">ORGANISM</div><h3>Confidence & value</h3><p class="muted">Risk reduction and online / physical store value.</p></div>',unsafe_allow_html=True)
    z.markdown('<div class="panel"><div class="kicker">RESPONSE</div><h3>Omnichannel & intent</h3><p class="muted">Channel behaviour and purchase intention.</p></div>',unsafe_allow_html=True)

elif page=="AR Experience Lab":
    st.subheader("AR experience lab")
    st.caption("Explore realism, accuracy, ease of use, product interaction and AR trust.")
    cols=[mapping[q] for q in CONSTRUCTS["AR Experience Quality"]]
    item_means=data[cols].mean()
    labels=[LABELS[q] for q in CONSTRUCTS["AR Experience Quality"]]
    fig=px.bar(x=labels,y=item_means.values,color=item_means.values,color_continuous_scale="Tealgrn",labels={"x":"Experience dimension","y":"Mean score"})
    fig.update_yaxes(range=[0,5],dtick=1)
    st.plotly_chart(theme_chart(fig),use_container_width=True)
    l,r=st.columns(2)
    with l:
        fig=px.histogram(data,x="AR Experience Quality",nbins=10,color_discrete_sequence=[C["accent"]])
        st.plotly_chart(theme_chart(fig),use_container_width=True)
    with r:
        val=alpha(data[cols])
        st.metric("AR quality Cronbach's alpha","Not estimable" if pd.isna(val) else f"{val:.3f}")
        st.caption("Reliability is a diagnostic. Interpret it alongside sample size, dimensionality and measurement quality.")

elif page=="Omnichannel Journey":
    st.subheader("Omnichannel journey intelligence")
    st.caption("Understand how shoppers combine digital discovery and physical evaluation.")
    if journey_col:
        counts=data[journey_col].fillna("Not specified").value_counts().rename_axis("Journey").reset_index(name="Respondents")
        counts["Share"]=counts["Respondents"]/counts["Respondents"].sum()
        fig=px.bar(counts,x="Journey",y="Respondents",color="Journey",text="Respondents",color_discrete_sequence=["#8B7CFF","#4EA5FF","#23C6A8","#F4B860"])
        st.plotly_chart(theme_chart(fig),use_container_width=True)
        st.dataframe(counts.assign(Share=counts["Share"].map(lambda v:f"{v:.1%}")),use_container_width=True,hide_index=True)
        web=counts.loc[counts["Journey"].str.contains("webroom",case=False),"Share"].sum()
        show=counts.loc[counts["Journey"].str.contains("showroom",case=False),"Share"].sum()
        p,q=st.columns(2); p.metric("Webrooming share",f"{web:.1%}"); q.metric("Showrooming share",f"{show:.1%}")
    else: st.info("Add a Shopping Journey column with values such as Webrooming, Showrooming, Mixed or Online-first.")
    if category_col and journey_col:
        cross=pd.crosstab(data[category_col],data[journey_col])
        st.markdown("#### Journey by product category")
        st.plotly_chart(theme_chart(px.imshow(cross,text_auto=True,aspect="auto",color_continuous_scale="Teal")),use_container_width=True)

elif page=="Customer Insights":
    st.subheader("Customer insights")
    st.caption("Compare construct scores across product categories, AR usage groups or journey types.")
    group=category_col or ar_col or journey_col or next((c for c in data.columns if 'purchase' in str(c).lower() and ('often' in str(c).lower() or 'frequency' in str(c).lower())),None)
    if group:
        grouped=data.groupby(group,dropna=False)[names].mean().reset_index()
        long=grouped.melt(id_vars=group,var_name="Construct",value_name="Mean score")
        fig=px.bar(long,x=group,y="Mean score",color="Construct",barmode="group",color_discrete_sequence=["#8B7CFF","#4EA5FF","#23C6A8","#F4B860","#E879A5","#8B9CF6"])
        fig.update_yaxes(range=[0,5],dtick=1)
        st.plotly_chart(theme_chart(fig),use_container_width=True)
        st.dataframe(grouped,use_container_width=True,hide_index=True)
    else: st.info("Add Product Category, AR Usage or Shopping Journey to compare customer groups.")
    st.markdown("#### Construct relationship explorer")
    st.plotly_chart(theme_chart(px.imshow(data[names].corr(),text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="RdBu_r",aspect="auto")),use_container_width=True)

elif page=="Research Analytics":
    st.subheader("Research analytics & hypothesis testing")
    st.caption("Exploratory inference using your six questionnaire constructs. Results describe associations, not causal effects.")
    hypotheses = [
        ("H1","Online Shopping Value","Purchase Intention"),
        ("H2","Physical Store Value","Purchase Intention"),
        ("H3","AR Experience Quality","Purchase Intention"),
        ("H4","AR Experience Quality","AR-Enabled Risk Reduction"),
        ("H5","AR-Enabled Risk Reduction","Purchase Intention"),
        ("H6","AR Experience Quality","Online Shopping Value"),
        ("H7","AR Experience Quality","Omnichannel Behaviour"),
        ("H8","AR-Enabled Risk Reduction","Omnichannel Behaviour"),
        ("H9","Omnichannel Behaviour","Purchase Intention"),
    ]
    st.markdown("#### Hypothesis testing")
    alpha_level=st.select_slider("Significance level (α)",options=[0.01,0.05,0.10],value=0.05,format_func=lambda x:f"{x:.2f}")
    rows=[]
    for h,x,y in hypotheses:
        pair=data[[x,y]].dropna()
        if len(pair)>=4 and pair[x].nunique()>1 and pair[y].nunique()>1:
            pearson_r,pearson_p=stats.pearsonr(pair[x],pair[y])
            spear_r,spear_p=stats.spearmanr(pair[x],pair[y])
            rows.append({"Hypothesis":h,"Relationship":f"{x} → {y}","N":len(pair),"Pearson r":pearson_r,"p-value":pearson_p,"Spearman ρ":spear_r,"Decision":"Supported" if pearson_p<alpha_level else "Not supported"})
        else:
            rows.append({"Hypothesis":h,"Relationship":f"{x} → {y}","N":len(pair),"Pearson r":np.nan,"p-value":np.nan,"Spearman ρ":np.nan,"Decision":"Insufficient variation"})
    hdf=pd.DataFrame(rows)
    st.dataframe(hdf,use_container_width=True,hide_index=True,column_config={"Pearson r":st.column_config.NumberColumn(format="%.3f"),"p-value":st.column_config.NumberColumn(format="%.4f"),"Spearman ρ":st.column_config.NumberColumn(format="%.3f")})
    st.caption("Pearson tests linear association; Spearman provides a rank-based sensitivity check. The decision column uses the selected unadjusted α. Consider multiple-testing correction before strong claims.")
    st.download_button("Download hypothesis results",hdf.to_csv(index=False).encode("utf-8"),"tryon_hypothesis_tests.csv","text/csv")
    st.markdown("#### Multiple regression model")
    st.write("Outcome: Purchase Intention · Predictors: the other five constructs.")
    predictors=["Online Shopping Value","Physical Store Value","AR Experience Quality","AR-Enabled Risk Reduction","Omnichannel Behaviour"]
    model_df=data[predictors+["Purchase Intention"]].dropna()
    if len(model_df)>len(predictors)+3:
        X=sm.add_constant(model_df[predictors])
        fit=sm.OLS(model_df["Purchase Intention"],X).fit()
        ci=fit.conf_int()
        coef=pd.DataFrame({"Predictor":fit.params.index,"Coefficient":fit.params.values,"Std. error":fit.bse.values,"p-value":fit.pvalues.values,"CI lower":ci[0].values,"CI upper":ci[1].values})
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Observations",f"{int(fit.nobs)}")
        m2.metric("R²",f"{fit.rsquared:.3f}")
        m3.metric("Adjusted R²",f"{fit.rsquared_adj:.3f}")
        m4.metric("Model p-value",f"{fit.f_pvalue:.4f}" if fit.f_pvalue is not None else "N/A")
        st.dataframe(coef.iloc[1:],use_container_width=True,hide_index=True,column_config={c:st.column_config.NumberColumn(format="%.4f") for c in ["Coefficient","Std. error","p-value","CI lower","CI upper"]})
        st.markdown("#### Regression diagnostics")
        residuals=fit.resid
        shapiro_p=stats.shapiro(residuals).pvalue if len(residuals)>=3 else np.nan
        try: bp_p=het_breuschpagan(residuals,fit.model.exog)[1]
        except Exception: bp_p=np.nan
        vif=[]
        for i,col in enumerate(predictors):
            try: vif.append({"Predictor":col,"VIF":variance_inflation_factor(model_df[predictors].astype(float).values,i)})
            except Exception: vif.append({"Predictor":col,"VIF":np.nan})
        d1,d2,d3=st.columns(3)
        d1.metric("Shapiro-Wilk p",f"{shapiro_p:.4f}" if pd.notna(shapiro_p) else "N/A")
        d2.metric("Breusch-Pagan p",f"{bp_p:.4f}" if pd.notna(bp_p) else "N/A")
        d3.metric("Durbin-Watson",f"{sm.stats.stattools.durbin_watson(residuals):.3f}")
        with st.expander("View multicollinearity (VIF)"):
            st.dataframe(pd.DataFrame(vif),use_container_width=True,hide_index=True)
        diag_fig=px.scatter(x=fit.fittedvalues,y=residuals,labels={"x":"Fitted values","y":"Residuals"},title="Residuals versus fitted values")
        diag_fig.add_hline(y=0,line_dash="dash",line_color="#E879A5")
        st.plotly_chart(theme_chart(diag_fig),use_container_width=True)
        st.download_button("Download regression summary",coef.to_csv(index=False).encode("utf-8"),"tryon_regression_coefficients.csv","text/csv")
        st.caption("Small convenience sample (N≈70): treat coefficients and diagnostics as exploratory. Likert composites are treated as approximately continuous; this modelling choice should be disclosed.")
    else:
        st.warning("Not enough complete observations to estimate the regression with the selected data.")

elif page=="Customer Segmentation":
    st.subheader("Exploratory customer segmentation")
    st.caption("K-means clustering on standardized construct scores. Segments are statistical profiles, not verified market personas.")
    k=st.slider("Number of segments",2,5,3)
    cluster_data=data[names].dropna()
    if len(cluster_data)>k+2:
        scaler=StandardScaler()
        scaled=scaler.fit_transform(cluster_data)
        km=KMeans(n_clusters=k,random_state=42,n_init=20)
        labels=km.fit_predict(scaled)
        sil=silhouette_score(scaled,labels) if len(np.unique(labels))>1 and len(cluster_data)>k else np.nan
        p1,p2,p3=st.columns(3)
        p1.metric("Respondents clustered",f"{len(cluster_data):,}")
        p2.metric("Segments",str(k))
        p3.metric("Silhouette score",f"{sil:.3f}" if pd.notna(sil) else "N/A")
        pca=PCA(n_components=2)
        coords=pca.fit_transform(scaled)
        plot_df=pd.DataFrame({"PC1":coords[:,0],"PC2":coords[:,1],"Segment":pd.Series(labels,index=cluster_data.index).map(lambda x:f"Segment {x+1}")})
        fig=px.scatter(plot_df,x="PC1",y="PC2",color="Segment",opacity=.82,color_discrete_sequence=["#7866B1","#5A9EAD","#D58BAF","#8C9ACB","#B69A72"],title="Respondent map (PCA projection)")
        st.plotly_chart(theme_chart(fig),use_container_width=True)
        prof=cluster_data.assign(Segment=[f"Segment {x+1}" for x in labels]).groupby("Segment")[names].mean().round(2)
        sizes=prof.copy()
        sizes.insert(0,"Respondents",pd.Series(labels).value_counts().sort_index().values)
        st.markdown("#### Segment profiles")
        st.dataframe(sizes,use_container_width=True)
        long=prof.reset_index().melt(id_vars="Segment",var_name="Construct",value_name="Mean score")
        fig=px.line_polar(long,r="Mean score",theta="Construct",color="Segment",line_close=True,color_discrete_sequence=["#7866B1","#5A9EAD","#D58BAF","#8C9ACB","#B69A72"])
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,5],showticklabels=True)))
        st.plotly_chart(theme_chart(fig),use_container_width=True)
        export=cluster_data.copy()
        export["Segment"]=[f"Segment {x+1}" for x in labels]
        st.download_button("Download respondent segment assignments",export.to_csv(index=True).encode("utf-8"),"tryon_customer_segments.csv","text/csv")
    else:
        st.warning("Not enough complete responses for the selected number of segments.")

elif page=="Strategy Simulator":
    st.subheader("Retail strategy simulator")
    st.caption("Illustrative what-if scenarios, not causal estimates or forecasts.")
    l,r=st.columns(2)
    with l:
        realism=st.slider("AR realism improvement",0,30,10,5)
        accuracy=st.slider("AR accuracy improvement",0,30,10,5)
    with r:
        ease=st.slider("Ease-of-use improvement",0,30,10,5)
        risk=st.slider("Risk-reduction support improvement",0,30,10,5)
    base=float(means["Purchase Intention"])
    scenario=min(5.0,base+((realism+accuracy+ease+risk)/300)*.8)
    a,b,c=st.columns(3)
    a.metric("Current purchase-intention mean",f"{base:.2f}/5")
    b.metric("Illustrative scenario index",f"{scenario:.2f}/5",delta=f"+{scenario-base:.2f}")
    c.metric("Combined improvement inputs",f"{realism+accuracy+ease+risk}%")
    fig=go.Figure([go.Bar(name="Baseline",x=["Purchase intention"],y=[base],marker_color="#8B7CFF"),go.Bar(name="Scenario",x=["Purchase intention"],y=[scenario],marker_color=C["accent"])])
    fig.update_layout(barmode="group",yaxis=dict(range=[0,5],title="Index / 5"))
    st.plotly_chart(theme_chart(fig),use_container_width=True)
    st.info("The scenario index is a transparent illustration from slider inputs. Replace it with a validated regression or experiment when suitable data is available.")

elif page=="Data Studio":
    st.subheader("Survey data studio")
    st.caption("Inspect completeness, construct reliability and respondent-level scores.")
    items=[mapping[q] for q in range(1,22)]
    item_frame=data[items]
    a,b,c=st.columns(3)
    a.metric("Respondent rows",f"{len(data):,}")
    b.metric("Average item completeness",f"{item_frame.notna().mean().mean():.1%}")
    c.metric("Missing item cells",f"{int(item_frame.isna().sum().sum()):,}")
    records=[]
    for name,nums in CONSTRUCTS.items():
        cc=[mapping[q] for q in nums]
        records.append({"Construct":name,"Items":len(cc),"Cronbach's alpha":alpha(data[cc]),"Mean score":data[name].mean()})
    rel=pd.DataFrame(records)
    st.markdown("#### Construct diagnostics")
    st.dataframe(rel,use_container_width=True,hide_index=True,column_config={"Cronbach's alpha":st.column_config.NumberColumn(format="%.3f"),"Mean score":st.column_config.NumberColumn(format="%.2f")})
    st.markdown("#### Respondent preview")
    st.dataframe(data.head(15),use_container_width=True,hide_index=True)
    st.download_button("Download scored dataset",data.to_csv(index=False).encode("utf-8"),"tryon_intelligence_scored_data.csv","text/csv")
    st.download_button("Download construct summary",rel.to_csv(index=False).encode("utf-8"),"tryon_construct_summary.csv","text/csv")

st.markdown("---")
st.caption("TryOn Intelligence · AR Shopping Analytics · Upload respondent-level data for actual analysis. Built-in demo data is synthetic and illustrative.")
