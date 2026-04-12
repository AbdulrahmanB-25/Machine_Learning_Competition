import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from rli_engine import (
    build_global_ranking, recommend, property_search,
    CATEGORY_MAP, NO_ROOM_CATEGORIES, TIER_LABELS,
    run_clustering_comparison, run_regression_comparison, run_classification_comparison,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA as PCA_viz

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Riyadh Livability Index", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #edf4f5 0%, #dbe9ec 100%); }
header { visibility: hidden; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1250px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #14353d 0%, #1d4953 100%); border-right: 1px solid rgba(255,255,255,0.08); }
section[data-testid="stSidebar"] * { color: white !important; }
.section-card { background: #ffffff; border: 1px solid rgba(20,53,61,0.08); border-radius: 22px; padding: 1.2rem; box-shadow: 0 10px 25px rgba(18,38,45,0.08); margin-bottom: 1rem; }
.brand-text { background: linear-gradient(90deg, #7c3aed, #14b8a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; }
.mini-step { background: #ffffff; border-radius: 18px; padding: 1rem; color: #14353d; border: 1px solid rgba(20,53,61,0.08); box-shadow: 0 8px 20px rgba(20,53,61,0.06); height: 180px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }
.mini-step .step-no { font-size: 0.85rem; font-weight: 800; color: #6b21a8; margin-bottom: 0.35rem; text-transform: uppercase; }
.mini-step .step-title { font-size: 1.05rem; font-weight: 800; color: #10333a; margin-bottom: 0.25rem; }
.mini-step .step-text { font-size: 0.85rem; color: #315760; }
.page-banner { border-radius: 24px; padding: 1.6rem 1.8rem; background: linear-gradient(90deg, rgba(10,34,40,0.92) 0%, rgba(20,73,79,0.82) 55%, rgba(88,137,145,0.52) 100%), url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1400&q=80'); background-size: cover; background-position: center; color: white; margin-bottom: 1rem; box-shadow: 0 10px 30px rgba(18,38,45,0.12); }
.page-banner h1 { margin: 0; font-size: 2.3rem; font-weight: 900; text-transform: uppercase; color: white !important; }
.page-banner p { margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.95) !important; font-size: 1rem; max-width: 760px; }
.stButton > button { border: none; border-radius: 14px; background: linear-gradient(90deg, #7c3aed 0%, #14b8a6 100%); color: white; font-weight: 800; padding: 0.75rem 1.2rem; }
div[data-testid="stMetric"] { background: #ffffff; border: 1px solid rgba(20,53,61,0.08); padding: 1rem; border-radius: 18px; box-shadow: 0 8px 20px rgba(20,53,61,0.06); }
div[data-testid="stMetric"] label, div[data-testid="stMetric"] div { color: #10333a !important; }
div[data-testid="stDataFrame"] { background: #ffffff; border: 1px solid rgba(20,53,61,0.08); border-radius: 18px; padding: 0.3rem; }
.small-note { color: #244851; font-size: 0.95rem; margin-top: -0.25rem; }
h1, h2, h3 { color: #10333a !important; }
p, label, span { color: #244851; }
.stat-row { display: flex; gap: 1.2rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-box { flex: 1; min-width: 120px; background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 1px solid rgba(20,53,61,0.06); border-radius: 16px; padding: 1rem; text-align: center; }
.stat-box .stat-value { font-size: 1.5rem; font-weight: 900; color: #10333a; }
.stat-box .stat-label { font-size: 0.8rem; color: #5c7a82; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_raw():
    csv_path = os.environ.get("RIYADH_CSV", "Riyadh_Master_Dataset.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)

@st.cache_resource
def load_pipeline(_df_raw):
    return build_global_ranking(_df_raw)

# =========================================================
# HOME PAGE
# =========================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="page-banner" style="min-height:160px;">
        <h1><span class="brand-text">Riyadh Livability Index & Recommendation</span></h1>
        <p>Find the right neighborhood using data-driven livability scores, ML-powered recommendations, and transparent comparisons.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### How It Works")
    a, b, c, d, e = st.columns(5)
    steps = [
        ("Step 1", "Buy or Rent", "Choose your intent — Villa, Apartment, Land, or 20+ categories."),
        ("Step 2", "Set Budget", "Define your min/max price range in SAR."),
        ("Step 3", "Lifestyle Priorities", "Adjust weights: Metro, Schools, Parks, Hospitals, Fiber."),
        ("Step 4", "Instant Matching", "K-Means finds the neighborhood archetype closest to your needs."),
        ("Step 5", "Top 3 Results", "Get tailored recommendations with match %, RLI score, and price."),
    ]
    for col, (no, title, text) in zip([a,b,c,d,e], steps):
        with col:
            st.markdown(f"""
            <div class="mini-step">
                <div class="step-no">{no}</div>
                <div class="step-title">{title}</div>
                <div class="step-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-value">348K</div><div class="stat-label">Property Listings</div></div>
        <div class="stat-box"><div class="stat-value">176</div><div class="stat-label">Neighborhoods</div></div>
        <div class="stat-box"><div class="stat-value">27K</div><div class="stat-label">Points of Interest</div></div>
        <div class="stat-box"><div class="stat-value">3K+</div><div class="stat-label">Transit Stops</div></div>
        <div class="stat-box"><div class="stat-value">4</div><div class="stat-label">Livability Pillars</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    x1, x2, x3 = st.columns([1.3, 1, 1.3])
    with x2:
        if st.button("Start Exploring", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
df_raw = load_raw()
if df_raw.empty:
    st.error("Dataset not found. Make sure Riyadh_Master_Dataset.csv exists.")
    st.stop()
pipe = load_pipeline(df_raw)
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Riyadh Livability Index")
st.sidebar.markdown("Adjust the importance of each pillar")
w_core = st.sidebar.slider("Core", 0.0, 1.0, 0.40)
w_mob = st.sidebar.slider("Mobility", 0.0, 1.0, 0.25)
w_well = st.sidebar.slider("Well-being", 0.0, 1.0, 0.20)
w_inf = st.sidebar.slider("Infrastructure", 0.0, 1.0, 0.15)
user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}
st.sidebar.markdown("---")
if st.sidebar.button("Back to Home", use_container_width=True):
    st.session_state.page = "home"
    st.rerun()

# =========================================================
# MAIN BANNER
# =========================================================
st.markdown("""
<div class="page-banner">
    <h1><span class="brand-text">Riyadh Livability Explorer</span></h1>
    <p>Explore rankings, search properties, and examine the ML models behind the system.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TABS — Each gets its own
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["City Ranking", "Property Search", "Clustering", "Price Prediction", "Price Tier"])

pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
pillar_cols = [f"pillar_{p}" for p in pillar_names]
colors = ["#f0c05a", "#4fc3f7", "#ff6b6b", "#66bb6a", "#ab47bc"]
CL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#10333a", size=14), legend=dict(font=dict(color="#10333a")))
PL = dict(polar=dict(radialaxis=dict(visible=True, color="#10333a", gridcolor="rgba(0,0,0,0.15)"), angularaxis=dict(color="#10333a")), **CL)

# =========================================================
# TAB 1: RANKING
# =========================================================
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top Neighborhoods in Riyadh")
    st.markdown('<div class="small-note">All neighborhoods rescored using your pillar weights.</div>', unsafe_allow_html=True)
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)
    top3 = df_city.head(3)
    c1, c2, c3 = st.columns(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        [c1,c2,c3][i].metric(f"Rank {['1st','2nd','3rd'][i]}", name if isinstance(name,str) else str(name), f"RLI {row['RLI']:.1f}")
    fig = go.Figure()
    for idx, (name, row) in enumerate(df_city.head(5).iterrows()):
        v = [row[c] for c in pillar_cols]
        fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself", name=name if isinstance(name,str) else str(name), line=dict(color=colors[idx%5]), opacity=0.7))
    fig.update_layout(title=dict(text="Pillar Profile — Top 5", font=dict(color="#10333a", size=18)), **PL)
    st.plotly_chart(fig, use_container_width=True)
    dc = ["rank","RLI","match_pct","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
    pr = [c for c in dc if c in df_city.columns]
    st.dataframe(df_city[pr].style.background_gradient(subset=["RLI"], cmap="YlGn").format("{:.2f}", subset=[c for c in pr if c!="rank"]), use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2: PROPERTY SEARCH
# =========================================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Personalized Recommendation")
    st.markdown('<div class="small-note">Filter by category, budget, rooms. K-Means matches you to the best archetype.</div>', unsafe_allow_html=True)
    with st.container(border=True):
        co1,co2,co3,co4 = st.columns(4)
        with co1:
            ac = df_raw["category"].value_counts()
            co = [k for k in sorted(CATEGORY_MAP.keys()) if k in ac.index]
            sel_cat = st.selectbox("Category", options=co, format_func=lambda x: f"{CATEGORY_MAP[x]} ({ac.get(x,0):,})", index=co.index(3) if 3 in co else 0)
        cl = CATEGORY_MAP.get(sel_cat,"")
        with co2: min_p = st.number_input("Min Budget (SAR)", value=0, step=50000, min_value=0)
        with co3: max_p = st.number_input("Max Budget (SAR)", value=2000000, step=50000, min_value=0)
        with co4:
            if cl not in NO_ROOM_CATEGORIES: min_r = st.number_input("Min Rooms", 0, 20, 3)
            else: min_r = 0; st.markdown("**Rooms:** N/A")
    s1,s2 = st.columns(2)
    with s1: pw = st.slider("Price Importance", 0.0, 1.0, 0.20, 0.05, help="0=livability only, 1=price only")
    with s2: cw = st.slider("Cluster Match Weight", 0.0, 0.50, 0.15, 0.05, help="K-Means archetype matching influence")
    result = property_search(df_raw=pipe["df_raw"], df_ranked=df_ranked, category=sel_cat, min_price=min_p, max_price=max_p, min_rooms=min_r, pillar_weights=user_weights, price_weight=pw, cluster_weight=cw)
    df_res = result["results"]; props_matched = result["properties_matched"]
    if df_res.empty:
        st.warning("No neighborhoods found. Try a higher budget, fewer rooms, or another category.")
    else:
        bc_id = result.get("best_cluster",-1); cs = result.get("cluster_scores",{})
        st.success(f"{props_matched:,} properties matched → {len(df_res)} neighborhoods qualify")
        if bc_id >= 0 and cs:
            ci1,ci2 = st.columns([1,2])
            with ci1: st.metric("Best-Fit Cluster", f"Cluster {bc_id}", f"Score {cs.get(bc_id,0):.4f}")
            with ci2: st.caption("K-Means matched your lifestyle priorities to the neighborhood archetype that best aligns with your pillar weights.")
        t3 = df_res.head(3); cs3 = st.columns(3)
        for i, (name, row) in enumerate(t3.iterrows()):
            lb = name if isinstance(name,str) else str(name)
            ct = f" (C{int(row['km_cluster'])})" if 'km_cluster' in row and pd.notna(row.get('km_cluster')) else ""
            cs3[i].metric(f"#{int(row['rank'])} — {lb}{ct}", f"Match {row['match_pct']:.0f}%", f"Avg {row.get('avg_price',0):,.0f} SAR")
        fig2 = go.Figure()
        for idx, (name, row) in enumerate(df_res.head(min(5,len(df_res))).iterrows()):
            v = [row.get(c,0) for c in pillar_cols]
            fig2.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself", name=name if isinstance(name,str) else str(name), line=dict(color=colors[idx%5]), opacity=0.7))
        fig2.update_layout(title=dict(text=f"Pillar Profile — Top for {cl}", font=dict(color="#10333a", size=18)), **PL)
        st.plotly_chart(fig2, use_container_width=True)
        scols = ["rank","match_pct","combined_score","RLI","price_score","cluster_fit","km_cluster","avg_price","median_price","filtered_count","avg_area","avg_rooms","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
        ps = [c for c in scols if c in df_res.columns]
        st.dataframe(df_res[ps].style.background_gradient(subset=["match_pct"], cmap="Blues").format({"match_pct":"{:.1f}%","combined_score":"{:.2f}","RLI":"{:.2f}","price_score":"{:.1f}","cluster_fit":"{:.1f}","km_cluster":"{:.0f}","avg_price":"{:,.0f}","median_price":"{:,.0f}","filtered_count":"{:,.0f}","avg_area":"{:,.0f}","avg_rooms":"{:.1f}","pillar_Core":"{:.3f}","pillar_Mobility":"{:.3f}","pillar_Well-being":"{:.3f}","pillar_Infrastructure":"{:.3f}"}), use_container_width=True, height=520)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 3: CLUSTERING
# =========================================================
with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Neighborhood Clustering — K-Means vs Hierarchical")
    st.markdown('<div class="small-note">Both algorithms tested at k=2, 3, 5. Metrics: Silhouette (↑), Calinski-Harabasz (↑), Davies-Bouldin (↓).</div>', unsafe_allow_html=True)
    with st.spinner("Running clustering comparison..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    bm = clr[bcl]
    m1,m2,m3 = st.columns(3)
    m1.metric("Best Model", bcl); m2.metric("Silhouette", f"{bm['silhouette']:.4f}"); m3.metric("Davies-Bouldin", f"{bm['davies_bouldin']:.4f}")
    cr = [{'Model':n,'Silhouette ↑':round(m['silhouette'],4),'Calinski-Harabasz ↑':round(m['calinski_harabasz'],1),'Davies-Bouldin ↓':round(m['davies_bouldin'],4)} for n,m in clr.items()]
    st.dataframe(pd.DataFrame(cr).style.highlight_max(subset=['Silhouette ↑','Calinski-Harabasz ↑'], color='#c8e6c9').highlight_min(subset=['Davies-Bouldin ↓'], color='#c8e6c9'), use_container_width=True)
    g1,g2 = st.columns(2)
    with g1:
        sd = {n:m['silhouette'] for n,m in clr.items()}
        fs = go.Figure(go.Bar(x=list(sd.keys()), y=list(sd.values()), marker_color=['#66bb6a' if n==bcl else '#4fc3f7' for n in sd]))
        fs.update_layout(title="Silhouette Score", yaxis_title="Score", height=400, **CL)
        st.plotly_chart(fs, use_container_width=True)
    with g2:
        pc = PCA_viz(n_components=2, random_state=42).fit_transform(Xsc)
        bl = clr[bcl]['labels']
        fp = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"Cluster {l}" for l in bl], labels={'x':'PC1','y':'PC2','color':'Cluster'}, title=f"PCA Scatter — {bcl}", hover_name=df_ranked.index.tolist())
        fp.update_layout(height=400, **CL)
        st.plotly_chart(fp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4: PRICE PREDICTION
# =========================================================
with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Property Price Prediction — Regression Models")
    st.markdown('<div class="small-note">Predicting price from area, category, rooms, and neighborhood features. 4 models compared.</div>', unsafe_allow_html=True)
    with st.spinner("Training regression models (sampling 30K rows)..."):
        rr, breg, rf, rs = run_regression_comparison(pipe['df_raw'])
    br = rr[breg]
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("Best Model", breg); r2.metric("R²", f"{br['R2']:.4f}"); r3.metric("MAE", f"{br['MAE']:,.0f} SAR"); r4.metric("RMSE", f"{br['RMSE']:,.0f} SAR")
    rrw = [{'Model':n,'R² ↑':m['R2'],'MAE (SAR) ↓':f"{m['MAE']:,.0f}",'RMSE (SAR) ↓':f"{m['RMSE']:,.0f}",'MAPE (%) ↓':f"{m['MAPE']:.1f}"} for n,m in rr.items()]
    st.dataframe(pd.DataFrame(rrw), use_container_width=True)
    rg1,rg2 = st.columns(2)
    with rg1:
        r2d = {n:m['R2'] for n,m in rr.items()}
        fr = go.Figure(go.Bar(x=list(r2d.keys()), y=list(r2d.values()), marker_color=['#66bb6a' if n==breg else '#4fc3f7' for n in r2d]))
        fr.update_layout(title="R² Comparison", yaxis_title="R²", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(fr, use_container_width=True)
    with rg2:
        ya=br['y_test']; yp=br['predictions']; si=np.random.RandomState(42).choice(len(ya),min(2000,len(ya)),replace=False)
        fsc=go.Figure()
        fsc.add_trace(go.Scatter(x=ya[si],y=yp[si],mode='markers',marker=dict(size=4,opacity=0.4,color='#4fc3f7'),name='Predictions'))
        mx=max(ya.max(),yp.max())
        fsc.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',line=dict(color='#ff6b6b',dash='dash'),name='Perfect'))
        fsc.update_layout(title=f"Actual vs Predicted — {breg}",xaxis_title="Actual (SAR)",yaxis_title="Predicted (SAR)",height=400,**CL)
        st.plotly_chart(fsc, use_container_width=True)
    bmod = br['model']
    if hasattr(bmod, 'feature_importances_'):
        imp = bmod.feature_importances_
        fi = sorted(zip(rf, imp), key=lambda x:-x[1])[:10]
        fim = go.Figure(go.Bar(x=[f[1] for f in fi][::-1], y=[f[0] for f in fi][::-1], orientation='h', marker_color='#f0c05a'))
        fim.update_layout(title=f"Top 10 Features — {breg}", xaxis_title="Importance", height=400, **CL)
        st.plotly_chart(fim, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 5: CLASSIFICATION
# =========================================================
with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Price Tier Classification — Budget / Mid / Premium / Luxury")
    st.markdown('<div class="small-note">Classifying properties into price quartiles. 3 models compared.</div>', unsafe_allow_html=True)
    with st.spinner("Training classification models (sampling 30K rows)..."):
        clsr, bclsn, clss = run_classification_comparison(pipe['df_raw'])
    bc = clsr[bclsn]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Best Model", bclsn); c2.metric("Accuracy", f"{bc['Accuracy']:.4f}"); c3.metric("F1 Score", f"{bc['F1']:.4f}"); c4.metric("Precision", f"{bc['Precision']:.4f}")
    clsrw = [{'Model':n,'Accuracy ↑':m['Accuracy'],'Precision ↑':m['Precision'],'Recall ↑':m['Recall'],'F1 ↑':m['F1']} for n,m in clsr.items()]
    st.dataframe(pd.DataFrame(clsrw), use_container_width=True)
    cg1,cg2 = st.columns(2)
    with cg1:
        f1d = {n:m['F1'] for n,m in clsr.items()}
        ff1 = go.Figure(go.Bar(x=list(f1d.keys()), y=list(f1d.values()), marker_color=['#66bb6a' if n==bclsn else '#4fc3f7' for n in f1d]))
        ff1.update_layout(title="F1 Comparison", yaxis_title="F1", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(ff1, use_container_width=True)
    with cg2:
        cmd = bc['confusion_matrix']
        fcm = px.imshow(cmd, text_auto=True, x=TIER_LABELS, y=TIER_LABELS, labels=dict(x="Predicted",y="Actual",color="Count"), title=f"Confusion Matrix — {bclsn}", color_continuous_scale="Blues")
        fcm.update_layout(height=400, **CL)
        st.plotly_chart(fcm, use_container_width=True)
    pcp = precision_score(bc['y_test'], bc['predictions'], average=None)
    pcr = recall_score(bc['y_test'], bc['predictions'], average=None)
    pcf = f1_score(bc['y_test'], bc['predictions'], average=None)
    pcrows = [{'Tier':t,'Precision':round(pcp[i],4),'Recall':round(pcr[i],4),'F1':round(pcf[i],4)} for i,t in enumerate(TIER_LABELS)]
    st.markdown(f"**Per-Class Metrics — {bclsn}**")
    st.dataframe(pd.DataFrame(pcrows), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
