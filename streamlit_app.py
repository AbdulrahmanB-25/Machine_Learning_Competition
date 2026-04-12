import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from rli_engine import (
    build_global_ranking, recommend, property_search,
    CATEGORY_MAP, NO_ROOM_CATEGORIES, TIER_LABELS, PILLARS,
    run_clustering_comparison, run_regression_comparison, run_classification_comparison,
)
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA as PCA_viz

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Riyadh DNA Explorer", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

# =========================================================
# CSS — All text forced dark on light backgrounds
# =========================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%); }

    /* Force all main-area text dark */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
    .stApp span, .stApp label, .stApp div, .stApp li {
        color: #1E293B !important;
    }

    .city-card {
        background: white; padding: 1.5rem; border-radius: 12px;
        border: 1px solid #E2E8F0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
        margin-bottom: 1rem; color: #1E293B !important;
    }
    .city-card h3 { color: #0F172A !important; margin: 0 0 0.4rem 0; }

    .hero-banner {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        padding: 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.12);
    }
    .hero-banner h1, .hero-banner p { color: white !important; }
    .hero-banner p { color: #CCFBF1 !important; }

    .section-header {
        color: #0F172A !important; font-weight: 800; font-size: 1.7rem;
        border-left: 6px solid #0D9488; padding-left: 1rem; margin: 1.5rem 0 1rem 0;
    }

    .note-box {
        background: #F0FDFA; border-left: 4px solid #2DD4BF;
        padding: 1rem; border-radius: 8px; color: #134E4A !important;
        margin-bottom: 1rem; font-size: 0.9rem;
    }
    .note-box b, .note-box i { color: #134E4A !important; }

    .winner-badge {
        background: linear-gradient(135deg, #059669, #0D9488);
        color: white !important; padding: 0.6rem 1.1rem; border-radius: 10px;
        font-weight: 800; display: inline-block; margin-bottom: 1rem;
    }
    .winner-badge * { color: white !important; }

    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem;
    }

    .match-card {
        background: white; border-radius: 14px; border: 1px solid #E2E8F0;
        border-top: 5px solid #0D9488; padding: 1.2rem; text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.06); margin-bottom: 0.5rem;
    }
    .match-card h3 { color: #0F172A !important; font-size: 1.05rem; margin: 0 0 0.3rem 0; }
    .match-card .score { font-size: 2rem; font-weight: 900; color: #0D9488 !important; }
    .match-card .detail { font-size: 0.85rem; color: #64748B !important; margin-top: 0.3rem; }

    .step-card {
        background: white; border-radius: 14px; border: 1px solid #E2E8F0;
        padding: 1.2rem; text-align: center; height: 170px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 3px 6px rgba(0,0,0,0.05);
    }
    .step-card .sn { font-size: 0.78rem; font-weight: 800; color: #0D9488 !important; text-transform: uppercase; }
    .step-card .st { font-size: 1rem; font-weight: 800; color: #0F172A !important; margin: 0.25rem 0; }
    .step-card .sd { font-size: 0.82rem; color: #64748B !important; }

    .stat-row { display: flex; gap: 0.8rem; margin: 1rem 0; flex-wrap: wrap; }
    .stat-item {
        flex: 1; min-width: 100px; background: white; border-radius: 12px;
        padding: 0.9rem; text-align: center; border: 1px solid #E2E8F0;
    }
    .stat-item .sv { font-size: 1.4rem; font-weight: 900; color: #0F172A !important; }
    .stat-item .sl { font-size: 0.75rem; color: #64748B !important; margin-top: 0.15rem; }

    /* Plotly chart text override */
    .js-plotly-plot .plotly .gtitle, .js-plotly-plot .plotly text { fill: #1E293B !important; }

    .stButton > button {
        border: none; border-radius: 12px;
        background: linear-gradient(90deg, #0D9488, #059669);
        color: white !important; font-weight: 700; padding: 0.7rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def get_data():
    csv = os.environ.get("RIYADH_CSV", "Riyadh_Master_Dataset.csv")
    if not os.path.exists(csv):
        return None, None
    df = pd.read_csv(csv)
    pipe = build_global_ranking(df)
    return df, pipe

df_raw, pipe = get_data()
if df_raw is None or pipe is None:
    st.error("Riyadh_Master_Dataset.csv not found."); st.stop()
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR — Navigation only (no sliders here)
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='color:white;margin-bottom:0;'>🏙️ Riyadh DNA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8;font-size:0.85rem;'>Urban Intelligence Dashboard</p>", unsafe_allow_html=True)
    st.write("---")
    page = st.radio("Navigate:", [
        "🏠 Home",
        "📊 City Ranking",
        "🎯 Recommender",
        "🧬 Clustering",
        "💸 Price Prediction",
        "🏷️ Price Tier",
    ])
    st.write("---")
    st.caption("RLI v2.4 · Saudi Vision 2030")

# =========================================================
# SHARED
# =========================================================
pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
pillar_cols = [f"pillar_{p}" for p in pillar_names]
COLORS = ['#0D9488', '#F59E0B', '#EF4444', '#8B5CF6', '#3B82F6']
CL = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF', font=dict(color='#1E293B', size=13))

# =========================================================
# HOME
# =========================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-banner">
        <h1 style='font-size:2.5rem;margin:0;'>RIYADH LIVABILITY INDEX</h1>
        <p style='font-size:1.1rem;margin-top:0.5rem;'>Helping residents find the right neighborhood using data-driven livability scores and ML-powered recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    steps = [
        ("Step 1","Buy or Rent","Choose from 23 property categories."),
        ("Step 2","Set Budget","Define your SAR price range."),
        ("Step 3","Priorities","Adjust weights: Schools, Metro, Parks..."),
        ("Step 4","ML Matching","K-Means finds your best archetype."),
        ("Step 5","Top Results","Ranked by livability + price + cluster fit."),
    ]
    for col, (n, t, d) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step-card"><div class="sn">{n}</div><div class="st">{t}</div><div class="sd">{d}</div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-item"><div class="sv">348K</div><div class="sl">Listings</div></div>
        <div class="stat-item"><div class="sv">176</div><div class="sl">Districts</div></div>
        <div class="stat-item"><div class="sv">27K</div><div class="sl">POI Venues</div></div>
        <div class="stat-item"><div class="sv">3K+</div><div class="sl">Transit Stops</div></div>
        <div class="stat-item"><div class="sv">4</div><div class="sl">Pillars</div></div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CITY RANKING — pillar weights live HERE
# =========================================================
elif page == "📊 City Ranking":
    st.markdown("<h2 class='section-header'>City-Wide Livability Ranking</h2>", unsafe_allow_html=True)

    with st.expander("⚙️ RLI Pillar Weights", expanded=False):
        wc1, wc2 = st.columns(2)
        with wc1:
            w_core = st.slider("Core (Schools / Health)", 0.0, 1.0, 0.40, key="r_c")
            w_mob  = st.slider("Mobility (Metro / Bus)", 0.0, 1.0, 0.25, key="r_m")
        with wc2:
            w_well = st.slider("Well-being (Parks / Dining)", 0.0, 1.0, 0.20, key="r_w")
            w_inf  = st.slider("Infrastructure (Fiber / Gov)", 0.0, 1.0, 0.15, key="r_i")
    uw = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

    df_city = recommend(df_ranked, user_weights=uw, top_n=0)
    top3 = df_city.head(3)
    mc = st.columns(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        mc[i].metric(f"{'🥇🥈🥉'[i]} Rank {i+1}", name if isinstance(name,str) else str(name), f"RLI {row['RLI']:.1f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for idx, (name, row) in enumerate(df_city.head(5).iterrows()):
            v = [row[c] for c in pillar_cols]
            fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself",
                          name=name if isinstance(name,str) else str(name), line=dict(color=COLORS[idx%5]), opacity=0.7))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), title="Pillar Profile — Top 5", height=400, **CL)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("""
        <div class='note-box'>
        <b>How RLI Works:</b><br>
        Each neighborhood is scored across 4 pillars, then boosted by <b>Shannon Entropy</b> — rewarding districts that are balanced across all services, not just strong in one area.
        </div>
        """, unsafe_allow_html=True)

    dc = ["rank","RLI","match_pct","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
    pr = [c for c in dc if c in df_city.columns]
    st.dataframe(df_city[pr].style.background_gradient(subset=["RLI"], cmap="GnBu").format("{:.2f}", subset=[c for c in pr if c!="rank"]), use_container_width=True, height=450)

# =========================================================
# RECOMMENDER — pillar weights live HERE too
# =========================================================
elif page == "🎯 Recommender":
    st.markdown("<h2 class='section-header'>Neighborhood Matchmaker</h2>", unsafe_allow_html=True)

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    r1,r2,r3,r4 = st.columns(4)
    with r1:
        ac = df_raw["category"].value_counts()
        co = [k for k in sorted(CATEGORY_MAP.keys()) if k in ac.index]
        sel_cat = st.selectbox("Property Type", co, format_func=lambda x: CATEGORY_MAP[x], index=co.index(3) if 3 in co else 0)
    cl = CATEGORY_MAP.get(sel_cat, "")
    with r2: min_p = st.number_input("Min Budget (SAR)", value=0, step=50000, min_value=0)
    with r3: max_p = st.number_input("Max Budget (SAR)", value=2000000, step=50000, min_value=0)
    with r4:
        if cl not in NO_ROOM_CATEGORIES: min_r = st.number_input("Min Rooms", 0, 20, 3)
        else: min_r = 0; st.markdown("**Rooms:** N/A")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("⚙️ Matching Parameters", expanded=False):
        pc1, pc2 = st.columns(2)
        with pc1:
            pw = st.slider("Price vs Quality", 0.0, 1.0, 0.20, 0.05)
            cw = st.slider("Cluster Match Weight", 0.0, 0.50, 0.15, 0.05)
        with pc2:
            w_core = st.slider("Core", 0.0, 1.0, 0.40, key="p_c")
            w_mob  = st.slider("Mobility", 0.0, 1.0, 0.25, key="p_m")
            w_well = st.slider("Well-being", 0.0, 1.0, 0.20, key="p_w")
            w_inf  = st.slider("Infrastructure", 0.0, 1.0, 0.15, key="p_i")
    uw = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

    result = property_search(df_raw, df_ranked, category=sel_cat, min_price=min_p, max_price=max_p, min_rooms=min_r, pillar_weights=uw, price_weight=pw, cluster_weight=cw)
    df_res = result["results"]

    if df_res.empty:
        st.warning("No matches. Try widening your filters.")
    else:
        bc_id = result.get("best_cluster", -1)
        st.success(f"**{result['properties_matched']:,}** properties → **{len(df_res)}** neighborhoods qualify")
        if bc_id >= 0:
            st.markdown(f"<div class='note-box'>K-Means identified <b>Cluster {bc_id}</b> as the best-fit archetype for your priorities.</div>", unsafe_allow_html=True)

        top3 = df_res.head(3); mc = st.columns(3)
        for i, (name, row) in enumerate(top3.iterrows()):
            lb = name if isinstance(name,str) else str(name)
            ct = f" · C{int(row['km_cluster'])}" if 'km_cluster' in row and pd.notna(row.get('km_cluster')) else ""
            with mc[i]:
                st.markdown(f'<div class="match-card"><h3>{lb}</h3><div class="score">{row["match_pct"]:.0f}%</div><div class="detail">RLI {row["RLI"]:.1f} · Avg {row.get("avg_price",0):,.0f} SAR{ct}</div></div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        for idx, (name, row) in enumerate(df_res.head(5).iterrows()):
            v = [row.get(c,0) for c in pillar_cols]
            fig2.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself", name=name if isinstance(name,str) else str(name), line=dict(color=COLORS[idx%5]), opacity=0.7))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"Top for {cl}", height=400, **CL)
        st.plotly_chart(fig2, use_container_width=True)

        scols = ["rank","match_pct","combined_score","RLI","price_score","cluster_fit","km_cluster","avg_price","filtered_count","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
        ps = [c for c in scols if c in df_res.columns]
        st.dataframe(df_res[ps].style.background_gradient(subset=["match_pct"], cmap="GnBu"), use_container_width=True, height=400)

# =========================================================
# CLUSTERING
# =========================================================
elif page == "🧬 Clustering":
    st.markdown("<h2 class='section-header'>Neighborhood Archetype Analysis</h2>", unsafe_allow_html=True)
    with st.spinner("Analyzing Urban DNA..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    st.markdown(f"<div class='winner-badge'>🏆 Best: {bcl} — Silhouette {clr[bcl]['silhouette']:.4f}</div>", unsafe_allow_html=True)

    m1,m2,m3 = st.columns(3)
    m1.metric("Best Model", bcl)
    m2.metric("Silhouette ↑", f"{clr[bcl]['silhouette']:.4f}")
    m3.metric("Davies-Bouldin ↓", f"{clr[bcl]['davies_bouldin']:.4f}")

    # Progress bars for each model
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    st.markdown("**Model Comparison**")
    for name, metrics in clr.items():
        tag = " ✅" if name == bcl else ""
        st.markdown(f"**{name}**{tag}")
        st.progress(min(metrics['silhouette'] / 0.25, 1.0))
        st.caption(f"Silhouette: {metrics['silhouette']:.4f} · CH: {metrics['calinski_harabasz']:.1f} · DB: {metrics['davies_bouldin']:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        sd = {n: m['silhouette'] for n,m in clr.items()}
        fs = go.Figure(go.Bar(x=list(sd.keys()), y=list(sd.values()), marker_color=['#059669' if n==bcl else '#CBD5E1' for n in sd]))
        fs.update_layout(title="Silhouette Scores", yaxis_title="Score", height=400, **CL)
        st.plotly_chart(fs, use_container_width=True)
    with g2:
        pc = PCA_viz(n_components=2, random_state=42).fit_transform(Xsc)
        fp = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"Cluster {l}" for l in clr[bcl]['labels']], title=f"PCA — {bcl}", hover_name=df_ranked.index.tolist(), color_discrete_sequence=COLORS)
        fp.update_layout(height=400, **CL)
        st.plotly_chart(fp, use_container_width=True)

    st.markdown("<div class='note-box'><b>What are clusters?</b> K-Means groups 176 neighborhoods into archetypes by service density, transit, and connectivity. The best cluster is used in the Recommender to match users to their ideal neighborhood type.</div>", unsafe_allow_html=True)

# =========================================================
# PRICE PREDICTION
# =========================================================
elif page == "💸 Price Prediction":
    st.markdown("<h2 class='section-header'>Market Value Prediction</h2>", unsafe_allow_html=True)
    with st.spinner("Training Price Oracle (30K sample)..."):
        rr, breg, rf, rs = run_regression_comparison(df_raw)
    br = rr[breg]
    st.markdown(f"<div class='winner-badge'>🏆 Best: {breg} — R² = {br['R2']:.4f}</div>", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Best Model", breg); m2.metric("R²", f"{br['R2']:.4f}"); m3.metric("MAE", f"{br['MAE']:,.0f} SAR"); m4.metric("RMSE", f"{br['RMSE']:,.0f} SAR")

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    rdf = pd.DataFrame([{'Model':n,'R² ↑':m['R2'],'MAE ↓':f"{m['MAE']:,.0f}",'RMSE ↓':f"{m['RMSE']:,.0f}",'MAPE ↓':f"{m['MAPE']:.1f}%"} for n,m in rr.items()])
    st.dataframe(rdf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        r2d = {n:m['R2'] for n,m in rr.items()}
        fr = go.Figure(go.Bar(x=list(r2d.keys()), y=list(r2d.values()), marker_color=['#059669' if n==breg else '#CBD5E1' for n in r2d]))
        fr.update_layout(title="R² Comparison", yaxis_title="R²", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(fr, use_container_width=True)
    with g2:
        ya=br['y_test']; yp=br['predictions']; si=np.random.RandomState(42).choice(len(ya),min(2000,len(ya)),replace=False)
        fsc=go.Figure()
        fsc.add_trace(go.Scatter(x=ya[si],y=yp[si],mode='markers',marker=dict(size=4,opacity=0.5,color='#0D9488'),name='Predictions'))
        mx=max(ya.max(),yp.max())
        fsc.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',line=dict(color='#EF4444',dash='dash'),name='Perfect'))
        fsc.update_layout(title=f"Actual vs Predicted — {breg}",xaxis_title="True (SAR)",yaxis_title="Predicted (SAR)",height=400,**CL)
        st.plotly_chart(fsc, use_container_width=True)

    bmod = br['model']
    if hasattr(bmod, 'feature_importances_'):
        imp = bmod.feature_importances_
        fi = sorted(zip(rf, imp), key=lambda x:-x[1])[:10]
        fim = go.Figure(go.Bar(x=[f[1] for f in fi][::-1], y=[f[0] for f in fi][::-1], orientation='h', marker_color='#F59E0B'))
        fim.update_layout(title=f"Top 10 Features — {breg}", xaxis_title="Importance", height=380, **CL)
        st.plotly_chart(fim, use_container_width=True)

    st.markdown("<div class='note-box'><b>Why this matters:</b> Random Forest explains ~92% of price variance using only neighborhood + property features. <b>Area</b> dominates, followed by category and service density.</div>", unsafe_allow_html=True)

# =========================================================
# PRICE TIER
# =========================================================
elif page == "🏷️ Price Tier":
    st.markdown("<h2 class='section-header'>Price Tier Classification</h2>", unsafe_allow_html=True)
    with st.spinner("Segmenting Market Tiers (30K sample)..."):
        clsr, bclsn, clss = run_classification_comparison(df_raw)
    bc = clsr[bclsn]
    st.markdown(f"<div class='winner-badge'>🏆 Best: {bclsn} — F1 = {bc['F1']:.4f}</div>", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Best Model", bclsn); m2.metric("Accuracy", f"{bc['Accuracy']:.4f}"); m3.metric("F1", f"{bc['F1']:.4f}"); m4.metric("Precision", f"{bc['Precision']:.4f}")

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    cdf = pd.DataFrame([{'Model':n,'Accuracy ↑':m['Accuracy'],'Precision ↑':m['Precision'],'Recall ↑':m['Recall'],'F1 ↑':m['F1']} for n,m in clsr.items()])
    st.dataframe(cdf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        f1d = {n:m['F1'] for n,m in clsr.items()}
        ff1 = go.Figure(go.Bar(x=list(f1d.keys()), y=list(f1d.values()), marker_color=['#059669' if n==bclsn else '#CBD5E1' for n in f1d]))
        ff1.update_layout(title="F1 Comparison", yaxis_title="F1", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(ff1, use_container_width=True)
    with g2:
        cm = bc['confusion_matrix']
        fcm = px.imshow(cm, text_auto=True, x=TIER_LABELS, y=TIER_LABELS, labels=dict(x="Predicted",y="Actual",color="Count"), title=f"Confusion Matrix — {bclsn}", color_continuous_scale="GnBu")
        fcm.update_layout(height=400, **CL)
        st.plotly_chart(fcm, use_container_width=True)

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    st.markdown(f"**Per-Class Metrics — {bclsn}**")
    pcp = precision_score(bc['y_test'], bc['predictions'], average=None)
    pcr = recall_score(bc['y_test'], bc['predictions'], average=None)
    pcf = f1_score(bc['y_test'], bc['predictions'], average=None)
    pcdf = pd.DataFrame([{'Tier':t,'Precision':round(pcp[i],4),'Recall':round(pcr[i],4),'F1':round(pcf[i],4)} for i,t in enumerate(TIER_LABELS)])
    st.dataframe(pcdf.style.background_gradient(subset=['F1'], cmap='GnBu'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='note-box'><b>Reading the matrix:</b> Diagonal = correct. Off-diagonal = errors. <b>Precision</b> = when it says Luxury, how often is it right? <b>Recall</b> = how many actual Luxury homes did we catch?</div>", unsafe_allow_html=True)
