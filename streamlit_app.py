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
st.set_page_config(page_title="Riyadh DNA Explorer", page_icon="🏙️", layout="wide")

# =========================================================
# CITY-STYLE CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F8FAFC; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #334155;
    }

    .city-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        color: white;
        padding: 2.5rem 3rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }

    .section-header {
        color: #0F172A;
        font-weight: 800;
        font-size: 1.8rem;
        border-left: 5px solid #0D9488;
        padding-left: 1rem;
        margin-bottom: 1.5rem;
    }

    .note-box {
        background-color: #F0FDFA;
        border-left: 4px solid #2DD4BF;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #134E4A;
        margin-bottom: 1rem;
    }

    .winner-badge {
        background: linear-gradient(135deg, #059669 0%, #0D9488 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        font-weight: 800;
        display: inline-block;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgb(0 0 0 / 0.05);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
    }

    .match-card {
        background: white;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        border-top: 5px solid #0D9488;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        text-align: center;
    }

    .match-card h3 { margin: 0 0 0.3rem 0; color: #0F172A; font-size: 1.1rem; }
    .match-card .match-score { font-size: 2rem; font-weight: 900; color: #0D9488; }
    .match-card .match-detail { font-size: 0.85rem; color: #64748B; margin-top: 0.3rem; }

    .step-card {
        background: white;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        text-align: center;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05);
    }

    .step-card .step-num { font-size: 0.8rem; font-weight: 800; color: #0D9488; text-transform: uppercase; }
    .step-card .step-title { font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0; }
    .step-card .step-desc { font-size: 0.85rem; color: #64748B; }

    .stat-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
    .stat-item { flex: 1; min-width: 100px; background: #F0FDFA; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #CCFBF1; }
    .stat-item .stat-val { font-size: 1.4rem; font-weight: 900; color: #0F172A; }
    .stat-item .stat-lbl { font-size: 0.75rem; color: #64748B; margin-top: 0.15rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# SHARED CHART LAYOUT
# =========================================================
CL = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
          font=dict(color='#0F172A', size=13), legend=dict(font=dict(color='#0F172A')))
COLORS = ['#0D9488', '#F59E0B', '#EF4444', '#8B5CF6', '#3B82F6']
pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
pillar_cols = [f"pillar_{p}" for p in pillar_names]

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    csv = os.environ.get("RIYADH_CSV", "Riyadh_Master_Dataset.csv")
    if not os.path.exists(csv):
        return pd.DataFrame(), None
    df = pd.read_csv(csv)
    pipe = build_global_ranking(df)
    return df, pipe

df_raw, pipe = load_data()
if df_raw.empty or pipe is None:
    st.error("Riyadh_Master_Dataset.csv not found.")
    st.stop()
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### **RIYADH DNA**")
    st.caption("Urban Intelligence Platform")
    st.write("---")

    page = st.radio("Navigate:", [
        "🏙️ City Overview",
        "🎯 Recommender",
        "🧬 Clustering",
        "💸 Price Prediction",
        "🏷️ Price Tier",
    ])

    st.write("---")
    st.markdown("**Pillar Weights**")
    w_core = st.slider("Core", 0.0, 1.0, 0.40, key="wc")
    w_mob = st.slider("Mobility", 0.0, 1.0, 0.25, key="wm")
    w_well = st.slider("Well-being", 0.0, 1.0, 0.20, key="ww")
    w_inf = st.slider("Infrastructure", 0.0, 1.0, 0.15, key="wi")
    user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

# =========================================================
# PAGE 1: CITY OVERVIEW
# =========================================================
if page == "🏙️ City Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1 style='color:white; margin:0; font-size:2.6rem;'>RIYADH DNA EXPLORER</h1>
        <p style='color:#CCFBF1; font-size:1.15rem; margin-top:0.5rem;'>The Scientific Blueprint of the 15-Minute City</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div class="stat-row">
        <div class="stat-item"><div class="stat-val">348K</div><div class="stat-lbl">Listings</div></div>
        <div class="stat-item"><div class="stat-val">176</div><div class="stat-lbl">Districts</div></div>
        <div class="stat-item"><div class="stat-val">27K</div><div class="stat-lbl">POI Venues</div></div>
        <div class="stat-item"><div class="stat-val">3K+</div><div class="stat-lbl">Transit Stops</div></div>
        <div class="stat-item"><div class="stat-val">4</div><div class="stat-lbl">Pillars</div></div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("#### How It Works")
    steps = [
        ("Step 1", "Buy or Rent", "Choose your property intent from 23 categories."),
        ("Step 2", "Set Budget", "Define min/max price range in SAR."),
        ("Step 3", "Priorities", "Adjust pillar weights for what matters to you."),
        ("Step 4", "ML Matching", "K-Means finds your best neighborhood archetype."),
        ("Step 5", "Top Results", "Get ranked recommendations with match scores."),
    ]
    scols = st.columns(5)
    for col, (num, title, desc) in zip(scols, steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # City ranking
    st.markdown("<h2 class='section-header'>City-Wide Livability Ranking</h2>", unsafe_allow_html=True)

    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)
    top3 = df_city.head(3)
    mc = st.columns(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        mc[i].metric(f"Rank {['1st','2nd','3rd'][i]}", name if isinstance(name,str) else str(name), f"RLI {row['RLI']:.1f}")

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure()
        for idx, (name, row) in enumerate(df_city.head(5).iterrows()):
            v = [row[c] for c in pillar_cols]
            fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself",
                          name=name if isinstance(name,str) else str(name), line=dict(color=COLORS[idx%5]), opacity=0.7))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, color="#0F172A"), angularaxis=dict(color="#0F172A")),
                          title="Pillar Profile — Top 5", height=420, **CL)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("""
        <div class='note-box'>
        <b>The RLI Formula</b><br>
        Our Livability Index applies <b>Shannon Entropy</b> to reward neighborhoods with <i>diverse</i> services.
        A district strong in only one pillar is penalized — the 15-Minute City ideal requires balance across health, mobility, well-being, and infrastructure.
        </div>
        """, unsafe_allow_html=True)

    dc = ["rank","RLI","match_pct","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
    pr = [c for c in dc if c in df_city.columns]
    st.dataframe(df_city[pr].style.background_gradient(subset=["RLI"], cmap="GnBu")
                 .format("{:.2f}", subset=[c for c in pr if c!="rank"]),
                 use_container_width=True, height=450)

# =========================================================
# PAGE 2: RECOMMENDER
# =========================================================
elif page == "🎯 Recommender":
    st.markdown("<h2 class='section-header'>Neighborhood Matchmaker</h2>", unsafe_allow_html=True)

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        ac = df_raw["category"].value_counts()
        co = [k for k in sorted(CATEGORY_MAP.keys()) if k in ac.index]
        sel_cat = st.selectbox("Looking for:", options=co, format_func=lambda x: CATEGORY_MAP[x],
                               index=co.index(3) if 3 in co else 0)
    cl = CATEGORY_MAP.get(sel_cat, "")
    with r2:
        min_p = st.number_input("Min Budget (SAR)", value=0, step=50000, min_value=0)
    with r3:
        max_p = st.number_input("Max Budget (SAR)", value=2000000, step=50000, min_value=0)
    with r4:
        if cl not in NO_ROOM_CATEGORIES:
            min_r = st.number_input("Min Rooms", 0, 20, 3)
        else:
            min_r = 0
            st.markdown("**Rooms:** N/A")
    st.markdown("</div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1: pw = st.slider("Price vs Quality", 0.0, 1.0, 0.20, 0.05)
    with s2: cw = st.slider("Cluster Match Weight", 0.0, 0.50, 0.15, 0.05)

    result = property_search(df_raw, df_ranked, category=sel_cat, min_price=min_p, max_price=max_p,
                             min_rooms=min_r, pillar_weights=user_weights, price_weight=pw, cluster_weight=cw)
    df_res = result["results"]

    if df_res.empty:
        st.warning("No matches found. Try widening your filters.")
    else:
        bc_id = result.get("best_cluster", -1)
        st.success(f"**{result['properties_matched']:,}** properties matched → **{len(df_res)}** neighborhoods qualify")

        if bc_id >= 0:
            st.markdown(f"<div class='note-box'>K-Means identified <b>Cluster {bc_id}</b> as the best-fit neighborhood archetype for your priorities.</div>", unsafe_allow_html=True)

        # Top 3 match cards
        top3 = df_res.head(3)
        mc = st.columns(3)
        for i, (name, row) in enumerate(top3.iterrows()):
            lb = name if isinstance(name,str) else str(name)
            ct = f" · C{int(row['km_cluster'])}" if 'km_cluster' in row and pd.notna(row.get('km_cluster')) else ""
            with mc[i]:
                st.markdown(f"""
                <div class="match-card">
                    <h3>{lb}</h3>
                    <div class="match-score">{row['match_pct']:.0f}%</div>
                    <div class="match-detail">RLI {row['RLI']:.1f} · Avg {row.get('avg_price',0):,.0f} SAR{ct}</div>
                </div>
                """, unsafe_allow_html=True)

        # Radar
        fig2 = go.Figure()
        for idx, (name, row) in enumerate(df_res.head(5).iterrows()):
            v = [row.get(c,0) for c in pillar_cols]
            fig2.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself",
                           name=name if isinstance(name,str) else str(name), line=dict(color=COLORS[idx%5]), opacity=0.7))
        fig2.update_layout(polar=dict(radialaxis=dict(visible=True)), title=f"Pillar Profile — Top for {cl}", height=420, **CL)
        st.plotly_chart(fig2, use_container_width=True)

        # Table
        scols = ["rank","match_pct","combined_score","RLI","price_score","cluster_fit","km_cluster",
                 "avg_price","median_price","filtered_count","avg_area","avg_rooms",
                 "pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
        ps = [c for c in scols if c in df_res.columns]
        st.dataframe(df_res[ps].style.background_gradient(subset=["match_pct"], cmap="GnBu")
                     .format({"match_pct":"{:.1f}%","combined_score":"{:.2f}","RLI":"{:.2f}","price_score":"{:.1f}",
                              "cluster_fit":"{:.1f}","km_cluster":"{:.0f}","avg_price":"{:,.0f}","median_price":"{:,.0f}",
                              "filtered_count":"{:,.0f}","avg_area":"{:,.0f}","avg_rooms":"{:.1f}",
                              "pillar_Core":"{:.3f}","pillar_Mobility":"{:.3f}","pillar_Well-being":"{:.3f}","pillar_Infrastructure":"{:.3f}"}),
                     use_container_width=True, height=500)

# =========================================================
# PAGE 3: CLUSTERING
# =========================================================
elif page == "🧬 Clustering":
    st.markdown("<h2 class='section-header'>Neighborhood Archetype Analysis</h2>", unsafe_allow_html=True)

    with st.spinner("Analyzing Urban DNA..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)

    st.markdown(f"<div class='winner-badge'>🏆 Best: {bcl} — Silhouette {clr[bcl]['silhouette']:.4f}</div>", unsafe_allow_html=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Best Model", bcl)
    m2.metric("Silhouette ↑", f"{clr[bcl]['silhouette']:.4f}")
    m3.metric("Davies-Bouldin ↓", f"{clr[bcl]['davies_bouldin']:.4f}")

    # Comparison table
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    cr_df = pd.DataFrame([{
        'Model': n, 'Silhouette ↑': round(m['silhouette'],4),
        'Calinski-Harabasz ↑': round(m['calinski_harabasz'],1),
        'Davies-Bouldin ↓': round(m['davies_bouldin'],4),
    } for n, m in clr.items()])
    st.dataframe(cr_df.style.highlight_max(subset=['Silhouette ↑','Calinski-Harabasz ↑'], color='#CCFBF1')
                 .highlight_min(subset=['Davies-Bouldin ↓'], color='#CCFBF1'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    g1, g2 = st.columns(2)
    with g1:
        sd = {n: m['silhouette'] for n, m in clr.items()}
        fs = go.Figure(go.Bar(x=list(sd.keys()), y=list(sd.values()),
                              marker_color=['#059669' if n==bcl else '#94A3B8' for n in sd]))
        fs.update_layout(title="Silhouette Score Comparison", yaxis_title="Score", height=420, **CL)
        st.plotly_chart(fs, use_container_width=True)
    with g2:
        pc = PCA_viz(n_components=2, random_state=42).fit_transform(Xsc)
        fp = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"Cluster {l}" for l in clr[bcl]['labels']],
                        labels={'x':'Urban Dimension 1','y':'Urban Dimension 2'},
                        title=f"PCA Visualization — {bcl}", hover_name=df_ranked.index.tolist(),
                        color_discrete_sequence=COLORS)
        fp.update_layout(height=420, **CL)
        st.plotly_chart(fp, use_container_width=True)

    st.markdown("""
    <div class='note-box'>
    <b>What are clusters?</b><br>
    Unsupervised K-Means groups Riyadh's 176 neighborhoods into archetypes based on service density, transit access, and connectivity.
    <b>Silhouette Score</b> measures how distinct each group is — higher means better-defined archetypes.
    The best cluster is used in the Recommender to match users to their ideal neighborhood type.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 4: PRICE PREDICTION
# =========================================================
elif page == "💸 Price Prediction":
    st.markdown("<h2 class='section-header'>Market Value Prediction</h2>", unsafe_allow_html=True)

    with st.spinner("Training Price Oracle (30K sample)..."):
        rr, breg, rf, rs = run_regression_comparison(df_raw)

    br = rr[breg]
    st.markdown(f"<div class='winner-badge'>🏆 Best: {breg} — R² = {br['R2']:.4f}</div>", unsafe_allow_html=True)

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", breg)
    m2.metric("R² Score", f"{br['R2']:.4f}")
    m3.metric("MAE", f"{br['MAE']:,.0f} SAR")
    m4.metric("RMSE", f"{br['RMSE']:,.0f} SAR")

    # Comparison table
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    rdf = pd.DataFrame([{
        'Model': n, 'R² ↑': m['R2'], 'MAE (SAR) ↓': f"{m['MAE']:,.0f}",
        'RMSE (SAR) ↓': f"{m['RMSE']:,.0f}", 'MAPE (%) ↓': f"{m['MAPE']:.1f}",
    } for n, m in rr.items()])
    st.dataframe(rdf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    g1, g2 = st.columns(2)
    with g1:
        r2d = {n: m['R2'] for n, m in rr.items()}
        fr = go.Figure(go.Bar(x=list(r2d.keys()), y=list(r2d.values()),
                              marker_color=['#059669' if n==breg else '#94A3B8' for n in r2d]))
        fr.update_layout(title="R² Score Comparison", yaxis_title="R²", yaxis_range=[0,1], height=420, **CL)
        st.plotly_chart(fr, use_container_width=True)
    with g2:
        ya = br['y_test']; yp = br['predictions']
        si = np.random.RandomState(42).choice(len(ya), min(2000, len(ya)), replace=False)
        fsc = go.Figure()
        fsc.add_trace(go.Scatter(x=ya[si], y=yp[si], mode='markers',
                                 marker=dict(size=4, opacity=0.4, color='#0D9488'), name='Predictions'))
        mx = max(ya.max(), yp.max())
        fsc.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode='lines',
                                  line=dict(color='#EF4444', dash='dash'), name='Perfect'))
        fsc.update_layout(title=f"Actual vs Predicted — {breg}", xaxis_title="True Price (SAR)",
                          yaxis_title="Predicted (SAR)", height=420, **CL)
        st.plotly_chart(fsc, use_container_width=True)

    # Feature importance
    bmod = br['model']
    if hasattr(bmod, 'feature_importances_'):
        imp = bmod.feature_importances_
        fi = sorted(zip(rf, imp), key=lambda x: -x[1])[:10]
        fim = go.Figure(go.Bar(x=[f[1] for f in fi][::-1], y=[f[0] for f in fi][::-1],
                               orientation='h', marker_color='#F59E0B'))
        fim.update_layout(title=f"Top 10 Features — {breg}", xaxis_title="Importance", height=400, **CL)
        st.plotly_chart(fim, use_container_width=True)

    st.markdown("""
    <div class='note-box'>
    <b>Why this matters:</b><br>
    The Random Forest model achieves R² ≈ 0.92, meaning it explains 92% of price variance using only neighborhood features and property attributes.
    <b>Area</b> is the strongest predictor, followed by category type and neighborhood service density.
    This validates that our livability features genuinely predict market value.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE 5: PRICE TIER CLASSIFICATION
# =========================================================
elif page == "🏷️ Price Tier":
    st.markdown("<h2 class='section-header'>Price Tier Classification</h2>", unsafe_allow_html=True)

    with st.spinner("Segmenting Market Tiers (30K sample)..."):
        clsr, bclsn, clss = run_classification_comparison(df_raw)

    bc = clsr[bclsn]
    st.markdown(f"<div class='winner-badge'>🏆 Best: {bclsn} — F1 = {bc['F1']:.4f}</div>", unsafe_allow_html=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", bclsn)
    m2.metric("Accuracy", f"{bc['Accuracy']:.4f}")
    m3.metric("F1 Score", f"{bc['F1']:.4f}")
    m4.metric("Precision", f"{bc['Precision']:.4f}")

    # Comparison table
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    cdf = pd.DataFrame([{
        'Model': n, 'Accuracy ↑': m['Accuracy'], 'Precision ↑': m['Precision'],
        'Recall ↑': m['Recall'], 'F1 ↑': m['F1'],
    } for n, m in clsr.items()])
    st.dataframe(cdf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Charts
    g1, g2 = st.columns(2)
    with g1:
        f1d = {n: m['F1'] for n, m in clsr.items()}
        ff1 = go.Figure(go.Bar(x=list(f1d.keys()), y=list(f1d.values()),
                               marker_color=['#059669' if n==bclsn else '#94A3B8' for n in f1d]))
        ff1.update_layout(title="F1 Score Comparison", yaxis_title="F1", yaxis_range=[0,1], height=420, **CL)
        st.plotly_chart(ff1, use_container_width=True)
    with g2:
        cm = bc['confusion_matrix']
        fcm = px.imshow(cm, text_auto=True, x=TIER_LABELS, y=TIER_LABELS,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        title=f"Confusion Matrix — {bclsn}", color_continuous_scale="Mint")
        fcm.update_layout(height=420, **CL)
        st.plotly_chart(fcm, use_container_width=True)

    # Per-class
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    st.markdown(f"**Per-Class Metrics — {bclsn}**")
    pcp = precision_score(bc['y_test'], bc['predictions'], average=None)
    pcr = recall_score(bc['y_test'], bc['predictions'], average=None)
    pcf = f1_score(bc['y_test'], bc['predictions'], average=None)
    pcdf = pd.DataFrame([{'Tier': t, 'Precision': round(pcp[i],4), 'Recall': round(pcr[i],4), 'F1': round(pcf[i],4)}
                         for i, t in enumerate(TIER_LABELS)])
    st.dataframe(pcdf.style.background_gradient(subset=['F1'], cmap='GnBu'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class='note-box'>
    <b>How to read the confusion matrix:</b><br>
    Each row is the <i>actual</i> price tier, each column is what the model <i>predicted</i>.
    Diagonal = correct predictions. Off-diagonal = misclassifications.
    <b>Precision:</b> When the model says "Luxury", how often is it right?
    <b>Recall:</b> How many actual Luxury homes did we catch?
    </div>
    """, unsafe_allow_html=True)
