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

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #edf4f5 0%, #dbe9ec 100%); }
header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }

/* Sidebar — gradient matching brand-text */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #7c3aed 0%, #14b8a6 100%) !important;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: white !important; }

.section-card { background: #ffffff; border: 1px solid rgba(20,53,61,0.08); border-radius: 22px; padding: 1.5rem; box-shadow: 0 10px 25px rgba(18,38,45,0.08); margin-bottom: 1.5rem; }
.brand-text { background: linear-gradient(90deg, #7c3aed, #14b8a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; }

.stat-row { display: flex; gap: 1.2rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-box { flex: 1; min-width: 160px; background: white; border: 1px solid rgba(20,53,61,0.1); border-radius: 16px; padding: 1.2rem; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.03); }
.stat-box .stat-value { font-size: 1.6rem; font-weight: 900; color: #10333a; }
.stat-box .stat-label { font-size: 0.82rem; color: #5c7a82; font-weight: 600; text-transform: uppercase; margin-top: 0.3rem; }

.page-banner { border-radius: 24px; padding: 2rem; background: linear-gradient(90deg, rgba(10,34,40,0.95) 0%, rgba(20,73,79,0.85) 60%), url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1400&q=80'); background-size: cover; color: white; margin-bottom: 2rem; }
.page-banner h1 { color: white !important; margin: 0; }
.page-banner p { color: rgba(255,255,255,0.9) !important; }

.note-box { background: linear-gradient(135deg, #f0fdfa 0%, #e6f9f5 100%); border-left: 4px solid #14b8a6; padding: 1rem 1.2rem; border-radius: 10px; color: #134e4a !important; margin: 1rem 0; }
.note-box b { color: #0f766e !important; }

.winner-pill { background: linear-gradient(90deg, #7c3aed, #14b8a6); color: white !important; padding: 0.5rem 1.2rem; border-radius: 20px; font-weight: 800; font-size: 0.9rem; display: inline-block; margin-bottom: 1rem; }

.metric-explain { background: white; border: 1px solid rgba(20,53,61,0.08); border-radius: 14px; padding: 0.9rem 1rem; margin-bottom: 0.8rem; }
.metric-explain .me-name { font-weight: 800; color: #10333a; font-size: 0.95rem; }
.metric-explain .me-val { font-size: 1.4rem; font-weight: 900; background: linear-gradient(90deg, #7c3aed, #14b8a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-explain .me-desc { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

.stButton > button { border: none; border-radius: 12px; background: linear-gradient(90deg, #7c3aed 0%, #14b8a6 100%); color: white; font-weight: 700; }
h1, h2, h3, h4 { color: #10333a !important; }
p, label { color: #244851 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_raw():
    csv_path = os.environ.get("RIYADH_CSV", "Riyadh_Master_Dataset.csv")
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

@st.cache_resource
def load_pipeline(_df_raw):
    return build_global_ranking(_df_raw)

df_raw = load_raw()
if df_raw.empty:
    st.error("Dataset not found."); st.stop()
pipe = load_pipeline(df_raw)
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🏙️ RLI Dashboard")
    page = st.radio("Navigation", ["Home", "City Ranking", "Property Search", "Clustering Analysis", "Price Prediction", "Price Tiering"])
    user_weights = {}
    if page in ["City Ranking", "Property Search"]:
        st.markdown("---")
        st.markdown("### Livability Weights")
        w_core = st.slider("Core", 0.0, 1.0, 0.40)
        w_mob = st.slider("Mobility", 0.0, 1.0, 0.25)
        w_well = st.slider("Well-being", 0.0, 1.0, 0.20)
        w_inf = st.slider("Infrastructure", 0.0, 1.0, 0.15)
        user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

# =========================================================
# SHARED
# =========================================================
pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
pillar_cols = [f"pillar_{p}" for p in pillar_names]
COLORS = ['#7c3aed', '#14b8a6', '#f59e0b', '#ef4444', '#3b82f6']
CL = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
          font=dict(color='#10333a', size=13),
          legend=dict(font=dict(color='#10333a')),
          xaxis=dict(color='#10333a', tickfont=dict(color='#10333a')),
          yaxis=dict(color='#10333a', tickfont=dict(color='#10333a')))

def metric_card(name, value, description):
    return f'<div class="metric-explain"><div class="me-name">{name}</div><div class="me-val">{value}</div><div class="me-desc">{description}</div></div>'

# =========================================================
# HOME
# =========================================================
if page == "Home":
    st.markdown("""
    <div class="page-banner">
        <h1><span class="brand-text">Riyadh Livability Index</span></h1>
        <p>Data-driven insights into the neighborhoods of Saudi Arabia's capital.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Quick Statistics")
    st.markdown("""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-value">348K</div><div class="stat-label">Listings</div></div>
        <div class="stat-box"><div class="stat-value">176</div><div class="stat-label">Districts</div></div>
        <div class="stat-box"><div class="stat-value">27K</div><div class="stat-label">POIs</div></div>
        <div class="stat-box"><div class="stat-value">3K+</div><div class="stat-label">Transit Stops</div></div>
        <div class="stat-box"><div class="stat-value">4</div><div class="stat-label">ML Models</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="note-box">
    <b>How it works:</b> Choose Buy or Rent → Set your budget → Adjust lifestyle priorities → Our K-Means algorithm matches you to the best neighborhood archetype → Get your <b>Top 3</b> tailored recommendations.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# CITY RANKING
# =========================================================
elif page == "City Ranking":
    st.subheader("🏆 Global Neighborhood Ranking")
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)
    c1, c2, c3 = st.columns(3)
    for i, (name, row) in enumerate(df_city.head(3).iterrows()):
        with [c1,c2,c3][i]:
            st.markdown(f'<div class="stat-box" style="border-top:4px solid {COLORS[i]};"><div class="stat-label">{"🥇🥈🥉"[i]} Rank {i+1}</div><div class="stat-value" style="font-size:1.1rem;">{name}</div><div style="color:#14b8a6;font-weight:bold;">RLI: {row["RLI"]:.1f}</div></div>', unsafe_allow_html=True)

    c_l, c_r = st.columns([3, 2])
    with c_l:
        fig = go.Figure()
        for idx, (name, row) in enumerate(df_city.head(5).iterrows()):
            v = [row[c] for c in pillar_cols]
            fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself", name=str(name), line=dict(color=COLORS[idx%5]), opacity=0.7))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, tickfont=dict(color='#10333a')), angularaxis=dict(tickfont=dict(color='#10333a'))), title=dict(text="Pillar Profile — Top 5", font=dict(color='#10333a')), height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#10333a'), legend=dict(font=dict(color='#10333a')))
        st.plotly_chart(fig, use_container_width=True)
    with c_r:
        st.markdown('<div class="note-box"><b>Understanding the Radar:</b><br>Each axis = one livability pillar. Balanced shapes = well-rounded neighborhoods. Shannon Entropy rewards balance.<br><br><b>Core</b> = Hospitals, schools, retail<br><b>Mobility</b> = Metro, bus, walkability<br><b>Well-being</b> = Parks, dining, sports<br><b>Infrastructure</b> = Fiber, government</div>', unsafe_allow_html=True)
    st.dataframe(df_city[['rank','RLI','pillar_Core','pillar_Mobility','pillar_Well-being','pillar_Infrastructure']].style.background_gradient(subset=["RLI"], cmap="YlGn"), use_container_width=True, height=450)

# =========================================================
# PROPERTY SEARCH
# =========================================================
elif page == "Property Search":
    st.subheader("🔍 Personalized Property Matcher")
    with st.container(border=True):
        f1,f2,f3,f4 = st.columns(4)
        with f1:
            ac = df_raw["category"].value_counts()
            co = [k for k in sorted(CATEGORY_MAP.keys()) if k in ac.index]
            sel_cat = st.selectbox("Category", co, format_func=lambda x: CATEGORY_MAP[x], index=co.index(3) if 3 in co else 0)
        cl = CATEGORY_MAP.get(sel_cat, "")
        min_p = f2.number_input("Min Budget", value=500000, step=50000)
        max_p = f3.number_input("Max Budget", value=3000000, step=50000)
        if cl not in NO_ROOM_CATEGORIES: min_r = f4.number_input("Min Rooms", 0, 20, 3)
        else: min_r = 0; f4.markdown("**Rooms:** N/A")

    result = property_search(df_raw=pipe["df_raw"], df_ranked=df_ranked, category=sel_cat, min_price=min_p, max_price=max_p, min_rooms=min_r, pillar_weights=user_weights)
    df_res = result["results"]
    if df_res.empty:
        st.warning("No matches. Try widening your filters.")
    else:
        bc_id = result.get("best_cluster", -1)
        st.success(f"**{result['properties_matched']:,}** properties → **{len(df_res)}** neighborhoods qualify")
        if bc_id >= 0:
            st.markdown(f'<div class="note-box">K-Means identified <b>Cluster {bc_id}</b> as the best-fit archetype for your priorities.</div>', unsafe_allow_html=True)
        for i, (name, row) in enumerate(df_res.head(3).iterrows()):
            lb = name if isinstance(name,str) else str(name)
            st.markdown(f'<div class="stat-box" style="border-left:4px solid {COLORS[i%5]};text-align:left;padding:1rem 1.5rem;"><div style="font-weight:900;font-size:1.1rem;color:#0f172a;">#{int(row["rank"])} {lb}</div><div style="background:linear-gradient(90deg,#7c3aed,#14b8a6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:900;font-size:1.3rem;">Match {row["match_pct"]:.0f}%</div><div style="color:#64748b;font-size:0.85rem;">RLI {row["RLI"]:.1f} · Avg {row.get("avg_price",0):,.0f} SAR · {int(row.get("filtered_count",0))} listings</div></div>', unsafe_allow_html=True)
        scols = ["rank","match_pct","combined_score","RLI","price_score","cluster_fit","km_cluster","avg_price","filtered_count","pillar_Core","pillar_Mobility","pillar_Well-being","pillar_Infrastructure"]
        ps = [c for c in scols if c in df_res.columns]
        st.dataframe(df_res[ps].style.background_gradient(subset=["match_pct"], cmap="YlGn"), use_container_width=True, height=400)

# =========================================================
# CLUSTERING
# =========================================================
elif page == "Clustering Analysis":
    st.subheader("🧬 Neighborhood Archetypes (Unsupervised)")
    with st.spinner("Analyzing clusters..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    bm = clr[bcl]
    st.markdown(f'<div class="winner-pill">🏆 Best Model: {bcl}</div>', unsafe_allow_html=True)

    mc1, mc2, mc3 = st.columns(3)
    with mc1: st.markdown(metric_card("Silhouette ↑", f"{bm['silhouette']:.4f}", "Cluster separation quality. Range -1 to 1. Higher = more distinct groups."), unsafe_allow_html=True)
    with mc2: st.markdown(metric_card("Calinski-Harabasz ↑", f"{bm['calinski_harabasz']:.1f}", "Between-cluster vs within-cluster variance ratio. Higher = tighter clusters."), unsafe_allow_html=True)
    with mc3: st.markdown(metric_card("Davies-Bouldin ↓", f"{bm['davies_bouldin']:.4f}", "Average similarity between clusters. Lower = more different from each other."), unsafe_allow_html=True)

    # Data preview — show what the clustering operates on
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**Neighborhood Feature Matrix (Head)**")
    st.caption("This is the aggregated data the clustering algorithm operates on — 176 neighborhoods × 16 livability features.")
    st.dataframe(df_ranked[pillar_cols + ['km_cluster','RLI']].head(10).style.background_gradient(cmap='YlGn'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**All Models Tested**")
    cr_df = pd.DataFrame([{'Model':n, 'Silhouette ↑':round(m['silhouette'],4), 'Calinski-Harabasz ↑':round(m['calinski_harabasz'],1), 'Davies-Bouldin ↓':round(m['davies_bouldin'],4)} for n,m in clr.items()])
    st.dataframe(cr_df.style.highlight_max(subset=['Silhouette ↑','Calinski-Harabasz ↑'], color='#ccfbf1').highlight_min(subset=['Davies-Bouldin ↓'], color='#ccfbf1'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        sd = {n:m['silhouette'] for n,m in clr.items()}
        fs = go.Figure(go.Bar(x=list(sd.keys()), y=list(sd.values()), marker_color=['#7c3aed' if n==bcl else '#cbd5e1' for n in sd], text=[f"{v:.3f}" for v in sd.values()], textposition='outside', textfont=dict(color='#10333a')))
        fs.update_layout(title=dict(text="Silhouette Score Comparison", font=dict(color='#10333a')), yaxis_title="Score", height=400, **CL)
        st.plotly_chart(fs, use_container_width=True)
    with g2:
        pc = PCA_viz(n_components=2, random_state=42).fit_transform(Xsc)
        fp = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"Cluster {l}" for l in bm['labels']], title=f"PCA Cluster Map — {bcl}", hover_name=df_ranked.index.tolist(), color_discrete_sequence=COLORS)
        fp.update_layout(height=400, **CL)
        st.plotly_chart(fp, use_container_width=True)

    st.markdown('<div class="note-box"><b>What does clustering do?</b> K-Means groups 176 neighborhoods into archetypes based on service density, transit, and connectivity — without labels. <b>Hierarchical clustering</b> (Ward linkage) is tested as an alternative. The best model by Silhouette Score is used in the Recommender to match users to their ideal neighborhood type.</div>', unsafe_allow_html=True)

# =========================================================
# PRICE PREDICTION
# =========================================================
elif page == "Price Prediction":
    st.subheader("📈 Property Price Prediction (Regression)")
    with st.spinner("Training models on 30K sample..."):
        rr, breg, rf, rs = run_regression_comparison(pipe['df_raw'])
    br = rr[breg]
    st.markdown(f'<div class="winner-pill">🏆 Best Model: {breg}</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.markdown(metric_card("R² Score ↑", f"{br['R2']:.4f}", "% of price variance explained. 1.0 = perfect prediction."), unsafe_allow_html=True)
    with mc2: st.markdown(metric_card("MAE ↓", f"{br['MAE']:,.0f} SAR", "Mean Absolute Error — average prediction offset."), unsafe_allow_html=True)
    with mc3: st.markdown(metric_card("RMSE ↓", f"{br['RMSE']:,.0f} SAR", "Root Mean Squared Error — penalizes large errors more."), unsafe_allow_html=True)
    with mc4: st.markdown(metric_card("MAPE ↓", f"{br['MAPE']:.1f}%", "Mean Absolute Percentage Error — average % off."), unsafe_allow_html=True)

    # Data preview — show sample of what the model trains on
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**Training Data Sample (Head)**")
    st.caption("The model predicts 'price' from these 20 features. Sampled 30K rows for training speed.")
    preview_cols = ['price', 'area', 'category', 'total_rooms', 'dining_cafe', 'bus_count', 'metro_count', 'Fiber_Available']
    pc_avail = [c for c in preview_cols if c in pipe['df_raw'].columns]
    st.dataframe(pipe['df_raw'][pc_avail].head(8).style.background_gradient(subset=['price'], cmap='YlGn'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**All Models Tested**")
    rdf = pd.DataFrame([{'Model':n, 'R² ↑':m['R2'], 'MAE ↓':f"{m['MAE']:,.0f}", 'RMSE ↓':f"{m['RMSE']:,.0f}", 'MAPE ↓':f"{m['MAPE']:.1f}%"} for n,m in rr.items()])
    st.dataframe(rdf, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        r2d = {n:m['R2'] for n,m in rr.items()}
        fr = go.Figure(go.Bar(x=list(r2d.keys()), y=list(r2d.values()), marker_color=['#7c3aed' if n==breg else '#cbd5e1' for n in r2d], text=[f"{v:.3f}" for v in r2d.values()], textposition='outside', textfont=dict(color='#10333a')))
        fr.update_layout(title=dict(text="R² Score Comparison", font=dict(color='#10333a')), yaxis_title="R²", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(fr, use_container_width=True)
    with g2:
        ya=br['y_test']; yp=br['predictions']; si=np.random.RandomState(42).choice(len(ya),min(2000,len(ya)),replace=False)
        fsc=go.Figure()
        fsc.add_trace(go.Scatter(x=ya[si],y=yp[si],mode='markers',marker=dict(size=4,opacity=0.4,color='#14b8a6'),name='Predictions'))
        mx=max(ya.max(),yp.max())
        fsc.add_trace(go.Scatter(x=[0,mx],y=[0,mx],mode='lines',line=dict(color='#ef4444',dash='dash'),name='Perfect'))
        fsc.update_layout(title=dict(text=f"Actual vs Predicted — {breg}", font=dict(color='#10333a')), xaxis_title="True Price (SAR)", yaxis_title="Predicted (SAR)", height=400, **CL)
        st.plotly_chart(fsc, use_container_width=True)

    bmod = br['model']
    if hasattr(bmod, 'feature_importances_'):
        imp = bmod.feature_importances_
        fi = sorted(zip(rf, imp), key=lambda x:-x[1])[:10]
        fim = go.Figure(go.Bar(y=[f[0] for f in fi][::-1], x=[f[1] for f in fi][::-1], orientation='h', marker=dict(color=[f[1] for f in fi][::-1], colorscale=[[0,'#e0e7ff'],[1,'#7c3aed']])))
        fim.update_layout(title=dict(text=f"Top 10 Features — {breg}", font=dict(color='#10333a')), xaxis_title="Importance", height=400, **CL)
        st.plotly_chart(fim, use_container_width=True)

    st.markdown('<div class="note-box"><b>Why this matters:</b> Random Forest explains ~92% of price variance. <b>Area</b> dominates, followed by category type and neighborhood service density. This validates that livability features genuinely predict market value.</div>', unsafe_allow_html=True)

# =========================================================
# PRICE TIERING
# =========================================================
elif page == "Price Tiering":
    st.subheader("🏷️ Price Tier Classification")
    with st.spinner("Classifying market tiers..."):
        clsr, bclsn, clss = run_classification_comparison(pipe['df_raw'])
    bc = clsr[bclsn]
    st.markdown(f'<div class="winner-pill">🏆 Best Model: {bclsn}</div>', unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: st.markdown(metric_card("Accuracy ↑", f"{bc['Accuracy']:.2%}", "% of all properties classified into the correct tier."), unsafe_allow_html=True)
    with mc2: st.markdown(metric_card("F1 Score ↑", f"{bc['F1']:.4f}", "Harmonic mean of Precision & Recall. Balances both errors."), unsafe_allow_html=True)
    with mc3: st.markdown(metric_card("Precision ↑", f"{bc['Precision']:.4f}", "When model says 'Luxury', how often is it right?"), unsafe_allow_html=True)
    with mc4: st.markdown(metric_card("Recall ↑", f"{bc['Recall']:.4f}", "Of all actual Luxury homes, how many did the model catch?"), unsafe_allow_html=True)

    # Data preview — show the tier distribution
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**Price Tier Distribution**")
    st.caption("Properties are split into 4 quartile-based tiers. The classifier learns to predict which tier a property belongs to.")
    df_tier_preview = pipe['df_raw'][['price','area','category','total_rooms']].head(8).copy()
    df_tier_preview['price_tier'] = pd.qcut(pipe['df_raw']['price'], 4, labels=TIER_LABELS).head(8).values
    st.dataframe(df_tier_preview.style.background_gradient(subset=['price'], cmap='YlGn'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("**All Models Tested**")
    cdf = pd.DataFrame([{'Model':n, 'Accuracy':m['Accuracy'], 'Precision':m['Precision'], 'Recall':m['Recall'], 'F1':m['F1']} for n,m in clsr.items()])
    st.dataframe(cdf.style.highlight_max(subset=['Accuracy','Precision','Recall','F1'], color='#ccfbf1'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        f1d = {n:m['F1'] for n,m in clsr.items()}
        ff1 = go.Figure(go.Bar(x=list(f1d.keys()), y=list(f1d.values()), marker_color=['#7c3aed' if n==bclsn else '#cbd5e1' for n in f1d], text=[f"{v:.3f}" for v in f1d.values()], textposition='outside', textfont=dict(color='#10333a')))
        ff1.update_layout(title=dict(text="F1 Score Comparison", font=dict(color='#10333a')), yaxis_title="F1", yaxis_range=[0,1], height=400, **CL)
        st.plotly_chart(ff1, use_container_width=True)
    with g2:
        cm = bc['confusion_matrix']
        fcm = px.imshow(cm, text_auto=True, x=TIER_LABELS, y=TIER_LABELS, labels=dict(x="Predicted",y="Actual",color="Count"), title=f"Confusion Matrix — {bclsn}", color_continuous_scale=[[0,'#f0e6ff'],[0.5,'#a78bfa'],[1,'#7c3aed']])
        fcm.update_layout(height=400, **CL)
        st.plotly_chart(fcm, use_container_width=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"**Per-Class Metrics — {bclsn}**")
    pcp = precision_score(bc['y_test'], bc['predictions'], average=None)
    pcr = recall_score(bc['y_test'], bc['predictions'], average=None)
    pcf = f1_score(bc['y_test'], bc['predictions'], average=None)
    pcdf = pd.DataFrame([{'Tier':t, 'Precision':round(pcp[i],4), 'Recall':round(pcr[i],4), 'F1':round(pcf[i],4)} for i,t in enumerate(TIER_LABELS)])
    st.dataframe(pcdf.style.background_gradient(subset=['Precision','Recall','F1'], cmap='BuPu'), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="note-box"><b>Reading the matrix:</b> Diagonal = correct predictions. Off-diagonal = misclassifications. The model struggles most between adjacent tiers (Mid vs Premium) — these boundaries are inherently fuzzy.<br><br><b>Budget</b> = bottom 25% · <b>Mid</b> = 25-50% · <b>Premium</b> = 50-75% · <b>Luxury</b> = top 25%.</div>', unsafe_allow_html=True)