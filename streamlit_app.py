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
from sklearn.decomposition import PCA as PCA_viz

# =========================================================
# 🏗️ 1. PAGE CONFIG & UI OVERRIDES
# =========================================================
st.set_page_config(
    page_title="Riyadh DNA Explorer", 
    page_icon="🏙️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "City" Aesthetic & Fixing Visibility
st.markdown("""
<style>
    /* Hide Streamlit Hamburger and Footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Background and Font */
    .stApp { background-color: #F8FAFC; }
    
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        color: white;
    }
    
    /* City Cards - EXPLICIT DARK TEXT FOR VISIBILITY */
    .city-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        color: #1E293B !important; /* Forces dark text on white background */
    }
    .city-card h1, .city-card h2, .city-card h3, .city-card p {
        color: #1E293B !important;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .section-header {
        color: #0F172A;
        font-weight: 800;
        font-size: 1.8rem;
        border-left: 6px solid #0D9488;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }

    /* Metric Branding */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🛡️ 2. DATA LOAD & CACHING
# =========================================================
@st.cache_data
def get_engine_data():
    csv_path = "Riyadh_Master_Dataset.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}")
        return None, None
    df = pd.read_csv(csv_path)
    pipeline = build_global_ranking(df)
    return df, pipeline

df_raw, pipe = get_engine_data()
df_ranked = pipe["df_ranked"] if pipe else None

# =========================================================
# 🧭 3. SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("<h2 style='color: white; margin-bottom:0;'>🏙️ Riyadh DNA</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem;'>Urban Intelligence Dashboard</p>", unsafe_allow_html=True)
    st.write("---")
    
    page = st.radio("Navigate Systems:", [
        "🏙️ City Overview", 
        "🎯 Personal Recommender", 
        "🧬 Clustering DNA", 
        "💸 Price Oracle", 
        "🏷️ Tier Classification"
    ])
    
    st.write("---")
    st.caption("Version 2.4 | Saudi Vision 2030 Data")

# =========================================================
# 🏙️ PAGE 1: CITY OVERVIEW
# =========================================================
if page == "🏙️ City Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1 style='color: white; margin:0;'>RIYADH LIVABILITY INDEX</h1>
        <p style='color: #CCFBF1;'>Scientific analysis of 176 neighborhoods across 4 urban pillars</p>
    </div>
    """, unsafe_allow_html=True)
    
    # RLI Parameters moved specifically to this section
    with st.expander("⚙️ Customize RLI Scoring Logic", expanded=False):
        st.write("Adjust weights to see how the city ranking shifts based on priorities:")
        cw1, cw2 = st.columns(2)
        with cw1:
            w_core = st.slider("Core Services (Schools/Health)", 0.0, 1.0, 0.40)
            w_mob = st.slider("Mobility (Metro/Bus/Traffic)", 0.0, 1.0, 0.25)
        with cw2:
            w_well = st.slider("Well-being (Parks/Fitness)", 0.0, 1.0, 0.20)
            w_inf = st.slider("Infrastructure (Connectivity/Utilities)", 0.0, 1.0, 0.15)
        user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

    st.markdown("<h2 class='section-header'>Top Districts by Livability</h2>", unsafe_allow_html=True)
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=10)
    st.dataframe(
        df_city[['rank', 'RLI', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being']].style.background_gradient(cmap="GnBu"), 
        use_container_width=True
    )

# =========================================================
# 🎯 PAGE 2: PERSONAL RECOMMENDER
# =========================================================
elif page == "🎯 Personal Recommender":
    st.markdown("<h2 class='section-header'>Neighborhood Matchmaker</h2>", unsafe_allow_html=True)
    
    # Local RLI Weights for the sidebar during this page
    with st.sidebar:
        st.write("---")
        st.markdown("### 🧬 Matching DNA")
        w_core = st.slider("Service Priority", 0.0, 1.0, 0.4)
        w_mob = st.slider("Transit Priority", 0.0, 1.0, 0.3)
        user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": 0.15, "Infrastructure": 0.15}

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        sel_cat = st.selectbox("Property Type", sorted(CATEGORY_MAP.keys()), format_func=lambda x: CATEGORY_MAP[x])
    with rc2:
        budget = st.slider("Budget (SAR)", 0, 5000000, (500000, 2000000))
    with rc3:
        p_weight = st.slider("Value vs Quality", 0.0, 1.0, 0.3)
    st.markdown("</div>", unsafe_allow_html=True)

    result = property_search(df_raw, df_ranked, category=sel_cat, min_price=budget[0], max_price=budget[1], pillar_weights=user_weights, price_weight=p_weight)
    
    if not result["results"].empty:
        st.write("### Recommended Matches")
        st.dataframe(result["results"][['match_pct', 'RLI', 'avg_price']].head(5), use_container_width=True)
    else:
        st.warning("No properties matched your criteria. Try widening your budget.")

# =========================================================
# 🧬 PAGE 3: CLUSTERING DNA (Statistics Layout)
# =========================================================
elif page == "🧬 Clustering DNA":
    st.markdown("<h2 class='section-header'>Urban Archetype Analysis</h2>", unsafe_allow_html=True)
    
    clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    
    # Stats Layout
    stat_col1, stat_col2 = st.columns([1, 2])
    
    with stat_col1:
        st.markdown("<div class='city-card'><h3>Model Comparison</h3>", unsafe_allow_html=True)
        for name, metrics in clr.items():
            st.write(f"**{name}**")
            st.progress(metrics['silhouette'])
            st.caption(f"Silhouette Score: {metrics['silhouette']:.3f}")
        st.markdown("<br><b>Winner:</b> <span style='color: #0D9488;'>{}</span>".format(bcl), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with stat_col2:
        pc = PCA_viz(n_components=2).fit_transform(Xsc)
        fig_cl = px.scatter(
            x=pc[:,0], y=pc[:,1], 
            color=[f"Cluster {l}" for l in clr[bcl]['labels']],
            title=f"Neighborhood Segments ({bcl})",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig_cl, use_container_width=True)

# =========================================================
# 💸 PAGE 4: PRICE ORACLE (Regression Stats)
# =========================================================
elif page == "💸 Price Oracle":
    st.markdown("<h2 class='section-header'>Real Estate Valuation Engine</h2>", unsafe_allow_html=True)
    
    rr, breg, rf, rs = run_regression_comparison(df_raw)
    
    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Algorithm", breg)
    m2.metric("Variance Explained (R²)", f"{rr[breg]['R2']:.1%}")
    m3.metric("Avg Prediction Error", f"{rr[breg]['MAE']:,.0f} SAR")

    # Chart Card
    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    st.write("### Prediction Accuracy Chart")
    y_test = rr[breg]['y_test']
    preds = rr[breg]['predictions']
    
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=y_test[:400], y=preds[:400], mode='markers', marker=dict(color='#0D9488', opacity=0.5), name="Predictions"))
    fig_reg.add_trace(go.Scatter(x=[0, y_test.max()], y=[0, y_test.max()], line=dict(color='red', dash='dash'), name="Ideal Path"))
    fig_reg.update_layout(xaxis_title="True Market Price", yaxis_title="Engine Prediction", template="plotly_white")
    st.plotly_chart(fig_reg, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 🏷️ PAGE 5: TIER CLASSIFICATION (Classification Stats)
# =========================================================
elif page == "🏷️ Tier Classification":
    st.markdown("<h2 class='section-header'>Property Tier Segmentation</h2>", unsafe_allow_html=True)
    
    clsr, bclsn, clss = run_classification_comparison(df_raw)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("<div class='city-card'><h3>Performance Matrix</h3>", unsafe_allow_html=True)
        # Convert classification results to display dataframe
        report_df = pd.DataFrame(clsr).T[['Accuracy', 'F1', 'Precision']]
        st.table(report_df.style.highlight_max(axis=0, color='#CCFBF1'))
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_t2:
        st.markdown("<div class='city-card'><h3>Confusion Matrix</h3>", unsafe_allow_html=True)
        cm = clsr[bclsn]['confusion_matrix']
        fig_cm = px.imshow(
            cm, text_auto=True, 
            x=TIER_LABELS, y=TIER_LABELS, 
            color_continuous_scale='GnBu',
            title=f"Prediction Confidence ({bclsn})"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)