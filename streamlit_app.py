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
# 🏗️ CITY-STYLE CONFIG & THEME
# =========================================================
st.set_page_config(page_title="Riyadh DNA Explorer", page_icon="🏙️", layout="wide")

# Custom CSS for the "City" Aesthetic
st.markdown("""
<style>
    /* Main Background and Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { background-color: #F8FAFC; }
    
    /* Navigation Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid #E2E8F0;
    }
    
    /* City Cards */
    .city-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        margin-bottom: 1rem;
    }
    
    /* Banners */
    .hero-banner {
        background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%);
        color: white;
        padding: 3rem;
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

    /* Research Notes Box */
    .note-box {
        background-color: #F0FDFA;
        border-left: 4px solid #2DD4BF;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #134E4A;
    }
    
    /* Metrics Branding */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #F1F5F9;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 🛡️ DATA CACHING
# =========================================================
@st.cache_data
def load_data():
    csv_path = "Riyadh_Master_Dataset.csv"
    df = pd.read_csv(csv_path)
    pipe = build_global_ranking(df)
    return df, pipe

df_raw, pipe = load_data()
df_ranked = pipe["df_ranked"]

# =========================================================
# 🧭 SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Riyadh_Metro_Logo.svg/1200px-Riyadh_Metro_Logo.svg.png", width=80)
    st.markdown("### **RIYADH DNA**\n*Urban Intelligence*")
    st.write("---")
    page = st.radio("Navigate City Systems:", [
        "🏙️ City Overview", 
        "🎯 Personal Recommender", 
        "🧬 Neighborhood DNA (Clustering)", 
        "💸 Price Oracle (Regression)", 
        "🏷️ Luxury Tiering (Classification)"
    ])
    
    st.write("---")
    st.markdown("### **Global Pillar Weights**")
    w_core = st.slider("Core Services", 0.0, 1.0, 0.40)
    w_mob = st.slider("Mobility", 0.0, 1.0, 0.25)
    w_well = st.slider("Well-being", 0.0, 1.0, 0.20)
    w_inf = st.slider("Infrastructure", 0.0, 1.0, 0.15)
    user_weights = {"Core": w_core, "Mobility": w_mob, "Well-being": w_well, "Infrastructure": w_inf}

# =========================================================
# 🏙️ PAGE 1: CITY OVERVIEW
# =========================================================
if page == "🏙️ City Overview":
    st.markdown("""
    <div class="hero-banner">
        <h1 style='color: white; margin:0;'>RIYADH DNA EXPLORER</h1>
        <p style='color: #CCFBF1; font-size: 1.2rem;'>The Scientific Blueprint of the 15-Minute City</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Listings", "348,521", "Market Volume")
    col2.metric("Districts", "176", "Geospatial Coverage")
    col3.metric("Data Features", "27", "Urban Variables")
    col4.metric("Engine", "RLI v2.1", "Active ML")

    st.markdown("<h2 class='section-header'>City-Wide Livability Ranking</h2>", unsafe_allow_html=True)
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=10)
    
    c_tab1, c_tab2 = st.columns([2, 1])
    with c_tab1:
        st.dataframe(df_city[['rank', 'RLI', 'pillar_Core', 'pillar_Mobility']].style.background_gradient(cmap="GnBu"), use_container_width=True, height=450)
    
    with c_tab2:
        st.markdown("""
        <div class='note-box'>
        <b>The RLI Formula:</b><br>
        Our Livability Index isn't just a sum. We apply <b>Shannon Entropy</b> to reward neighborhoods with <i>diverse</i> services. 
        <br><br>A district with only shops but no schools is penalized to encourage the "15-Minute City" ideal.
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 🎯 PAGE 2: RECOMMENDER
# =========================================================
elif page == "🎯 Personal Recommender":
    st.markdown("<h2 class='section-header'>Neighborhood Matchmaker</h2>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='city-card'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            sel_cat = st.selectbox("I am looking for:", options=sorted(CATEGORY_MAP.keys()), format_func=lambda x: CATEGORY_MAP[x])
        with r2:
            budget = st.slider("Budget (SAR)", 0, 5000000, (500000, 2000000))
        with r3:
            p_weight = st.slider("Price vs. Quality Weight", 0.0, 1.0, 0.3)
        st.markdown("</div>", unsafe_allow_html=True)

    result = property_search(df_raw, df_ranked, category=sel_cat, min_price=budget[0], max_price=budget[1], pillar_weights=user_weights, price_weight=p_weight)
    
    if not result["results"].empty:
        st.success(f"Found {result['properties_matched']} matching properties across {len(result['results'])} neighborhoods.")
        
        top_matches = result["results"].head(3)
        mc1, mc2, mc3 = st.columns(3)
        cols = [mc1, mc2, mc3]
        for i, (name, row) in enumerate(top_matches.iterrows()):
            with cols[i]:
                st.markdown(f"""
                <div class='city-card' style='border-top: 5px solid #0D9488;'>
                    <h3 style='margin:0;'>{name}</h3>
                    <p style='color: #64748B;'>Match Score: <b>{row['match_pct']}%</b></p>
                    <hr>
                    <small>Avg Price: {row['avg_price']:,.0f} SAR</small><br>
                    <small>Livability: {row['RLI']}/100</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No matches found for your criteria.")

# =========================================================
# 🧬 PAGE 3: CLUSTERING
# =========================================================
elif page == "🧬 Neighborhood DNA (Clustering)":
    st.markdown("<h2 class='section-header'>Neighborhood Archetype Analysis</h2>", unsafe_allow_html=True)
    
    with st.spinner("Analyzing Urban DNA..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    
    st.markdown(f"### Best Performer: **{bcl}**")
    
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write("**Model Matrix**")
        cr_df = pd.DataFrame([{'Model':n,'Silhouette ↑':m['silhouette'],'Davies-Bouldin ↓':m['davies_bouldin']} for n,m in clr.items()])
        st.table(cr_df)
        
        st.markdown("""
        <div class='note-box'>
        <b>Clustering Note:</b><br>
        We use Unsupervised Learning to group Riyadh into "Archetypes" (e.g., Luxury Residential, Emerging Infrastructure, Commercial Hubs). 
        <b>Silhouette Score</b> measures how distinct these groups are.
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        pc = PCA_viz(n_components=2).fit_transform(Xsc)
        fig = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"DNA {l}" for l in clr[bcl]['labels']], title="PCA Clustering Visualization", labels={'x':'Urban Dimension 1', 'y':'Urban Dimension 2'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 💸 PAGE 4: PRICE ORACLE
# =========================================================
elif page == "💸 Price Oracle (Regression)":
    st.markdown("<h2 class='section-header'>Market Value Prediction</h2>", unsafe_allow_html=True)
    
    with st.spinner("Training Market Oracle..."):
        rr, breg, rf, rs = run_regression_comparison(df_raw)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Dominant Model", breg)
    m2.metric("Prediction Accuracy (R²)", f"{rr[breg]['R2']:.2%}")
    m3.metric("Avg Error (MAE)", f"{rr[breg]['MAE']:,.0f} SAR")

    st.markdown("<div class='city-card'>", unsafe_allow_html=True)
    st.write("**Regression Comparison Table**")
    st.dataframe(pd.DataFrame(rr).T[['R2', 'MAE', 'MAPE (%) ↓']].style.highlight_max(axis=0, subset=['R2']))
    st.markdown("</div>", unsafe_allow_html=True)

    # Error Visualization
    st.markdown("### Actual vs. Predicted Values")
    br = rr[breg]
    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(x=br['y_test'][:500], y=br['predictions'][:500], mode='markers', marker=dict(color='#0D9488', opacity=0.5)))
    fig_reg.add_trace(go.Scatter(x=[0, max(br['y_test'])], y=[0, max(br['y_test'])], line=dict(color='red', dash='dash')))
    fig_reg.update_layout(xaxis_title="True Market Price", yaxis_title="Oracle Prediction")
    st.plotly_chart(fig_reg, use_container_width=True)

# =========================================================
# 🏷️ PAGE 5: LUXURY TIERING
# =========================================================
elif page == "🏷️ Luxury Tiering (Classification)":
    st.markdown("<h2 class='section-header'>Price Tier Classification</h2>", unsafe_allow_html=True)
    
    with st.spinner("Segmenting Market Tiers..."):
        clsr, bclsn, clss = run_classification_comparison(df_raw)
    
    st.info(f"The **{bclsn}** model is the most effective at identifying Budget vs. Luxury properties.")

    # Show Confusion Matrix
    cc1, cc2 = st.columns(2)
    with cc1:
        st.write("**Performance Matrix**")
        st.table(pd.DataFrame(clsr).T[['Accuracy', 'F1', 'Precision']])
        
        st.markdown("""
        <div class='note-box'>
        <b>Matrix Guide:</b><br>
        <b>Precision:</b> When the model says 'Luxury', how often is it right?<br>
        <b>Recall:</b> How many of the actual 'Luxury' homes did we catch?
        </div>
        """, unsafe_allow_html=True)

    with cc2:
        st.write("**Confusion Matrix (Heatmap)**")
        cm = clsr[bclsn]['confusion_matrix']
        fig_cm = px.imshow(cm, text_auto=True, x=TIER_LABELS, y=TIER_LABELS, color_continuous_scale='Mint')
        st.plotly_chart(fig_cm, use_container_width=True)