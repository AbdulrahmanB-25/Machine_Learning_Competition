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
# PAGE CONFIG & STATE
# =========================================================
st.set_page_config(page_title="Riyadh Livability Index", page_icon="🏙️", layout="wide", initial_sidebar_state="expanded")

# =========================================================
# GLOBAL STYLE (Fixing text visibility & card consistency)
# =========================================================
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #edf4f5 0%, #dbe9ec 100%); }
header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }

/* Sidebar Styling */
section[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #14353d 0%, #1d4953 100%); 
    border-right: 1px solid rgba(255,255,255,0.08); 
}
section[data-testid="stSidebar"] * { color: white !important; }

/* Dashboard Cards */
.section-card { 
    background: #ffffff; 
    border: 1px solid rgba(20,53,61,0.08); 
    border-radius: 22px; 
    padding: 1.5rem; 
    box-shadow: 0 10px 25px rgba(18,38,45,0.08); 
    margin-bottom: 1.5rem; 
}

.brand-text { background: linear-gradient(90deg, #7c3aed, #14b8a6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; }

/* Stats & Metrics */
.stat-row { display: flex; gap: 1.2rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-box { 
    flex: 1; 
    min-width: 180px; 
    background: white; 
    border: 1px solid rgba(20,53,61,0.1); 
    border-radius: 16px; 
    padding: 1.2rem; 
    text-align: center; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
}
.stat-box .stat-value { font-size: 1.6rem; font-weight: 900; color: #10333a; }
.stat-box .stat-label { font-size: 0.85rem; color: #5c7a82; font-weight: 600; text-transform: uppercase; margin-top: 0.3rem; }

/* Banner */
.page-banner { 
    border-radius: 24px; 
    padding: 2rem; 
    background: linear-gradient(90deg, rgba(10,34,40,0.95) 0%, rgba(20,73,79,0.85) 60%), url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1400&q=80'); 
    background-size: cover; 
    color: white; 
    margin-bottom: 2rem; 
}
.page-banner h1 { color: white !important; margin: 0; }
.page-banner p { color: rgba(255,255,255,0.9) !important; }

/* Buttons */
.stButton > button { border: none; border-radius: 12px; background: linear-gradient(90deg, #7c3aed 0%, #14b8a6 100%); color: white; font-weight: 700; }

h1, h2, h3, h4 { color: #10333a !important; }
p, label, .small-note { color: #244851 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA LOADING
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
    st.error("Dataset not found.")
    st.stop()
pipe = load_pipeline(df_raw)
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR NAVIGATION & PARAMETERS
# =========================================================
with st.sidebar:
    st.markdown("## 🏙️ RLI Dashboard")
    page = st.radio("Navigation", ["Home", "City Ranking", "Property Search", "Clustering Analysis", "Price Prediction", "Price Tiering"])
    
    # RLI Weights - Only show on relevant pages
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
# PAGE ROUTING
# =========================================================

# --- HOME PAGE ---
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
        <div class="stat-box"><div class="stat-value">4</div><div class="stat-label">ML Models</div></div>
    </div>
    """, unsafe_allow_html=True)

# --- CITY RANKING ---
elif page == "City Ranking":
    st.subheader("🏆 Global Neighborhood Ranking")
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)
    
    # Top 3 Metrics
    c1, c2, c3 = st.columns(3)
    top3 = df_city.head(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        with [c1, c2, c3][i]:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Rank {i+1}</div>
                <div class="stat-value" style="font-size:1.2rem;">{name}</div>
                <div style="color:#14b8a6; font-weight:bold;">RLI: {row['RLI']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

    fig = go.Figure()
    pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
    pillar_cols = [f"pillar_{p}" for p in pillar_names]
    for idx, (name, row) in enumerate(df_city.head(5).iterrows()):
        v = [row[c] for c in pillar_cols]
        fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=pillar_names+[pillar_names[0]], fill="toself", name=str(name)))
    
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_city.style.background_gradient(subset=["RLI"], cmap="YlGn"), use_container_width=True)

# --- PROPERTY SEARCH ---
elif page == "Property Search":
    st.subheader("🔍 Personalized Property Matcher")
    # Filters
    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        sel_cat = f1.selectbox("Category", options=sorted(CATEGORY_MAP.keys()), format_func=lambda x: CATEGORY_MAP[x])
        min_p = f2.number_input("Min Budget (SAR)", value=500000)
        max_p = f3.number_input("Max Budget (SAR)", value=3000000)
    
    result = property_search(df_raw=pipe["df_raw"], df_ranked=df_ranked, category=sel_cat, min_price=min_p, max_price=max_p, pillar_weights=user_weights)
    st.dataframe(result["results"], use_container_width=True)

# --- CLUSTERING ---
elif page == "Clustering Analysis":
    st.subheader("🤖 Neighborhood Archetypes (Clustering)")
    with st.spinner("Analyzing clusters..."):
        clr, Xsc, bcl = run_clustering_comparison(df_ranked)
    
    bm = clr[bcl]
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-label">Best Model</div><div class="stat-value" style="font-size:1.2rem;">{bcl}</div></div>
        <div class="stat-box"><div class="stat-label">Silhouette</div><div class="stat-value">{bm['silhouette']:.4f}</div></div>
        <div class="stat-box"><div class="stat-label">Davies-Bouldin</div><div class="stat-value">{bm['davies_bouldin']:.4f}</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    # PCA Plot
    pc = PCA_viz(n_components=2).fit_transform(Xsc)
    fig_pca = px.scatter(x=pc[:,0], y=pc[:,1], color=[f"Cluster {l}" for l in bm['labels']], title="Cluster Map (PCA)")
    col1.plotly_chart(fig_pca, use_container_width=True)
    col2.dataframe(pd.DataFrame(clr).T, use_container_width=True)

# --- PRICE PREDICTION ---
elif page == "Price Prediction":
    st.subheader("📈 Price Regression Analysis")
    with st.spinner("Training models..."):
        rr, breg, rf, rs = run_regression_comparison(pipe['df_raw'])
    
    br = rr[breg]
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-label">Best Regression</div><div class="stat-value" style="font-size:1.2rem;">{breg}</div></div>
        <div class="stat-box"><div class="stat-label">R² Accuracy</div><div class="stat-value">{br['R2']:.4f}</div></div>
        <div class="stat-box"><div class="stat-label">Avg Error (MAE)</div><div class="stat-value">{br['MAE']:,.0f}</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(px.scatter(x=br['y_test'], y=br['predictions'], labels={'x':'Actual','y':'Predicted'}, title="Accuracy: Actual vs Predicted"), use_container_width=True)

# --- PRICE TIER ---
elif page == "Price Tiering":
    st.subheader("🏷️ Price Tier Classification")
    with st.spinner("Classifying..."):
        clsr, bclsn, clss = run_classification_comparison(pipe['df_raw'])
    
    bc = clsr[bclsn]
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box"><div class="stat-label">Classifier</div><div class="stat-value" style="font-size:1.2rem;">{bclsn}</div></div>
        <div class="stat-box"><div class="stat-label">Accuracy</div><div class="stat-value">{bc['Accuracy']:.2%}</div></div>
        <div class="stat-box"><div class="stat-label">F1-Score</div><div class="stat-value">{bc['F1']:.4f}</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    fig_cm = px.imshow(bc['confusion_matrix'], text_auto=True, x=TIER_LABELS, y=TIER_LABELS, title="Confusion Matrix", color_continuous_scale="Blues")
    st.plotly_chart(fig_cm, use_container_width=True)