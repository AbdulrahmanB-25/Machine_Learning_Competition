import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

# Ensure the engine is in the path if it's a separate file, 
# though we assume the engine logic is imported or available.
sys.path.insert(0, os.path.dirname(__file__))
from rli_engine import run_full_pipeline, recommend, PILLARS, ALL_RLI_FEATURES

# ── 1. Page Config & Styling ──────────────────────────────────────────────────
st.set_page_config(
    page_title='Riyadh Livability Index',
    page_icon='City',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0a0e27; }
    .stMetric { background: linear-gradient(135deg, #0f1333 0%, #1a1f3e 100%);
                padding: 1rem; border-radius: 10px; border: 1px solid #2a2f4e; }
    h1 { color: #f0c05a !important; }
    h2, h3 { color: #4fc3f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── 2. Data Loading ───────────────────────────────────────────────────────────
@st.cache_data
def load_pipeline():
    # Update this to your actual file name
    csv_path = 'Riyadh_Master_Dataset.csv'
    if not os.path.exists(csv_path):
        st.error(f"Dataset {csv_path} not found!")
        st.stop()
    return run_full_pipeline(csv_path)

pipe = load_pipeline()
df_raw = pipe['df_raw']

# ── 3. Sidebar: Pillar Weighting ──────────────────────────────────────────────
st.sidebar.title("⚖️ Pillar Weights")
st.sidebar.info("Adjust the importance of each urban pillar.")

w_core = st.sidebar.slider('Core Essentials', 0.0, 1.0, 0.40)
w_mob  = st.sidebar.slider('Mobility', 0.0, 1.0, 0.25)
w_well = st.sidebar.slider('Well-being', 0.0, 1.0, 0.20)
w_inf  = st.sidebar.slider('Infrastructure', 0.0, 1.0, 0.15)

# Normalize weights so they sum to 1.0
total_w = w_core + w_mob + w_well + w_inf
weights = {
    'Core': w_core / total_w,
    'Mobility': w_mob / total_w,
    'Well-being': w_well / total_w,
    'Infrastructure': w_inf / total_w
}

# ── 4. Main Header ────────────────────────────────────────────────────────────
st.title("🏙️ Riyadh Livability Index (RLI)")
st.markdown("### Neighborhood DNA — The 15-Minute City Analysis")

# ── 5. NEW SECTION: Property Preferences ──────────────────────────────────────
with st.expander("🎯 Filter by Property Preferences", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_map = {1: "Apartment (Rent)", 3: "Villa", 6: "Apartment (Sale)", 2: "Land"}
        selected_cat = st.selectbox("Property Type", options=list(category_map.keys()), 
                                    format_func=lambda x: category_map[x])
    
    with col2:
        max_price = st.slider("Maximum Price (SAR)", 10000, 5000000, 1500000, step=50000)
        
    with col3:
        if selected_cat in [1, 3, 6]:
            min_rooms = st.number_input("Minimum Rooms", 0, 10, 3)
        else:
            min_rooms = 0
            st.write("Rooms N/A for this type")

# ── 6. Compute Recommendations ────────────────────────────────────────────────
# Using the engine's recommend function with our new filters
df_rec = recommend(
    df_raw, 
    user_weights=weights, 
    category=selected_cat, 
    max_price=max_price, 
    min_rooms=min_rooms,
    top_n=10
)

if df_rec.empty:
    st.warning("⚠️ No properties match your criteria. Try broadening your filters.")
    st.stop()

# ── 7. Visualization: Radar Chart ─────────────────────────────────────────────
st.subheader("📊 Neighborhood DNA Comparison (Top 3)")
fig = go.Figure()

top_3 = df_rec.head(3)
for idx, row in top_3.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['pillar_Core'], row['pillar_Mobility'], row['pillar_Well-being'], row['pillar_Infrastructure']],
        theta=['Core', 'Mobility', 'Well-being', 'Infrastructure'],
        fill='toself',
        name=idx
    ))

fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ── 8. Results Table ───────────────────────────────────────────────────────────
st.subheader("📋 Top Neighborhood Rankings")

# Formatting for display
df_display = df_rec[['RLI', 'match_pct', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']].copy()
df_display.columns = ['RLI Score', 'Match Pct', 'Core', 'Mobility', 'Well-being', 'Infrastructure']

st.dataframe(
    df_display.style.format("{:.2f}").background_gradient(subset=['Match Pct'], cmap='YlGn'),
    use_container_width=True
)

# ── 9. Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("RLI Engine v2.0 | Data-driven urban planning for Riyadh")
