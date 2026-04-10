import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. CORE ENGINE LOGIC (Internalized for Stability) ---
EPSILON = 1e-9
H_MAX = np.log(4)

PILLARS = {
    'Core': {'weight': 0.40, 'features': ['med_facilities', 'edu_primary', 'essential_retail', 'religious']},
    'Mobility': {'weight': 0.25, 'features': ['bus_count', 'metro_count', 'pedestrian', 'connectivity_score']},
    'Well-being': {'weight': 0.20, 'features': ['dining_cafe', 'parks_green', 'sports_play', 'fitness_care']},
    'Infrastructure': {'weight': 0.15, 'features': ['Fiber_Available', 'gov_civil', 'malls_shopping', 'edu_higher']}
}
ALL_RLI_FEATURES = [f for p in PILLARS.values() for f in p['features']]

def compute_rli(df_in, pillar_weights=None):
    pw = pillar_weights or {p: v['weight'] for p, v in PILLARS.items()}
    total_w = sum(pw.values())
    pw = {k: v / total_w for k, v in pw.items()}

    scaler_mm = MinMaxScaler()
    # Ensure all features exist in the dataframe
    available_features = [f for f in ALL_RLI_FEATURES if f in df_in.columns]
    scaled_vals = scaler_mm.fit_transform(df_in[available_features])
    df_sc = pd.DataFrame(scaled_vals, columns=available_features, index=df_in.index)

    pillar_scores = {}
    for pname, pinfo in PILLARS.items():
        feats = [f for f in pinfo['features'] if f in df_in.columns]
        pillar_scores[pname] = df_sc[feats].mean(axis=1) * pw[pname]

    pillar_df = pd.DataFrame(pillar_scores, index=df_in.index)
    weighted_sum = pillar_df.sum(axis=1)
    
    pillar_props = pillar_df.div(pillar_df.sum(axis=1) + EPSILON, axis=0)
    H = -(pillar_props * np.log(pillar_props + EPSILON)).sum(axis=1)
    diversity_mult = 1 + (H / H_MAX)
    raw_rli = weighted_sum * diversity_mult

    r_min, r_max = raw_rli.min(), raw_rli.max()
    rli_100 = ((raw_rli - r_min) / (r_max - r_min + EPSILON)) * 100

    result = df_in.copy()
    for pname in PILLARS:
        result[f'pillar_{pname}'] = pillar_scores[pname].values
    result['RLI'] = rli_100.round(2).values
    result['rank'] = result['RLI'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('RLI', ascending=False)

def get_recommendations(df_raw, user_weights, category, max_price, min_rooms):
    # Filter-First Logic
    temp_df = df_raw.copy()
    if category != "All":
        temp_df = temp_df[temp_df['category'] == category]
    temp_df = temp_df[temp_df['price'] <= max_price]
    if 'total_rooms' in temp_df.columns and min_rooms > 0:
        temp_df = temp_df[temp_df['total_rooms'] >= min_rooms]

    if temp_df.empty:
        return pd.DataFrame()

    # Aggregate filtered properties to neighborhood level
    df_agg = temp_df.groupby('neighborhood')[ALL_RLI_FEATURES].mean().fillna(0)
    scored = compute_rli(df_agg, pillar_weights=user_weights)
    scored['match_pct'] = (scored['RLI'] / (scored['RLI'].max() + EPSILON) * 100).round(1)
    return scored

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title='Riyadh Livability Index', layout='wide')

st.title("🏙️ Riyadh Livability Index (RLI)")
st.markdown("### Find the best neighborhood based on urban science and your budget")

# Data Loading
@st.cache_data
def load_data():
    csv_path = 'Riyadh_Master_Dataset.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.error("Dataset not found. Please ensure 'Riyadh_Master_Dataset.csv' is in the repository.")
    st.stop()

# --- SIDEBAR: WEIGHTS ---
st.sidebar.header("⚖️ Lifestyle Priorities")
w_core = st.sidebar.slider('Core Essentials (Schools/Clinics)', 0.0, 1.0, 0.40)
w_mob  = st.sidebar.slider('Mobility (Metro/Bus)', 0.0, 1.0, 0.25)
w_well = st.sidebar.slider('Well-being (Parks/Dining)', 0.0, 1.0, 0.20)
w_inf  = st.sidebar.slider('Infrastructure (Fiber/Gov)', 0.0, 1.0, 0.15)
weights = {'Core': w_core, 'Mobility': w_mob, 'Well-being': w_well, 'Infrastructure': w_inf}

# --- MAIN PANEL: PROPERTY FILTERS ---
with st.container(border=True):
    st.subheader("🎯 Property Preferences")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cat_options = ["All"] + sorted(list(df_raw['category'].unique()))
        selected_cat = st.selectbox("Property Type", options=cat_options)
    
    with col2:
        max_p = st.slider("Maximum Budget (SAR)", 
                          int(df_raw['price'].min()), 
                          int(df_raw['price'].max()), 
                          int(df_raw['price'].mean()))
    
    with col3:
        rooms = st.number_input("Minimum Rooms", 0, 10, 0)

# --- EXECUTION ---
df_rec = get_recommendations(df_raw, weights, selected_cat, max_p, rooms)

if df_rec.empty:
    st.warning("No neighborhoods match these specific property criteria. Try increasing your budget or changing the category.")
else:
    # Top 3 Radar Chart
    st.subheader("📊 Neighborhood DNA Comparison")
    top_3 = df_rec.head(3)
    fig = go.Figure()
    for idx, row in top_3.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['pillar_Core'], row['pillar_Mobility'], row['pillar_Well-being'], row['pillar_Infrastructure']],
            theta=['Core', 'Mobility', 'Well-being', 'Infrastructure'],
            fill='toself', name=idx
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # Results Table
    st.subheader("📋 Recommended Neighborhoods")
    cols_to_show = ['RLI', 'match_pct', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']
    st.dataframe(
        df_rec[cols_to_show].style.format("{:.2f}").background_gradient(subset=['match_pct'], cmap='YlGn'),
        use_container_width=True
    )
