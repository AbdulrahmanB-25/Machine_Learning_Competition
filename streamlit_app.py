import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --- 1. CORE ENGINE LOGIC ---
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

    # Normalize to 0-100 scale
    r_min, r_max = raw_rli.min(), raw_rli.max()
    rli_100 = ((raw_rli - r_min) / (r_max - r_min + EPSILON)) * 100

    result = df_in.copy()
    for pname in PILLARS:
        result[f'pillar_{pname}'] = pillar_scores[pname].values
    result['RLI'] = rli_100.round(2)
    # Correct Ranking Logic
    result['Rank'] = result['RLI'].rank(ascending=False, method='min').astype(int)
    return result.sort_values('Rank')

# --- 2. STREAMLIT CONFIG & DATA ---
st.set_page_config(page_title='RLI Explorer', layout='wide')

@st.cache_data
def load_data():
    csv_path = 'Riyadh_Master_Dataset.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df_raw = load_data()

if df_raw.empty:
    st.error("Riyadh_Master_Dataset.csv not found!")
    st.stop()

# --- 3. SIDEBAR (Global Weights) ---
st.sidebar.header("⚖️ RLI Weighting")
w_core = st.sidebar.slider('Core Essentials', 0.0, 1.0, 0.40)
w_mob  = st.sidebar.slider('Mobility', 0.0, 1.0, 0.25)
w_well = st.sidebar.slider('Well-being', 0.0, 1.0, 0.20)
w_inf  = st.sidebar.slider('Infrastructure', 0.0, 1.0, 0.15)
user_weights = {'Core': w_core, 'Mobility': w_mob, 'Well-being': w_well, 'Infrastructure': w_inf}

# --- 4. MAIN INTERFACE (TABS) ---
st.title("🏙️ Riyadh Livability Index")

tab1, tab2 = st.tabs(["🏆 City-Wide Ranking", "🎯 Personalized Search"])

# --- TAB 1: CITY RANKING ---
with tab1:
    st.header("General Neighborhood Rankings")
    st.write("Ranking all neighborhoods based on total service availability and your weights.")
    
    # Process all data
    df_agg_all = df_raw.groupby('neighborhood')[ALL_RLI_FEATURES].mean().fillna(0)
    df_city_ranked = compute_rli(df_agg_all, pillar_weights=user_weights)
    
    # Metrics
    top_n = df_city_ranked.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Top Neighborhood", top_n.name)
    c2.metric("Highest RLI Score", f"{top_n['RLI']}")
    c3.metric("Neighborhoods Scored", len(df_city_ranked))

    st.dataframe(
        df_city_ranked[['Rank', 'RLI', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']]
        .style.format("{:.2f}", subset=['RLI', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure'])
        .background_gradient(subset=['RLI'], cmap='YlGn'),
        use_container_width=True
    )

# --- TAB 2: PROPERTY SEARCH ---
with tab2:
    st.header("Find My Neighborhood")
    st.write("Filter the city to find areas that have the specific properties you need.")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_list = ["All"] + sorted(list(df_raw['category'].unique()))
            sel_cat = st.selectbox("Property Type", options=cat_list)
        with col2:
            max_price = st.number_input("Max Price (SAR)", value=1000000, step=50000)
        with col3:
            min_rms = st.number_input("Min Rooms", 0, 10, 0)

    # Filtered Logic
    df_filt = df_raw.copy()
    if sel_cat != "All":
        df_filt = df_filt[df_filt['category'] == sel_cat]
    df_filt = df_filt[df_filt['price'] <= max_price]
    if min_rms > 0 and 'total_rooms' in df_filt.columns:
        df_filt = df_filt[df_filt['total_rooms'] >= min_rms]

    if df_filt.empty:
        st.warning("No properties match these criteria. Try increasing your budget.")
    else:
        df_agg_filt = df_filt.groupby('neighborhood')[ALL_RLI_FEATURES].mean().fillna(0)
        df_search_results = compute_rli(df_agg_filt, pillar_weights=user_weights)
        
        # Add a Match % relative to the best neighborhood in this specific search
        df_search_results['Match %'] = (df_search_results['RLI'] / (df_search_results['RLI'].max() + EPSILON) * 100).round(1)

        # Radar Chart for Search results
        st.subheader(f"Top 3 Results for {sel_cat}")
        top_3_search = df_search_results.head(3)
        fig = go.Figure()
        for idx, row in top_3_search.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['pillar_Core'], row['pillar_Mobility'], row['pillar_Well-being'], row['pillar_Infrastructure']],
                theta=['Core', 'Mobility', 'Well-being', 'Infrastructure'],
                fill='toself', name=idx
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_search_results[['Rank', 'RLI', 'Match %', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']]
            .style.format("{:.2f}").background_gradient(subset=['Match %'], cmap='Blues'),
            use_container_width=True
        )
