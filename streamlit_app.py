import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys
from sklearn.preprocessing import MinMaxScaler

# --- 1. CORE ENGINE LOGIC (Internalized) ---
EPSILON = 1e-9
H_MAX = np.log(4)

PILLARS = {
    'Core': {'weight': 0.40, 'features': ['med_facilities', 'edu_primary', 'essential_retail', 'religious']},
    'Mobility': {'weight': 0.25, 'features': ['bus_count', 'metro_count', 'pedestrian', 'connectivity_score']},
    'Well-being': {'weight': 0.20, 'features': ['dining_cafe', 'parks_green', 'sports_play', 'fitness_care']},
    'Infrastructure': {'weight': 0.15, 'features': ['Fiber_Available', 'gov_civil', 'malls_shopping', 'edu_higher']}
}
ALL_RLI_FEATURES = [f for p in PILLARS.values() for f in p['features']]

CATEGORY_MAP = {
    1: "Apartment (Rent)", 2: "Land (Sell)", 3: "Villa (Sell)", 4: "Floor (Rent)",
    5: "Villa (Rent)", 6: "Apartment (Sell)", 7: "Building (Sell)", 8: "Store (Rent)",
    9: "House (Sell)", 10: "Esterahah (Sell)", 11: "House (Rent)", 12: "Farm (Sell)",
    13: "Esterahah (Rent)", 14: "Office (Rent)", 15: "Land (Rent)", 16: "Building (Rent)",
    17: "Warehouse (Rent)", 18: "Campsite (Rent)", 19: "Room (Rent)", 20: "Store (Sell)",
    21: "Furnished Apartment", 22: "Floor (Sell)", 23: "Chalet (Rent)"
}

# Categories that typically DO NOT have rooms
NO_ROOM_CATEGORIES = [2, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 20]

def compute_rli(df_in, pillar_weights=None):
    if df_in.empty: return pd.DataFrame()
    
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
    
    result = df_in.copy()
    for pname in PILLARS:
        result[f'pillar_{pname}'] = pillar_scores[pname].values
    
    raw_rli = weighted_sum * diversity_mult
    r_min, r_max = raw_rli.min(), raw_rli.max()
    result['RLI'] = (((raw_rli - r_min) / (r_max - r_min + EPSILON)) * 100).round(2)
    result['Rank'] = result['RLI'].rank(ascending=False, method='min').astype(int)
    
    return result.sort_values('Rank')

# --- 2. PAGE SETUP & DATA ---
st.set_page_config(page_title="Riyadh Livability Explorer", layout="wide")

@st.cache_data
def get_data():
    csv_path = 'Riyadh_Master_Dataset.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df_raw = get_data()

if df_raw.empty:
    st.error("Error: 'Riyadh_Master_Dataset.csv' not found.")
    st.stop()

# --- 3. SIDEBAR (Global Weights) ---
st.sidebar.title("⚖️ RLI DNA Weights")
st.sidebar.markdown("Adjust how much each pillar matters to you.")
w_core = st.sidebar.slider('Core (Health/Edu)', 0.0, 1.0, 0.40)
w_mob  = st.sidebar.slider('Mobility (Transit)', 0.0, 1.0, 0.25)
w_well = st.sidebar.slider('Well-being (Parks)', 0.0, 1.0, 0.20)
w_inf  = st.sidebar.slider('Infrastructure (Fiber)', 0.0, 1.0, 0.15)
u_weights = {'Core': w_core, 'Mobility': w_mob, 'Well-being': w_well, 'Infrastructure': w_inf}

# --- 4. MAIN INTERFACE ---
st.title("🏙️ Riyadh Livability Index (RLI)")

tab1, tab2 = st.tabs(["🏆 City Ranking", "🎯 Property Search"])

# --- TAB 1: CITY RANKING ---
with tab1:
    st.header("Top Neighborhoods in Riyadh")
    # Aggregate all data for a general view
    df_all = df_raw.groupby('neighborhood')[ALL_RLI_FEATURES].mean().fillna(0)
    df_city = compute_rli(df_all, u_weights)
    
    top_3 = df_city.head(3)
    c1, c2, c3 = st.columns(3)
    for i, (name, row) in enumerate(top_3.iterrows()):
        cols = [c1, c2, c3]
        cols[i].metric(f"Rank {i+1}", name, f"{row['RLI']} RLI")

    st.dataframe(
        df_city[['Rank', 'RLI', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']]
        .style.background_gradient(subset=['RLI'], cmap='YlGn').format("{:.2f}", subset=['RLI','pillar_Core','pillar_Mobility','pillar_Well-being','pillar_Infrastructure']),
        use_container_width=True
    )

# --- TAB 2: PROPERTY SEARCH ---
with tab2:
    st.header("Personalized Recommendation")
    st.write("Filter for neighborhoods that have exactly what you need.")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            sel_cat = st.selectbox("Category", options=list(CATEGORY_MAP.keys()), format_func=lambda x: CATEGORY_MAP[x])
        with col2:
            max_p = st.number_input("Maximum Budget (SAR)", value=1000000, step=50000)
        with col3:
            # Check if rooms are relevant
            if sel_cat not in NO_ROOM_CATEGORIES:
                min_r = st.number_input("Min Rooms", 0, 10, 3)
            else:
                min_r = 0
                st.write("Rooms: N/A for this type")

    # Apply Filters
    df_f = df_raw[df_raw['category'] == sel_cat].copy()
    df_f = df_f[df_f['price'] <= max_p]
    if min_r > 0 and 'total_rooms' in df_f.columns:
        df_f = df_f[df_f['total_rooms'] >= min_r]

    if df_f.empty:
        st.warning("No neighborhoods found with those specific criteria. Try increasing your budget.")
    else:
        df_agg_f = df_f.groupby('neighborhood')[ALL_RLI_FEATURES].mean().fillna(0)
        df_search = compute_rli(df_agg_f, u_weights)
        df_search['Match %'] = (df_search['RLI'] / (df_search['RLI'].max() + EPSILON) * 100).round(1)

        st.subheader(f"Best Results for {CATEGORY_MAP[sel_cat]}")
        
        # Radar Chart
        fig = go.Figure()
        for idx, row in df_search.head(3).iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['pillar_Core'], row['pillar_Mobility'], row['pillar_Well-being'], row['pillar_Infrastructure']],
                theta=['Core', 'Mobility', 'Well-being', 'Infrastructure'], fill='toself', name=idx
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_search[['Rank', 'RLI', 'Match %', 'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']]
            .style.background_gradient(subset=['Match %'], cmap='Blues').format("{:.2f}"),
            use_container_width=True
        )
