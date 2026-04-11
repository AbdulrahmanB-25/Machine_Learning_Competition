"""
Riyadh Livability Index — Streamlit Dashboard
===============================================
Two tabs:
  1. City Ranking  — Global RLI ranking of all 176 neighborhoods.
  2. Property Search — Filter by category + price + rooms → recommended neighborhoods.

Both tabs respect the sidebar pillar weight sliders.
Imports all ML logic from rli_engine.py (single source of truth).

Run
---
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

from rli_engine import (
    build_global_ranking, recommend, property_search,
    PILLARS, ALL_RLI_FEATURES, CATEGORY_MAP, NO_ROOM_CATEGORIES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Riyadh Livability Explorer",
    page_icon="🏙️",
    layout="wide",
)


@st.cache_data
def load_raw():
    """Load the master dataset (cached across reruns)."""
    csv_path = os.environ.get('RIYADH_CSV', 'Riyadh_Master_Dataset.csv')
    if not os.path.exists(csv_path):
        # Fallback path
        alt = '/mnt/user-data/uploads/Riyadh_Master_Dataset.csv'
        if os.path.exists(alt):
            csv_path = alt
        else:
            return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_resource
def load_pipeline(_df_raw):
    """Build the global ranking pipeline (cached, runs once)."""
    return build_global_ranking(_df_raw)


df_raw = load_raw()

if df_raw.empty:
    st.error(
        "Riyadh_Master_Dataset.csv not found. "
        "Place it in the same folder as streamlit_app.py or set RIYADH_CSV env var."
    )
    st.stop()

pipe = load_pipeline(df_raw)
df_ranked = pipe['df_ranked']


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Pillar Weights
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("RLI DNA Weights")
st.sidebar.markdown("Adjust how much each pillar matters to you.")

w_core = st.sidebar.slider('Core (Health / Education)',   0.0, 1.0, 0.40, 0.05)
w_mob  = st.sidebar.slider('Mobility (Transit)',          0.0, 1.0, 0.25, 0.05)
w_well = st.sidebar.slider('Well-being (Parks / Dining)', 0.0, 1.0, 0.20, 0.05)
w_inf  = st.sidebar.slider('Infrastructure (Fiber)',      0.0, 1.0, 0.15, 0.05)

user_weights = {
    'Core':           w_core,
    'Mobility':       w_mob,
    'Well-being':     w_well,
    'Infrastructure': w_inf,
}

# Show normalized weights
total_w = sum(user_weights.values())
if total_w > 0:
    st.sidebar.caption(
        "Normalized: " +
        " | ".join(f"{k[:4]} {v/total_w:.0%}" for k, v in user_weights.items())
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

st.title("Riyadh Livability Index (RLI)")

tab1, tab2 = st.tabs(["City Ranking", "Property Search"])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: CITY RANKING
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Top Neighborhoods in Riyadh")
    st.caption("All 176 neighborhoods scored with your pillar weights. No property filters applied.")

    # Re-score with user weights
    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)  # 0 = all

    # ── Top 3 medals ──
    top3 = df_city.head(3)
    medals = ["1st", "2nd", "3rd"]
    cols = st.columns(3)
    for i, (name, row) in enumerate(top3.iterrows()):
        cols[i].metric(
            label=f"Rank {medals[i]}",
            value=name if isinstance(name, str) else str(name),
            delta=f"RLI {row['RLI']:.1f}",
        )

    # ── Radar chart: top 5 ──
    top5 = df_city.head(5)
    pillar_names = ['Core', 'Mobility', 'Well-being', 'Infrastructure']
    pillar_cols  = [f'pillar_{p}' for p in pillar_names]

    fig_radar = go.Figure()
    colors = ['#f0c05a', '#4fc3f7', '#ff6b6b', '#66bb6a', '#ab47bc']
    for idx, (name, row) in enumerate(top5.iterrows()):
        vals = [row[c] for c in pillar_cols]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=pillar_names + [pillar_names[0]],
            fill='toself',
            name=name if isinstance(name, str) else str(name),
            line=dict(color=colors[idx % len(colors)]),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Pillar Profile — Top 5 Neighborhoods",
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Full table ──
    display_cols = ['rank', 'RLI', 'match_pct',
                    'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure']
    present = [c for c in display_cols if c in df_city.columns]
    st.dataframe(
        df_city[present]
        .style.background_gradient(subset=['RLI'], cmap='YlGn')
        .format("{:.2f}", subset=[c for c in present if c != 'rank']),
        use_container_width=True,
        height=500,
    )


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: PROPERTY SEARCH (Recommendation System)
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("Personalized Recommendation")
    st.caption(
        "Filter for your property type, budget, and room needs. "
        "The system finds neighborhoods that have matching listings AND score high on livability."
    )

    # ── Filter controls ──
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Build options from categories that actually exist in the data
            available_cats = df_raw['category'].value_counts()
            cat_options = [k for k in sorted(CATEGORY_MAP.keys()) if k in available_cats.index]
            sel_cat = st.selectbox(
                "Property Category",
                options=cat_options,
                format_func=lambda x: f"{CATEGORY_MAP[x]}  ({available_cats.get(x, 0):,})",
                index=cat_options.index(3) if 3 in cat_options else 0,
            )

        cat_label = CATEGORY_MAP.get(sel_cat, '')
        with col2:
            min_p = st.number_input("Min Budget (SAR)", value=0, step=50_000, min_value=0)

        with col3:
            max_p = st.number_input("Max Budget (SAR)", value=2_000_000, step=50_000, min_value=0)

        with col4:
            if cat_label not in NO_ROOM_CATEGORIES:
                min_r = st.number_input("Min Rooms", 0, 20, 3)
            else:
                min_r = 0
                st.markdown("**Rooms:** N/A for this type")

    price_w = st.slider(
        "Price Importance",
        0.0, 1.0, 0.20, 0.05,
        help="0 = only livability matters, 1 = only price fit matters",
    )

    # ── Run search ──
    result = property_search(
        df_raw=pipe['df_raw'],
        df_ranked=df_ranked,
        category=sel_cat,
        min_price=min_p,
        max_price=max_p,
        min_rooms=min_r,
        pillar_weights=user_weights,
        price_weight=price_w,
    )

    df_res = result['results']
    props_matched = result['properties_matched']

    if df_res.empty:
        st.warning(
            "No neighborhoods found with those criteria. "
            "Try increasing your budget, lowering room count, or picking a different category."
        )
    else:
        # ── Search summary ──
        st.success(
            f"**{props_matched:,}** properties matched → "
            f"**{len(df_res)}** neighborhoods qualify "
            f"({len(df_ranked) - len(df_res)} eliminated)"
        )

        # ── Top 3 medals ──
        top3s = df_res.head(3)
        cols_s = st.columns(3)
        for i, (name, row) in enumerate(top3s.iterrows()):
            label = name if isinstance(name, str) else str(name)
            cols_s[i].metric(
                label=f"#{int(row['rank'])} — {label}",
                value=f"Match {row['match_pct']:.0f}%",
                delta=f"Avg {row.get('avg_price', 0):,.0f} SAR",
            )

        # ── Radar chart: top 5 ──
        top5s = df_res.head(min(5, len(df_res)))
        fig_r2 = go.Figure()
        for idx, (name, row) in enumerate(top5s.iterrows()):
            vals = [row.get(c, 0) for c in pillar_cols]
            fig_r2.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=pillar_names + [pillar_names[0]],
                fill='toself',
                name=name if isinstance(name, str) else str(name),
                line=dict(color=colors[idx % len(colors)]),
                opacity=0.7,
            ))
        fig_r2.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title=f"Pillar Profile — Top {len(top5s)} for {cat_label}",
            height=450,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

        # ── Results table ──
        search_cols = [
            'rank', 'match_pct', 'combined_score', 'RLI', 'price_score',
            'avg_price', 'median_price', 'filtered_count', 'avg_area', 'avg_rooms',
            'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure',
        ]
        present_s = [c for c in search_cols if c in df_res.columns]
        st.dataframe(
            df_res[present_s]
            .style.background_gradient(subset=['match_pct'], cmap='Blues')
            .format({
                'match_pct': '{:.1f}%',
                'combined_score': '{:.2f}',
                'RLI': '{:.2f}',
                'price_score': '{:.1f}',
                'avg_price': '{:,.0f}',
                'median_price': '{:,.0f}',
                'filtered_count': '{:,.0f}',
                'avg_area': '{:,.0f}',
                'avg_rooms': '{:.1f}',
                'pillar_Core': '{:.3f}',
                'pillar_Mobility': '{:.3f}',
                'pillar_Well-being': '{:.3f}',
                'pillar_Infrastructure': '{:.3f}',
            }),
            use_container_width=True,
            height=500,
        )
