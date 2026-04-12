import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from rli_engine import (
    build_global_ranking,
    recommend,
    property_search,
    CATEGORY_MAP,
    NO_ROOM_CATEGORIES,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Riyadh Livability Index",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =========================================================
# GLOBAL STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #edf4f5 0%, #dbe9ec 100%);
    }

    header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14353d 0%, #1d4953 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-title {
        font-size: 3.6rem;
        font-weight: 900;
        letter-spacing: 1px;
        color: white;
        margin-bottom: 0.4rem;
        line-height: 1;
        text-transform: uppercase;
    }

    .sub-title {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 1.6rem;
    }

    .cover-wrap {
        min-height: 86vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .cover-card {
        width: 100%;
        min-height: 78vh;
        border: 3px solid #7c3aed;
        box-shadow: 0 0 0 2px rgba(124,58,237,0.2) inset;
        border-radius: 0;
        overflow: hidden;
        position: relative;
        background:
            linear-gradient(to top, rgba(10,30,35,0.72) 0%, rgba(10,30,35,0.24) 30%, rgba(10,30,35,0.02) 65%),
            linear-gradient(180deg, rgba(92,139,149,0.72) 0%, rgba(153,191,197,0.46) 100%),
            url('https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?auto=format&fit=crop&w=1600&q=80');
        background-size: cover;
        background-position: center;
    }

    .cover-overlay {
        position: absolute;
        inset: 0;
        padding: 2rem 2.2rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .brand-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
    }

    .brand-left {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        color: white;
        font-weight: 700;
        font-size: 1rem;
    }

    .brand-icon {
        width: 42px;
        height: 42px;
        border: 2px solid rgba(255,255,255,0.85);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        backdrop-filter: blur(4px);
    }

    .brand-right {
        color: rgba(16,28,40,0.98);
        text-align: right;
        font-size: 0.92rem;
        font-weight: 800;
        line-height: 1.2;
        background: rgba(255,255,255,0.45);
        padding: 0.35rem 0.55rem;
        border-radius: 10px;
        backdrop-filter: blur(4px);
    }

    .center-content {
        text-align: center;
        margin-top: -2.5rem;
    }

    .team-row {
        margin-top: 1.5rem;
        display: flex;
        justify-content: center;
        gap: 4rem;
        flex-wrap: wrap;
        color: white;
        font-size: 0.96rem;
        font-weight: 600;
    }

    .team-row div {
        text-align: left;
        min-width: 130px;
    }

    .start-button-wrap {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid rgba(20,53,61,0.08);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 10px 25px rgba(18, 38, 45, 0.08);
        margin-bottom: 1rem;
    }
    .brand-text {
        background: linear-gradient(90deg, #7c3aed, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
    }

    .mini-step {
        background: #ffffff;
        border-radius: 18px;
        padding: 1rem;
        color: #14353d;
        border: 1px solid rgba(20,53,61,0.08);
        box-shadow: 0 8px 20px rgba(20,53,61,0.06);

        /* المهم */
        height: 160px;                
        display: flex;
        flex-direction: column;
        justify-content: center;       
        align-items: center;           
        text-align: center;
    }

    .mini-step .step-no {
        font-size: 0.85rem;
        font-weight: 800;
        color: #6b21a8;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }

    .mini-step .step-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #10333a;
        margin-bottom: 0.25rem;
    }

    .mini-step .step-text {
        font-size: 0.92rem;
        color: #315760;
    }

    .page-banner {
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        background:
            linear-gradient(90deg, rgba(10, 34, 40, 0.92) 0%, rgba(20, 73, 79, 0.82) 55%, rgba(88, 137, 145, 0.52) 100%),
            url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=1400&q=80');
        background-size: cover;
        background-position: center;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(18, 38, 45, 0.12);
    }

    .page-banner h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 900;
        text-transform: uppercase;
        color: white !important;
    }

    .page-banner p {
        margin: 0.5rem 0 0 0;
        color: rgba(255,255,255,0.95) !important;
        font-size: 1rem;
        max-width: 760px;
    }

    .stButton > button {
        border: none;
        border-radius: 14px;
        background: linear-gradient(90deg, #7c3aed 0%, #14b8a6 100%);
        color: white;
        font-weight: 800;
        padding: 0.75rem 1.2rem;
    }

    .stDownloadButton > button {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid rgba(20,53,61,0.08);
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(20,53,61,0.06);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: #10333a !important;
    }

    div[data-testid="stDataFrame"] {
        background: #ffffff;
        border: 1px solid rgba(20,53,61,0.08);
        border-radius: 18px;
        padding: 0.3rem;
    }

    .small-note {
        color: #244851;
        font-size: 0.95rem;
        margin-top: -0.25rem;
    }

    h1, h2, h3 {
        color: #10333a !important;
    }

    p, label, span {
        color: #244851;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_raw():
    csv_path = os.environ.get("RIYADH_CSV", "Riyadh_Master_Dataset.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


@st.cache_resource
def load_pipeline(_df_raw):
    return build_global_ranking(_df_raw)

# =========================================================
# HOME PAGE
# =========================================================
if st.session_state.page == "home":
    st.markdown(
        """
        <div class="page-banner">
            <h1><span class="brand-text">Riyadh Livability Index</span></h1>
            <p>
                Find the right neighborhood in Riyadh using livability scores,
                property filters, and clear comparisons.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    with a:
        st.markdown(
            """
            <div class="mini-step">
                <div class="step-no">Step 1</div>
                <div class="step-title">Set your needs</div>
                <div class="step-text">Adjust pillars and define your housing criteria.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="mini-step">
                <div class="step-no">Step 2</div>
                <div class="step-title">See best matches</div>
                <div class="step-text">Get ranked neighborhoods based on your priorities.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c:
        st.markdown(
            """
            <div class="mini-step">
                <div class="step-no">Step 3</div>
                <div class="step-title">Understand each place</div>
                <div class="step-text">Compare results using scores, charts, and property fit.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    x1, x2, x3 = st.columns([1.3, 1, 1.3])
    with x2:
        if st.button("Start Finding Neighborhoods", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

    st.stop()

# =========================================================
# LOAD DATA AFTER ENTERING APP
# =========================================================
df_raw = load_raw()

if df_raw.empty:
    st.error("Dataset not found. Make sure Riyadh_Master_Dataset.csv exists.")
    st.stop()

pipe = load_pipeline(df_raw)
df_ranked = pipe["df_ranked"]

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Riyadh Livability Index")
st.sidebar.markdown("Adjust the importance of each pillar")

w_core = st.sidebar.slider("Core", 0.0, 1.0, 0.40)
w_mob = st.sidebar.slider("Mobility", 0.0, 1.0, 0.25)
w_well = st.sidebar.slider("Well-being", 0.0, 1.0, 0.20)
w_inf = st.sidebar.slider("Infrastructure", 0.0, 1.0, 0.15)

user_weights = {
    "Core": w_core,
    "Mobility": w_mob,
    "Well-being": w_well,
    "Infrastructure": w_inf,
}

st.sidebar.markdown("---")
if st.sidebar.button("Back to Home", use_container_width=True):
    st.session_state.page = "home"
    st.rerun()

# =========================================================
# MAIN PAGE
# =========================================================
st.markdown(
    """
    <div class="page-banner">
        <h1><span class="brand-text">Riyadh Livability Explorer</h1>
        <p>
            Explore neighborhood rankings and search properties using your own priorities.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["City Ranking", "Property Search"])

# =========================================================
# SHARED CHART DATA
# =========================================================
pillar_names = ["Core", "Mobility", "Well-being", "Infrastructure"]
pillar_cols = [f"pillar_{p}" for p in pillar_names]
colors = ["#f0c05a", "#4fc3f7", "#ff6b6b", "#66bb6a", "#ab47bc"]

# =========================================================
# TAB 1
# =========================================================
with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top Neighborhoods in Riyadh")
    st.markdown(
        '<div class="small-note">All neighborhoods are rescored using your pillar weights.</div>',
        unsafe_allow_html=True,
    )

    df_city = recommend(df_ranked, user_weights=user_weights, top_n=0)

    top3 = df_city.head(3)
    medals = ["1st", "2nd", "3rd"]
    cols = st.columns(3)

    for i, (name, row) in enumerate(top3.iterrows()):
        cols[i].metric(
            label=f"Rank {medals[i]}",
            value=name if isinstance(name, str) else str(name),
            delta=f"RLI {row['RLI']:.1f}",
        )

    top5 = df_city.head(5)
    fig_radar = go.Figure()

    for idx, (name, row) in enumerate(top5.iterrows()):
        vals = [row[c] for c in pillar_cols]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=pillar_names + [pillar_names[0]],
                fill="toself",
                name=name if isinstance(name, str) else str(name),
                line=dict(color=colors[idx % len(colors)]),
                opacity=0.7,
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                color="#10333a",        
                gridcolor="rgba(0,0,0,0.15)"
            ),
            angularaxis=dict(
                color="#10333a"         
            )
        ),

        font=dict(
            color="#10333a",            
            size=14
        ),

        legend=dict(
            font=dict(color="#10333a")  
        ),

        title=dict(
            text="Pillar Profile of Top 5 Neighborhoods",
            font=dict(color="#10333a", size=18)
        ),

        paper_bgcolor="rgba(0,0,0,0)",  
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_radar, use_container_width=True)

    display_cols = [
        "rank",
        "RLI",
        "match_pct",
        "pillar_Core",
        "pillar_Mobility",
        "pillar_Well-being",
        "pillar_Infrastructure",
    ]
    present = [c for c in display_cols if c in df_city.columns]

    st.dataframe(
        df_city[present]
        .style.background_gradient(subset=["RLI"], cmap="YlGn")
        .format("{:.2f}", subset=[c for c in present if c != "rank"]),
        use_container_width=True,
        height=520,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 2
# =========================================================
with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Personalized Recommendation")
    st.markdown(
        '<div class="small-note">Filter by category, budget, and room needs, then combine property fit with livability.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            available_cats = df_raw["category"].value_counts()
            cat_options = [k for k in sorted(CATEGORY_MAP.keys()) if k in available_cats.index]

            sel_cat = st.selectbox(
                "Property Category",
                options=cat_options,
                format_func=lambda x: f"{CATEGORY_MAP[x]} ({available_cats.get(x, 0):,})",
                index=cat_options.index(3) if 3 in cat_options else 0,
            )

        cat_label = CATEGORY_MAP.get(sel_cat, "")

        with col2:
            min_p = st.number_input("Min Budget (SAR)", value=0, step=50000, min_value=0)

        with col3:
            max_p = st.number_input("Max Budget (SAR)", value=2000000, step=50000, min_value=0)

        with col4:
            if cat_label not in NO_ROOM_CATEGORIES:
                min_r = st.number_input("Min Rooms", 0, 20, 3)
            else:
                min_r = 0
                st.markdown("**Rooms:** N/A")

    price_w = st.slider(
        "Price Importance",
        0.0,
        1.0,
        0.20,
        0.05,
        help="0 means livability matters more. 1 means price fit matters more.",
    )

    cluster_w = st.slider(
        "Cluster Matching Weight",
        0.0,
        0.50,
        0.15,
        0.05,
        help="How much K-Means neighborhood archetype matching influences the recommendation.",
    )

    result = property_search(
        df_raw=pipe["df_raw"],
        df_ranked=df_ranked,
        category=sel_cat,
        min_price=min_p,
        max_price=max_p,
        min_rooms=min_r,
        pillar_weights=user_weights,
        price_weight=price_w,
        cluster_weight=cluster_w,
    )

    df_res = result["results"]
    props_matched = result["properties_matched"]

    if df_res.empty:
        st.warning(
            "No neighborhoods found with these filters. Try a higher budget, fewer rooms, or another category."
        )
    else:
        best_cluster = result.get("best_cluster", -1)
        cluster_scores = result.get("cluster_scores", {})

        st.success(
            f"{props_matched:,} properties matched. {len(df_res)} neighborhoods qualify."
        )

        # ── Cluster matching info ──
        if best_cluster >= 0 and cluster_scores:
            cluster_info_cols = st.columns([2, 3])
            with cluster_info_cols[0]:
                st.metric(
                    label="Best-Fit Cluster",
                    value=f"Cluster {best_cluster}",
                    delta=f"Score {cluster_scores.get(best_cluster, 0):.4f}",
                )
            with cluster_info_cols[1]:
                st.caption("K-Means matched your lifestyle priorities to the neighborhood archetype that best aligns with your pillar weights.")

        top3s = df_res.head(3)
        cols_s = st.columns(3)

        for i, (name, row) in enumerate(top3s.iterrows()):
            label = name if isinstance(name, str) else str(name)
            cluster_tag = f" (C{int(row['km_cluster'])})" if 'km_cluster' in row and pd.notna(row.get('km_cluster')) else ""
            cols_s[i].metric(
                label=f"#{int(row['rank'])} — {label}{cluster_tag}",
                value=f"Match {row['match_pct']:.0f}%",
                delta=f"Avg {row.get('avg_price', 0):,.0f} SAR",
            )

        top5s = df_res.head(min(5, len(df_res)))
        fig_r2 = go.Figure()

        for idx, (name, row) in enumerate(top5s.iterrows()):
            vals = [row.get(c, 0) for c in pillar_cols]
            fig_r2.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=pillar_names + [pillar_names[0]],
                    fill="toself",
                    name=name if isinstance(name, str) else str(name),
                    line=dict(color=colors[idx % len(colors)]),
                    opacity=0.7,
                )
            )

        fig_r2.update_layout(
            polar=dict(
        radialaxis=dict(
            visible=True,
            color="#10333a",        
            gridcolor="rgba(0,0,0,0.15)"
        ),
        angularaxis=dict(
            color="#10333a"         
        )
            ),

            font=dict(
                color="#10333a",            
                size=14
            ),

            legend=dict(
                font=dict(color="#10333a")  
            ),

            title=dict(
                text="Pillar Profile of Top 5 Neighborhoods",
                font=dict(color="#10333a", size=18)
            ),

            paper_bgcolor="rgba(0,0,0,0)",  
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig_r2, use_container_width=True)

        search_cols = [
            "rank",
            "match_pct",
            "combined_score",
            "RLI",
            "price_score",
            "cluster_fit",
            "km_cluster",
            "avg_price",
            "median_price",
            "filtered_count",
            "avg_area",
            "avg_rooms",
            "pillar_Core",
            "pillar_Mobility",
            "pillar_Well-being",
            "pillar_Infrastructure",
        ]
        present_s = [c for c in search_cols if c in df_res.columns]

        st.dataframe(
            df_res[present_s]
            .style.background_gradient(subset=["match_pct"], cmap="Blues")
            .format(
                {
                    "match_pct": "{:.1f}%",
                    "combined_score": "{:.2f}",
                    "RLI": "{:.2f}",
                    "price_score": "{:.1f}",
                    "cluster_fit": "{:.1f}",
                    "km_cluster": "{:.0f}",
                    "avg_price": "{:,.0f}",
                    "median_price": "{:,.0f}",
                    "filtered_count": "{:,.0f}",
                    "avg_area": "{:,.0f}",
                    "avg_rooms": "{:.1f}",
                    "pillar_Core": "{:.3f}",
                    "pillar_Mobility": "{:.3f}",
                    "pillar_Well-being": "{:.3f}",
                    "pillar_Infrastructure": "{:.3f}",
                }
            ),
            use_container_width=True,
            height=520,
        )

    st.markdown("</div>", unsafe_allow_html=True)