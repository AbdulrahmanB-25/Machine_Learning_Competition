"""
Riyadh Livability Index — Streamlit Dashboard
==============================================
Mirrors RLI_Engine.ipynb logic exactly.

• 4 pillar weight sliders (auto-normalized)
• Match % per neighborhood (% of top scorer's RLI)
• Top-5 PCA scatter with star markers
• Pillar DNA radar chart
• Full 176-neighborhood ranking table
• Imputation audit panel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from rli_engine import run_full_pipeline, recommend, PILLARS, ALL_RLI_FEATURES

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Riyadh Livability Index',
    page_icon='🏙️',
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

# ── Palette (matches notebook) ────────────────────────────────────────────────
GOLD, CYAN, CORAL, MINT, PURPLE = '#f0c05a', '#4fc3f7', '#ff6b6b', '#66bb6a', '#ab47bc'
PALETTE = [GOLD, CYAN, CORAL, MINT, PURPLE]


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Running ML pipeline on 346K rows...')
def load_pipeline():
    for path in [
        '/mnt/user-data/uploads/Riyadh_Master_Dataset.csv',
        os.path.join(os.path.dirname(__file__), 'Riyadh_Master_Dataset.csv'),
    ]:
        if os.path.exists(path):
            return run_full_pipeline(path)
    st.error('Riyadh_Master_Dataset.csv not found.')
    st.stop()

pipe = load_pipeline()


# ── Sidebar: Pillar Weight Sliders ────────────────────────────────────────────
st.sidebar.title('🎛️ Pillar Weights')
st.sidebar.caption('Drag sliders to set importance per pillar. Weights are auto-normalized to sum to 1.')

w_core = st.sidebar.slider('🏥 Core (Medical · Schools · Retail · Mosques)',
                            0.0, 1.0, 0.40, 0.05)
w_mob  = st.sidebar.slider('🚌 Mobility (Bus · Metro · Walkability · Connectivity)',
                            0.0, 1.0, 0.25, 0.05)
w_wb   = st.sidebar.slider('🌳 Well-being (Dining · Parks · Sports · Fitness)',
                            0.0, 1.0, 0.20, 0.05)
w_inf  = st.sidebar.slider('🏗️ Infrastructure (Fiber · Govt · Malls · Universities)',
                            0.0, 1.0, 0.15, 0.05)

user_weights = {'Core': w_core, 'Mobility': w_mob, 'Well-being': w_wb, 'Infrastructure': w_inf}
total_w = sum(user_weights.values()) or 1.0
user_weights = {k: v / total_w for k, v in user_weights.items()}

st.sidebar.markdown('---')
st.sidebar.markdown('**Normalized Weights:**')
for k, v in user_weights.items():
    st.sidebar.markdown(f'- **{k}**: {v:.0%}')


# ── Compute Recommendations ──────────────────────────────────────────────────
df_rec = recommend(pipe['df_imputed'], user_weights=user_weights, top_n=176)

# Attach PCA + cluster from base pipeline
base = pipe['df_scored'][['PC1', 'PC2', 'km_cluster']]
df_rec = df_rec.join(base, rsuffix='_drop')
df_rec.drop(columns=[c for c in df_rec.columns if c.endswith('_drop')],
            inplace=True, errors='ignore')


# ── Header ────────────────────────────────────────────────────────────────────
st.title('🏙️ Riyadh Livability Index (RLI)')
st.markdown(
    'Neighborhood scoring via **Cluster-Centroid Imputation** → '
    '**4-Pillar Weighted Sum** × **Shannon Entropy Diversity Multiplier** → '
    '**0–100 min-max normalization** (best neighborhood = 100).'
)


# ── Top 5 Metrics ────────────────────────────────────────────────────────────
st.subheader('🏆 Top 5 Matches')
top5 = df_rec.head(5)

cols = st.columns(5)
for i, (idx, row) in enumerate(top5.iterrows()):
    with cols[i]:
        name = idx.replace(' Dist.', '') if isinstance(idx, str) else str(idx)
        st.metric(
            label=f'#{i+1}  {name}',
            value=f'{row["match_pct"]:.0f}%',
            delta=f'RLI {row["RLI"]:.1f}',
        )


# ── PCA Scatter — Top 5 Highlighted ──────────────────────────────────────────
st.subheader('📊 PCA Scatter — Top 5 Highlighted')

df_plot = df_rec.copy()
df_plot['label'] = df_plot.index.map(lambda n: n.replace(' Dist.', '') if isinstance(n, str) else str(n))
df_plot['is_top5'] = False
df_plot.iloc[:5, df_plot.columns.get_loc('is_top5')] = True

fig = go.Figure()

# All neighborhoods
other = df_plot[~df_plot['is_top5']]
fig.add_trace(go.Scatter(
    x=other['PC1'], y=other['PC2'],
    mode='markers',
    marker=dict(size=6, color=other['RLI'], colorscale='Viridis',
                opacity=0.5, line=dict(width=0.3, color='#2a2f4e'),
                colorbar=dict(title='RLI', thickness=15)),
    text=other['label'],
    hovertemplate='<b>%{text}</b><br>RLI: %{marker.color:.1f}<extra></extra>',
    name='All Neighborhoods',
))

# Top 5
t5 = df_plot[df_plot['is_top5']]
fig.add_trace(go.Scatter(
    x=t5['PC1'], y=t5['PC2'],
    mode='markers+text',
    marker=dict(size=16, color=GOLD, symbol='star',
                line=dict(width=1.5, color=CORAL)),
    text=t5['label'],
    textposition='top center',
    textfont=dict(size=10, color=GOLD),
    hovertemplate='<b>%{text}</b><extra>Top 5</extra>',
    name='Top 5 Matches',
))

fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0a0e27', plot_bgcolor='#0a0e27',
    height=500,
    xaxis_title='PC1', yaxis_title='PC2',
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    margin=dict(l=40, r=40, t=60, b=40),
)
st.plotly_chart(fig, use_container_width=True)


# ── Pillar DNA Radar — Top 5 ─────────────────────────────────────────────────
st.subheader('🕸️ Pillar DNA Radar — Top 5')

pillar_names = list(PILLARS.keys())
fig_radar = go.Figure()

for i, (idx, row) in enumerate(top5.iterrows()):
    vals = [row[f'pillar_{p}'] for p in pillar_names] + [row[f'pillar_{pillar_names[0]}']]
    name = idx.replace(' Dist.', '') if isinstance(idx, str) else str(idx)
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=pillar_names + [pillar_names[0]],
        fill='toself',
        name=f'#{i+1} {name}',
        line=dict(color=PALETTE[i]),
        opacity=0.6,
    ))

fig_radar.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0a0e27',
    polar=dict(bgcolor='#0f1333', radialaxis=dict(visible=True, color='#8b8fa3')),
    height=420,
    margin=dict(l=60, r=60, t=40, b=40),
)
st.plotly_chart(fig_radar, use_container_width=True)


# ── Full Ranking Table ────────────────────────────────────────────────────────
st.subheader('📋 Full Neighborhood Ranking')

display_cols = [
    'match_pct', 'RLI', 'rank',
    'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure',
    'H_entropy', 'km_cluster',
]
df_display = df_rec[[c for c in display_cols if c in df_rec.columns]].copy()
df_display.index = df_display.index.map(lambda n: n.replace(' Dist.', '') if isinstance(n, str) else n)
df_display.index.name = 'Neighborhood'
df_display.columns = [c.replace('pillar_', '').replace('_', ' ').title() for c in df_display.columns]

st.dataframe(
    df_display.style.format({
        'Match Pct': '{:.1f}%', 'Rli': '{:.2f}',
        'Core': '{:.4f}', 'Mobility': '{:.4f}',
        'Well-Being': '{:.4f}', 'Infrastructure': '{:.4f}',
        'H Entropy': '{:.4f}',
    }).background_gradient(subset=['Match Pct'], cmap='YlOrRd'),
    use_container_width=True, height=600,
)


# ── Imputation Audit ─────────────────────────────────────────────────────────
with st.expander('🔧 Data Repair Summary — Cluster-Centroid Imputation'):
    df_orig = pipe['df_neighborhoods']
    df_imp = pipe['df_imputed']

    rows = []
    for col in ALL_RLI_FEATURES:
        if col in df_orig.columns:
            n_zeros = int((df_orig[col] == 0).sum())
            n_fixed = int(((df_orig[col] == 0) & (df_imp[col] != 0)).sum())
            if n_zeros > 0:
                rows.append({'Feature': col, 'Zeros Found': n_zeros, 'Repaired': n_fixed})

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index('Feature'), use_container_width=True)
    else:
        st.info('No zero-value repairs needed.')

    k = pipe['df_scored']['km_cluster'].nunique()
    st.markdown(f'''
    **Method:** For each of the 16 RLI features, zeros are replaced with the
    mean of that feature within the neighborhood's K-Means cluster (k={k}).
    This prevents neighborhoods from being penalized when their cluster peers
    have non-zero values for the same service.
    ''')


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('---')
st.caption(
    'RLI Engine · Aggregate 346K rows → K-Means (k=2) → Centroid Impute 1,291 zeros → '
    'Min-Max Scale → 4-Pillar Weighted Sum × Shannon Entropy Multiplier → '
    '0–100 Normalization (best = 100)'
)
