"""
Riyadh Livability Index (RLI) — Core Engine
============================================
Extracted from RLI_Engine.ipynb (final draft).

Pipeline: Aggregate → K-Means Cluster → Centroid-Impute Zeros
         → 4-Pillar Weighted Score × Shannon Entropy → Rank

RLI_i = (Σ w_j · x̂_ij) × [1 + H_i / H_max]
    x̂  = Min-Max scaled features (0–1)
    w  = User-defined or default pillar weights
    H  = Shannon Entropy  −Σ p·ln(p)  across 4 pillars
    H_max = ln(4)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Constants ─────────────────────────────────────────────────────────────────
EPSILON = 1e-9
H_MAX = np.log(4)  # ln(4) — theoretical max Shannon entropy for 4 pillars

# ── Feature Groups (notebook cell 5) ──────────────────────────────────────────
PILLAR_COLS = [
    'dining_cafe', 'med_facilities', 'health_retail', 'fitness_care',
    'edu_primary', 'edu_higher', 'religious', 'essential_retail',
    'parks_green', 'sports_play', 'pedestrian', 'resort_rural_retreats',
    'gov_civil', 'malls_shopping',
]
CONNECTIVITY_COLS = ['Fiber_Available', 'FWA_Available', 'Mobile_Available', 'connectivity_score']
TRANSIT_COLS = ['bus_count', 'metro_count']

# ── 4 RLI Pillars (notebook cell 7) ──────────────────────────────────────────
PILLARS = {
    'Core': {
        'weight': 0.40,
        'features': ['med_facilities', 'edu_primary', 'essential_retail', 'religious'],
    },
    'Mobility': {
        'weight': 0.25,
        'features': ['bus_count', 'metro_count', 'pedestrian', 'connectivity_score'],
    },
    'Well-being': {
        'weight': 0.20,
        'features': ['dining_cafe', 'parks_green', 'sports_play', 'fitness_care'],
    },
    'Infrastructure': {
        'weight': 0.15,
        'features': ['Fiber_Available', 'gov_civil', 'malls_shopping', 'edu_higher'],
    },
}

ALL_RLI_FEATURES = sorted({f for p in PILLARS.values() for f in p['features']})

# 24 ML features for clustering (notebook cell 10)
ML_COLS = PILLAR_COLS + CONNECTIVITY_COLS + TRANSIT_COLS + [
    'property_count', 'median_price', 'median_area', 'n_categories',
]


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation (notebook cell 5)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_to_neighborhoods(df_raw: pd.DataFrame) -> pd.DataFrame:
    """346K property rows → 176 neighborhood rows × 26 features."""
    neigh_const = PILLAR_COLS + CONNECTIVITY_COLS + TRANSIT_COLS
    df_nc = df_raw.groupby('neighborhood')[neigh_const].first()
    df_ps = df_raw.groupby('neighborhood').agg(
        property_count=('property_id', 'count'),
        median_price=('price', 'median'),
        median_area=('area', 'median'),
        n_categories=('category', 'nunique'),
    ).round(1)
    df_coords = df_raw.groupby('neighborhood')[['lat', 'lng']].mean()
    df = pd.concat([df_nc, df_ps, df_coords], axis=1)
    df[CONNECTIVITY_COLS] = df[CONNECTIVITY_COLS].fillna(0)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# K-Means Clustering (notebook cell 10)
# ══════════════════════════════════════════════════════════════════════════════

def fit_clusters(df: pd.DataFrame, k: int = 2, random_state: int = 42):
    """StandardScale 24 features → K-Means (k=2) → PCA 2D."""
    scaler_std = StandardScaler()
    X_scaled = scaler_std.fit_transform(df[ML_COLS])

    km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
    labels = km.fit_predict(X_scaled)

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_2d = pca_2d.fit_transform(X_scaled)

    return labels, km, scaler_std, pca_2d, X_2d


# ══════════════════════════════════════════════════════════════════════════════
# Cluster-Centroid Imputation (notebook cell 14)
# ══════════════════════════════════════════════════════════════════════════════

def cluster_centroid_impute(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Replace 0-valued RLI features with their cluster mean (vectorized)."""
    df_out = df.copy()
    df_out['_cluster'] = labels

    for col in ALL_RLI_FEATURES:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(float)

    for col in ALL_RLI_FEATURES:
        if col not in df_out.columns:
            continue
        cluster_means = df_out.groupby('_cluster')[col].transform('mean')
        mask_zero = df_out[col] == 0
        df_out.loc[mask_zero, col] = cluster_means[mask_zero]

    df_out.drop(columns=['_cluster'], inplace=True)
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# RLI Scoring (notebook cell 18 — exact logic)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rli(df_in: pd.DataFrame, pillar_weights: dict | None = None):
    """
    Compute Riyadh Livability Index for all neighborhoods.

    Parameters
    ----------
    df_in : DataFrame with ALL_RLI_FEATURES columns
    pillar_weights : dict like {'Core': 0.4, ...} or None for defaults

    Returns
    -------
    df_result : DataFrame with RLI, pillar scores, entropy, rank — sorted by rank
    scaler_mm : fitted MinMaxScaler
    """
    pw = pillar_weights or {p: v['weight'] for p, v in PILLARS.items()}
    total_w = sum(pw.values())
    pw = {k: v / total_w for k, v in pw.items()}  # auto-normalize

    # Step 1: Min-Max scale
    scaler_mm = MinMaxScaler()
    scaled_vals = scaler_mm.fit_transform(df_in[ALL_RLI_FEATURES])
    df_sc = pd.DataFrame(scaled_vals, columns=ALL_RLI_FEATURES, index=df_in.index)

    # Step 2: Weighted pillar scores (vectorized)
    pillar_scores = {}
    for pname, pinfo in PILLARS.items():
        feats = pinfo['features']
        w = pw[pname]
        pillar_scores[pname] = df_sc[feats].mean(axis=1) * w

    pillar_df = pd.DataFrame(pillar_scores, index=df_in.index)
    weighted_sum = pillar_df.sum(axis=1)  # Σ w_j * x̂_ij

    # Step 3: Shannon Entropy across 4 pillars
    pillar_props = pillar_df.div(pillar_df.sum(axis=1) + EPSILON, axis=0)
    H = -(pillar_props * np.log(pillar_props + EPSILON)).sum(axis=1)

    # Step 4: Diversity multiplier [1 + H/H_max]
    diversity_mult = 1 + (H / H_MAX)

    # Step 5: RLI = weighted_sum × diversity_multiplier
    raw_rli = weighted_sum * diversity_mult

    # Step 6: Final Normalization (0–100 min-max)
    r_min, r_max = raw_rli.min(), raw_rli.max()
    rli_100 = ((raw_rli - r_min) / (r_max - r_min)) * 100

    # Assemble result
    result = df_in.copy()
    for pname in PILLARS:
        result[f'pillar_{pname}'] = pillar_scores[pname].values
    result['H_entropy'] = H.values
    result['diversity_mult'] = diversity_mult.values
    result['raw_rli'] = raw_rli.values
    result['RLI'] = rli_100.round(2).values
    result['rank'] = result['RLI'].rank(ascending=False, method='min').astype(int)

    return result.sort_values('rank'), scaler_mm


# ══════════════════════════════════════════════════════════════════════════════
# Recommendation (notebook cell 32)
# ══════════════════════════════════════════════════════════════════════════════

def recommend(df_imputed: pd.DataFrame, user_weights: dict | None = None, top_n: int = 10):
    """Re-score with custom pillar weights, return top_n with match_pct."""
    scored, _ = compute_rli(df_imputed, pillar_weights=user_weights)
    scored['match_pct'] = (scored['RLI'] / scored['RLI'].max() * 100).round(1)
    return scored.sort_values('RLI', ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(csv_path: str):
    """End-to-end: load → aggregate → cluster → impute → score."""
    df_raw = pd.read_csv(csv_path)
    df = aggregate_to_neighborhoods(df_raw)

    labels, km, scaler_std, pca_2d, X_2d = fit_clusters(df)
    df['km_cluster'] = labels
    df['PC1'] = X_2d[:, 0]
    df['PC2'] = X_2d[:, 1]

    df_imputed = cluster_centroid_impute(df, labels)

    df_scored, scaler_mm = compute_rli(df_imputed)
    df_scored['km_cluster'] = df.loc[df_scored.index, 'km_cluster']
    df_scored['PC1'] = df.loc[df_scored.index, 'PC1']
    df_scored['PC2'] = df.loc[df_scored.index, 'PC2']

    return {
        'df_raw': df_raw,
        'df_neighborhoods': df,
        'df_imputed': df_imputed,
        'df_scored': df_scored,
        'km_model': km,
        'pca_2d': pca_2d,
        'scaler_std': scaler_std,
        'scaler_mm': scaler_mm,
        'X_2d': X_2d,
        'labels': labels,
    }
