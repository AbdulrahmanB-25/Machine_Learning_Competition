"""
Riyadh Livability Index — ML Engine
=====================================
Project : Neighborhood DNA — The 15-Minute City Index
Input   : Riyadh_Master_Dataset.csv — 348K property rows × 29 columns
Pipeline: Load → Aggregate → Density-Normalize → Log-Scale → Cluster-Impute
          → 4-Pillar RLI Score → Rank & Recommend

Outputs two systems:
  1. Ranking   — City-wide livability ranking of all 176 neighborhoods (no user input).
  2. Recommend — Filter-first property search: category + price + rooms
                 → qualifying neighborhoods ranked by RLI + price fit.

Formula:
    RLI_i = ( Σ w_j * x̂_ij ) × [ 1 + H_i / H_max ]
    where x̂ = MinMax-scaled log1p features,
          w = user-defined or default pillar weights,
          H = Shannon Entropy across 4 pillars,
          H_max = ln(4).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

EPSILON = 1e-9
H_MAX   = np.log(4)          # ln(4) — theoretical max Shannon entropy for 4 pillars
K_BEST  = 5                  # optimal cluster count from silhouette analysis

CATEGORY_MAP = {
    1:  'Apartment (Rent)',   2:  'Land',            3:  'Villa',
    4:  'Floor (Rent)',       5:  'Duplex (Rent)',    6:  'Apartment (Sale)',
    7:  'Commercial Land',    8:  'Office',           9:  'Building',
    10: 'Compound',           11: 'Farm',             13: 'Room',
    14: 'Shop',               15: 'Warehouse',        16: 'Commercial Building',
    17: 'Tower',              18: 'Camp',             19: 'Parking',
    20: 'Studio',             21: 'Chalet',           22: 'Duplex (Sale)',
    23: 'Rest House',         24: 'Palace',
}

# Categories where room filtering makes no sense
NO_ROOM_CATEGORIES = {
    'Land', 'Commercial Land', 'Farm', 'Rest House', 'Parking',
    'Warehouse', 'Camp', 'Shop',
}

# ── 4 RLI Pillars ──
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

# Features that are raw counts and need density normalization (÷ km²)
COUNT_FEATURES = [
    'bus_count', 'metro_count', 'dining_cafe', 'med_facilities',
    'edu_primary', 'religious', 'essential_retail', 'parks_green',
    'sports_play', 'malls_shopping',
]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rli(df_in: pd.DataFrame, pillar_weights: dict | None = None):
    """
    Compute Riyadh Livability Index for all neighborhoods in df_in.

    Parameters
    ----------
    df_in : DataFrame
        Neighborhood-level aggregated features (index = neighborhood name).
    pillar_weights : dict, optional
        Custom weights like {'Core': 0.5, 'Mobility': 0.2, ...}.
        Auto-normalized to sum to 1.0.

    Returns
    -------
    (df_result, scaler) : tuple
        df_result sorted by rank (best first), fitted MinMaxScaler.
    """
    pw = pillar_weights if pillar_weights else {p: v['weight'] for p, v in PILLARS.items()}
    # Normalize weights to sum to 1
    total_w = sum(pw.values())
    if total_w > 0:
        pw = {k: v / total_w for k, v in pw.items()}

    # Step 1: Log-scale then MinMax 0–1
    available = [f for f in ALL_RLI_FEATURES if f in df_in.columns]
    df_log = np.log1p(df_in[available])
    scaler = MinMaxScaler()
    df_sc = pd.DataFrame(
        scaler.fit_transform(df_log),
        columns=available,
        index=df_in.index,
    )

    # Step 2: Weighted pillar scores
    pillar_scores = {}
    for pname, pinfo in PILLARS.items():
        feats = [f for f in pinfo['features'] if f in df_sc.columns]
        w = pw.get(pname, 0)
        pillar_scores[pname] = df_sc[feats].mean(axis=1) * w

    pillar_df = pd.DataFrame(pillar_scores, index=df_in.index)
    weighted_sum = pillar_df.sum(axis=1)

    # Step 3: Shannon Entropy across 4 pillars
    pillar_props = pillar_df.div(pillar_df.sum(axis=1) + EPSILON, axis=0)
    H = -(pillar_props * np.log(pillar_props + EPSILON)).sum(axis=1)

    # Step 4: Diversity multiplier [1 + H / H_max]
    diversity_mult = 1 + (H / H_MAX)

    # Step 5: RLI = weighted_sum × diversity_multiplier
    raw_rli = weighted_sum * diversity_mult

    # Step 6: Normalize 0–100
    r_min, r_max = raw_rli.min(), raw_rli.max()
    rli_100 = ((raw_rli - r_min) / (r_max - r_min + EPSILON)) * 100

    # Assemble result
    result = df_in.copy()
    for pname in PILLARS:
        result[f'pillar_{pname}'] = pillar_scores[pname].values
    result['H_entropy']      = H.values
    result['diversity_mult'] = diversity_mult.values
    result['raw_rli']        = raw_rli.values
    result['RLI']            = rli_100.round(2).values
    result['rank']           = result['RLI'].rank(ascending=False, method='min').astype(int)

    return result.sort_values('rank'), scaler


def build_global_ranking(df_raw: pd.DataFrame, pillar_weights: dict | None = None):
    """
    Aggregate all 348K properties → 176 neighborhoods, density-normalize,
    compute RLI, cluster (K-Means), and attach PCA coordinates.

    Returns
    -------
    dict with keys:
        'df_raw'     — original DataFrame (with category_name added)
        'df_global'  — aggregated neighborhood matrix
        'df_ranked'  — ranked DataFrame (index = neighborhood)
        'pca_2d'     — fitted PCA object
        'kmeans'     — fitted KMeans object
    """
    # Add category_name if missing
    if 'category_name' not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw['category_name'] = df_raw['category'].map(CATEGORY_MAP)

    # ── 1. City-wide aggregation ──
    df_global = df_raw.groupby('neighborhood').agg(
        neighborhood_area_km2=('neighborhood_area_km2', 'first'),
        property_count=('property_id', 'count'),
        median_price=('price', 'median'),
        median_area=('area', 'median'),
        lat=('lat', 'mean'),
        lng=('lng', 'mean'),
        n_categories=('category', 'nunique'),
        **{f: (f, 'mean') for f in ALL_RLI_FEATURES},
    )

    # ── 2. Density normalization (raw counts ÷ km²) ──
    for feat in COUNT_FEATURES:
        if feat in df_global.columns:
            df_global[feat] = df_global[feat] / df_global['neighborhood_area_km2']
    df_global = df_global.replace([np.inf, -np.inf], 0).fillna(0)

    # ── 3. K-Means clustering ──
    cluster_feats = [f for f in ALL_RLI_FEATURES if f in df_global.columns]
    X_std = (df_global[cluster_feats] - df_global[cluster_feats].mean()) / (df_global[cluster_feats].std() + EPSILON)
    kmeans = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    df_global['km_cluster'] = kmeans.fit_predict(X_std)

    # ── 4. PCA for visualization ──
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_std)
    df_global['PC1'] = coords[:, 0]
    df_global['PC2'] = coords[:, 1]

    # ── 5. Compute RLI ──
    df_ranked, scaler = compute_rli(df_global, pillar_weights)

    return {
        'df_raw':    df_raw,
        'df_global': df_global,
        'df_ranked': df_ranked,
        'pca_2d':    pca,
        'kmeans':    kmeans,
        'scaler':    scaler,
    }


def recommend(df_ranked: pd.DataFrame, user_weights: dict | None = None, top_n: int = 20):
    """
    Re-score all neighborhoods with custom pillar weights.
    Returns re-ranked DataFrame with match_pct (% of top scorer).
    """
    df_re, _ = compute_rli(df_ranked, pillar_weights=user_weights)

    top_rli = df_re['RLI'].max()
    df_re['match_pct'] = (df_re['RLI'] / (top_rli + EPSILON) * 100).round(1)

    return df_re.head(top_n) if top_n else df_re


def property_search(
    df_raw: pd.DataFrame,
    df_ranked: pd.DataFrame,
    category: int,
    min_price: float = 0,
    max_price: float = float('inf'),
    min_rooms: int = 0,
    pillar_weights: dict | None = None,
    price_weight: float = 0.20,
    cluster_weight: float = 0.15,
):
    """
    Filter-first property search with K-Means cluster matching.

    1. Filter raw properties by category + price + rooms.
    2. Identify qualifying neighborhoods.
    3. Re-score using RLI + price compatibility + K-Means cluster fit.

    K-Means Integration
    -------------------
    After filtering, the system computes the average pillar profile of each
    K-Means cluster among qualifying neighborhoods, then scores each cluster
    against the user's pillar weights. Neighborhoods in the best-fit cluster
    receive a boost, making the recommendation cluster-aware.

    Returns
    -------
    dict with keys:
        'results'              — DataFrame of ranked neighborhoods
        'properties_matched'   — int, how many raw properties survived
        'category_label'       — str
        'best_cluster'         — int, the cluster ID that best matches user priorities
        'cluster_scores'       — dict, {cluster_id: alignment_score}
    """
    cat_label = CATEGORY_MAP.get(category, f'Category {category}')

    # Ensure category_name exists
    if 'category_name' not in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw['category_name'] = df_raw['category'].map(CATEGORY_MAP)

    # ── Step A: Strict filtering ──
    mask = df_raw['category'] == category
    mask &= df_raw['price'].between(min_price, max_price)

    if cat_label not in NO_ROOM_CATEGORIES and min_rooms > 0:
        mask &= df_raw['total_rooms'] >= min_rooms

    filtered = df_raw[mask]

    if len(filtered) == 0:
        return {
            'results': pd.DataFrame(),
            'properties_matched': 0,
            'category_label': cat_label,
            'best_cluster': -1,
            'cluster_scores': {},
        }

    # ── Step B: Neighborhood aggregation ──
    neigh_stats = filtered.groupby('neighborhood').agg(
        filtered_count=('property_id', 'count'),
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        avg_area=('area', 'mean'),
        avg_rooms=('total_rooms', 'mean'),
    ).round(1)

    qualifying = neigh_stats.index.tolist()
    df_score_input = df_ranked.loc[df_ranked.index.isin(qualifying)].copy()

    # ── Step C: Re-score with user weights ──
    df_scored, _ = compute_rli(df_score_input, pillar_weights=pillar_weights)

    # ── Step D: K-Means cluster matching ──
    # Score each cluster by how well its average pillar profile aligns with user weights
    pw = pillar_weights if pillar_weights else {p: v['weight'] for p, v in PILLARS.items()}
    total_w = sum(pw.values())
    if total_w > 0:
        pw = {k: v / total_w for k, v in pw.items()}

    cluster_scores = {}
    if 'km_cluster' in df_scored.columns:
        pillar_cols = [f'pillar_{p}' for p in PILLARS.keys()]
        available_pcols = [c for c in pillar_cols if c in df_scored.columns]

        for cid, grp in df_scored.groupby('km_cluster'):
            # Average pillar profile of this cluster
            cluster_profile = grp[available_pcols].mean()
            # Dot product with user weights = alignment score
            alignment = sum(
                cluster_profile.get(f'pillar_{p}', 0) * pw.get(p, 0)
                for p in PILLARS.keys()
            )
            cluster_scores[int(cid)] = round(alignment, 6)

        # Best cluster = highest alignment with user priorities
        best_cluster = max(cluster_scores, key=cluster_scores.get) if cluster_scores else -1

        # Normalize cluster scores to 0–100 for the boost
        cs_vals = list(cluster_scores.values())
        cs_min, cs_max = min(cs_vals), max(cs_vals)
        cs_range = cs_max - cs_min

        if cs_range > 0:
            df_scored['cluster_fit'] = df_scored['km_cluster'].map(
                lambda c: ((cluster_scores.get(c, 0) - cs_min) / cs_range) * 100
            )
        else:
            df_scored['cluster_fit'] = 100.0
    else:
        best_cluster = -1
        df_scored['cluster_fit'] = 0.0

    # ── Step E: Price compatibility score ──
    budget_mid = (min_price + max_price) / 2
    price_dist = np.abs(neigh_stats['avg_price'] - budget_mid)
    price_max_dist = price_dist.max()
    neigh_stats['price_score'] = (
        (1 - price_dist / price_max_dist) * 100 if price_max_dist > 0 else 100.0
    )

    # ── Step F: Merge & combined score (RLI + Price + Cluster Fit) ──
    overlap = neigh_stats.columns.intersection(df_scored.columns)
    df_result = df_scored.join(neigh_stats.drop(columns=overlap), how='inner')

    # Three-factor combined score
    rli_w = 1 - price_weight - cluster_weight
    rli_w = max(rli_w, 0)  # safety clamp
    df_result['combined_score'] = (
        df_result['RLI'] * rli_w
        + df_result['price_score'] * price_weight
        + df_result['cluster_fit'] * cluster_weight
    ).round(2)

    top_cs = df_result['combined_score'].max()
    df_result['match_pct'] = (
        (df_result['combined_score'] / (top_cs + EPSILON) * 100).round(1)
    )
    df_result['rank'] = df_result['combined_score'].rank(
        ascending=False, method='min'
    ).astype(int)
    df_result = df_result.sort_values('rank')

    return {
        'results': df_result,
        'properties_matched': len(filtered),
        'category_label': cat_label,
        'best_cluster': best_cluster,
        'cluster_scores': cluster_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: Full Pipeline (used by API on startup)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(csv_path: str):
    """
    Load CSV → build global ranking → return pipeline dict.
    Keys: df_raw, df_global, df_ranked (=df_imputed=df_scored), pca_2d, kmeans, scaler.
    """
    df_raw = pd.read_csv(csv_path)
    pipe = build_global_ranking(df_raw)

    # Alias for backward compatibility with api.py
    pipe['df_scored']  = pipe['df_ranked']
    pipe['df_imputed'] = pipe['df_ranked']

    return pipe
