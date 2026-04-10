"""
Riyadh Livability Index — REST API  (Flask)
============================================
Mirrors RLI_Engine.ipynb + RLI_Property_Search.ipynb logic exactly.

Endpoints
---------
GET  /score      → City-wide ranking with default scientific weights.
POST /recommend  → User-defined pillar weights, re-runs compute_rli, returns sorted list with match %.
POST /search     → Filter-first property search: category + price + rooms → qualifying neighborhoods ranked by RLI + price fit.
GET  /pca        → PCA 2D coordinates + cluster labels for scatter visualization.
GET  /neighborhoods → All 176 neighborhood names.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from rli_engine import run_full_pipeline, recommend, compute_rli, PILLARS, ALL_RLI_FEATURES

CATEGORY_MAP = {
    1: 'Apartment (Rent)', 2: 'Land', 3: 'Villa', 4: 'Floor (Rent)',
    5: 'Duplex (Rent)', 6: 'Apartment (Sale)', 7: 'Commercial Land',
    8: 'Office', 9: 'Building', 10: 'Compound', 11: 'Farm',
    13: 'Room', 14: 'Shop', 15: 'Warehouse', 16: 'Commercial Building',
    17: 'Tower', 18: 'Camp', 19: 'Parking', 20: 'Studio',
    21: 'Chalet', 22: 'Duplex (Sale)', 23: 'Rest House', 24: 'Palace',
}

app = Flask(__name__)

# ── Locate dataset ────────────────────────────────────────────────────────────
_CANDIDATES = [
    os.environ.get('RIYADH_CSV', ''),
    '/mnt/user-data/uploads/Riyadh_Master_Dataset.csv',
    os.path.join(os.path.dirname(__file__), 'Riyadh_Master_Dataset.csv'),
]
CSV_PATH = next((p for p in _CANDIDATES if p and os.path.exists(p)), None)

PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        if CSV_PATH is None:
            raise FileNotFoundError('Riyadh_Master_Dataset.csv not found. Set RIYADH_CSV env var.')
        PIPELINE = run_full_pipeline(CSV_PATH)
    return PIPELINE


# ── Helpers ───────────────────────────────────────────────────────────────────

def scored_to_records(df_scored: pd.DataFrame, top_n: int | None = None) -> list:
    """Convert scored DataFrame to JSON-safe list of dicts."""
    cols = [
        'neighborhood', 'RLI', 'rank', 'km_cluster',
        'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure',
        'H_entropy', 'diversity_mult',
    ]
    if 'match_pct' in df_scored.columns:
        cols.append('match_pct')
    if 'PC1' in df_scored.columns:
        cols.extend(['PC1', 'PC2'])

    out = df_scored.reset_index()
    present = [c for c in cols if c in out.columns]
    out = out[present].copy()

    for c in out.select_dtypes(include=[np.floating]).columns:
        out[c] = out[c].round(4)

    records = out.to_dict(orient='records')
    return records[:top_n] if top_n else records


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'Riyadh Livability Index API',
        'endpoints': {
            'GET /score': 'City-wide ranking with default weights. ?top=N for top N.',
            'POST /recommend': 'Custom-weighted recommendations. Body: {core, mobility, wellbeing, infrastructure}.',
            'POST /search': 'Filter-first property search. Body: {category, min_price, max_price, min_rooms, weights, price_weight}.',
            'GET /pca': 'PCA 2D coordinates + cluster labels.',
            'GET /neighborhoods': 'List all 176 neighborhoods.',
            'GET /categories': 'Property category codes and labels.',
        },
    })


@app.route('/categories', methods=['GET'])
def categories():
    """Available property categories with counts."""
    pipe = get_pipeline()
    counts = pipe['df_raw']['category'].value_counts().to_dict()
    return jsonify({
        'categories': {
            str(k): {'label': v, 'count': counts.get(k, 0)}
            for k, v in sorted(CATEGORY_MAP.items())
            if counts.get(k, 0) > 0
        }
    })


@app.route('/score', methods=['GET'])
def score():
    """City-wide ranking using default scientific weights.

    RLI is normalized 0–100 (min-max on raw_rli). The neighborhood with the
    highest raw score gets RLI = 100.00, the lowest gets RLI = 0.00.
    """
    pipe = get_pipeline()
    df_s = pipe['df_scored']
    top_n = request.args.get('top', None, type=int)
    return jsonify({
        'description': 'Riyadh Livability Index — default scientific weights',
        'normalization': '0-100 min-max on raw RLI (best = 100, worst = 0)',
        'pillars': {k: v['weight'] for k, v in PILLARS.items()},
        'total_neighborhoods': len(df_s),
        'rankings': scored_to_records(df_s, top_n),
    })


@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    """User-weighted recommendations.

    Accepts pillar weights from sliders, re-runs compute_rli with those weights,
    and returns all neighborhoods sorted best-to-worst with a match_pct column
    (percentage of the top scorer's RLI).
    """
    pipe = get_pipeline()
    data = request.get_json(force=True)

    user_weights = {
        'Core':           data.get('core', 0.40),
        'Mobility':       data.get('mobility', 0.25),
        'Well-being':     data.get('wellbeing', 0.20),
        'Infrastructure': data.get('infrastructure', 0.15),
    }

    df_rec = recommend(pipe['df_imputed'], user_weights=user_weights, top_n=176)

    # Attach PCA coords + cluster from the base pipeline
    base = pipe['df_scored'][['PC1', 'PC2', 'km_cluster']]
    df_rec = df_rec.join(base, rsuffix='_drop')
    df_rec.drop(columns=[c for c in df_rec.columns if c.endswith('_drop')], inplace=True, errors='ignore')

    return jsonify({
        'description': 'Custom-weighted RLI recommendations (sorted best → worst)',
        'user_weights': user_weights,
        'total': len(df_rec),
        'recommendations': scored_to_records(df_rec),
    })


@app.route('/search', methods=['POST'])
def search_endpoint():
    """Filter-first property search.

    Filters raw properties by category, price range, and room count,
    then scores qualifying neighborhoods using RLI + price compatibility.
    """
    pipe = get_pipeline()
    data = request.get_json(force=True)

    category = data.get('category', 3)
    min_price = data.get('min_price', 0)
    max_price = data.get('max_price', float('inf'))
    min_rooms = data.get('min_rooms', 0)
    price_weight = data.get('price_weight', 0.20)
    user_weights = data.get('weights', None)

    if user_weights:
        user_weights = {
            'Core': user_weights.get('core', user_weights.get('Core', 0.4)),
            'Mobility': user_weights.get('mobility', user_weights.get('Mobility', 0.25)),
            'Well-being': user_weights.get('wellbeing', user_weights.get('Well-being', 0.2)),
            'Infrastructure': user_weights.get('infrastructure', user_weights.get('Infrastructure', 0.15)),
        }

    df_raw = pipe['df_raw']
    df_imputed = pipe['df_imputed']

    # Step A: Strict Filtering
    mask = df_raw['category'] == category
    mask &= df_raw['price'].between(min_price, max_price)
    if min_rooms > 0:
        mask &= df_raw['total_rooms'] >= min_rooms

    filtered = df_raw[mask]

    if len(filtered) == 0:
        return jsonify({
            'category': category,
            'category_label': CATEGORY_MAP.get(category, f'Category {category}'),
            'price_range': f'{min_price:,.0f} – {max_price:,.0f} SAR',
            'min_rooms': min_rooms,
            'properties_found': 0,
            'neighborhoods_found': 0,
            'message': 'No properties match your criteria. Try widening your filters.',
            'results': [],
        })

    # Step B: Neighborhood Aggregation
    neigh_stats = filtered.groupby('neighborhood').agg(
        filtered_count=('property_id', 'count'),
        avg_price=('price', 'mean'),
        median_price=('price', 'median'),
        avg_area=('area', 'mean'),
        avg_rooms=('total_rooms', 'mean'),
    ).round(1)

    qualifying = neigh_stats.index.tolist()
    df_score_input = df_imputed.loc[df_imputed.index.isin(qualifying)].copy()

    # Step C: Scoring
    df_scored, _ = compute_rli(df_score_input, pillar_weights=user_weights)

    budget_mid = (min_price + max_price) / 2
    price_dist = np.abs(neigh_stats['avg_price'] - budget_mid)
    price_max_dist = price_dist.max()
    neigh_stats['price_score'] = (1 - price_dist / price_max_dist) * 100 if price_max_dist > 0 else 100.0

    overlap_cols = neigh_stats.columns.intersection(df_scored.columns)
    df_result = df_scored.join(neigh_stats.drop(columns=overlap_cols), how='inner')

    rli_w = 1 - price_weight
    df_result['combined_score'] = (df_result['RLI'] * rli_w + df_result['price_score'] * price_weight).round(2)
    top_cs = df_result['combined_score'].max()
    df_result['match_pct'] = (df_result['combined_score'] / top_cs * 100).round(1) if top_cs > 0 else 0.0
    df_result['rank'] = df_result['combined_score'].rank(ascending=False, method='min').astype(int)
    df_result = df_result.sort_values('rank')

    # Build response records
    out_cols = [
        'rank', 'match_pct', 'combined_score', 'RLI', 'price_score',
        'avg_price', 'median_price', 'filtered_count', 'avg_area', 'avg_rooms',
        'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure',
        'H_entropy', 'km_cluster',
    ]
    out = df_result.reset_index()
    present = [c for c in ['neighborhood'] + out_cols if c in out.columns]
    out = out[present]
    for c in out.select_dtypes(include=[np.floating]).columns:
        out[c] = out[c].round(4)

    return jsonify({
        'category': category,
        'category_label': CATEGORY_MAP.get(category, f'Category {category}'),
        'price_range': f'{min_price:,.0f} – {max_price:,.0f} SAR',
        'min_rooms': min_rooms,
        'total_properties': len(df_raw),
        'properties_after_filter': len(filtered),
        'neighborhoods_found': len(df_result),
        'neighborhoods_eliminated': 176 - len(df_result),
        'results': out.to_dict(orient='records'),
    })


@app.route('/neighborhoods', methods=['GET'])
def neighborhoods():
    pipe = get_pipeline()
    names = pipe['df_scored'].index.tolist()
    return jsonify({'count': len(names), 'neighborhoods': names})


@app.route('/pca', methods=['GET'])
def pca_data():
    """PCA 2D coordinates for scatter plot visualization."""
    pipe = get_pipeline()
    df = pipe['df_scored'].reset_index()
    records = df[['neighborhood', 'PC1', 'PC2', 'km_cluster', 'RLI']].round(4).to_dict(orient='records')
    return jsonify({
        'explained_variance': pipe['pca_2d'].explained_variance_ratio_.round(4).tolist(),
        'data': records,
    })


if __name__ == '__main__':
    print('Pre-loading pipeline...')
    get_pipeline()
    print('API ready → http://0.0.0.0:8000')
    app.run(host='0.0.0.0', port=8000, debug=False)
