"""
Riyadh Livability Index — REST API  (FastAPI)
===============================================
Mirrors RLI_Engine.ipynb logic exactly via rli_engine.py.

Endpoints
---------
GET  /                → API info & available endpoints.
GET  /categories      → Property category codes, labels, and counts.
GET  /neighborhoods   → All 176 neighborhood names.
GET  /ranking         → City-wide ranking with default scientific weights. ?top=N for top N.
POST /recommend       → User-defined pillar weights → re-scored ranking with match %.
POST /search          → Filter-first property search: category + price + rooms
                        → qualifying neighborhoods ranked by combined RLI + price score.
GET  /pca             → PCA 2D coordinates + cluster labels for scatter visualization.

Run
---
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os

from rli_engine import (
    run_full_pipeline, recommend, compute_rli, property_search,
    PILLARS, ALL_RLI_FEATURES, CATEGORY_MAP, NO_ROOM_CATEGORIES,
)

# ═══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Riyadh Livability Index API",
    description="Neighborhood DNA — Ranking & Recommendation engine for Riyadh properties.",
    version="2.0.0",
)

# ── Locate dataset ──
_CANDIDATES = [
    os.environ.get('RIYADH_CSV', ''),
    os.path.join(os.path.dirname(__file__), 'Riyadh_Master_Dataset.csv'),
    '/mnt/user-data/uploads/Riyadh_Master_Dataset.csv',
]
CSV_PATH = next((p for p in _CANDIDATES if p and os.path.exists(p)), None)

PIPELINE = None


def get_pipeline():
    """Lazy-load the full pipeline on first request."""
    global PIPELINE
    if PIPELINE is None:
        if CSV_PATH is None:
            raise FileNotFoundError(
                'Riyadh_Master_Dataset.csv not found. '
                'Place it next to api.py or set RIYADH_CSV env var.'
            )
        PIPELINE = run_full_pipeline(CSV_PATH)
    return PIPELINE


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    core:           float = Field(0.40, ge=0, le=1, description="Core pillar weight (health, education, retail, religious)")
    mobility:       float = Field(0.25, ge=0, le=1, description="Mobility pillar weight (transit, pedestrian, connectivity)")
    wellbeing:      float = Field(0.20, ge=0, le=1, description="Well-being pillar weight (dining, parks, sports, fitness)")
    infrastructure: float = Field(0.15, ge=0, le=1, description="Infrastructure pillar weight (fiber, gov, malls, higher-edu)")
    top_n:          int   = Field(20, ge=1, le=176, description="Number of results to return")


class SearchRequest(BaseModel):
    category:     int   = Field(..., description="Category ID (e.g. 1=Apartment Rent, 3=Villa)")
    min_price:    float = Field(0, ge=0, description="Minimum budget in SAR")
    max_price:    float = Field(10_000_000, ge=0, description="Maximum budget in SAR")
    min_rooms:    int   = Field(0, ge=0, description="Minimum room count (ignored for land/farm/etc.)")
    price_weight: float = Field(0.20, ge=0, le=1, description="How much price fit matters vs. livability (0–1)")
    core:           float = Field(0.40, ge=0, le=1)
    mobility:       float = Field(0.25, ge=0, le=1)
    wellbeing:      float = Field(0.20, ge=0, le=1)
    infrastructure: float = Field(0.15, ge=0, le=1)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def scored_to_records(df: pd.DataFrame, top_n: int | None = None) -> list[dict]:
    """Convert scored DataFrame to JSON-safe list of dicts."""
    cols = [
        'neighborhood', 'RLI', 'rank', 'km_cluster',
        'pillar_Core', 'pillar_Mobility', 'pillar_Well-being', 'pillar_Infrastructure',
        'H_entropy', 'diversity_mult',
    ]
    optional = ['match_pct', 'PC1', 'PC2', 'combined_score', 'price_score',
                'avg_price', 'median_price', 'filtered_count', 'avg_area', 'avg_rooms']
    cols += [c for c in optional if c in df.columns]

    out = df.reset_index()
    present = [c for c in cols if c in out.columns]
    out = out[present].copy()

    for c in out.select_dtypes(include=[np.floating]).columns:
        out[c] = out[c].round(4)

    records = out.to_dict(orient='records')
    return records[:top_n] if top_n else records


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    return {
        'name': 'Riyadh Livability Index API',
        'version': '2.0.0',
        'endpoints': {
            'GET  /categories':    'Property category codes, labels, and counts.',
            'GET  /neighborhoods':  'All 176 neighborhood names.',
            'GET  /ranking':       'City-wide ranking (default weights). ?top=N.',
            'POST /recommend':     'Custom-weighted recommendations.',
            'POST /search':        'Filter-first property search.',
            'GET  /pca':           'PCA 2D coordinates + cluster labels.',
        },
    }


@app.get("/categories")
def categories():
    """Available property categories with counts from the dataset."""
    pipe = get_pipeline()
    counts = pipe['df_raw']['category'].value_counts().to_dict()
    return {
        'categories': {
            str(k): {'label': v, 'count': counts.get(k, 0)}
            for k, v in sorted(CATEGORY_MAP.items())
            if counts.get(k, 0) > 0
        }
    }


@app.get("/neighborhoods")
def neighborhoods():
    """List all scored neighborhoods."""
    pipe = get_pipeline()
    names = sorted(pipe['df_ranked'].index.tolist())
    return {'count': len(names), 'neighborhoods': names}


@app.get("/ranking")
def ranking(top: int | None = None):
    """
    City-wide RLI ranking using default scientific weights.
    RLI normalized 0–100 (best = 100, worst = 0).
    """
    pipe = get_pipeline()
    df = pipe['df_ranked']
    return {
        'description': 'Riyadh Livability Index — default scientific weights',
        'normalization': '0-100 min-max on raw RLI (best=100, worst=0)',
        'pillars': {k: v['weight'] for k, v in PILLARS.items()},
        'total_neighborhoods': len(df),
        'rankings': scored_to_records(df, top),
    }


@app.post("/recommend")
def recommend_endpoint(body: RecommendRequest):
    """
    Re-score all neighborhoods with user-defined pillar weights.
    Returns sorted list with match_pct (% of top scorer).
    """
    pipe = get_pipeline()

    user_weights = {
        'Core':           body.core,
        'Mobility':       body.mobility,
        'Well-being':     body.wellbeing,
        'Infrastructure': body.infrastructure,
    }

    df_rec = recommend(pipe['df_ranked'], user_weights=user_weights, top_n=body.top_n)

    # Attach PCA coords + cluster from the base pipeline
    base_cols = ['PC1', 'PC2', 'km_cluster']
    base_present = [c for c in base_cols if c in pipe['df_ranked'].columns]
    if base_present:
        base = pipe['df_ranked'][base_present]
        df_rec = df_rec.join(base, rsuffix='_drop')
        df_rec.drop(columns=[c for c in df_rec.columns if c.endswith('_drop')],
                     inplace=True, errors='ignore')

    return {
        'description': 'Custom-weighted RLI recommendations',
        'user_weights': user_weights,
        'total': len(df_rec),
        'recommendations': scored_to_records(df_rec),
    }


@app.post("/search")
def search_endpoint(body: SearchRequest):
    """
    Filter-first property search.
    Filters raw properties by category, price range, and room count,
    then scores qualifying neighborhoods using RLI + price compatibility.
    """
    pipe = get_pipeline()

    user_weights = {
        'Core':           body.core,
        'Mobility':       body.mobility,
        'Well-being':     body.wellbeing,
        'Infrastructure': body.infrastructure,
    }

    result = property_search(
        df_raw=pipe['df_raw'],
        df_ranked=pipe['df_ranked'],
        category=body.category,
        min_price=body.min_price,
        max_price=body.max_price,
        min_rooms=body.min_rooms,
        pillar_weights=user_weights,
        price_weight=body.price_weight,
    )

    df_res = result['results']
    cat_label = result['category_label']

    if df_res.empty:
        return {
            'category': body.category,
            'category_label': cat_label,
            'price_range': f'{body.min_price:,.0f} – {body.max_price:,.0f} SAR',
            'min_rooms': body.min_rooms,
            'properties_found': 0,
            'neighborhoods_found': 0,
            'message': 'No properties match your criteria. Try widening your filters.',
            'results': [],
        }

    return {
        'category': body.category,
        'category_label': cat_label,
        'price_range': f'{body.min_price:,.0f} – {body.max_price:,.0f} SAR',
        'min_rooms': body.min_rooms,
        'total_properties': len(pipe['df_raw']),
        'properties_after_filter': result['properties_matched'],
        'neighborhoods_found': len(df_res),
        'neighborhoods_eliminated': len(pipe['df_ranked']) - len(df_res),
        'results': scored_to_records(df_res),
    }


@app.get("/pca")
def pca_data():
    """PCA 2D coordinates for scatter plot visualization."""
    pipe = get_pipeline()
    df = pipe['df_ranked'].reset_index()
    cols = ['neighborhood', 'PC1', 'PC2', 'km_cluster', 'RLI']
    present = [c for c in cols if c in df.columns]
    records = df[present].round(4).to_dict(orient='records')
    return {
        'explained_variance': pipe['pca_2d'].explained_variance_ratio_.round(4).tolist(),
        'data': records,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_event():
    """Pre-load the pipeline so the first request is fast."""
    try:
        get_pipeline()
        print("Pipeline loaded successfully.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
