# Riyadh Livability Index

An end-to-end machine learning system that scores **176 Riyadh neighborhoods** on livability by fusing four data layers — real estate, urban services, public transit, and internet connectivity — into a single **Riyadh Livability Index (RLI)**.

The system delivers two outputs:

1. **Ranking** — A city-wide livability ranking of all neighborhoods with no user input required.
2. **Recommendation** — A filter-first property search where users specify category, budget, and room count to get personalized neighborhood recommendations ranked by livability + price fit.

Both are served through a **FastAPI** backend and a **Streamlit** dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [RLI Formula](#rli-formula)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
- [API Reference](#api-reference)
- [Notebooks](#notebooks)
- [Tech Stack](#tech-stack)

---

## Project Overview

The "15-Minute City" concept asks: can residents reach essential services — healthcare, education, groceries, parks, transit — within 15 minutes? This project operationalizes that question for Riyadh by building a composite livability score from real data.

**Scale:** 348K property listings across 176 neighborhoods, enriched with 27K Foursquare venues, 3,010 bus stops, 83 metro stations, and internet coverage for 189 neighborhoods.

**Approach:** No labeled "livability" target exists, so the system uses unsupervised methods — PCA for dimensionality reduction, K-Means clustering for neighborhood archetypes, and a weighted composite index with Shannon entropy diversity boosting.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYERS                              │
│                                                                 │
│  Real Estate    Services     Metro    Bus     Internet           │
│  (423K rows)    (27K venues) (83 st.) (3,010) (189 nbhds)       │
│      │              │           │       │         │              │
│      ▼              ▼           ▼       ▼         ▼              │
│  ┌──────────────────────────────────────────────────┐           │
│  │         5 EDA Notebooks (Clean & Profile)        │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │   Merge_Master_Dataset.ipynb (Spatial Join)      │           │
│  │   → Riyadh_Master_Dataset.csv (348K × 29)       │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │   ML_Clustering_Master.ipynb                     │           │
│  │   PCA + K-Means (k=5) + Hierarchical Clustering  │           │
│  └──────────────────────┬───────────────────────────┘           │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │   RLI_Engine.ipynb → rli_engine.py               │           │
│  │   4-Pillar Scoring + Entropy Boost + Search      │           │
│  └──────────┬───────────────────────┬───────────────┘           │
│             ▼                       ▼                           │
│     ┌──────────────┐      ┌─────────────────┐                  │
│     │  api.py       │      │ streamlit_app.py │                  │
│     │  (FastAPI)    │      │ (Dashboard)      │                  │
│     └──────────────┘      └─────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Pipeline

### Layer 1 — Real Estate (423K listings)

Cleaned from raw property data. Drops internal IDs, timestamps, and metadata columns. Adds floor thresholds (price < 500 SAR and area < 10 m² flagged as test/spam). 23 property categories from apartments to palaces.

### Layer 2 — Services (27K Foursquare venues)

559 venue categories collapsed into **14 livability pillars**: dining/cafe, medical facilities, health retail, fitness, primary education, higher education, religious, essential retail, parks/green spaces, sports/play, pedestrian infrastructure, resort/rural retreats, government/civil, and malls/shopping.

### Layer 3 — Transit

- **Metro**: 94 entries deduplicated to 83 unique physical stations across 6 lines.
- **Bus**: 3,010 stops across 88 routes. Shelter types range from basic poles (62%) to air-conditioned shelters (23%).

### Layer 4 — Internet Connectivity (189 neighborhoods)

Mobile coverage is universal (100%). FWA covers 93%. Fiber is the real differentiator at only 29%. Each neighborhood gets a `connectivity_score` (0–3).

### Merge

All layers are spatially joined via GeoJSON district boundaries using a two-pass strategy (point-in-polygon + nearest-neighbor fallback). The result is `Riyadh_Master_Dataset.csv` — 348K property rows × 29 columns, each row enriched with its neighborhood's service counts, transit proximity, and connectivity profile.

### Clustering

PCA across 24 features (14 service + 4 connectivity + 2 transit + 4 property stats) followed by K-Means (k=5, chosen via silhouette analysis) and hierarchical clustering (Ward's linkage) for validation. The 5 clusters represent neighborhood archetypes from service-rich urban cores to peripheral service deserts.

---

## RLI Formula

$$RLI_i = \left( \sum_j w_j \cdot \hat{x}_{ij} \right) \times \left[ 1 + \frac{H_i}{H_{max}} \right]$$

| Symbol | Meaning |
|---|---|
| $\hat{x}$ | MinMax-scaled `log1p` features (0–1) |
| $w$ | User-defined or default pillar weights |
| $H$ | Shannon Entropy $-\sum p \ln p$ across 4 pillars |
| $H_{max}$ | $\ln(4)$ — theoretical max diversity for 4 pillars |

### The 4 Pillars

| Pillar | Default Weight | Features |
|---|---|---|
| **Core** | 40% | Medical facilities, primary education, essential retail, religious |
| **Mobility** | 25% | Bus count, metro count, pedestrian infrastructure, connectivity score |
| **Well-being** | 20% | Dining/cafe, parks/green, sports/play, fitness |
| **Infrastructure** | 15% | Fiber availability, government/civil, malls/shopping, higher education |

The **entropy multiplier** rewards balanced neighborhoods — a neighborhood strong in all 4 pillars scores higher than one that's exceptional in only one.

### Recommendation (Property Search)

The filter-first pipeline:

1. User specifies category, price range, minimum rooms.
2. System discards every property that doesn't match.
3. Surviving neighborhoods are scored using RLI + price compatibility.
4. **Combined Score** = RLI × (1 − price_weight) + Price Score × price_weight.

Categories like Land, Farm, and Commercial Land automatically skip room filtering.

---

## Repository Structure

```
Machine_Learning_Competition/
│
├── EDA_Services.ipynb              # Layer 2: 27K Foursquare venues → 14 pillars
├── EDA_Riyadh_Real_Estate.ipynb    # Layer 1: 423K property listings
├── EDA_Riyadh_Metro.ipynb          # Layer 3a: 83 metro stations
├── EDA_Riyadh_Bus.ipynb            # Layer 3b: 3,010 bus stops
├── EDA_Riyadh_Internet.ipynb       # Layer 4: 189 neighborhoods connectivity
│
├── Merge_Master_Dataset.ipynb      # Spatial join → Riyadh_Master_Dataset.csv
├── ML_Clustering_Master.ipynb      # PCA + K-Means + Hierarchical clustering
├── RLI_Engine.ipynb                # Scoring + search prototype (notebook version)
│
├── rli_engine.py                   # ML engine module (single source of truth)
├── api.py                          # FastAPI REST API
├── streamlit_app.py                # Streamlit dashboard
├── requirements.txt                # Python dependencies
│
├── Cleaned_DataSets/               # Outputs from EDA notebooks
│   ├── Cleaned_Riyadh_Real_Estate.csv
│   ├── Cleaned_Riyadh_Services.csv
│   ├── Cleaned_Riyadh_Bus.csv
│   ├── Cleaned_Riyadh_Metro.csv
│   └── Cleaned_Riyadh_Internet.csv
│
├── Riyadh_Master_Dataset.csv       # Final merged dataset (348K × 29)
└── README.md
```

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- `Riyadh_Master_Dataset.csv` in the project root (or set `RIYADH_CSV` env var)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will pre-load the pipeline on startup. Visit `http://localhost:8000/docs` for interactive Swagger documentation.

### Run the Dashboard

```bash
streamlit run streamlit_app.py
```

Opens a browser with two tabs: City Ranking and Property Search.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info and available endpoints |
| `/categories` | GET | Property categories with listing counts |
| `/neighborhoods` | GET | All 176 neighborhood names |
| `/ranking` | GET | City-wide RLI ranking (default weights). `?top=N` for top N |
| `/recommend` | POST | Custom pillar weights → re-scored ranking with match % |
| `/search` | POST | Filter-first property search |
| `/pca` | GET | PCA 2D coordinates + cluster labels |

### POST /search — Example

```json
{
    "category": 3,
    "min_price": 500000,
    "max_price": 2000000,
    "min_rooms": 6,
    "price_weight": 0.20,
    "core": 0.40,
    "mobility": 0.25,
    "wellbeing": 0.20,
    "infrastructure": 0.15
}
```

**Response** includes matching property count, qualifying neighborhoods, and each neighborhood's rank, match %, RLI score, average price, and pillar breakdown.

### POST /recommend — Example

```json
{
    "core": 0.15,
    "mobility": 0.50,
    "wellbeing": 0.25,
    "infrastructure": 0.10,
    "top_n": 10
}
```

---

## Notebooks

All notebooks follow a consistent template: dark theme, shared color palette (`GOLD`, `CYAN`, `CORAL`, `MINT`, `PURPLE`), cleaning log with summary chart, and markdown narration per section.

### Standard EDA Pipeline

1. **Load & Initial Profile** — shape, dtypes, nulls, describe
2. **Feature Selection** — keep needed columns, document rationale for every dropped column
3. **Data Processing** — types, nulls, duplicates, coordinate bounds
4. **Data Analysis** — distributions, spatial patterns, correlations
5. **Fix What Analysis Reveals** — outliers, mappings, spatial joins
6. **Export** as `Cleaned_*.csv`

| Notebook | Input | Output | Key Stat |
|---|---|---|---|
| `EDA_Riyadh_Real_Estate` | 423K listings | Cleaned RE CSV | 23 property categories |
| `EDA_Services` | 27K venues | Cleaned Services CSV | 559 categories → 14 pillars |
| `EDA_Riyadh_Metro` | 94 entries | Cleaned Metro CSV | 83 unique stations, 6 lines |
| `EDA_Riyadh_Bus` | 3,010 stops | Cleaned Bus CSV | 88 routes, 62% basic shelters |
| `EDA_Riyadh_Internet` | 189 neighborhoods | Cleaned Internet CSV | Fiber at 29% is the gap |
| `Merge_Master_Dataset` | 5 cleaned CSVs | Master CSV (348K × 29) | Spatial join via GeoJSON |
| `ML_Clustering_Master` | Master CSV | Clustered CSV | k=5 optimal, 5 archetypes |
| `RLI_Engine` | Master CSV | Ranked + Search results | RLI 0–100, entropy-boosted |

---

## Tech Stack

- **Data Processing**: pandas, NumPy, GeoPandas
- **ML**: scikit-learn (PCA, K-Means, MinMaxScaler, StandardScaler, silhouette analysis)
- **Scoring**: Custom weighted composite index with Shannon entropy diversity multiplier
- **API**: FastAPI + Pydantic + Uvicorn
- **Dashboard**: Streamlit + Plotly
- **Visualization**: Matplotlib, Seaborn (notebooks), Plotly (dashboard)

---

## Property Categories

| ID | Category | ID | Category |
|---|---|---|---|
| 1 | Apartment (Rent) | 13 | Room |
| 2 | Land | 14 | Shop |
| 3 | Villa | 15 | Warehouse |
| 4 | Floor (Rent) | 16 | Commercial Building |
| 5 | Duplex (Rent) | 17 | Tower |
| 6 | Apartment (Sale) | 18 | Camp |
| 7 | Commercial Land | 19 | Parking |
| 8 | Office | 20 | Studio |
| 9 | Building | 21 | Chalet |
| 10 | Compound | 22 | Duplex (Sale) |
| 11 | Farm | 23 | Rest House |
| — | — | 24 | Palace |
