# Riyadh Livability Index & Smart Recommendation Engine

An end-to-end machine learning system that scores **176 Riyadh neighborhoods** on livability by fusing four data layers — real estate, urban services, public transit, and internet connectivity — into a single **Riyadh Livability Index (RLI)**.

## Live Demo

**[riyadh-livability-index-and-smart-recommendation-engine.streamlit.app](https://riyadh-livability-index-and-smart-recommendation-engine.streamlit.app/)**

---

## Team

| GitHub | Name |
|---|---|
| [AbdulrahmanB-25](https://github.com/AbdulrahmanB-25) | Abdulrahman K B |
| [RaghadDasman](https://github.com/RaghadDasman) | Raghad Alsubaie |
| [BananAlnemri](https://github.com/BananAlnemri) | Banan Alnemri |
| [TheScientisiSaad](https://github.com/TheScientisiSaad) | Saad Alshahrani |

---

## What It Does

The system delivers five capabilities through a Streamlit dashboard:

1. **City Ranking** — All 176 neighborhoods ranked by livability with adjustable pillar weights.
2. **Smart Recommendation** — Filter by property type, budget, and rooms. K-Means cluster matching finds the best neighborhood archetype for your priorities and returns top results with match %, RLI score, and price fit.
3. **Clustering Analysis** — K-Means vs Hierarchical comparison (k=2,3,5) with Silhouette, Calinski-Harabasz, and Davies-Bouldin metrics.
4. **Price Prediction** — Linear, Ridge, Random Forest, and Gradient Boosting regression models compared on R², MAE, RMSE, and MAPE.
5. **Price Tier Classification** — Logistic Regression, Random Forest, and Gradient Boosting classifiers compared on Accuracy, Precision, Recall, F1, and Confusion Matrix.

---

## ML Model Results

| Problem | Algorithms Tested | Best Model | Key Metric |
|---|---|---|---|
| **Clustering** (Unsupervised) | K-Means (k=2,3,5), Hierarchical (k=2,3,5) | K-Means (k=2) | Silhouette = 0.1761 |
| **Price Prediction** (Regression) | Linear, Ridge, Random Forest, Gradient Boosting | Random Forest | R² = 0.919 |
| **Price Tier** (Classification) | Logistic Regression, Random Forest, Gradient Boosting | Gradient Boosting | F1 = 0.877 |

The best clustering model is used in the Recommender to match users to their ideal neighborhood archetype. The recommendation combines three factors:

**Combined Score = RLI × 65% + Price Score × 20% + Cluster Fit × 15%**

---

## The 15-Minute City Concept

Can residents reach everything they need — healthcare, education, groceries, parks, transit — within 15 minutes? This project measures that for every Riyadh neighborhood by building a composite livability score from real data.

**Scale:** 348K property listings across 176 neighborhoods, enriched with 27K Foursquare venues, 3,010 bus stops, 83 metro stations, and internet coverage for 189 neighborhoods.

---

## RLI Formula

$$RLI_i = \left( \sum_j w_j \cdot \hat{x}_{ij} \right) \times \left[ 1 + \frac{H_i}{H_{max}} \right]$$

| Symbol | Meaning |
|---|---|
| $\hat{x}$ | MinMax-scaled log1p features (0–1) |
| $w$ | User-defined or default pillar weights |
| $H$ | Shannon Entropy across 4 pillars |
| $H_{max}$ | ln(4) — theoretical max diversity |

### The 4 Pillars

| Pillar | Default Weight | Features |
|---|---|---|
| **Core** | 40% | Medical facilities, primary education, essential retail, religious |
| **Mobility** | 25% | Bus count, metro count, pedestrian infrastructure, connectivity score |
| **Well-being** | 20% | Dining/cafe, parks/green, sports/play, fitness |
| **Infrastructure** | 15% | Fiber availability, government/civil, malls/shopping, higher education |

The **entropy multiplier** rewards balanced neighborhoods — strong in all 4 pillars scores higher than exceptional in only one.

---

## User Journey

1. **Intent Selection** — Buy or Rent (choose from 23 property categories)
2. **Budget Constraint** — Set min/max price range in SAR
3. **Lifestyle Priorities** — Adjust pillar weights: Metro, Schools, Parks, Hospitals, Fiber
4. **Instant Matching** — K-Means finds the closest neighborhood archetype in the data
5. **Output** — Top 3 tailored recommendations with match %, livability score, and average price

---

## Data Pipeline

### Layer 1 — Real Estate (423K listings)
Cleaned from raw property data. Floor thresholds applied (price < 500 SAR and area < 10 m² flagged as spam). 23 property categories from apartments to palaces.

### Layer 2 — Services (27K Foursquare venues)
559 venue categories collapsed into 14 livability pillars. Spatially joined to neighborhoods via GeoJSON district boundaries.

### Layer 3 — Transit
Metro: 83 unique stations across 6 lines. Bus: 3,010 stops across 88 routes.

### Layer 4 — Internet Connectivity (189 neighborhoods)
Mobile = 100%. FWA = 93%. Fiber = 29% (the key differentiator). Each neighborhood gets a connectivity_score (0–3).

### Merge
All layers spatially joined via two-pass strategy (point-in-polygon + nearest-neighbor fallback) → `Riyadh_Master_Dataset.csv` (348K × 29).

---

## Tech Stack

- **Data Processing:** pandas, NumPy, GeoPandas
- **ML:** scikit-learn (K-Means, PCA, Random Forest, Gradient Boosting, Logistic Regression, Silhouette/CH/DB metrics)
- **Scoring:** Custom weighted composite index with Shannon entropy diversity multiplier
- **Dashboard:** Streamlit + Plotly
- **Visualization:** Matplotlib, Seaborn (notebooks), Plotly (dashboard)

---

## Notebooks

All notebooks follow a consistent template: dark theme, shared color palette, cleaning log with chart, and markdown narration.

| Notebook | Purpose | Key Output |
|---|---|---|
| `EDA_Riyadh_Real_Estate` | Layer 1: 423K property listings | Cleaned RE CSV |
| `EDA_Services` | Layer 2: 27K Foursquare venues → 14 pillars | Cleaned Services CSV |
| `EDA_Riyadh_Metro` | Layer 3a: 83 metro stations | Cleaned Metro CSV |
| `EDA_Riyadh_Bus` | Layer 3b: 3,010 bus stops | Cleaned Bus CSV |
| `EDA_Riyadh_Internet` | Layer 4: 189 neighborhoods connectivity | Cleaned Internet CSV |
| `Merge_Master_Dataset` | Spatial join → master dataset | Riyadh_Master_Dataset.csv (348K × 29) |
| `ML_Clustering_Master` | PCA + K-Means + Hierarchical | Clustered neighborhood profiles |
| `RLI_Engine` | Scoring + search + ML comparison | Same code as rli_engine.py |

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