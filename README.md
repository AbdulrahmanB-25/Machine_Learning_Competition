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

## ML Model Results

### Unsupervised — Neighborhood Clustering

K-Means and Hierarchical clustering tested at k=2, 3, and 5 on 176 neighborhoods using 16 standardized livability features.

| Model | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ |
|---|---|---|---|
| **K-Means (k=2)** | **0.1761** | **24.3** | 2.1091 |
| K-Means (k=3) | 0.1753 | 23.5 | 2.1103 |
| K-Means (k=5) | 0.1664 | 21.2 | **1.5823** |
| Hierarchical (k=2) | 0.1317 | 17.3 | 2.2956 |
| Hierarchical (k=3) | 0.1460 | 17.9 | 1.7413 |
| Hierarchical (k=5) | 0.1331 | 18.7 | 1.8536 |

**Winner: K-Means (k=2)** — highest Silhouette score. The best cluster is used in the Recommender to match users to their ideal neighborhood archetype.

**Metric definitions:**
- **Silhouette Score** (−1 to 1): How well-separated clusters are. Higher = more distinct groups.
- **Calinski-Harabasz**: Ratio of between-cluster to within-cluster variance. Higher = tighter clusters.
- **Davies-Bouldin**: Average similarity between clusters. Lower = clusters are more different from each other.

---

### Supervised — Property Price Prediction (Regression)

Four algorithms trained on 30K sampled properties to predict price from 20 features (area, category, rooms, neighborhood services, transit, connectivity). Outliers trimmed at 1st/99th percentile.

| Model | R² ↑ | MAE (SAR) ↓ | RMSE (SAR) ↓ | MAPE (%) ↓ |
|---|---|---|---|---|
| Linear Regression | 0.4905 | 720,134 | 975,932 | 884.3 |
| Ridge Regression | 0.4905 | 720,134 | 975,931 | 884.3 |
| **Random Forest** | **0.9190** | **210,313** | **389,045** | **56.7** |
| Gradient Boosting | 0.9101 | 243,901 | 410,010 | 106.3 |

**Winner: Random Forest (R² = 0.919)** — explains 92% of price variance. Area is the strongest predictor, followed by category type and neighborhood service density.

**Metric definitions:**
- **R²** (0 to 1): Proportion of variance explained by the model. 1.0 = perfect.
- **MAE**: Mean Absolute Error — on average, predictions are off by this SAR amount.
- **RMSE**: Root Mean Squared Error — penalizes large errors more heavily than MAE.
- **MAPE**: Mean Absolute Percentage Error — average percentage the prediction is off.

---

### Supervised — Price Tier Classification

Three classifiers trained to categorize properties into quartile-based tiers: Budget (bottom 25%), Mid (25–50%), Premium (50–75%), Luxury (top 25%).

| Model | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 ↑ |
|---|---|---|---|---|
| Logistic Regression | 0.6360 | 0.6311 | 0.6360 | 0.6321 |
| Random Forest | 0.8708 | 0.8719 | 0.8708 | 0.8713 |
| **Gradient Boosting** | **0.8774** | **0.8786** | **0.8774** | **0.8779** |

**Winner: Gradient Boosting (F1 = 0.878)** — classifies 87.7% of properties into the correct price tier.

**Metric definitions:**
- **Accuracy**: Percentage of all properties classified into the correct tier.
- **Precision**: When the model says "Luxury", how often is it actually Luxury?
- **Recall**: Of all actual Luxury homes, how many did the model catch?
- **F1 Score**: Harmonic mean of Precision and Recall. Balances both types of errors.

---

### Recommendation Engine — Three-Factor Scoring

The recommendation system combines all three ML components into a single combined score:

$$Score = RLI \times w_{rli} + PriceScore \times w_{price} + ClusterFit \times w_{cluster}$$

| Factor | Default Weight | Source |
|---|---|---|
| **RLI** (Livability) | 65% | 4-pillar weighted + entropy boost |
| **Price Score** (Budget fit) | 20% | Distance from budget midpoint |
| **Cluster Fit** (Archetype match) | 15% | K-Means cluster alignment with user priorities |

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