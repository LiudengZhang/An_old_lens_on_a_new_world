# Rice Datathon 2026 - RevPAR Growth Prediction
## BroadVail Capital Partners Finance Track

Predicting apartment RevPAR (Revenue Per Available Room) growth based on neighborhood amenities and the "X-Minute City" concept.

## Results Summary

| Metric | Value |
|--------|-------|
| **Holdout RMSE** | 0.1353 |
| **CV RMSE** | 0.0714 |
| **Best Strategy** | Ridge Stacking (diverse ensemble) |
| **Models Used** | 12 (5 GBDT + 2 Linear + 1 Forest + 4 Feature-Subset) |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run notebooks in order
# 01_prepare_data.ipynb  -> Combines raw data files
# 02_preprocessing.ipynb -> Feature engineering
# 03_full_ensemble.ipynb -> Model training & submission
```

## Project Structure

```
03_Rice_Datathon_Colab/
├── data/
│   ├── raw/                    # Original competition files
│   ├── processed/              # Feature-engineered data
│   └── submissions/            # Final submission CSVs
├── extra/
│   └── outputs/
│       ├── predictions_*/      # Cached model predictions
│       └── feature_importance.csv
├── notebooks/
│   ├── 01_prepare_data.ipynb   # Data loading & cleaning
│   ├── 02_preprocessing.ipynb  # Feature engineering
│   └── 03_full_ensemble.ipynb  # Model training & ensemble
├── requirements.txt
└── README.md
```

## Methodology

### 1. Feature Engineering

**Property Features:**
- Age polynomial features (age_squared, age_log, age_cubed)
- Renovation signals (never_renovated, years_since_renov, recently_renovated)
- Vintage categories and value tiers

**Geographic Features:**
- MSA ring position (downtown, inner suburb, outer suburb)
- Sunbelt/Texas indicators
- Market-level target encoding

**Economic Features:**
- Rent percentiles and affordability index
- Supply growth metrics
- Own-vs-rent spread

**Amenity Features:**
- Food/dining counts by category
- Healthcare and grocery access
- AARP livability scores
- Walkable dining interaction (walkability x food options)

**Correlation-Based Pruning:**
- Dropped 8 highly correlated features (threshold > 0.80)
- Kept most informative feature per cluster

### 2. Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RIDGE STACKING ENSEMBLE                      │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   FULL MODELS (5)                       │   │
│   │  LightGBM, XGBoost, CatBoost, HistGB, ExtraTrees       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              +                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              FEATURE-SUBSET MODELS (4)                  │   │
│   │  LGB_property, LGB_geographic, LGB_economic, LGB_amenities │
│   └─────────────────────────────────────────────────────────┘   │
│                              +                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  LINEAR MODELS (2)                      │   │
│   │              Ridge, ElasticNet                          │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Validation Strategy

- **15% holdout** by UBID (property-level split, ~150 properties)
- **5-fold GroupKFold CV** by UBID (no property leakage)
- **Target clipping** at 99th percentile to reduce outlier influence

### 4. Ensemble Strategies Tested

| Strategy | CV RMSE | Holdout RMSE |
|----------|---------|--------------|
| diverse_ridge (winner) | 0.1323 | **0.1353** |
| full_only | 0.1323 | 0.1353 |
| diverse_lgb | 0.1323 | 0.1356 |
| best_single (CatBoost) | 0.1331 | 0.1361 |

## Key Findings

### Top Features (by importance)

1. **rent_percentile** - Property's rent position in market
2. **ownrent_avg_rent** - Average rent in own-vs-rent comparison
3. **property_age** - Years since construction
4. **supply_growth_pct** - New supply in trade area
5. **msa_norm_dist** - Distance from metro center

### Pre-COVID vs Post-COVID Shifts

| Feature Category | Pre-COVID Importance | Post-COVID Importance |
|-----------------|---------------------|----------------------|
| Downtown proximity | High | Moderate |
| Amenity access | Moderate | High |
| Property age | Moderate | High |
| Supply growth | Low | High |

**Key Insight:** Post-COVID, amenity access within the trade area became more predictive than downtown proximity, consistent with the "15-minute city" hypothesis and hybrid work patterns.

### Model Diversity Analysis

Feature-subset models achieved low correlation with full models (0.23-0.34), providing genuine diversity for ensemble improvement. However, individual subset models were weaker (RMSE 0.15-0.22), so the ensemble benefit was modest.

## Configuration

In `03_full_ensemble.ipynb`:

```python
DRY_RUN = False       # True for quick testing
RESUME = False        # True to load cached predictions
N_FOLDS = 5           # CV folds
HOLDOUT_FRACTION = 0.15
```

## Reproducibility

All models use `random_state=42`. To reproduce results:

1. Run notebooks in order (01 → 02 → 03)
2. Set `RESUME = False` to retrain from scratch
3. Submission will be saved to `data/submissions/`

## Files

- **Submission:** `data/submissions/sub_diverse_ridge_cv0.0714_20260124_215345.csv`
- **Feature Importance:** `extra/outputs/feature_importance.csv`
- **Model Predictions:** `extra/outputs/predictions_*/`

## Team

Rice Datathon 2026 - Finance Track
