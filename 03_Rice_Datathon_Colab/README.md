# Rice Datathon 2025 - REVPAR Growth Prediction

Predicting hotel REVPAR (Revenue Per Available Room) growth using ensemble machine learning.

## Project Structure

```
03_Rice_Datathon_Colab/
├── data/
│   ├── raw/
│   │   ├── competition_data/     # Original competition files
│   │   ├── train.csv             # Combined training data (generated)
│   │   └── test.csv              # Scoring data (generated)
│   ├── processed/
│   │   ├── train_clean.csv       # Feature-engineered data
│   │   ├── test_clean.csv
│   │   └── categorical_features.json
│   └── submissions/              # Final submission CSVs
├── extra/
│   ├── models/
│   │   └── tabpfn_cache/         # TabPFN model cache (offline mode)
│   └── outputs/
│       ├── predictions_*/        # Model predictions
│       ├── tuning/               # Tuned parameters
│       └── feature_importance.csv
├── notebooks/
│   ├── 01_prepare_data.ipynb     # Data loading & cleaning
│   ├── 02_preprocessing.ipynb    # Feature engineering
│   └── 03_full_ensemble.ipynb    # Model training & ensemble
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Notebooks in Order

1. **01_prepare_data.ipynb** - Combines raw data files
2. **02_preprocessing.ipynb** - Creates features
3. **03_full_ensemble.ipynb** - Trains models and creates submission

### 3. For Google Colab

Upload the folder to Google Drive, then:

1. Open notebooks in Colab
2. The code auto-detects Colab and mounts your Drive
3. Run cells in order

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FINAL ENSEMBLE                           │
│                                                                 │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│   │  GBDT Blend   │  │ Linear Blend  │  │ Forest Blend  │       │
│   │ LGB,XGB,Cat,  │  │ Ridge,        │  │ ExtraTrees    │       │
│   │ HistGB        │  │ ElasticNet    │  │               │       │
│   └───────────────┘  └───────────────┘  └───────────────┘       │
│                                                                 │
│   + Optional: TabPFN (6 subsets), KNN (6 subsets)              │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### Feature Engineering
- Property age features (polynomial, log, vintage categories)
- Renovation signals (never_renovated, years_since_renov)
- COVID period indicators (is_pre_covid, is_post_covid)
- Geographic features (is_sunbelt, msa_ring_num)
- Drivetime features (suburb_tight_amenities)
- Fold-aware target encoding

### Model Improvements
- **Subset Training**: TabPFN/KNN trained separately for each (time_window x trade_area) combination
- **Regularized Ensemble**: Entropy + variance regularization prevents overfitting
- **Target Clipping**: 99th percentile clipping reduces outlier influence

### Validation
- 15% holdout by UBID (property-level split)
- 5-fold GroupKFold CV by UBID
- Overfitting gap analysis

## Configuration

In `03_full_ensemble.ipynb`:

```python
DRY_RUN = True    # Quick test mode (few trials, fast training)
RESUME = False    # Load cached predictions if available
N_FOLDS = 5       # CV folds
HOLDOUT_FRACTION = 0.15  # Holdout percentage
```

## Competition Details

- **Target**: REVPAR_GROWTH_PCT (pre-COVID: 2015-2020, post-COVID: 2022-2025)
- **Metric**: RMSE
- **Key Insight**: Pre vs Post COVID patterns differ significantly

## Best Practices

1. Always run notebooks in order (01 → 02 → 03)
2. Use `DRY_RUN = True` for initial testing
3. Check holdout RMSE before submitting
4. Monitor overfitting gap (should be < 10%)
