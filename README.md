# Rice Datathon 2026 - RevPAR Growth Prediction

## BroadVail Capital Partners Finance Track

Predicting apartment RevPAR (Revenue Per Available Room) growth using neighborhood amenities and the **"X-Minute City"** concept.

## Results

| Metric | Value |
|--------|-------|
| **Holdout RMSE** | 0.1353 |
| **Cross-Validation RMSE** | 0.0714 |
| **Best Strategy** | Ridge Stacking (12 diverse models) |
| **Key Insight** | Post-COVID, amenity access surpassed downtown proximity as the primary driver of RevPAR growth |

## Project Structure

```
Project_8_Rice_Datathon/
├── 03_Rice_Datathon_Colab/          # ML Pipeline (Google Colab compatible)
│   ├── notebooks/
│   │   ├── 01_prepare_data.ipynb    # Data loading & cleaning
│   │   ├── 02_preprocessing.ipynb   # Feature engineering
│   │   └── 03_full_ensemble.ipynb   # Model training & ensemble
│   ├── data/
│   │   ├── raw/                     # Original competition files
│   │   ├── processed/               # Feature-engineered data
│   │   └── submissions/             # Final submission CSVs
│   └── requirements.txt
│
├── 04_Chatbot/                       # AI-powered Query System
│   ├── app.py                       # Streamlit application
│   ├── query_engine.py              # Multi-layer query processing
│   ├── visualizations.py            # Dynamic chart generation
│   └── data/                        # Pre-computed findings & summaries
│
├── 05_Highlights/                    # Publication-ready figures
│   ├── create_summary_figure_*.py   # Figure generation scripts
│   └── fig_summary_nature.png       # Summary visualization
│
└── README.md
```

## Quick Start

### 1. ML Pipeline (Colab or Local)

```bash
# Install dependencies
cd 03_Rice_Datathon_Colab
pip install -r requirements.txt

# Run notebooks in order
# 01_prepare_data.ipynb  -> Combines raw data files
# 02_preprocessing.ipynb -> Feature engineering
# 03_full_ensemble.ipynb -> Model training & submission
```

### 2. Chatbot Application

```bash
cd 04_Chatbot
pip install streamlit openai pandas plotly

# Run the app
streamlit run app.py
```

Access at `http://localhost:8501`

**Example questions:**
- "What are the top 10 most important features?"
- "How did COVID change apartment preferences?"
- "Which drivetime definition works best?"

## Methodology

### Data

- **Source**: BroadVail Capital Partners apartment market data
- **Target**: RevPAR growth percentage
- **Time periods**: Pre-COVID (2015-2020) and Post-COVID (2022-2025)
- **Trade areas**: 10-minute, 15-minute, and 30-minute drivetimes

### Feature Engineering

| Category | Features |
|----------|----------|
| **Property** | Age, vintage, renovation status, unit count, class tier |
| **Geographic** | MSA ring position, downtown proximity, Sunbelt/Texas indicators |
| **Economic** | Rent percentile, supply growth, affordability index |
| **Amenities** | Food/dining counts, healthcare access, walkability scores, AARP livability |

### Model Architecture

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
│   │  Property, Geographic, Economic, Amenities              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              +                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  LINEAR MODELS (2)                      │   │
│   │              Ridge, ElasticNet                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              +                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │               DISTANCE MODEL (1)                        │   │
│   │                    KNN                                  │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Validation Strategy

- **15% holdout** by property ID (UBID) - no property leakage
- **5-fold GroupKFold CV** by property ID
- **Target clipping** at 99th percentile to reduce outlier influence

## Key Findings

### Top Predictive Features

1. **rent_percentile** - Property's rent position in market
2. **ownrent_avg_rent** - Average rent in own-vs-rent comparison
3. **property_age** - Years since construction
4. **supply_growth_pct** - New supply in trade area
5. **msa_norm_dist** - Distance from metro center

### Pre-COVID vs Post-COVID Shift

| Feature Category | Pre-COVID | Post-COVID |
|-----------------|-----------|------------|
| Downtown proximity | High importance | Moderate |
| Amenity access | Moderate | **High importance** |
| Property age | Moderate | High |
| Supply growth | Low | High |

**Key Insight**: Post-COVID, amenity access within the trade area became more predictive than downtown proximity, consistent with the **"15-minute city" hypothesis** and hybrid work patterns.

### Confidence Analysis

The ensemble provides prediction confidence based on model agreement:
- **High confidence** (ensemble std < 0.022): Models strongly agree
- **Medium confidence** (0.022 - 0.033): Use with caution
- **Low confidence** (> 0.033): Investigate further

## Chatbot Features

The AI-powered query system uses a 3-layer architecture:

| Layer | Description | Speed |
|-------|-------------|-------|
| **Layer 1** | Pre-computed findings (semantic search) | Fast |
| **Layer 2** | Structured data queries (aggregated stats) | Medium |
| **Layer 3** | Raw data analysis (on-demand computation) | Slow |

Features:
- Natural language queries about the data
- Interactive visualizations (Plotly)
- Prediction confidence explorer
- Market comparison tools

## Requirements

### ML Pipeline
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
optuna>=3.4.0
```

### Chatbot
```
streamlit
openai
pandas
plotly
```

## Reproducibility

All models use `random_state=42`. To reproduce:

1. Run notebooks in order: `01 → 02 → 03`
2. Set `RESUME = False` in notebook to retrain from scratch
3. Submissions saved to `data/submissions/`

## Team

Rice Datathon 2026 - Finance Track
