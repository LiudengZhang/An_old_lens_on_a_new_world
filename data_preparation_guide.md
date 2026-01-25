# Data Preparation Guide: BroadVail Datathon Query System

## Project Overview

We are building an AI-powered data query website for the Rice Datathon 2026 BroadVail Capital track. Users can ask questions in natural language, and the system queries data in layers based on complexity:

- **Layer 1**: Pre-computed analysis results (fast, low cost)
- **Layer 2**: Model prediction results (medium cost)
- **Layer 3**: Raw data analysis (high cost, requires user confirmation)

### OpenAI API Token Usage

- **Default**: We provide a shared OpenAI API token for testing (from BroadVail's $25 credit)
- **Optional**: Users can input their own OpenAI API key to use the system
- The tiered query system is designed to minimize token consumption
- Layer 3 queries (raw data analysis) will prompt users to confirm before proceeding due to higher token cost

---

## File 1: key_findings.json

**Purpose**: Store pre-analyzed key findings for fast answers to common questions

**Format**:
```json
[
  {
    "id": "finding_001",
    "category": "covid_impact",
    "question_patterns": [
      "post covid feature importance",
      "how did covid change preferences",
      "what matters after pandemic"
    ],
    "answer": "Your analysis conclusion in complete sentences...",
    "supporting_data": {
      "metric_name": "value"
    },
    "source": "analysis_source_description"
  }
]
```

**Required question categories**:

1. **COVID Impact**
   - Pre vs Post COVID feature importance changes
   - Which amenities became more/less important
   - Quartile transition analysis (only ~20% stayed in same quartile)

2. **Geographic/City**
   - City performance comparisons
   - Best/worst performing cities
   - Submarket characteristics

3. **Drivetime**
   - 10min vs 15min vs 30min explanatory power
   - Which drivetime works best for which neighborhood type

4. **Amenity**
   - Top 10 most important amenities
   - Impact by amenity category (food, retail, parks, transit, etc.)
   
5. **Model Performance**
   - Overall RMSE, R²
   - Which properties are well/poorly predicted
   - Pre vs Post prediction difficulty

**Example entry**:
```json
{
  "id": "finding_001",
  "category": "covid_impact",
  "question_patterns": [
    "quartile change",
    "performance persistence",
    "which properties maintained performance"
  ],
  "answer": "Apartment performance persistence across COVID is very low. According to quartile transition matrix analysis, only about 20% of properties (195/996) remained in the same performance quartile. Notably, 93 properties fell from top quartile (Q4) to bottom (Q1), while 89 rose from Q1 to Q4, indicating COVID significantly reshaped rental market competition.",
  "supporting_data": {
    "persistence_rate": "19.6%",
    "q4_to_q1_count": 93,
    "q1_to_q4_count": 89,
    "total_properties": 996
  },
  "source": "quartile_transition_matrix_analysis"
}
```

**Requirements**:
- Prepare at least 15-20 findings
- Include various phrasings in question_patterns
- Answers should be complete, professional analysis conclusions
- Include specific numbers for support

---

## File 2: feature_importance.csv

**Purpose**: Model feature importance for feature-related queries

**Format**:
```csv
feature_name,feature_category,importance_pre,importance_post,importance_change_pct,rank_pre,rank_post,drivetime,description
restaurant_count,amenity_food,0.152,0.083,-45.4,3,12,15min,Number of restaurants within 15-min drive
grocery_count,amenity_retail,0.048,0.127,164.6,15,4,15min,Number of grocery stores within 15-min drive
park_count,amenity_recreation,0.031,0.089,187.1,22,8,10min,Number of parks within 10-min drive
cbd_distance,location,0.203,0.118,-41.9,1,6,NA,Distance to central business district
median_income,demographics,0.089,0.095,6.7,8,9,NA,Area median household income
```

**Required columns**:
- `feature_name`: Feature name (match original data column names)
- `feature_category`: Category (amenity_food, amenity_retail, amenity_recreation, amenity_transport, location, demographics, housing, supply)
- `importance_pre`: Pre-COVID importance score
- `importance_post`: Post-COVID importance score
- `importance_change_pct`: Change percentage
- `rank_pre`: Pre-COVID rank
- `rank_post`: Post-COVID rank
- `drivetime`: Applicable drivetime (10min/15min/30min/NA)
- `description`: Feature description

**Requirements**:
- Include all features used in modeling
- Normalize importance scores to 0-1
- Sort by importance_post descending

---

## File 3: city_summary.csv

**Purpose**: City-level aggregate statistics

**Format**:
```csv
city,state,property_count,avg_revpar_growth_pre,avg_revpar_growth_post,median_revpar_growth_pre,median_revpar_growth_post,std_revpar_growth_pre,std_revpar_growth_post,top_amenity_pre,top_amenity_post,avg_prediction_error,quartile_up_pct,quartile_down_pct,quartile_same_pct
Phoenix-Mesa-Scottsdale,AZ,156,0.123,0.081,0.115,0.078,0.045,0.052,cbd_distance,grocery_count,0.032,28.5,35.2,36.3
Houston,TX,203,0.101,0.112,0.098,0.105,0.038,0.041,restaurant_count,park_count,0.028,32.1,29.8,38.1
```

**Required columns**:
- `city`: City name (match original data)
- `state`: State abbreviation
- `property_count`: Number of properties in city
- `avg_revpar_growth_pre`: Pre-COVID average RevPAR growth
- `avg_revpar_growth_post`: Post-COVID average RevPAR growth
- `median_revpar_growth_pre`: Pre-COVID median
- `median_revpar_growth_post`: Post-COVID median
- `std_revpar_growth_pre`: Pre-COVID standard deviation
- `std_revpar_growth_post`: Post-COVID standard deviation
- `top_amenity_pre`: Most important amenity pre-COVID
- `top_amenity_post`: Most important amenity post-COVID
- `avg_prediction_error`: Average prediction error for city
- `quartile_up_pct`: Percentage of properties that moved up in quartile
- `quartile_down_pct`: Percentage that moved down
- `quartile_same_pct`: Percentage that stayed same

**Requirements**:
- Cover all cities in dataset
- Keep 3-4 decimal places
- Sort by property_count descending

---

## File 4: submarket_summary.csv

**Purpose**: Finer-grained submarket-level statistics

**Format**:
```csv
city,submarket,property_count,avg_revpar_growth_pre,avg_revpar_growth_post,growth_change,positioning_label,dominant_amenity_pre,dominant_amenity_post,avg_drivetime_preference
Phoenix-Mesa-Scottsdale,Deer Valley,23,0.134,0.092,-0.042,Upper-Mid,cbd_distance,grocery_count,15min
Phoenix-Mesa-Scottsdale,Downtown Phoenix,18,0.156,0.067,-0.089,Premium,restaurant_count,restaurant_count,10min
```

**Required columns**:
- `city`: City name
- `submarket`: Submarket name
- `property_count`: Number of properties
- `avg_revpar_growth_pre`: Pre-COVID average growth
- `avg_revpar_growth_post`: Post-COVID average growth
- `growth_change`: Growth change (post - pre)
- `positioning_label`: Market positioning (Premium/Upper-Mid/Mid-Market/Value-Oriented)
- `dominant_amenity_pre`: Dominant amenity pre-COVID
- `dominant_amenity_post`: Dominant amenity post-COVID
- `avg_drivetime_preference`: Best explanatory drivetime

---

## File 5: predictions.csv

**Purpose**: Model predictions for property-level queries

**Format**:
```csv
property_id,city,submarket,lat,lng,time_period,drivetime,actual_revpar_growth,predicted_revpar_growth,residual,abs_error,quartile_actual,quartile_predicted,is_outperformer,prediction_confidence
P00001,Phoenix-Mesa-Scottsdale,Deer Valley,33.6845,-112.0853,post,15min,0.082,0.075,0.007,0.007,3,3,false,high
P00001,Phoenix-Mesa-Scottsdale,Deer Valley,33.6845,-112.0853,pre,15min,0.134,0.128,0.006,0.006,4,4,true,high
P00002,Houston,Midtown,29.7475,-95.3942,post,15min,0.156,0.142,0.014,0.014,4,4,true,medium
```

**Required columns**:
- `property_id`: Unique property identifier
- `city`: City
- `submarket`: Submarket
- `lat`: Latitude (if available)
- `lng`: Longitude (if available)
- `time_period`: Time period (pre/post)
- `drivetime`: Drivetime definition (10min/15min/30min)
- `actual_revpar_growth`: Actual RevPAR growth rate
- `predicted_revpar_growth`: Predicted RevPAR growth rate
- `residual`: Residual (actual - predicted)
- `abs_error`: Absolute error
- `quartile_actual`: Actual quartile (1-4)
- `quartile_predicted`: Predicted quartile (1-4)
- `is_outperformer`: Whether property is an outperformer (boolean)
- `prediction_confidence`: Prediction confidence (high/medium/low)

**Requirements**:
- One row per property per time_period per drivetime combination
- Include all properties from training set
- Keep 4-5 decimal places

---

## File 6: model_performance.json

**Purpose**: Overall model performance metrics

**Format**:
```json
{
  "model_info": {
    "model_type": "XGBoost / Random Forest / etc.",
    "n_features": 45,
    "training_samples": 5000,
    "cv_folds": 5
  },
  "overall_performance": {
    "rmse": 0.045,
    "mae": 0.032,
    "r2": 0.72,
    "mape": 12.5
  },
  "by_time_period": {
    "pre": {
      "rmse": 0.038,
      "mae": 0.027,
      "r2": 0.78,
      "sample_count": 2500
    },
    "post": {
      "rmse": 0.052,
      "mae": 0.038,
      "r2": 0.65,
      "sample_count": 2500
    }
  },
  "by_drivetime": {
    "10min": {"rmse": 0.041, "r2": 0.74, "sample_count": 1666},
    "15min": {"rmse": 0.044, "r2": 0.73, "sample_count": 1667},
    "30min": {"rmse": 0.049, "r2": 0.69, "sample_count": 1667}
  },
  "by_city": {
    "Phoenix-Mesa-Scottsdale": {"rmse": 0.042, "r2": 0.75},
    "Houston": {"rmse": 0.039, "r2": 0.77}
  },
  "quartile_accuracy": {
    "exact_match_rate": 0.45,
    "within_one_quartile_rate": 0.82,
    "confusion_matrix": [
      [120, 45, 20, 10],
      [40, 110, 50, 15],
      [15, 55, 105, 40],
      [10, 20, 50, 125]
    ]
  }
}
```

---

## File 7: drivetime_analysis.csv

**Purpose**: Comparison of different drivetime definitions

**Format**:
```csv
drivetime,model_r2,model_rmse,avg_amenity_count,best_for_category,explanation
10min,0.74,0.041,45,downtown/urban core,Best for high-density urban core areas with walking/short trips
15min,0.73,0.044,120,suburban/mixed,Best for mixed-use areas balancing convenience and reach
30min,0.69,0.049,350,outer suburb/exurban,Best for outer suburbs where residents are used to longer commutes
```

---

## File 8: amenity_analysis.csv

**Purpose**: Detailed analysis of amenity types

**Format**:
```csv
amenity_type,amenity_category,avg_count_10min,avg_count_15min,avg_count_30min,correlation_with_revpar_pre,correlation_with_revpar_post,correlation_change,interpretation
restaurant,food_dining,12.5,45.2,180.3,0.35,0.18,-0.17,Restaurant density correlation decreased post-COVID reflecting reduced workday dining out due to remote work
grocery,retail_essential,2.1,5.8,15.2,0.15,0.42,0.27,Grocery accessibility became more important post-COVID reflecting increased time at home
park,recreation,3.2,8.5,22.1,0.12,0.38,0.26,Park importance increased significantly reflecting lifestyle changes
gym,recreation,1.8,5.2,12.8,0.22,0.25,0.03,Gym importance remained relatively stable
public_transit,transportation,4.5,12.3,35.6,0.45,0.28,-0.17,Public transit importance decreased reflecting remote work trends
```

---

## File 9: data_dictionary.csv

**Purpose**: Field descriptions for raw data

**Format**:
```csv
column_name,data_type,description,is_feature,is_target,category,example_value
property_id,string,Unique property identifier,no,no,identifier,P00001
city,string,Metropolitan area name,no,no,location,Phoenix-Mesa-Scottsdale
submarket,string,Submarket within city,no,no,location,Deer Valley
time_window_tag,string,Time period indicator,no,no,time,pre/post
time_window_label,string,Detailed time window,no,no,time,2015_to_2020Feb
drivetime_minutes,integer,Drive time radius in minutes,no,no,spatial,15
REVPAR_GROWTH_2015_2020_PCT,float,RevPAR growth rate 2015-2020,no,yes (when pre),target,0.125
REVPAR_GROWTH_2022_2025_PCT,float,RevPAR growth rate 2022-2025,no,yes (when post),target,0.083
restaurant_count,integer,Number of restaurants in trade area,yes,no,amenity_food,45
grocery_count,integer,Number of grocery stores,yes,no,amenity_retail,5
```

**Requirements**:
- Cover all columns in original data
- Clearly mark which are features vs targets
- Provide clear descriptions

---

## File 10: training_data.csv

**Purpose**: Raw training data (for Layer 3 queries)

Use the `training.csv` provided by the competition directly. Ensure:
- File is complete
- UTF-8 encoding
- No extra blank rows

---

## File Checklist Summary

Prepare and package these 10 files:

| Filename | Format | Priority | Purpose |
|----------|--------|----------|---------|
| key_findings.json | JSON | Required | Pre-computed key findings |
| feature_importance.csv | CSV | Required | Feature importance |
| city_summary.csv | CSV | Required | City-level summary |
| submarket_summary.csv | CSV | Recommended | Submarket-level summary |
| predictions.csv | CSV | Required | Model predictions |
| model_performance.json | JSON | Required | Model performance metrics |
| drivetime_analysis.csv | CSV | Recommended | Drivetime comparison |
| amenity_analysis.csv | CSV | Recommended | Amenity detailed analysis |
| data_dictionary.csv | CSV | Required | Data dictionary |
| training_data.csv | CSV | Required | Raw training data |

---

## Data Quality Checklist

Before submitting, verify:

- [ ] All CSV files use UTF-8 encoding
- [ ] All CSV files have column headers in first row
- [ ] No null values in numeric columns (fill with 0 or NA and document)
- [ ] JSON files are valid (check with online JSON validator)
- [ ] property_id is consistent across files
- [ ] City and submarket names are spelled consistently
- [ ] feature_name matches original data column names exactly
- [ ] All percentages in decimal format (0.125, not 12.5%)

---

## Modeling Recommendations

To generate the above data, consider:

1. **Use interpretable models**: XGBoost, Random Forest, or Linear Regression with regularization
2. **Record feature importance**: Use model's built-in importance or SHAP values
3. **Train separate pre and post models**: Easier to compare feature importance changes
4. **Evaluate each drivetime separately**: Understand which spatial definition is most effective
5. **Save cross-validation results**: Provides more robust performance estimates

---

## Delivery Format

Package all files as `datathon_query_data.zip` with this structure:

```
datathon_query_data/
├── key_findings.json
├── feature_importance.csv
├── city_summary.csv
├── submarket_summary.csv
├── predictions.csv
├── model_performance.json
├── drivetime_analysis.csv
├── amenity_analysis.csv
├── data_dictionary.csv
└── training_data.csv
```

Upload when ready and I will build the query system.
