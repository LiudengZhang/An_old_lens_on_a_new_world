"""
Generate chatbot data files from notebook outputs.

Run this AFTER the notebooks (01, 02, 03) have completed.

Usage:
    cd /priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/04_Chatbot
    python generate_chatbot_data.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent / "03_Rice_Datathon_Colab"
CHATBOT_DATA = Path(__file__).parent / "data"

# Input paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "extra" / "outputs"
COMPETITION_DIR = RAW_DIR / "competition_data"

# =============================================================================
# HOLDOUT SPLIT CONFIG (must match notebook 03_full_ensemble.ipynb)
# =============================================================================
SEED = 42
HOLDOUT_FRACTION = 0.15

# Ensure output directory exists
CHATBOT_DATA.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("GENERATING CHATBOT DATA FILES")
print("=" * 60)
print(f"Project root: {PROJECT_ROOT}")
print(f"Output dir: {CHATBOT_DATA}")


# =============================================================================
# [1] LOAD INPUTS
# =============================================================================

print("\n[1] Loading inputs...")

# Load raw train data
train_raw_path = RAW_DIR / "train.csv"
if train_raw_path.exists():
    train_raw = pd.read_csv(train_raw_path)
    print(f"  train.csv: {train_raw.shape}")
else:
    print(f"  ERROR: {train_raw_path} not found. Run notebook 01 first.")
    train_raw = None

# Load processed train data
train_clean_path = PROCESSED_DIR / "train_clean.csv"
if train_clean_path.exists():
    train_clean = pd.read_csv(train_clean_path)
    print(f"  train_clean.csv: {train_clean.shape}")
else:
    print(f"  WARNING: {train_clean_path} not found. Some features may be missing.")
    train_clean = None

# Load feature importance from models
feat_imp_path = OUTPUTS_DIR / "feature_importance.csv"
if feat_imp_path.exists():
    feat_imp_raw = pd.read_csv(feat_imp_path)
    print(f"  feature_importance.csv: {feat_imp_raw.shape}")
else:
    print(f"  WARNING: {feat_imp_path} not found. Will compute from data.")
    feat_imp_raw = None

# Load OOF predictions
oof_dir = OUTPUTS_DIR / "predictions_oof"
oof_predictions = {}
if oof_dir.exists():
    for npy_file in oof_dir.glob("*.npy"):
        model_name = npy_file.stem
        oof_predictions[model_name] = np.load(npy_file)
        print(f"  OOF {model_name}: {oof_predictions[model_name].shape}")
else:
    print(f"  WARNING: {oof_dir} not found. Run notebook 03 first.")

# Load dictionary
dict_path = COMPETITION_DIR / "dictionary.csv"
if dict_path.exists():
    data_dict = pd.read_csv(dict_path)
    print(f"  dictionary.csv: {data_dict.shape}")
else:
    print(f"  WARNING: {dict_path} not found.")
    data_dict = None


# =============================================================================
# [2] GENERATE feature_importance.csv
# =============================================================================

print("\n[2] Generating feature_importance.csv...")

if feat_imp_raw is not None and train_clean is not None:
    # Get feature columns - only numeric ones
    exclude_cols = ['target', 'target_clipped', 'target_log', 'id']
    numeric_cols = train_clean.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Use existing feature importance as base, then compute pre/post correlations
    try:
        # Map existing importance to feature columns (overall model importance)
        imp_map = dict(zip(feat_imp_raw['feature'], feat_imp_raw['avg_importance']))
        importance_overall = np.array([imp_map.get(f, 0) for f in feature_cols])

        # Normalize overall importance
        if importance_overall.sum() > 0:
            importance_overall = importance_overall / importance_overall.sum()

        # Compute pre/post correlations with target to get differentiated importance
        # Merge time_window_tag from train_raw if not in train_clean
        if 'time_window_tag' not in train_clean.columns and train_raw is not None:
            train_clean['time_window_tag'] = train_raw['time_window_tag'].values[:len(train_clean)]

        pre_mask = train_clean['time_window_tag'] == 'pre'
        post_mask = train_clean['time_window_tag'] == 'post'

        # Compute absolute correlations for pre and post periods
        importance_pre = []
        importance_post = []
        for f in feature_cols:
            if f in train_clean.columns:
                pre_corr = abs(train_clean.loc[pre_mask, [f, 'target']].corr().iloc[0, 1])
                post_corr = abs(train_clean.loc[post_mask, [f, 'target']].corr().iloc[0, 1])
                importance_pre.append(pre_corr if not np.isnan(pre_corr) else 0)
                importance_post.append(post_corr if not np.isnan(post_corr) else 0)
            else:
                importance_pre.append(0)
                importance_post.append(0)

        importance_pre = np.array(importance_pre)
        importance_post = np.array(importance_post)

        # Normalize to sum to 1
        if importance_pre.sum() > 0:
            importance_pre = importance_pre / importance_pre.sum()
        if importance_post.sum() > 0:
            importance_post = importance_post / importance_post.sum()

        # Compute ranks
        rank_pre = (-importance_pre).argsort().argsort() + 1
        rank_post = (-importance_post).argsort().argsort() + 1

        # Compute percentage change
        change_pct = np.where(
            importance_pre > 0.001,
            (importance_post - importance_pre) / importance_pre * 100,
            np.where(importance_post > 0.001, 100, 0)  # New importance = +100%
        )

        # Categorize features
        def categorize_feature(name):
            name_lower = name.lower()
            if 'food' in name_lower or 'restaurant' in name_lower or 'grocery' in name_lower:
                return 'amenity_food'
            elif 'park' in name_lower or 'recreation' in name_lower:
                return 'amenity_recreation'
            elif 'transit' in name_lower or 'transport' in name_lower:
                return 'amenity_transport'
            elif 'aarp' in name_lower:
                return 'livability'
            elif 'age' in name_lower or 'year' in name_lower or 'renov' in name_lower:
                return 'property'
            elif 'rent' in name_lower or 'mortgage' in name_lower or 'supply' in name_lower:
                return 'housing'
            elif 'msa' in name_lower or 'city' in name_lower or 'state' in name_lower or 'market' in name_lower:
                return 'location'
            elif 'drv' in name_lower or 'drivetime' in name_lower:
                return 'spatial'
            else:
                return 'other'

        # Determine drivetime
        def get_drivetime(name):
            if 'drv10' in name.lower() or '10min' in name.lower():
                return '10min'
            elif 'drv15' in name.lower() or '15min' in name.lower():
                return '15min'
            elif 'drv30' in name.lower() or '30min' in name.lower():
                return '30min'
            else:
                return 'NA'

        # Build DataFrame
        feat_imp_df = pd.DataFrame({
            'feature_name': feature_cols,
            'feature_category': [categorize_feature(f) for f in feature_cols],
            'importance_pre': importance_pre,
            'importance_post': importance_post,
            'importance_change_pct': change_pct,
            'rank_pre': rank_pre,
            'rank_post': rank_post,
            'drivetime': [get_drivetime(f) for f in feature_cols],
            'description': feature_cols,  # Placeholder
        })

        # Sort by post importance
        feat_imp_df = feat_imp_df.sort_values('importance_post', ascending=False)
        feat_imp_df.to_csv(CHATBOT_DATA / "feature_importance.csv", index=False)
        print(f"  Saved feature_importance.csv ({len(feat_imp_df)} features)")

    except ImportError:
        print("  WARNING: LightGBM not available. Using raw importance.")
        feat_imp_raw.to_csv(CHATBOT_DATA / "feature_importance.csv", index=False)
else:
    print("  SKIPPED: Missing inputs")


# =============================================================================
# [2.5] LOAD TRANSFORM PARAMETERS
# =============================================================================

print("\n[2.5] Loading transform parameters...")

transform_params_path = OUTPUTS_DIR / 'transform_params.json'
if transform_params_path.exists():
    with open(transform_params_path, 'r') as f:
        TRANSFORM_PARAMS = json.load(f)
    print(f"  Transform: {TRANSFORM_PARAMS}")
else:
    TRANSFORM_PARAMS = {'method': 'none', 'shift': 0}
    print("  No transform params found, using identity")

def inverse_transform(y, params):
    """Inverse transform predictions from log space to original space.

    Forward transform: y_log = log1p(y + shift)
    Inverse transform: y = expm1(y_log) - shift
    """
    if params['method'] == 'log':
        return np.expm1(y) - params['shift']
    return y


# =============================================================================
# [3] GENERATE predictions.csv
# =============================================================================

print("\n[3] Generating predictions.csv...")

if oof_predictions and train_raw is not None:
    # Filter to only use good models (exclude broken/low-quality ones)
    GOOD_MODELS = ['lgb', 'xgb', 'cat', 'hist', 'extra_trees', 'knn', 'ridge']
    good_preds = []
    for model_name in GOOD_MODELS:
        if model_name in oof_predictions:
            pred = oof_predictions[model_name]
            # Skip if all zeros or near-zero variance
            if pred.std() > 0.01 and (pred == 0).sum() < len(pred) * 0.5:
                good_preds.append(pred)
                print(f"  Using model: {model_name}")

    if not good_preds:
        print("  WARNING: No good models found, using all predictions")
        good_preds = list(oof_predictions.values())

    # Stack predictions for ensemble statistics
    good_preds_array = np.array(good_preds)  # Shape: (n_models, n_samples)

    # Average ensemble predictions from good models only
    ensemble_pred_raw = np.mean(good_preds_array, axis=0)

    # Calculate ensemble standard deviation (model disagreement)
    ensemble_std = np.std(good_preds_array, axis=0)
    print(f"  Ensemble of {len(good_preds)} models")
    print(f"  Ensemble std range: [{ensemble_std.min():.4f}, {ensemble_std.max():.4f}]")

    # CRITICAL FIX: Inverse transform predictions from log space
    ensemble_pred = inverse_transform(ensemble_pred_raw, TRANSFORM_PARAMS)
    print(f"  Predictions inverted from log space: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")

    n_preds = len(ensemble_pred)

    # Get target
    target_col = 'target'
    if target_col not in train_raw.columns:
        # Compute target from raw columns
        train_raw['target'] = np.where(
            train_raw['time_window_tag'] == 'pre',
            train_raw.get('REVPAR_GROWTH_2015_2020_PCT', 0),
            train_raw.get('REVPAR_GROWTH_2022_2025_PCT', 0)
        )

    # CRITICAL FIX: Recreate holdout split to get correct row alignment
    # OOF predictions are for training rows (NOT holdout), based on UBID split
    np.random.seed(SEED)
    ubids = train_raw['UBID'].unique()
    np.random.shuffle(ubids)
    n_holdout = int(len(ubids) * HOLDOUT_FRACTION)
    holdout_ubids = set(ubids[:n_holdout])
    train_mask = ~train_raw['UBID'].isin(holdout_ubids)

    # Select only the training rows that correspond to OOF predictions
    train_subset = train_raw[train_mask].reset_index(drop=True).copy()
    print(f"  Recreated holdout split: {len(train_subset)} training rows (holdout: {(~train_mask).sum()})")

    # Verify alignment
    if len(train_subset) != n_preds:
        print(f"  WARNING: Row count mismatch! train_subset={len(train_subset)}, predictions={n_preds}")
    else:
        print(f"  Row alignment verified: {n_preds} rows")

    # Build predictions DataFrame
    predictions_df = pd.DataFrame({
        'property_id': train_subset['UBID'],
        'city': train_subset.get('mrkt_name', train_subset.get('city', 'Unknown')),
        'submarket': train_subset.get('submrkt_name', 'Unknown'),
        'time_period': train_subset['time_window_tag'],
        'drivetime': train_subset['trade_area_label'].str.replace('drv', '') + 'min',
        'actual_revpar_growth': train_subset['target'],
        'predicted_revpar_growth': ensemble_pred,
    })

    # Compute metrics
    predictions_df['residual'] = predictions_df['actual_revpar_growth'] - predictions_df['predicted_revpar_growth']
    predictions_df['abs_error'] = predictions_df['residual'].abs()

    # Compute quartiles
    predictions_df['quartile_actual'] = pd.qcut(
        predictions_df['actual_revpar_growth'],
        q=4, labels=[1, 2, 3, 4], duplicates='drop'
    ).astype(int)
    predictions_df['quartile_predicted'] = pd.qcut(
        predictions_df['predicted_revpar_growth'],
        q=4, labels=[1, 2, 3, 4], duplicates='drop'
    ).astype(int)

    # Outperformer flag
    predictions_df['is_outperformer'] = (predictions_df['quartile_actual'] >= 3).astype(str).str.lower()

    # Ensemble variance (model disagreement) - usable without labels
    predictions_df['ensemble_std'] = ensemble_std

    # Confidence based on ensemble variance (lower std = higher confidence)
    # This can be computed WITHOUT knowing actual values
    std_pcts = predictions_df['ensemble_std'].rank(pct=True)
    predictions_df['prediction_confidence'] = pd.cut(
        std_pcts, bins=[0, 0.33, 0.66, 1.0], labels=['high', 'medium', 'low']
    )

    predictions_df.to_csv(CHATBOT_DATA / "predictions.csv", index=False)
    print(f"  Saved predictions.csv ({len(predictions_df)} rows)")
else:
    print("  SKIPPED: Missing OOF predictions or train data")


# =============================================================================
# [4] GENERATE model_performance.json
# =============================================================================

print("\n[4] Generating model_performance.json...")

if oof_predictions and train_raw is not None:
    # Use the train_subset already created with correct holdout alignment
    # (train_subset was defined in section [3] with proper UBID-based split)
    y_true = train_subset['target'].values
    y_pred = ensemble_pred

    # Remove NaN
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_mask]
    y_pred_clean = y_pred[valid_mask]

    # Overall metrics
    overall_rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    overall_mae = mean_absolute_error(y_true_clean, y_pred_clean)
    overall_r2 = r2_score(y_true_clean, y_pred_clean)

    # By time period
    by_period = {}
    for period in ['pre', 'post']:
        mask = train_subset['time_window_tag'].values[valid_mask] == period
        if mask.sum() > 0:
            by_period[period] = {
                'rmse': float(np.sqrt(mean_squared_error(y_true_clean[mask], y_pred_clean[mask]))),
                'mae': float(mean_absolute_error(y_true_clean[mask], y_pred_clean[mask])),
                'r2': float(r2_score(y_true_clean[mask], y_pred_clean[mask])),
                'sample_count': int(mask.sum())
            }

    # By drivetime
    by_drivetime = {}
    for drv in ['drv10', 'drv15', 'drv30']:
        mask = train_subset['trade_area_label'].values[valid_mask] == drv
        if mask.sum() > 0:
            by_drivetime[drv.replace('drv', '') + 'min'] = {
                'rmse': float(np.sqrt(mean_squared_error(y_true_clean[mask], y_pred_clean[mask]))),
                'r2': float(r2_score(y_true_clean[mask], y_pred_clean[mask])),
                'sample_count': int(mask.sum())
            }

    # By city (top 10)
    by_city = {}
    city_col = train_subset.get('mrkt_name', train_subset.get('city'))
    if city_col is not None:
        cities = city_col.values[valid_mask]
        for city in pd.Series(cities).value_counts().head(10).index:
            mask = cities == city
            if mask.sum() > 10:
                by_city[city] = {
                    'rmse': float(np.sqrt(mean_squared_error(y_true_clean[mask], y_pred_clean[mask]))),
                    'r2': float(r2_score(y_true_clean[mask], y_pred_clean[mask]))
                }

    # Quartile accuracy
    q_actual = pd.qcut(y_true_clean, q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)
    q_pred = pd.qcut(y_pred_clean, q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)
    exact_match = (q_actual == q_pred).mean()
    within_one = (np.abs(q_actual - q_pred) <= 1).mean()

    # Confusion matrix
    confusion = np.zeros((4, 4), dtype=int)
    for a, p in zip(q_actual, q_pred):
        confusion[a, p] += 1

    performance = {
        'model_info': {
            'model_type': 'Ensemble (LGB+XGB+CatBoost+others)',
            'n_features': len(feature_cols) if 'feature_cols' in dir() else 0,
            'training_samples': int(len(y_true_clean)),
            'cv_folds': 5
        },
        'overall_performance': {
            'rmse': float(overall_rmse),
            'mae': float(overall_mae),
            'r2': float(overall_r2),
            'mape': float(np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100)
        },
        'by_time_period': by_period,
        'by_drivetime': by_drivetime,
        'by_city': by_city,
        'quartile_accuracy': {
            'exact_match_rate': float(exact_match),
            'within_one_quartile_rate': float(within_one),
            'confusion_matrix': confusion.tolist()
        }
    }

    with open(CHATBOT_DATA / "model_performance.json", 'w') as f:
        json.dump(performance, f, indent=2)
    print(f"  Saved model_performance.json (R²={overall_r2:.3f}, RMSE={overall_rmse:.4f})")
else:
    print("  SKIPPED: Missing inputs")


# =============================================================================
# [5] GENERATE city_summary.csv
# =============================================================================

print("\n[5] Generating city_summary.csv...")

if train_raw is not None:
    city_col = 'mrkt_name' if 'mrkt_name' in train_raw.columns else 'city'
    state_col = 'state' if 'state' in train_raw.columns else None

    # Separate pre and post
    pre_data = train_raw[train_raw['time_window_tag'] == 'pre']
    post_data = train_raw[train_raw['time_window_tag'] == 'post']

    # Aggregate by city
    city_stats = []
    for city in train_raw[city_col].unique():
        pre_city = pre_data[pre_data[city_col] == city]['target']
        post_city = post_data[post_data[city_col] == city]['target']

        state = train_raw[train_raw[city_col] == city][state_col].iloc[0] if state_col else 'NA'

        city_stats.append({
            'city': city,
            'state': state,
            'property_count': train_raw[train_raw[city_col] == city]['UBID'].nunique(),
            'avg_revpar_growth_pre': pre_city.mean() if len(pre_city) > 0 else None,
            'avg_revpar_growth_post': post_city.mean() if len(post_city) > 0 else None,
            'median_revpar_growth_pre': pre_city.median() if len(pre_city) > 0 else None,
            'median_revpar_growth_post': post_city.median() if len(post_city) > 0 else None,
            'std_revpar_growth_pre': pre_city.std() if len(pre_city) > 0 else None,
            'std_revpar_growth_post': post_city.std() if len(post_city) > 0 else None,
        })

    city_df = pd.DataFrame(city_stats)
    city_df = city_df.sort_values('property_count', ascending=False)
    city_df.to_csv(CHATBOT_DATA / "city_summary.csv", index=False)
    print(f"  Saved city_summary.csv ({len(city_df)} cities)")
else:
    print("  SKIPPED: Missing train data")


# =============================================================================
# [6] GENERATE submarket_summary.csv
# =============================================================================

print("\n[6] Generating submarket_summary.csv...")

if train_raw is not None:
    city_col = 'mrkt_name' if 'mrkt_name' in train_raw.columns else 'city'
    submarket_col = 'submrkt_name' if 'submrkt_name' in train_raw.columns else None

    if submarket_col:
        pre_data = train_raw[train_raw['time_window_tag'] == 'pre']
        post_data = train_raw[train_raw['time_window_tag'] == 'post']

        submarket_stats = []
        for (city, submarket), group in train_raw.groupby([city_col, submarket_col]):
            pre_sub = pre_data[(pre_data[city_col] == city) & (pre_data[submarket_col] == submarket)]['target']
            post_sub = post_data[(post_data[city_col] == city) & (post_data[submarket_col] == submarket)]['target']

            avg_pre = pre_sub.mean() if len(pre_sub) > 0 else 0
            avg_post = post_sub.mean() if len(post_sub) > 0 else 0

            # Determine positioning
            if avg_post > 0.15:
                positioning = 'Premium'
            elif avg_post > 0.08:
                positioning = 'Upper-Mid'
            elif avg_post > 0:
                positioning = 'Mid-Market'
            else:
                positioning = 'Value-Oriented'

            submarket_stats.append({
                'city': city,
                'submarket': submarket,
                'property_count': group['UBID'].nunique(),
                'avg_revpar_growth_pre': avg_pre,
                'avg_revpar_growth_post': avg_post,
                'growth_change': avg_post - avg_pre,
                'positioning_label': positioning,
            })

        submarket_df = pd.DataFrame(submarket_stats)
        submarket_df = submarket_df.sort_values('property_count', ascending=False)
        submarket_df.to_csv(CHATBOT_DATA / "submarket_summary.csv", index=False)
        print(f"  Saved submarket_summary.csv ({len(submarket_df)} submarkets)")
    else:
        print("  SKIPPED: No submarket column found")
else:
    print("  SKIPPED: Missing train data")


# =============================================================================
# [7] GENERATE drivetime_analysis.csv
# =============================================================================

print("\n[7] Generating drivetime_analysis.csv...")

if oof_predictions and train_raw is not None:
    # Use the train_subset already created with correct holdout alignment
    train_drv_subset = train_subset  # Reuse properly aligned data
    y_true_drv = train_drv_subset['target'].values
    y_pred_drv = ensemble_pred

    drivetime_stats = []
    for drv in ['drv10', 'drv15', 'drv30']:
        mask = (train_drv_subset['trade_area_label'] == drv).values
        mask = mask & ~np.isnan(y_true_drv) & ~np.isnan(y_pred_drv)

        if mask.sum() > 0:
            r2 = r2_score(y_true_drv[mask], y_pred_drv[mask])
            rmse = np.sqrt(mean_squared_error(y_true_drv[mask], y_pred_drv[mask]))

            # Count amenities (columns containing '_ta' or 'num_' that are numeric)
            numeric_cols = train_drv_subset.select_dtypes(include=[np.number]).columns
            amenity_cols = [c for c in numeric_cols if '_ta' in c.lower() or 'num_' in c.lower()]
            avg_amenities = train_drv_subset.loc[mask, amenity_cols].mean().mean() if amenity_cols else 0

            # Determine best use case
            drv_min = drv.replace('drv', '')
            if drv_min == '10':
                best_for = 'downtown/urban core'
                explanation = 'Best for high-density urban core areas with walking/short trips'
            elif drv_min == '15':
                best_for = 'suburban/mixed'
                explanation = 'Best for mixed-use areas balancing convenience and reach'
            else:
                best_for = 'outer suburb/exurban'
                explanation = 'Best for outer suburbs where residents are used to longer commutes'

            drivetime_stats.append({
                'drivetime': drv_min + 'min',
                'model_r2': round(r2, 3),
                'model_rmse': round(rmse, 4),
                'avg_amenity_count': round(avg_amenities, 1),
                'best_for_category': best_for,
                'explanation': explanation
            })

    drivetime_df = pd.DataFrame(drivetime_stats)
    drivetime_df.to_csv(CHATBOT_DATA / "drivetime_analysis.csv", index=False)
    print(f"  Saved drivetime_analysis.csv ({len(drivetime_df)} rows)")
else:
    print("  SKIPPED: Missing inputs")


# =============================================================================
# [8] GENERATE amenity_analysis.csv
# =============================================================================

print("\n[8] Generating amenity_analysis.csv...")

if train_raw is not None:
    # Find amenity columns
    amenity_patterns = ['num_', 'food_count', 'food_total', 'food_nearest']
    amenity_cols = []
    for col in train_raw.columns:
        for pattern in amenity_patterns:
            if pattern in col.lower():
                amenity_cols.append(col)
                break

    if amenity_cols:
        pre_data = train_raw[train_raw['time_window_tag'] == 'pre']
        post_data = train_raw[train_raw['time_window_tag'] == 'post']

        amenity_stats = []
        for col in amenity_cols[:30]:  # Limit to 30 amenities
            # Correlations
            pre_corr = pre_data[[col, 'target']].dropna().corr().iloc[0, 1] if len(pre_data) > 10 else 0
            post_corr = post_data[[col, 'target']].dropna().corr().iloc[0, 1] if len(post_data) > 10 else 0

            # Determine category
            if 'food' in col.lower() or 'restaurant' in col.lower():
                category = 'food_dining'
            elif 'grocery' in col.lower():
                category = 'retail_essential'
            elif 'park' in col.lower():
                category = 'recreation'
            elif 'gas' in col.lower():
                category = 'convenience'
            elif 'social' in col.lower() or 'senior' in col.lower() or 'childcare' in col.lower():
                category = 'services'
            else:
                category = 'other'

            # Generate interpretation
            change = post_corr - pre_corr
            if change > 0.1:
                interpretation = f"{col} became more important post-COVID"
            elif change < -0.1:
                interpretation = f"{col} became less important post-COVID"
            else:
                interpretation = f"{col} importance remained stable"

            amenity_stats.append({
                'amenity_type': col,
                'amenity_category': category,
                'avg_count_10min': train_raw[train_raw['trade_area_label'] == 'drv10'][col].mean(),
                'avg_count_15min': train_raw[train_raw['trade_area_label'] == 'drv15'][col].mean(),
                'avg_count_30min': train_raw[train_raw['trade_area_label'] == 'drv30'][col].mean(),
                'correlation_with_revpar_pre': round(pre_corr, 3),
                'correlation_with_revpar_post': round(post_corr, 3),
                'correlation_change': round(change, 3),
                'interpretation': interpretation
            })

        amenity_df = pd.DataFrame(amenity_stats)
        amenity_df = amenity_df.sort_values('correlation_change', ascending=False)
        amenity_df.to_csv(CHATBOT_DATA / "amenity_analysis.csv", index=False)
        print(f"  Saved amenity_analysis.csv ({len(amenity_df)} amenities)")
    else:
        print("  SKIPPED: No amenity columns found")
else:
    print("  SKIPPED: Missing train data")


# =============================================================================
# [9] COPY/REFORMAT data_dictionary.csv
# =============================================================================

print("\n[9] Generating data_dictionary.csv...")

if data_dict is not None:
    # Add additional columns
    data_dict['is_feature'] = data_dict['column_name'].apply(
        lambda x: 'no' if x in ['UBID', 'id', 'target', 'REVPAR'] or 'REVPAR' in str(x) else 'yes'
    )
    data_dict['is_target'] = data_dict['column_name'].apply(
        lambda x: 'yes' if 'REVPAR_GROWTH' in str(x) and 'PCT' in str(x) else 'no'
    )
    data_dict['category'] = data_dict['scope'].fillna('other')

    # Rename columns to match guide
    data_dict = data_dict.rename(columns={
        'column_name': 'column_name',
        'description': 'description',
    })

    data_dict.to_csv(CHATBOT_DATA / "data_dictionary.csv", index=False)
    print(f"  Saved data_dictionary.csv ({len(data_dict)} columns)")
else:
    print("  SKIPPED: No dictionary found")


# =============================================================================
# [10] CREATE training_data.csv (symlink or copy)
# =============================================================================

print("\n[10] Creating training_data.csv...")

training_data_path = CHATBOT_DATA / "training_data.csv"
if train_raw is not None:
    # Save a sample (first 10000 rows to save space)
    train_raw.head(10000).to_csv(training_data_path, index=False)
    print(f"  Saved training_data.csv (10000 sample rows)")
else:
    print("  SKIPPED: No train data")


# =============================================================================
# VALIDATION & SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

required_files = [
    "key_findings.json",
    "feature_importance.csv",
    "city_summary.csv",
    "submarket_summary.csv",
    "predictions.csv",
    "model_performance.json",
    "drivetime_analysis.csv",
    "amenity_analysis.csv",
    "data_dictionary.csv",
    "training_data.csv",
]

all_present = True
for f in required_files:
    path = CHATBOT_DATA / f
    if path.exists():
        size = path.stat().st_size
        print(f"  ✅ {f} ({size:,} bytes)")
    else:
        print(f"  ❌ {f} MISSING")
        all_present = False

print("\n" + "=" * 60)
if all_present:
    print("✅ ALL FILES READY - Chatbot data generation complete!")
else:
    print("⚠️  Some files missing. Check notebook outputs.")
print("=" * 60)
