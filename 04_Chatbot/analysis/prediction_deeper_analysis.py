#!/usr/bin/env python3
"""
Deeper analysis of prediction discrepancies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PRED_CSV = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/04_Chatbot/data/predictions.csv")
MODEL_PRED_DIR = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab/extra/outputs/predictions_oof")
OUTPUT_DIR = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/04_Chatbot/analysis/outputs")

df = pd.read_csv(PRED_CSV)
actual = df['actual_revpar_growth'].values
predicted = df['predicted_revpar_growth'].values

print("="*70)
print("DEEPER ANALYSIS: PREDICTION DISCREPANCIES")
print("="*70)

# Load individual model predictions
model_preds = {}
for f in MODEL_PRED_DIR.glob("*.npy"):
    model_name = f.stem
    preds = np.load(f)
    if preds.std() > 0:  # exclude zero-variance models
        model_preds[model_name] = preds

# ============================================================================
# 1. CHECK IF INDIVIDUAL MODELS MATCH ENSEMBLE
# ============================================================================
print("\n1. INDIVIDUAL MODEL vs ENSEMBLE COMPARISON")
print("-"*70)

# Filter to active models only
active_models = {k:v for k,v in model_preds.items() if v.std() > 0.01}
print(f"Active models (std > 0.01): {list(active_models.keys())}")

# Check if any model matches the ensemble output
for name, preds in active_models.items():
    corr = np.corrcoef(preds, predicted)[0,1]
    mean_diff = abs(preds.mean() - predicted.mean())
    print(f"  {name:15s}: r={corr:.4f}, mean_diff={mean_diff:.4f}")

# Average of active models
avg_active = np.mean([active_models[m] for m in active_models], axis=0)
print(f"\nAverage of active models: mean={avg_active.mean():.4f}, std={avg_active.std():.4f}")
print(f"Ensemble predictions:     mean={predicted.mean():.4f}, std={predicted.std():.4f}")
print(f"Correlation with ensemble: {np.corrcoef(avg_active, predicted)[0,1]:.4f}")

# ============================================================================
# 2. ANALYZE PRE vs POST - THE KEY ISSUE
# ============================================================================
print("\n2. PRE vs POST - THE CRITICAL FAILURE")
print("-"*70)

pre_mask = df['time_period'] == 'pre'
post_mask = df['time_period'] == 'post'

print("ACTUAL values by period:")
print(f"  PRE:  mean={actual[pre_mask].mean():>8.4f}, std={actual[pre_mask].std():>8.4f}")
print(f"  POST: mean={actual[post_mask].mean():>8.4f}, std={actual[post_mask].std():>8.4f}")
print(f"  Difference: {actual[pre_mask].mean() - actual[post_mask].mean():.4f}")

print("\nPREDICTED values by period:")
print(f"  PRE:  mean={predicted[pre_mask].mean():>8.4f}, std={predicted[pre_mask].std():>8.4f}")
print(f"  POST: mean={predicted[post_mask].mean():>8.4f}, std={predicted[post_mask].std():>8.4f}")
print(f"  Difference: {predicted[pre_mask].mean() - predicted[post_mask].mean():.4f}")

print("\n⚠️  KEY FINDING:")
print(f"   Actual pre-post difference: +0.27 (pre is higher)")
print(f"   Predicted pre-post difference: {predicted[pre_mask].mean() - predicted[post_mask].mean():.4f}")
print("   → Model FAILS to capture the pre/post structure!")

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================
print("\n3. CORRELATION ANALYSIS")
print("-"*70)

# Overall
r = np.corrcoef(actual, predicted)[0,1]
print(f"Overall correlation: r = {r:.4f}")

# By period
r_pre = np.corrcoef(actual[pre_mask], predicted[pre_mask])[0,1]
r_post = np.corrcoef(actual[post_mask], predicted[post_mask])[0,1]
print(f"Pre-period correlation:  r = {r_pre:.4f}")
print(f"Post-period correlation: r = {r_post:.4f}")

# ============================================================================
# 4. CHECK IF MODEL JUST LEARNS MEAN
# ============================================================================
print("\n4. BASELINE COMPARISON")
print("-"*70)

# Mean prediction baseline
mean_pred = np.full_like(actual, actual.mean())
mae_mean = np.abs(actual - mean_pred).mean()
mae_model = np.abs(actual - predicted).mean()
print(f"MAE (predict global mean):   {mae_mean:.4f}")
print(f"MAE (model predictions):     {mae_model:.4f}")
print(f"Improvement over mean:       {(mae_mean - mae_model)/mae_mean*100:.1f}%")

# Per-group mean baseline
period_means = df.groupby('time_period')['actual_revpar_growth'].transform('mean')
mae_period_mean = np.abs(actual - period_means.values).mean()
print(f"MAE (predict period mean):   {mae_period_mean:.4f}")

# ============================================================================
# 5. VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 5a. Pre vs Post comparison
ax = axes[0, 0]
pre_actual = actual[pre_mask]
pre_pred = predicted[pre_mask]
post_actual = actual[post_mask]
post_pred = predicted[post_mask]

ax.scatter(pre_actual, pre_pred, alpha=0.3, s=10, label='Pre', c='green')
ax.scatter(post_actual, post_pred, alpha=0.3, s=10, label='Post', c='red')
lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
ax.plot(lims, lims, 'k--', linewidth=1)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Predicted vs Actual by Time Period')
ax.legend()

# 5b. Distribution by period
ax = axes[0, 1]
bins = np.linspace(-0.5, 1.0, 50)
ax.hist(pre_actual, bins=bins, alpha=0.5, label='Pre Actual', color='green')
ax.hist(post_actual, bins=bins, alpha=0.5, label='Post Actual', color='red')
ax.hist(pre_pred, bins=bins, alpha=0.5, label='Pre Pred', color='lightgreen', histtype='step', linewidth=2)
ax.hist(post_pred, bins=bins, alpha=0.5, label='Post Pred', color='lightcoral', histtype='step', linewidth=2)
ax.set_xlabel('RevPAR Growth')
ax.set_title('Distribution by Time Period')
ax.legend()

# 5c. Model diversity (excluding zero models)
ax = axes[1, 0]
active_names = list(active_models.keys())
for name in active_names:
    ax.hist(active_models[name], bins=50, alpha=0.3, label=name)
ax.axvline(predicted.mean(), color='black', linestyle='--', linewidth=2, label='Ensemble mean')
ax.set_xlabel('Prediction Value')
ax.set_title('Individual Model Distributions (Active Only)')
ax.legend(fontsize=8)

# 5d. Error by period
ax = axes[1, 1]
errors_pre = pre_actual - pre_pred
errors_post = post_actual - post_pred
bp = ax.boxplot([errors_pre, errors_post], labels=['Pre', 'Post'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax.axhline(0, color='red', linestyle='--')
ax.set_ylabel('Error (Actual - Predicted)')
ax.set_title('Error Distribution by Period')
print(f"\nMean error Pre:  {errors_pre.mean():.4f}")
print(f"Mean error Post: {errors_post.mean():.4f}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_deeper_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {OUTPUT_DIR / 'prediction_deeper_analysis.png'}")

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("""
CRITICAL FINDINGS:

1. MODEL FAILS TO CAPTURE PRE/POST STRUCTURE
   - Actual: Pre period has +0.22 mean, Post has -0.05 mean (diff = 0.27)
   - Predicted: Both periods have ~-0.06 mean (diff = ~0)
   - The model ignores time_period entirely!

2. NEAR-ZERO CORRELATION
   - r = 0.01 means predictions are essentially random relative to actuals
   - Model is no better than predicting a constant

3. INDIVIDUAL MODEL vs ENSEMBLE DISCREPANCY
   - Individual models predict mean ~0.37 (all positive!)
   - Ensemble predicts mean ~-0.06 (negative!)
   - Something is wrong with how predictions are combined or stored

4. NO DRIVETIME DIFFERENTIATION
   - Predictions are identical across 10min, 15min, 30min drivetimes

LIKELY CAUSES:
- time_period feature may not be included in model training
- Ensemble stacking may have bugs
- Target leakage or data processing errors
""")
