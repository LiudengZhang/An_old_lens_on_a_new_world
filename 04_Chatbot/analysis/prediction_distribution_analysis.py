#!/usr/bin/env python3
"""
Analysis of prediction vs actual value distributions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PRED_CSV = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/04_Chatbot/data/predictions.csv")
MODEL_PRED_DIR = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab/extra/outputs/predictions_oof")
OUTPUT_DIR = Path("/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/04_Chatbot/analysis/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(PRED_CSV)
print("="*60)
print("PREDICTION DISTRIBUTION ANALYSIS")
print("="*60)

# ============================================================================
# 1. OVERALL DISTRIBUTION COMPARISON
# ============================================================================
print("\n1. OVERALL STATISTICS")
print("-"*60)

actual = df['actual_revpar_growth']
predicted = df['predicted_revpar_growth']

print(f"{'Metric':<25} {'Actual':>15} {'Predicted':>15}")
print("-"*55)
print(f"{'Mean':<25} {actual.mean():>15.4f} {predicted.mean():>15.4f}")
print(f"{'Std':<25} {actual.std():>15.4f} {predicted.std():>15.4f}")
print(f"{'Min':<25} {actual.min():>15.4f} {predicted.min():>15.4f}")
print(f"{'25%':<25} {actual.quantile(0.25):>15.4f} {predicted.quantile(0.25):>15.4f}")
print(f"{'Median':<25} {actual.median():>15.4f} {predicted.median():>15.4f}")
print(f"{'75%':<25} {actual.quantile(0.75):>15.4f} {predicted.quantile(0.75):>15.4f}")
print(f"{'Max':<25} {actual.max():>15.4f} {predicted.max():>15.4f}")
print(f"{'Range':<25} {actual.max()-actual.min():>15.4f} {predicted.max()-predicted.min():>15.4f}")

# Variance ratio
var_ratio = predicted.var() / actual.var()
print(f"\nVariance Ratio (Pred/Actual): {var_ratio:.4f}")
print(f"  → Model captures {var_ratio*100:.1f}% of actual variance range")

# ============================================================================
# 2. IS MODEL PREDICTING NEAR-CONSTANT?
# ============================================================================
print("\n2. NEAR-CONSTANT PREDICTION CHECK")
print("-"*60)

pred_iqr = predicted.quantile(0.75) - predicted.quantile(0.25)
actual_iqr = actual.quantile(0.75) - actual.quantile(0.25)
print(f"Actual IQR: {actual_iqr:.4f}")
print(f"Predicted IQR: {pred_iqr:.4f}")
print(f"IQR Ratio (Pred/Actual): {pred_iqr/actual_iqr:.4f}")

# Count predictions within narrow bands
within_01 = ((predicted > -0.1) & (predicted < 0.1)).sum() / len(predicted) * 100
within_02 = ((predicted > -0.2) & (predicted < 0.2)).sum() / len(predicted) * 100
print(f"\nPredictions within [-0.1, 0.1]: {within_01:.1f}%")
print(f"Predictions within [-0.2, 0.2]: {within_02:.1f}%")

if var_ratio < 0.2:
    print("\n⚠️  WARNING: Model predicts near-constant values (low variance)")
else:
    print("\n✓ Model shows reasonable prediction variance")

# ============================================================================
# 3. ANALYSIS BY TIME_PERIOD
# ============================================================================
print("\n3. PREDICTIONS BY TIME_PERIOD (pre vs post)")
print("-"*60)

for period in df['time_period'].unique():
    subset = df[df['time_period'] == period]
    print(f"\n{period.upper()} period (n={len(subset)}):")
    print(f"  Actual:    mean={subset['actual_revpar_growth'].mean():>8.4f}, std={subset['actual_revpar_growth'].std():>8.4f}")
    print(f"  Predicted: mean={subset['predicted_revpar_growth'].mean():>8.4f}, std={subset['predicted_revpar_growth'].std():>8.4f}")
    print(f"  MAE: {subset['abs_error'].mean():.4f}")

# Statistical test
from scipy import stats
pre_pred = df[df['time_period']=='pre']['predicted_revpar_growth']
post_pred = df[df['time_period']=='post']['predicted_revpar_growth']
t_stat, p_val = stats.ttest_ind(pre_pred, post_pred)
print(f"\nT-test (pre vs post predictions): t={t_stat:.4f}, p={p_val:.4e}")
if p_val < 0.05:
    print("  → Predictions significantly differ between pre/post")
else:
    print("  → Predictions do NOT significantly differ between pre/post")

# ============================================================================
# 4. ANALYSIS BY DRIVETIME
# ============================================================================
print("\n4. PREDICTIONS BY DRIVETIME")
print("-"*60)

for dt in sorted(df['drivetime'].unique()):
    subset = df[df['drivetime'] == dt]
    print(f"\n{dt} (n={len(subset)}):")
    print(f"  Actual:    mean={subset['actual_revpar_growth'].mean():>8.4f}, std={subset['actual_revpar_growth'].std():>8.4f}")
    print(f"  Predicted: mean={subset['predicted_revpar_growth'].mean():>8.4f}, std={subset['predicted_revpar_growth'].std():>8.4f}")
    print(f"  MAE: {subset['abs_error'].mean():.4f}")

# ANOVA test
drivetime_groups = [df[df['drivetime']==dt]['predicted_revpar_growth'] for dt in df['drivetime'].unique()]
f_stat, p_val_anova = stats.f_oneway(*drivetime_groups)
print(f"\nANOVA (predictions across drivetimes): F={f_stat:.4f}, p={p_val_anova:.4e}")
if p_val_anova < 0.05:
    print("  → Predictions significantly differ across drivetimes")
else:
    print("  → Predictions do NOT significantly differ across drivetimes")

# ============================================================================
# 5. INDIVIDUAL MODEL PREDICTIONS
# ============================================================================
print("\n5. INDIVIDUAL MODEL PREDICTION DIVERSITY")
print("-"*60)

model_preds = {}
for f in MODEL_PRED_DIR.glob("*.npy"):
    model_name = f.stem
    preds = np.load(f)
    model_preds[model_name] = preds
    print(f"{model_name:15s}: mean={preds.mean():>8.4f}, std={preds.std():>8.4f}, range=[{preds.min():.4f}, {preds.max():.4f}]")

# Calculate correlation between model predictions
print("\n5b. CORRELATION BETWEEN MODELS")
print("-"*60)
model_names = list(model_preds.keys())
pred_matrix = np.column_stack([model_preds[m] for m in model_names])
corr_matrix = np.corrcoef(pred_matrix.T)

print("Pairwise correlations (top 5 most different):")
pairs = []
for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        pairs.append((model_names[i], model_names[j], corr_matrix[i,j]))
pairs.sort(key=lambda x: x[2])
for m1, m2, corr in pairs[:5]:
    print(f"  {m1} vs {m2}: r={corr:.4f}")
print("...")
for m1, m2, corr in pairs[-3:]:
    print(f"  {m1} vs {m2}: r={corr:.4f}")

avg_corr = np.mean([p[2] for p in pairs])
print(f"\nAverage pairwise correlation: {avg_corr:.4f}")
if avg_corr > 0.9:
    print("  → Models predict very similar values (low diversity)")
elif avg_corr > 0.7:
    print("  → Models show moderate diversity")
else:
    print("  → Models show high diversity")

# ============================================================================
# 6. VISUALIZATION
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 6a. Distribution comparison
ax = axes[0, 0]
ax.hist(actual, bins=40, alpha=0.6, label='Actual', color='blue', density=True)
ax.hist(predicted, bins=40, alpha=0.6, label='Predicted', color='red', density=True)
ax.axvline(actual.mean(), color='blue', linestyle='--', linewidth=2)
ax.axvline(predicted.mean(), color='red', linestyle='--', linewidth=2)
ax.set_xlabel('RevPAR Growth')
ax.set_ylabel('Density')
ax.set_title('Distribution: Actual vs Predicted')
ax.legend()

# 6b. Scatter plot
ax = axes[0, 1]
ax.scatter(actual, predicted, alpha=0.3, s=10)
ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Actual RevPAR Growth')
ax.set_ylabel('Predicted RevPAR Growth')
ax.set_title('Predicted vs Actual')
ax.legend()

# 6c. By time_period
ax = axes[0, 2]
positions = [0, 1]
bp_data = [df[df['time_period']=='pre']['predicted_revpar_growth'],
           df[df['time_period']=='post']['predicted_revpar_growth']]
bp = ax.boxplot(bp_data, positions=positions, patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_xticks(positions)
ax.set_xticklabels(['Pre', 'Post'])
ax.set_ylabel('Predicted RevPAR Growth')
ax.set_title('Predictions by Time Period')

# 6d. By drivetime
ax = axes[1, 0]
drivetimes = sorted(df['drivetime'].unique())
bp_data = [df[df['drivetime']==dt]['predicted_revpar_growth'] for dt in drivetimes]
bp = ax.boxplot(bp_data, patch_artist=True)
colors = ['lightblue', 'lightyellow', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(drivetimes)
ax.set_ylabel('Predicted RevPAR Growth')
ax.set_title('Predictions by Drivetime')

# 6e. Model comparison
ax = axes[1, 1]
model_means = [(m, model_preds[m].mean()) for m in model_names]
model_means.sort(key=lambda x: x[1])
models = [m[0] for m in model_means]
means = [m[1] for m in model_means]
stds = [model_preds[m].std() for m in models]
y_pos = np.arange(len(models))
ax.barh(y_pos, means, xerr=stds, alpha=0.7, capsize=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Mean Prediction')
ax.set_title('Individual Model Predictions')
ax.axvline(0, color='black', linestyle='-', alpha=0.3)

# 6f. Residual pattern
ax = axes[1, 2]
ax.scatter(predicted, df['residual'], alpha=0.3, s=10)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel('Predicted Value')
ax.set_ylabel('Residual (Actual - Predicted)')
ax.set_title('Residual vs Predicted')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_distribution_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nFigure saved: {OUTPUT_DIR / 'prediction_distribution_analysis.png'}")

# ============================================================================
# 7. SUMMARY DIAGNOSIS
# ============================================================================
print("\n" + "="*60)
print("SUMMARY DIAGNOSIS")
print("="*60)

issues = []
if var_ratio < 0.3:
    issues.append(f"Model captures only {var_ratio*100:.1f}% of actual variance")
if pred_iqr/actual_iqr < 0.5:
    issues.append(f"Predicted IQR is only {pred_iqr/actual_iqr*100:.1f}% of actual IQR")
if avg_corr > 0.85:
    issues.append(f"Models have high avg correlation ({avg_corr:.2f}) - low diversity")

if issues:
    print("\nISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ Model appears to be capturing reasonable variance")

# Correlation with actual
corr_with_actual = np.corrcoef(actual, predicted)[0,1]
print(f"\nCorrelation (predicted vs actual): {corr_with_actual:.4f}")
print(f"R-squared: {corr_with_actual**2:.4f}")

print("\n" + "="*60)
