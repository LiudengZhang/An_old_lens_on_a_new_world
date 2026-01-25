#!/usr/bin/env python3
"""
Rice Datathon 2026 - Publication-Quality Summary Figure
Using SHAP values from the model instead of correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# NATURE CANCER STYLE CONFIGURATION
# =============================================================================

NC_DOUBLE_COL_MM = 183
MM_TO_INCH = 1 / 25.4
DPI = 300
NC_DOUBLE_COL = NC_DOUBLE_COL_MM * MM_TO_INCH

COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
}

def apply_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'axes.linewidth': 0.5,
        'pdf.fonttype': 42,
        'savefig.dpi': DPI,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def value_to_color(val, max_abs=None):
    """Map value to blue (negative) or orange (positive) with intensity."""
    import matplotlib.colors as mcolors
    if max_abs is None:
        max_abs = abs(val) if val != 0 else 1
    norm_val = np.clip(val / max_abs, -1, 1)
    blue = np.array(mcolors.to_rgb('#0072B2'))
    orange = np.array(mcolors.to_rgb('#E69F00'))
    white = np.array([1.0, 1.0, 1.0])
    if norm_val >= 0:
        return white * (1 - norm_val) + orange * norm_val
    else:
        return white * (1 + norm_val) + blue * (-norm_val)

# =============================================================================
# LOAD DATA AND COMPUTE SHAP
# =============================================================================

DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')

# Load feature importance (model-based)
feat_imp = pd.read_csv(f'{DATA_DIR}/extra/outputs/feature_importance.csv')
feat_imp_dict = dict(zip(feat_imp['feature'], feat_imp['avg_importance']))

# Define masks
pre_mask = train['is_pre_covid'] == 1
post_mask = train['is_post_covid'] == 1
post_data = train[post_mask].copy()

print("Computing SHAP values using LightGBM...")

# Train a LightGBM model for SHAP (faster and cleaner than AutoGluon for SHAP)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import shap

# Prepare features (use numeric columns only for simplicity)
feature_cols = [c for c in train.columns if c not in ['target', 'target_clipped']
                and train[c].dtype in ['int64', 'float64']]

# Remove columns with too many NaN
feature_cols = [c for c in feature_cols if train[c].isna().sum() / len(train) < 0.3]

X = train[feature_cols].fillna(0)
y = train['target']

# Train LightGBM
print(f"Training LightGBM on {len(feature_cols)} features...")
model = lgb.LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          verbose=-1, n_jobs=-1, random_state=42)
model.fit(X, y)

# Compute SHAP values
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Create DataFrame with SHAP values
shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=train.index)

# Add metadata
shap_df['is_pre_covid'] = train['is_pre_covid'].values
shap_df['is_post_covid'] = train['is_post_covid'].values

# Healthcare feature
healthcare_col = 'aarp_met_health_hospital'
if healthcare_col not in shap_df.columns:
    print(f"Warning: {healthcare_col} not in SHAP columns")
    healthcare_col = [c for c in shap_df.columns if 'health_hospital' in c.lower()][0]

print(f"Using healthcare column: {healthcare_col}")

apply_nature_style()

# =============================================================================
# CREATE FIGURE
# =============================================================================

fig = plt.figure(figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * 0.65))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                       height_ratios=[1, 1.1])

# =============================================================================
# Panel A: Healthcare SHAP - Pre vs Post COVID
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])

shap_pre = shap_df.loc[pre_mask, healthcare_col].mean()
shap_post = shap_df.loc[post_mask, healthcare_col].mean()

# Bootstrap CI for SHAP means
def bootstrap_mean_ci(values, n_boot=1000):
    values = values.dropna().values
    means = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)

shap_pre_mean, shap_pre_lo, shap_pre_hi = bootstrap_mean_ci(shap_df.loc[pre_mask, healthcare_col])
shap_post_mean, shap_post_lo, shap_post_hi = bootstrap_mean_ci(shap_df.loc[post_mask, healthcare_col])

max_abs_a = max(abs(shap_pre_mean), abs(shap_post_mean))
colors_a = [value_to_color(shap_pre_mean, max_abs_a), value_to_color(shap_post_mean, max_abs_a)]
yerr_a = [[shap_pre_mean - shap_pre_lo, shap_post_mean - shap_post_lo],
          [shap_pre_hi - shap_pre_mean, shap_post_hi - shap_post_mean]]

bars_a = ax_a.bar([0, 1], [shap_pre_mean, shap_post_mean], color=colors_a,
                  yerr=yerr_a, capsize=3, width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_a.axhline(y=0, color='black', linewidth=0.5)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(['Pre-COVID\n(2015-2020)', 'Post-COVID\n(2022-2025)'])
ax_a.set_ylabel('Healthcare SHAP\n(contribution to RevPAR)')
ax_a.set_title('Healthcare effect flipped', fontweight='bold')

# =============================================================================
# Panel B: Healthcare SHAP by Segment
# =============================================================================
ax_b = fig.add_subplot(gs[0, 1])

# Create segments in post_data
post_indices = train[post_mask].index
post_data_shap = shap_df.loc[post_indices].copy()
post_data_shap['segment'] = 'Other'

# Use same segment definitions
senior_mask_idx = (
    (train.loc[post_indices, 'aarp_met_health_hospital'] > train.loc[post_indices, 'aarp_met_health_hospital'].quantile(0.6)) &
    (train.loc[post_indices, 'areaperunit'] > 850) & (train.loc[post_indices, 'areaperunit'] < 1000) &
    (train.loc[post_indices, 'property_age'] > 25)
)
family_mask_idx = (
    (train.loc[post_indices, 'areaperunit'] > 1000) &
    (train.loc[post_indices, 'msa_ring'].isin(['innersuburb', 'outersuburb']))
)
young_mask_idx = (
    (train.loc[post_indices, 'areaperunit'] < 850) &
    (train.loc[post_indices, 'msa_ring'].isin(['downtown', 'outercore'])) &
    (train.loc[post_indices, 'property_age'] < 25)
)

post_data_shap.loc[senior_mask_idx[senior_mask_idx].index, 'segment'] = 'Senior'
post_data_shap.loc[family_mask_idx[family_mask_idx].index & ~senior_mask_idx[senior_mask_idx].index, 'segment'] = 'Family'
post_data_shap.loc[young_mask_idx[young_mask_idx].index & ~senior_mask_idx[senior_mask_idx].index & ~family_mask_idx[family_mask_idx].index, 'segment'] = 'Young Prof.'

segment_shap = []
for seg in ['Young Prof.', 'Family', 'Senior']:
    subset = post_data_shap[post_data_shap['segment'] == seg]
    if len(subset) >= 100:
        mean, lo, hi = bootstrap_mean_ci(subset[healthcare_col])
        segment_shap.append({'Segment': seg, 'SHAP': mean, 'CI_lo': lo, 'CI_hi': hi})

seg_df = pd.DataFrame(segment_shap)
max_abs_b = seg_df['SHAP'].abs().max()
seg_colors = [value_to_color(s, max_abs_b) for s in seg_df['SHAP']]
yerr_b = [seg_df['SHAP'] - seg_df['CI_lo'], seg_df['CI_hi'] - seg_df['SHAP']]

bars_b = ax_b.bar(range(len(seg_df)), seg_df['SHAP'], color=seg_colors,
                  yerr=yerr_b, capsize=3, width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_b.set_xticks(range(len(seg_df)))
ax_b.set_xticklabels(seg_df['Segment'])
ax_b.axhline(y=0, color='black', linewidth=0.5)
ax_b.set_ylabel('Healthcare SHAP')
ax_b.set_title('Young professionals drive effect', fontweight='bold')

# =============================================================================
# Panel C: Feature Importance - Healthcare vs Others (Ablation-like)
# =============================================================================
ax_c = fig.add_subplot(gs[0, 2])

# Show top features importance from model
top_features = [
    ('health_hospital\n×post', feat_imp_dict.get('health_hospital_x_post', 0)),
    ('healthcare\naccess', feat_imp_dict.get('aarp_met_health_hospital', 0)),
    ('rent\npercentile', feat_imp_dict.get('rent_percentile', 0)),
    ('age×post', feat_imp_dict.get('age_x_post', 0)),
]

labels_c = [f[0] for f in top_features]
values_c = [f[1] * 100 for f in top_features]  # Convert to percentage
max_abs_c = max(values_c)
colors_c = [value_to_color(v, max_abs_c) for v in values_c]

bars_c = ax_c.bar(range(len(labels_c)), values_c, color=colors_c,
                  width=0.6, edgecolor='black', linewidth=0.8)

ax_c.set_xticks(range(len(labels_c)))
ax_c.set_xticklabels(labels_c)
ax_c.set_ylabel('Feature importance (%)')
ax_c.set_title('Healthcare dominates model', fontweight='bold')

# =============================================================================
# Panel D: Healthcare SHAP by State
# =============================================================================
ax_d = fig.add_subplot(gs[1, 0])

state_shap = []
for state in train['state'].unique():
    pre_idx = (train['state'] == state) & pre_mask
    post_idx = (train['state'] == state) & post_mask
    if pre_idx.sum() >= 500 and post_idx.sum() >= 500:
        shap_pre_state = shap_df.loc[pre_idx, healthcare_col].mean()
        shap_post_state = shap_df.loc[post_idx, healthcare_col].mean()
        shift = shap_post_state - shap_pre_state
        state_shap.append({'State': state, 'Shift': shift})

state_df = pd.DataFrame(state_shap).sort_values('Shift', ascending=True)
max_abs_d = state_df['Shift'].abs().max()
bar_colors_d = [value_to_color(s, max_abs_d) for s in state_df['Shift']]

bars_d = ax_d.barh(range(len(state_df)), state_df['Shift'],
                   color=bar_colors_d, height=0.6, edgecolor='black', linewidth=0.8)

ax_d.set_yticks(range(len(state_df)))
ax_d.set_yticklabels(state_df['State'])
ax_d.axvline(x=0, color='black', linewidth=0.5)
ax_d.set_xlabel('SHAP shift (post - pre)')
ax_d.set_title('Geographic variation', fontweight='bold')

# =============================================================================
# Panel E: Healthcare SHAP by Property Type
# =============================================================================
ax_e = fig.add_subplot(gs[1, 1:])

# Create property segments
post_data_shap['class_group'] = train.loc[post_indices, 'type_main'].apply(lambda x: x[0] if pd.notna(x) else 'X')
post_data_shap['location_type'] = train.loc[post_indices, 'msa_ring'].apply(
    lambda x: 'Urban' if x in ['downtown', 'outercore'] else 'Suburban'
)
post_data_shap['age_group'] = pd.cut(train.loc[post_indices, 'property_age'],
                                      bins=[0, 15, 30, 100],
                                      labels=['Modern', 'Mid-age', 'Older'])

segment_results = []

# By Class
for cls in ['A', 'B', 'C']:
    subset = post_data_shap[post_data_shap['class_group'] == cls]
    if len(subset) >= 100:
        mean, lo, hi = bootstrap_mean_ci(subset[healthcare_col])
        segment_results.append({'Category': 'Class', 'Segment': f'Class {cls}',
                               'SHAP': mean, 'CI_lo': lo, 'CI_hi': hi})

# By Location
for loc in ['Urban', 'Suburban']:
    subset = post_data_shap[post_data_shap['location_type'] == loc]
    if len(subset) >= 100:
        mean, lo, hi = bootstrap_mean_ci(subset[healthcare_col])
        segment_results.append({'Category': 'Location', 'Segment': loc,
                               'SHAP': mean, 'CI_lo': lo, 'CI_hi': hi})

# By Age
for age in ['Modern', 'Mid-age', 'Older']:
    subset = post_data_shap[post_data_shap['age_group'] == age]
    if len(subset) >= 100:
        mean, lo, hi = bootstrap_mean_ci(subset[healthcare_col])
        segment_results.append({'Category': 'Age', 'Segment': age,
                               'SHAP': mean, 'CI_lo': lo, 'CI_hi': hi})

seg_results_df = pd.DataFrame(segment_results)

# Build grouped bar chart
categories = ['Class', 'Location', 'Age']
x_positions = []
x_labels = []
colors = []
values = []
yerr_lo = []
yerr_hi = []
pos = 0

max_abs_e = seg_results_df['SHAP'].abs().max()

for cat in categories:
    cat_data = seg_results_df[seg_results_df['Category'] == cat]
    for i, row in enumerate(cat_data.itertuples()):
        x_positions.append(pos)
        x_labels.append(row.Segment)
        colors.append(value_to_color(row.SHAP, max_abs_e))
        values.append(row.SHAP)
        yerr_lo.append(row.SHAP - row.CI_lo)
        yerr_hi.append(row.CI_hi - row.SHAP)
        pos += 1
    pos += 0.5

yerr_e = [yerr_lo, yerr_hi]
bars_e = ax_e.bar(x_positions, values, color=colors, yerr=yerr_e, capsize=3,
                  width=0.7, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_e.set_xticks(x_positions)
ax_e.set_xticklabels(x_labels, fontsize=6)
ax_e.axhline(y=0, color='black', linewidth=0.5)
ax_e.set_ylabel('Healthcare SHAP')
ax_e.set_title('Which properties benefit most from healthcare access?', fontweight='bold')

# Add category labels
cat_positions = {'Class': 1, 'Location': 4.25, 'Age': 7.5}
for cat, xpos in cat_positions.items():
    ax_e.text(xpos, ax_e.get_ylim()[1] * 0.95, cat, ha='center', va='top',
              fontsize=7, fontweight='bold', color='#444444')

# =============================================================================
# SAVE
# =============================================================================

plt.savefig(f'{OUTPUT_DIR}/fig_summary_shap.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}/fig_summary_shap.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\nSaved: {OUTPUT_DIR}/fig_summary_shap.png")
print(f"Saved: {OUTPUT_DIR}/fig_summary_shap.pdf")

# Print statistics
print("\n=== Panel A: Pre vs Post SHAP ===")
print(f"Pre-COVID:  {shap_pre_mean:.6f} [{shap_pre_lo:.6f}, {shap_pre_hi:.6f}]")
print(f"Post-COVID: {shap_post_mean:.6f} [{shap_post_lo:.6f}, {shap_post_hi:.6f}]")

print("\n=== Panel E: Property Segments ===")
print(seg_results_df.to_string(index=False))

print("\nDone!")
