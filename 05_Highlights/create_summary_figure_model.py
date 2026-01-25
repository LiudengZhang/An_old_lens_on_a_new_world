#!/usr/bin/env python3
"""
Rice Datathon 2026 - Publication-Quality Summary Figure
Using model feature importance instead of correlations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

NC_DOUBLE_COL_MM = 183
MM_TO_INCH = 1 / 25.4
NC_DOUBLE_COL = NC_DOUBLE_COL_MM * MM_TO_INCH

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
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def value_to_color(val, max_abs=None):
    """Map value to blue (negative) or orange (positive)."""
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

def get_healthcare_importance(X, y, n_bootstrap=20):
    """Train LightGBM and get healthcare feature importance with bootstrap CI."""
    healthcare_col = 'aarp_met_health_hospital'
    if healthcare_col not in X.columns:
        return None, None, None

    importances = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1,
                                   verbose=-1, n_jobs=-1, random_state=None)
        model.fit(X_boot, y_boot)

        feat_imp = dict(zip(X.columns, model.feature_importances_))
        total_imp = sum(model.feature_importances_)
        if total_imp > 0:
            importances.append(feat_imp.get(healthcare_col, 0) / total_imp)

    if len(importances) == 0:
        return None, None, None

    return np.mean(importances), np.percentile(importances, 2.5), np.percentile(importances, 97.5)

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')

# Load existing feature importance
feat_imp = pd.read_csv(f'{DATA_DIR}/extra/outputs/feature_importance.csv')
feat_imp_dict = dict(zip(feat_imp['feature'], feat_imp['avg_importance']))

pre_mask = train['is_pre_covid'] == 1
post_mask = train['is_post_covid'] == 1

# Prepare features for LightGBM
feature_cols = [c for c in train.columns if c not in ['target', 'target_clipped']
                and train[c].dtype in ['int64', 'float64']
                and train[c].isna().sum() / len(train) < 0.3]

apply_nature_style()

# =============================================================================
# CREATE FIGURE
# =============================================================================

fig = plt.figure(figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * 0.65))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                       height_ratios=[1, 1.1])

# =============================================================================
# Panel A: Healthcare Importance - Pre vs Post
# =============================================================================
print("Panel A: Computing healthcare importance pre vs post...")
ax_a = fig.add_subplot(gs[0, 0])

# Train models on pre and post separately
X_pre = train.loc[pre_mask, feature_cols].fillna(0)
y_pre = train.loc[pre_mask, 'target']
X_post = train.loc[post_mask, feature_cols].fillna(0)
y_post = train.loc[post_mask, 'target']

imp_pre, ci_pre_lo, ci_pre_hi = get_healthcare_importance(X_pre, y_pre)
imp_post, ci_post_lo, ci_post_hi = get_healthcare_importance(X_post, y_post)

# Convert to percentage
imp_pre *= 100
ci_pre_lo *= 100
ci_pre_hi *= 100
imp_post *= 100
ci_post_lo *= 100
ci_post_hi *= 100

max_abs_a = max(imp_pre, imp_post)
colors_a = [value_to_color(imp_pre, max_abs_a), value_to_color(imp_post, max_abs_a)]
yerr_a = [[imp_pre - ci_pre_lo, imp_post - ci_post_lo],
          [ci_pre_hi - imp_pre, ci_post_hi - imp_post]]

bars_a = ax_a.bar([0, 1], [imp_pre, imp_post], color=colors_a,
                  yerr=yerr_a, capsize=3, width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_a.axhline(y=0, color='black', linewidth=0.5)
ax_a.set_xticks([0, 1])
ax_a.set_xticklabels(['Pre-COVID\n(2015-2020)', 'Post-COVID\n(2022-2025)'])
ax_a.set_ylabel('Healthcare importance (%)')
ax_a.set_title('Healthcare importance increased', fontweight='bold')

print(f"  Pre: {imp_pre:.2f}% [{ci_pre_lo:.2f}, {ci_pre_hi:.2f}]")
print(f"  Post: {imp_post:.2f}% [{ci_post_lo:.2f}, {ci_post_hi:.2f}]")

# =============================================================================
# Panel B: Healthcare Importance by Segment
# =============================================================================
print("Panel B: Computing by segment...")
ax_b = fig.add_subplot(gs[0, 1])

post_data = train[post_mask].copy()

# Create segments
post_data['segment'] = 'Other'
senior_mask_s = (
    (post_data['aarp_met_health_hospital'] > post_data['aarp_met_health_hospital'].quantile(0.6)) &
    (post_data['areaperunit'] > 850) & (post_data['areaperunit'] < 1000) &
    (post_data['property_age'] > 25)
)
family_mask_s = (
    (post_data['areaperunit'] > 1000) &
    (post_data['msa_ring'].isin(['innersuburb', 'outersuburb']))
)
young_mask_s = (
    (post_data['areaperunit'] < 850) &
    (post_data['msa_ring'].isin(['downtown', 'outercore'])) &
    (post_data['property_age'] < 25)
)

post_data.loc[senior_mask_s, 'segment'] = 'Senior'
post_data.loc[family_mask_s & ~senior_mask_s, 'segment'] = 'Family'
post_data.loc[young_mask_s & ~senior_mask_s & ~family_mask_s, 'segment'] = 'Young Prof.'

segment_results_b = []
for seg in ['Young Prof.', 'Family', 'Senior']:
    subset = post_data[post_data['segment'] == seg]
    if len(subset) >= 100:
        X_seg = subset[feature_cols].fillna(0)
        y_seg = subset['target']
        imp, lo, hi = get_healthcare_importance(X_seg, y_seg)
        if imp is not None:
            segment_results_b.append({'Segment': seg, 'Importance': imp*100, 'CI_lo': lo*100, 'CI_hi': hi*100})
            print(f"  {seg}: {imp*100:.2f}%")

seg_df_b = pd.DataFrame(segment_results_b)
max_abs_b = seg_df_b['Importance'].max()
seg_colors_b = [value_to_color(s, max_abs_b) for s in seg_df_b['Importance']]
yerr_b = [seg_df_b['Importance'] - seg_df_b['CI_lo'], seg_df_b['CI_hi'] - seg_df_b['Importance']]

bars_b = ax_b.bar(range(len(seg_df_b)), seg_df_b['Importance'], color=seg_colors_b,
                  yerr=yerr_b, capsize=3, width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_b.set_xticks(range(len(seg_df_b)))
ax_b.set_xticklabels(seg_df_b['Segment'])
ax_b.axhline(y=0, color='black', linewidth=0.5)
ax_b.set_ylabel('Healthcare importance (%)')
ax_b.set_title('Segment-specific importance', fontweight='bold')

# =============================================================================
# Panel C: Top Features from Full Model
# =============================================================================
print("Panel C: Top features...")
ax_c = fig.add_subplot(gs[0, 2])

top_features = [
    ('health×post', feat_imp_dict.get('health_hospital_x_post', 0) * 100),
    ('time\nwindow', feat_imp_dict.get('time_window_tag', 0) * 100),
    ('market', feat_imp_dict.get('mrkt_name_te', 0) * 100),
    ('rent\nlevel', feat_imp_dict.get('rent_percentile', 0) * 100),
]

labels_c = [f[0] for f in top_features]
values_c = [f[1] for f in top_features]
max_abs_c = max(values_c)
colors_c = [value_to_color(v, max_abs_c) for v in values_c]

bars_c = ax_c.bar(range(len(labels_c)), values_c, color=colors_c,
                  width=0.6, edgecolor='black', linewidth=0.8)

ax_c.set_xticks(range(len(labels_c)))
ax_c.set_xticklabels(labels_c)
ax_c.set_ylabel('Feature importance (%)')
ax_c.set_title('Top model features', fontweight='bold')

# =============================================================================
# Panel D: Healthcare Importance by State
# =============================================================================
print("Panel D: Computing by state...")
ax_d = fig.add_subplot(gs[1, 0])

state_results = []
for state in train['state'].unique():
    state_post = train[(train['state'] == state) & post_mask]
    if len(state_post) >= 300:
        X_state = state_post[feature_cols].fillna(0)
        y_state = state_post['target']
        imp, lo, hi = get_healthcare_importance(X_state, y_state, n_bootstrap=10)
        if imp is not None:
            state_results.append({'State': state, 'Importance': imp*100})
            print(f"  {state}: {imp*100:.2f}%")

state_df = pd.DataFrame(state_results).sort_values('Importance', ascending=True)
max_abs_d = state_df['Importance'].max()
bar_colors_d = [value_to_color(s, max_abs_d) for s in state_df['Importance']]

bars_d = ax_d.barh(range(len(state_df)), state_df['Importance'],
                   color=bar_colors_d, height=0.6, edgecolor='black', linewidth=0.8)

ax_d.set_yticks(range(len(state_df)))
ax_d.set_yticklabels(state_df['State'])
ax_d.axvline(x=0, color='black', linewidth=0.5)
ax_d.set_xlabel('Healthcare importance (%)')
ax_d.set_title('Geographic variation', fontweight='bold')

# =============================================================================
# Panel E: Healthcare Importance by Property Type
# =============================================================================
print("Panel E: Computing by property type...")
ax_e = fig.add_subplot(gs[1, 1:])

post_data['class_group'] = post_data['type_main'].apply(lambda x: x[0] if pd.notna(x) else 'X')
post_data['location_type'] = post_data['msa_ring'].apply(
    lambda x: 'Urban' if x in ['downtown', 'outercore'] else 'Suburban'
)
post_data['age_group'] = pd.cut(post_data['property_age'],
                                 bins=[0, 15, 30, 100],
                                 labels=['Modern', 'Mid-age', 'Older'])

segment_results_e = []

# By Class
for cls in ['A', 'B', 'C']:
    subset = post_data[post_data['class_group'] == cls]
    if len(subset) >= 100:
        X_seg = subset[feature_cols].fillna(0)
        y_seg = subset['target']
        imp, lo, hi = get_healthcare_importance(X_seg, y_seg)
        if imp is not None:
            segment_results_e.append({'Category': 'Class', 'Segment': f'Class {cls}',
                                     'Importance': imp*100, 'CI_lo': lo*100, 'CI_hi': hi*100})
            print(f"  Class {cls}: {imp*100:.2f}%")

# By Location
for loc in ['Urban', 'Suburban']:
    subset = post_data[post_data['location_type'] == loc]
    if len(subset) >= 100:
        X_seg = subset[feature_cols].fillna(0)
        y_seg = subset['target']
        imp, lo, hi = get_healthcare_importance(X_seg, y_seg)
        if imp is not None:
            segment_results_e.append({'Category': 'Location', 'Segment': loc,
                                     'Importance': imp*100, 'CI_lo': lo*100, 'CI_hi': hi*100})
            print(f"  {loc}: {imp*100:.2f}%")

# By Age
for age in ['Modern', 'Mid-age', 'Older']:
    subset = post_data[post_data['age_group'] == age]
    if len(subset) >= 100:
        X_seg = subset[feature_cols].fillna(0)
        y_seg = subset['target']
        imp, lo, hi = get_healthcare_importance(X_seg, y_seg)
        if imp is not None:
            segment_results_e.append({'Category': 'Age', 'Segment': age,
                                     'Importance': imp*100, 'CI_lo': lo*100, 'CI_hi': hi*100})
            print(f"  {age}: {imp*100:.2f}%")

seg_results_df = pd.DataFrame(segment_results_e)

# Build grouped bar chart
categories = ['Class', 'Location', 'Age']
x_positions = []
x_labels = []
colors = []
values = []
yerr_lo = []
yerr_hi = []
pos = 0

max_abs_e = seg_results_df['Importance'].max()

for cat in categories:
    cat_data = seg_results_df[seg_results_df['Category'] == cat]
    for row in cat_data.itertuples():
        x_positions.append(pos)
        x_labels.append(row.Segment)
        colors.append(value_to_color(row.Importance, max_abs_e))
        values.append(row.Importance)
        yerr_lo.append(row.Importance - row.CI_lo)
        yerr_hi.append(row.CI_hi - row.Importance)
        pos += 1
    pos += 0.5

yerr_e = [yerr_lo, yerr_hi]
bars_e = ax_e.bar(x_positions, values, color=colors, yerr=yerr_e, capsize=3,
                  width=0.7, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_e.set_xticks(x_positions)
ax_e.set_xticklabels(x_labels, fontsize=6)
ax_e.axhline(y=0, color='black', linewidth=0.5)
ax_e.set_ylabel('Healthcare importance (%)')
ax_e.set_title('Which properties: healthcare matters most?', fontweight='bold')

# Add category labels
cat_positions = {'Class': 1, 'Location': 4.25, 'Age': 7.5}
for cat, xpos in cat_positions.items():
    ax_e.text(xpos, ax_e.get_ylim()[1] * 0.95, cat, ha='center', va='top',
              fontsize=7, fontweight='bold', color='#444444')

# =============================================================================
# SAVE
# =============================================================================

plt.savefig(f'{OUTPUT_DIR}/fig_summary_model.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}/fig_summary_model.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"\nSaved: {OUTPUT_DIR}/fig_summary_model.png")
print(f"Saved: {OUTPUT_DIR}/fig_summary_model.pdf")

print("\n=== Summary ===")
print(seg_results_df.to_string(index=False))
print("\nDone!")
