#!/usr/bin/env python3
"""
Rice Datathon 2026 - 05.1_Highlights
Combined 2-Panel Figure (Nature Cancer Style):
  Panel A: Feature Importance (Post-COVID Interactions vs Base)
  Panel B: Model Confidence by Asset Type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# NATURE CANCER STYLE CONFIGURATION
# =============================================================================
NC_SINGLE_COL_MM = 89
NC_DOUBLE_COL_MM = 183
MM_TO_INCH = 1 / 25.4
DPI = 300

NC_DOUBLE_COL = NC_DOUBLE_COL_MM * MM_TO_INCH

# Colorblind-safe palette (Okabe-Ito) - matching reference figure
COLORS = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'dark_gray': '#444444',
    'light_orange': '#F5D78E',
}

SEED = 42

# Paths
PROJECT_ROOT = Path('/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon')
DATA_DIR = PROJECT_ROOT / '03_Rice_Datathon_Colab'
OUTPUT_DIR = PROJECT_ROOT / '05.1_Highlights'


def apply_nature_style():
    """Apply Nature Cancer figure style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'legend.title_fontsize': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.75,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': DPI,
        'figure.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def importance_to_color(is_interaction):
    """Map importance to color. Interactions = yellow, base = blue."""
    if is_interaction:
        return COLORS['orange']  # Yellow-orange
    else:
        return COLORS['blue']  # Blue


def get_category_color(category):
    """Get color for each category - matching heatmap style palettes."""
    # Class: Blue tones (from RdBu)
    # Location: Red/vermillion tones (from RdBu)
    # Age: Green tones (from PiYG)
    CATEGORY_COLORS = {
        'Class': '#4393C3',      # Blue (from RdBu)
        'Location': '#D6604D',   # Red (from RdBu)
        'Age': '#4DAC26',        # Green (from PiYG)
    }
    return CATEGORY_COLORS.get(category, '#999999')


# =============================================================================
# LOAD DATA
# =============================================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

train = pd.read_csv(DATA_DIR / 'data/processed/train_clean.csv')
importance_df = pd.read_csv(DATA_DIR / 'extra/outputs/feature_importance.csv')

print(f"Total records: {len(train)}")
print(f"Features with importance: {len(importance_df)}")


# =============================================================================
# PREPARE PANEL A DATA: Feature Importance
# =============================================================================
print("\n--- Preparing Panel A: Feature Importance ---")

# Identify interaction terms
importance_df['is_interaction'] = importance_df['feature'].str.contains('_x_post|_x_class|_x_sunbelt', regex=True)
importance_df['is_time_indicator'] = importance_df['feature'].isin([
    'time_window_tag', 'time_window_label', 'is_pre_covid', 'is_post_covid'
])

interaction_features = importance_df[importance_df['is_interaction']].copy()
base_features = importance_df[~importance_df['is_interaction'] & ~importance_df['is_time_indicator']].copy()

# Display name mapping
DISPLAY_NAMES = {
    'health_hospital_x_post': 'Hospital Access\n× Post-COVID',
    'age_x_post': 'Property Age\n× Post-COVID',
    'age_x_class_d': 'Age × Class D',
    'value_x_sunbelt': 'Value × Sun Belt',
    'suburb_tight_x_post': 'Suburban × Post',
    'mrkt_name_te': 'Market',
    'rent_percentile': 'Rent Percentile',
    'zip_metro': 'Zip-Metro',
    'state_te': 'State',
    'aarp_met_health_hospital': 'Hospital Access',
    'aarp_met_house_access_step': 'Housing Access',
    'type_main_te': 'Property Type',
    'numunits': 'Unit Count',
    'aarp_met_health_smoke': 'Smoking Rate',
    'aarp_met_health_obese': 'Obesity Rate',
    'yearbuilt': 'Year Built',
    'healthcare_access': 'Healthcare Index',
    'property_age': 'Property Age',
}

def get_display_name(feat):
    if feat in DISPLAY_NAMES:
        return DISPLAY_NAMES[feat]
    name = feat.replace('_', ' ').replace('aarp met ', '').replace('aarp score ', '')
    return name.title()[:18]

# Select top features
top_interactions = interaction_features.nlargest(3, 'avg_importance')
top_base = base_features.nlargest(7, 'avg_importance')

plot_data_a = pd.concat([top_interactions, top_base]).copy()
plot_data_a['display'] = plot_data_a['feature'].apply(get_display_name)
plot_data_a = plot_data_a.sort_values('avg_importance', ascending=True)

print(f"  Interactions: {len(top_interactions)}, Base: {len(top_base)}")


# =============================================================================
# PREPARE PANEL B DATA: Model Confidence (Uncertainty)
# =============================================================================
print("\n--- Preparing Panel B: Model Confidence ---")

# Load OOF predictions
OOF_DIR = DATA_DIR / 'extra/outputs/predictions_oof'
MODELS = ['lgb', 'xgb', 'cat', 'hist', 'extra_trees', 'ridge', 'elasticnet', 'knn']

oof_preds = {}
for model in MODELS:
    path = OOF_DIR / f'{model}.npy'
    if path.exists():
        oof_preds[model] = np.load(path)

# Recreate holdout split
HOLDOUT_FRACTION = 0.15
train_raw = pd.read_csv(DATA_DIR / 'data/raw/train.csv')
ubids = train_raw['UBID'].unique()
np.random.seed(SEED)
np.random.shuffle(ubids)
n_holdout = int(len(ubids) * HOLDOUT_FRACTION)
holdout_ubids = set(ubids[:n_holdout])
train_mask = ~train_raw['UBID'].isin(holdout_ubids)

train_oof = train[train_mask.values].reset_index(drop=True)
print(f"  Train OOF subset: {len(train_oof)}")

# Compute ensemble uncertainty
pred_matrix = np.column_stack([oof_preds[m] for m in oof_preds.keys()])
ensemble_std = pred_matrix.std(axis=1)

train_oof['ensemble_std'] = ensemble_std

# Define segments
train_oof['class_group'] = train_oof['type_main'].apply(lambda x: x[0] if pd.notna(x) and len(str(x)) > 0 else 'X')
train_oof['location'] = train_oof['msa_ring'].apply(
    lambda x: 'Urban' if x in ['downtown', 'outercore'] else 'Suburban' if x in ['innersuburb', 'outersuburb'] else 'Other'
)
train_oof['age_group'] = pd.cut(train_oof['property_age'],
                                 bins=[0, 15, 30, 100],
                                 labels=['Modern', 'Mid-age', 'Older'])

def calc_uncertainty_with_ci(df, group_col, n_bootstrap=500):
    """Calculate mean uncertainty with bootstrap CI."""
    results = []
    for name, group in df.groupby(group_col):
        if len(group) >= 50:
            vals = group['ensemble_std'].values
            mean_val = vals.mean()

            # Bootstrap CI
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(vals, size=len(vals), replace=True)
                boot_means.append(sample.mean())
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)

            results.append({
                'segment': name,
                'n': len(group),
                'uncertainty': mean_val,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
            })
    return pd.DataFrame(results)

# Calculate for each dimension
class_metrics = calc_uncertainty_with_ci(train_oof, 'class_group')
class_metrics = class_metrics[class_metrics['segment'].isin(['A', 'B', 'C'])]
class_metrics['segment'] = 'Class ' + class_metrics['segment']
class_metrics['category'] = 'Class'

location_metrics = calc_uncertainty_with_ci(train_oof, 'location')
location_metrics = location_metrics[location_metrics['segment'] != 'Other']
location_metrics['category'] = 'Location'

age_metrics = calc_uncertainty_with_ci(train_oof, 'age_group')
age_metrics['category'] = 'Age'

all_metrics = pd.concat([class_metrics, location_metrics, age_metrics], ignore_index=True)
median_uncertainty = all_metrics['uncertainty'].median()

print(f"  Segments: {len(all_metrics)}")


# =============================================================================
# CREATE COMBINED FIGURE
# =============================================================================
print("\n--- Creating Combined Figure ---")
apply_nature_style()

fig = plt.figure(figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * 0.45))
gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4, width_ratios=[1, 1.2])


# -----------------------------------------------------------------------------
# Panel A: Feature Importance (RIGHT)
# -----------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 1])

y_pos = np.arange(len(plot_data_a))
bar_colors_a = [importance_to_color(row['is_interaction'])
                for _, row in plot_data_a.iterrows()]

bars_a = ax_a.barh(y_pos, plot_data_a['avg_importance'],
                   color=bar_colors_a, edgecolor='black', linewidth=0.8, height=0.7)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(plot_data_a['display'])
ax_a.set_xlabel('Feature importance')
ax_a.set_title('What drives RevPAR prediction?', fontweight='bold')

# Ensure spines match Nature Cancer style
ax_a.spines['left'].set_linewidth(0.5)
ax_a.spines['bottom'].set_linewidth(0.5)

# Legend
legend_a = [
    Patch(facecolor=COLORS['orange'], edgecolor='black', label='Post-COVID interaction'),
    Patch(facecolor=COLORS['blue'], edgecolor='black', label='Base feature'),
]
ax_a.legend(handles=legend_a, loc='lower right', frameon=False)


# -----------------------------------------------------------------------------
# Panel B: Model Confidence (Uncertainty) - Boxplot (LEFT)
# -----------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 0])

# Prepare data for boxplot
categories = ['Class', 'Location', 'Age']
box_data = []
box_positions = []
box_labels = []
box_colors = []
pos = 0

for cat in categories:
    cat_data = all_metrics[all_metrics['category'] == cat].sort_values('uncertainty')
    for _, row in cat_data.iterrows():
        seg_name = row['segment']
        # Get raw data for this segment
        if cat == 'Class':
            mask = train_oof['class_group'] == seg_name.replace('Class ', '')
        elif cat == 'Location':
            mask = train_oof['location'] == seg_name
        else:  # Age
            mask = train_oof['age_group'] == seg_name

        segment_data = train_oof.loc[mask, 'ensemble_std'].values
        box_data.append(segment_data)
        box_positions.append(pos)
        box_labels.append(seg_name)
        box_colors.append(get_category_color(cat))  # Color by category
        pos += 1
    pos += 0.5  # Gap between categories

# Create boxplot (Nature Cancer style - matching Panel F)
bp = ax_b.boxplot(
    box_data,
    positions=box_positions,
    widths=0.6,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(linewidth=0.8),
    whiskerprops=dict(color='black', linewidth=0.8),
    capprops=dict(color='black', linewidth=0.8),
    flierprops=dict(marker='o', markerfacecolor='black', markersize=2,
                   linestyle='none', markeredgecolor='black', alpha=0.5),
    medianprops=dict(color='black', linewidth=1.2)
)

# Color the boxes
for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(0.8)
    patch.set_alpha(0.8)

# Ensure spines match Nature Cancer style
ax_b.spines['left'].set_linewidth(0.5)
ax_b.spines['bottom'].set_linewidth(0.5)

ax_b.set_xticks(box_positions)
ax_b.set_xticklabels(box_labels, rotation=30, ha='right')
ax_b.set_ylabel('Model uncertainty\n(ensemble std)')
ax_b.set_ylim(0, 0.1)
ax_b.set_title('Which assets are hardest to predict?', fontweight='bold')

# Add y-axis grid (Project 4 style)
ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)

# Legend - show category colors
legend_b = [
    Patch(facecolor=get_category_color('Class'), edgecolor='black', label='Class'),
    Patch(facecolor=get_category_color('Location'), edgecolor='black', label='Location'),
    Patch(facecolor=get_category_color('Age'), edgecolor='black', label='Age'),
]
ax_b.legend(handles=legend_b, loc='upper right', frameon=False)


# =============================================================================
# SAVE
# =============================================================================
plt.tight_layout()

fig.savefig(OUTPUT_DIR / 'fig_panels_ab.png', dpi=DPI, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print(f"\nSaved: {OUTPUT_DIR / 'fig_panels_ab.png'}")

# Clean up old files
import os
for old_file in ['feature_importance_pre_post.png', 'prediction_confidence.png']:
    old_path = OUTPUT_DIR / old_file
    if old_path.exists():
        os.remove(old_path)
        print(f"Removed: {old_file}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nPanel A - Key Features:")
for _, row in plot_data_a.tail(5).iloc[::-1].iterrows():
    tag = "[Interaction]" if row['is_interaction'] else "[Base]"
    print(f"  {row['display'].replace(chr(10), ' ')}: {row['avg_importance']:.1%} {tag}")

print("\nPanel B - Model Uncertainty:")
easiest = all_metrics.nsmallest(2, 'uncertainty')
hardest = all_metrics.nlargest(2, 'uncertainty')
print("  Easiest to predict (low uncertainty):")
for _, row in easiest.iterrows():
    print(f"    {row['segment']}: {row['uncertainty']:.4f}")
print("  Hardest to predict (high uncertainty):")
for _, row in hardest.iterrows():
    print(f"    {row['segment']}: {row['uncertainty']:.4f}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
