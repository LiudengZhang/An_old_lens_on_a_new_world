#!/usr/bin/env python3
"""
Rice Datathon 2026 - Publication-Quality Summary Figure
Using Nature Cancer style specifications
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

NC_SINGLE_COL_MM = 89
NC_DOUBLE_COL_MM = 183
NC_MAX_HEIGHT_MM = 170
MM_TO_INCH = 1 / 25.4
DPI = 300

NC_SINGLE_COL = NC_SINGLE_COL_MM * MM_TO_INCH
NC_DOUBLE_COL = NC_DOUBLE_COL_MM * MM_TO_INCH
NC_MAX_HEIGHT = NC_MAX_HEIGHT_MM * MM_TO_INCH

# Colorblind-safe palette (Okabe-Ito)
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'vermillion': '#D55E00',
    'sky_blue': '#56B4E9',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'gray': '#999999',
    'dark_gray': '#444444',
}

def fisher_z_test(r1, n1, r2, n2):
    """
    Test if two correlations are significantly different using Fisher's z-transformation.
    Returns: z-statistic, p-value (two-tailed)
    """
    from scipy import stats

    # Fisher z-transformation
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))

    # Standard error of difference
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))

    # Z-statistic
    z_stat = (z1 - z2) / se

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value

def bootstrap_correlation_ci(x, y, n_bootstrap=1000, ci=95):
    """
    Compute bootstrap confidence interval for correlation.
    Returns: (correlation, lower_ci, upper_ci)
    """
    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    # Original correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Bootstrap
    n = len(x)
    boot_corrs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_corr = np.corrcoef(x[idx], y[idx])[0, 1]
        if not np.isnan(boot_corr):
            boot_corrs.append(boot_corr)

    # Confidence interval
    lower = np.percentile(boot_corrs, (100 - ci) / 2)
    upper = np.percentile(boot_corrs, 100 - (100 - ci) / 2)

    return corr, lower, upper

def correlation_to_color(corr, max_abs=0.4):
    """
    Map correlation value to color.
    Negative → Blue, Positive → Orange
    Intensity proportional to magnitude.
    """
    import matplotlib.colors as mcolors

    # Normalize correlation to [-1, 1] range based on max_abs
    norm_val = np.clip(corr / max_abs, -1, 1)

    # Define colors
    blue = np.array(mcolors.to_rgb('#0072B2'))
    orange = np.array(mcolors.to_rgb('#E69F00'))
    white = np.array([1.0, 1.0, 1.0])

    if norm_val >= 0:
        # Interpolate white -> orange
        intensity = norm_val
        color = white * (1 - intensity) + orange * intensity
    else:
        # Interpolate white -> blue
        intensity = -norm_val
        color = white * (1 - intensity) + blue * intensity

    return color

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

def add_panel_label(ax, label, x=-0.15, y=1.1):
    """Add panel label (lowercase, bold) per Nature Cancer style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top', ha='left')

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')

pre_mask = train['is_pre_covid'] == 1
post_mask = train['is_post_covid'] == 1
post_data = train[post_mask].copy()

apply_nature_style()

# =============================================================================
# CREATE FIGURE (2 rows x 3 columns, but panel e spans bottom right 2 cols)
# =============================================================================

fig = plt.figure(figsize=(NC_DOUBLE_COL, NC_DOUBLE_COL * 0.65))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4,
                       height_ratios=[1, 1.1])

# =============================================================================
# Panel A: Pre vs Post COVID Flip
# =============================================================================
ax_a = fig.add_subplot(gs[0, 0])

pre_health = train.loc[pre_mask, 'aarp_met_health_hospital'].values
pre_target = train.loc[pre_mask, 'target'].values
post_health = train.loc[post_mask, 'aarp_met_health_hospital'].values
post_target = train.loc[post_mask, 'target'].values

corr_pre, ci_pre_lo, ci_pre_hi = bootstrap_correlation_ci(pre_health, pre_target)
corr_post, ci_post_lo, ci_post_hi = bootstrap_correlation_ci(post_health, post_target)

x_pos = [0, 1]
colors_a = [correlation_to_color(corr_pre), correlation_to_color(corr_post)]
yerr_a = [[corr_pre - ci_pre_lo, corr_post - ci_post_lo],
          [ci_pre_hi - corr_pre, ci_post_hi - corr_post]]
bars_a = ax_a.bar(x_pos, [corr_pre, corr_post],
                  color=colors_a, yerr=yerr_a, capsize=3,
                  width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_a.axhline(y=0, color='black', linewidth=0.5)
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(['Pre-COVID\n(2015-2020)', 'Post-COVID\n(2022-2025)'])
ax_a.set_ylabel('Correlation with\nRevPAR growth')
ax_a.set_ylim(-0.25, 0.35)
ax_a.set_title('Healthcare effect flipped', fontweight='bold')


# =============================================================================
# Panel B: Who Drives It - Segment Sensitivity (now position 3)
# =============================================================================
ax_b = fig.add_subplot(gs[0, 2])

# Create segments
post_data['segment'] = 'Other'
senior_mask = (
    (post_data['aarp_met_health_hospital'] > post_data['aarp_met_health_hospital'].quantile(0.6)) &
    (post_data['areaperunit'] > 850) & (post_data['areaperunit'] < 1000) &
    (post_data['property_age'] > 25)
)
family_mask = (
    (post_data['areaperunit'] > 1000) &
    (post_data['msa_ring'].isin(['innersuburb', 'outersuburb']))
)
young_mask = (
    (post_data['areaperunit'] < 850) &
    (post_data['msa_ring'].isin(['downtown', 'outercore'])) &
    (post_data['property_age'] < 25)
)

post_data.loc[senior_mask, 'segment'] = 'Senior'
post_data.loc[family_mask & ~senior_mask, 'segment'] = 'Family'
post_data.loc[young_mask & ~senior_mask & ~family_mask, 'segment'] = 'Young Prof.'

# Calculate correlations with bootstrap CI
segment_corrs = []
for seg in ['Young Prof.', 'Family', 'Senior']:
    subset = post_data[post_data['segment'] == seg]
    if len(subset) >= 100:
        health = subset['aarp_met_health_hospital'].values
        target = subset['target'].values
        corr, ci_lo, ci_hi = bootstrap_correlation_ci(health, target)
        segment_corrs.append({'Segment': seg, 'Correlation': corr, 'CI_lo': ci_lo, 'CI_hi': ci_hi})

seg_df = pd.DataFrame(segment_corrs)
seg_colors = [correlation_to_color(c) for c in seg_df['Correlation']]
yerr_b = [seg_df['Correlation'] - seg_df['CI_lo'], seg_df['CI_hi'] - seg_df['Correlation']]

bars_b = ax_b.bar(range(len(seg_df)), seg_df['Correlation'],
                  color=seg_colors, yerr=yerr_b, capsize=3,
                  width=0.6, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_b.set_xticks(range(len(seg_df)))
ax_b.set_xticklabels(seg_df['Segment'])
ax_b.axhline(y=0, color='black', linewidth=0.5)
ax_b.set_ylabel('Healthcare\nsensitivity')
ax_b.set_title('Young professionals drive effect', fontweight='bold')
ax_b.set_ylim(-0.15, 0.45)


# =============================================================================
# Panel C: Confounder Check (now position 2)
# =============================================================================
ax_c = fig.add_subplot(gs[0, 1])

# Simple correlation with bootstrap CI
health_vals = post_data['aarp_met_health_hospital'].values
target_vals = post_data['target'].values
simple_corr, simple_ci_lo, simple_ci_hi = bootstrap_correlation_ci(health_vals, target_vals)

# Partial correlation (bootstrap the residuals)
urban_controls = ['aarp_met_prox_trans', 'msa_norm_dist', 'aarp_met_trans_walk']
urban_controls = [c for c in urban_controls if c in post_data.columns]
X = post_data[urban_controls].dropna()
y_hospital = post_data.loc[X.index, 'aarp_met_health_hospital']
y_target = post_data.loc[X.index, 'target']

X_mat = np.column_stack([np.ones(len(X)), X.values])
hospital_resid = y_hospital.values - X_mat @ np.linalg.lstsq(X_mat, y_hospital.values, rcond=None)[0]
target_resid = y_target.values - X_mat @ np.linalg.lstsq(X_mat, y_target.values, rcond=None)[0]
partial_corr, partial_ci_lo, partial_ci_hi = bootstrap_correlation_ci(hospital_resid, target_resid)

colors_c = [correlation_to_color(simple_corr), correlation_to_color(partial_corr)]
yerr_c = [[simple_corr - simple_ci_lo, partial_corr - partial_ci_lo],
          [simple_ci_hi - simple_corr, partial_ci_hi - partial_corr]]
bars_c = ax_c.bar([0, 1], [simple_corr, partial_corr],
                  color=colors_c, yerr=yerr_c, capsize=3,
                  width=0.5, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['Simple', 'Controlled'])
ax_c.set_ylabel('Correlation')
ax_c.set_title('Effect persists after controls', fontweight='bold')
ax_c.set_ylim(0, 0.36)

# Fisher z-test: is simple significantly different from controlled?
n_simple = len(post_data.dropna(subset=['aarp_met_health_hospital', 'target']))
n_partial = len(X)
z_stat, p_val = fisher_z_test(simple_corr, n_simple, partial_corr, n_partial)

# Add p-value annotation
if p_val < 0.001:
    p_text = 'p < 0.001'
elif p_val < 0.01:
    p_text = f'p = {p_val:.3f}'
else:
    p_text = f'p = {p_val:.2f}'

# Add "ns" or p-value between bars
y_max = max(simple_corr, partial_corr) + 0.06
ax_c.plot([0, 0, 1, 1], [y_max - 0.01, y_max, y_max, y_max - 0.01],
          color='black', linewidth=0.8)
ax_c.text(0.5, y_max + 0.01, f'ns ({p_text})', ha='center', va='bottom', fontsize=6)


# =============================================================================
# Panel D: Geographic Variation
# =============================================================================
ax_d = fig.add_subplot(gs[1, 0])

state_data = []
for state in train['state'].unique():
    pre_sub = train[(train['state'] == state) & pre_mask]
    post_sub = train[(train['state'] == state) & post_mask]
    if len(pre_sub) >= 500 and len(post_sub) >= 500:
        # Bootstrap the shift
        pre_health = pre_sub['aarp_met_health_hospital'].values
        pre_target = pre_sub['target'].values
        post_health = post_sub['aarp_met_health_hospital'].values
        post_target = post_sub['target'].values

        corr_pre, _, _ = bootstrap_correlation_ci(pre_health, pre_target, n_bootstrap=500)
        corr_post, _, _ = bootstrap_correlation_ci(post_health, post_target, n_bootstrap=500)

        # Bootstrap the shift directly
        n_boot = 500
        shifts = []
        for _ in range(n_boot):
            idx_pre = np.random.choice(len(pre_health), size=len(pre_health), replace=True)
            idx_post = np.random.choice(len(post_health), size=len(post_health), replace=True)
            c_pre = np.corrcoef(pre_health[idx_pre], pre_target[idx_pre])[0, 1]
            c_post = np.corrcoef(post_health[idx_post], post_target[idx_post])[0, 1]
            if not (np.isnan(c_pre) or np.isnan(c_post)):
                shifts.append(c_post - c_pre)
        ci_lo = np.percentile(shifts, 2.5)
        ci_hi = np.percentile(shifts, 97.5)

        state_data.append({'State': state, 'Shift': corr_post - corr_pre, 'CI_lo': ci_lo, 'CI_hi': ci_hi})

state_df = pd.DataFrame(state_data).sort_values('Shift', ascending=True)
bar_colors_d = [correlation_to_color(s, max_abs=0.5) for s in state_df['Shift']]
xerr_d = [state_df['Shift'] - state_df['CI_lo'], state_df['CI_hi'] - state_df['Shift']]

bars_d = ax_d.barh(range(len(state_df)), state_df['Shift'],
                   color=bar_colors_d, xerr=xerr_d, capsize=2,
                   height=0.6, edgecolor='black', linewidth=0.8,
                   error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_d.set_yticks(range(len(state_df)))
ax_d.set_yticklabels(state_df['State'])
ax_d.axvline(x=0, color='black', linewidth=0.5)
ax_d.set_xlabel('Post-COVID shift')
ax_d.set_title('Geographic variation', fontweight='bold')


# =============================================================================
# Panel E: Stratified Healthcare Correlation (which properties benefit most)
# =============================================================================
ax_e = fig.add_subplot(gs[1, 1:])

# Create property segments
post_data['class_group'] = post_data['type_main'].apply(lambda x: x[0] if pd.notna(x) else 'X')
post_data['location_type'] = post_data['msa_ring'].apply(
    lambda x: 'Urban' if x in ['downtown', 'outercore'] else 'Suburban'
)
post_data['age_group'] = pd.cut(post_data['property_age'],
                                 bins=[0, 15, 30, 100],
                                 labels=['Modern', 'Mid-age', 'Older'])

# Calculate healthcare-RevPAR correlation for each segment combination with bootstrap CI
segment_results = []

# By Property Class
for cls in ['A', 'B', 'C']:
    subset = post_data[post_data['class_group'] == cls]
    if len(subset) >= 100:
        health = subset['aarp_met_health_hospital'].values
        target = subset['target'].values
        corr, ci_lo, ci_hi = bootstrap_correlation_ci(health, target, n_bootstrap=500)
        segment_results.append({'Category': 'Class', 'Segment': f'Class {cls}',
                               'Correlation': corr, 'CI_lo': ci_lo, 'CI_hi': ci_hi})

# By Location
for loc in ['Urban', 'Suburban']:
    subset = post_data[post_data['location_type'] == loc]
    if len(subset) >= 100:
        health = subset['aarp_met_health_hospital'].values
        target = subset['target'].values
        corr, ci_lo, ci_hi = bootstrap_correlation_ci(health, target, n_bootstrap=500)
        segment_results.append({'Category': 'Location', 'Segment': loc,
                               'Correlation': corr, 'CI_lo': ci_lo, 'CI_hi': ci_hi})

# By Age
for age in ['Modern', 'Mid-age', 'Older']:
    subset = post_data[post_data['age_group'] == age]
    if len(subset) >= 100:
        health = subset['aarp_met_health_hospital'].values
        target = subset['target'].values
        corr, ci_lo, ci_hi = bootstrap_correlation_ci(health, target, n_bootstrap=500)
        segment_results.append({'Category': 'Age', 'Segment': age,
                               'Correlation': corr, 'CI_lo': ci_lo, 'CI_hi': ci_hi})

seg_results_df = pd.DataFrame(segment_results)

# Create grouped bar chart with correlation-based colors and error bars
categories = ['Class', 'Location', 'Age']

x_positions = []
x_labels = []
colors = []
values = []
yerr_lo = []
yerr_hi = []
pos = 0

for cat in categories:
    cat_data = seg_results_df[seg_results_df['Category'] == cat]
    for i, row in enumerate(cat_data.itertuples()):
        x_positions.append(pos)
        x_labels.append(row.Segment)
        colors.append(correlation_to_color(row.Correlation))
        values.append(row.Correlation)
        yerr_lo.append(row.Correlation - row.CI_lo)
        yerr_hi.append(row.CI_hi - row.Correlation)
        pos += 1
    pos += 0.5  # Gap between categories

yerr_e = [yerr_lo, yerr_hi]
bars_e = ax_e.bar(x_positions, values, color=colors, yerr=yerr_e, capsize=3,
                  width=0.7, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 0.8, 'capthick': 0.8})

ax_e.set_xticks(x_positions)
ax_e.set_xticklabels(x_labels, rotation=0, fontsize=6)
ax_e.axhline(y=0, color='black', linewidth=0.5)
ax_e.set_ylabel('Healthcare-RevPAR\ncorrelation')
ax_e.set_title('Which properties benefit most from healthcare access?', fontweight='bold')

# Add category labels at top
cat_positions = {'Class': 1, 'Location': 4.25, 'House-age': 7.5}
for cat, xpos in cat_positions.items():
    ax_e.text(xpos, ax_e.get_ylim()[1] * 0.95, cat, ha='center', va='top',
              fontsize=7, fontweight='bold', color=COLORS['dark_gray'])


# =============================================================================
# SAVE
# =============================================================================

plt.savefig(f'{OUTPUT_DIR}/fig_summary_nature.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}/fig_summary_nature.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"Saved: {OUTPUT_DIR}/fig_summary_nature.png")
print(f"Saved: {OUTPUT_DIR}/fig_summary_nature.pdf")

# Print summary statistics
print("\n=== Panel E Statistics ===")
print(seg_results_df.to_string(index=False))
print("\nDone!")
