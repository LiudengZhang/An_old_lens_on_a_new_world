#!/usr/bin/env python3
"""
Rice Datathon 2026 - Summary Figure
5 panels capturing all key findings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

# Load data
print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')

pre_mask = train['is_pre_covid'] == 1
post_mask = train['is_post_covid'] == 1
post_data = train[post_mask].copy()

# Create figure with custom grid
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# ============================================================================
# Panel A: Pre vs Post COVID Flip (top-left)
# ============================================================================
ax_a = fig.add_subplot(gs[0, 0])

corr_pre = train.loc[pre_mask, ['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
corr_post = train.loc[post_mask, ['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]

periods = ['Pre-COVID\n(2015-2020)', 'Post-COVID\n(2022-2025)']
corrs = [corr_pre, corr_post]
colors_a = ['#3498db', '#e74c3c']

bars_a = ax_a.bar(periods, corrs, color=colors_a, edgecolor='white', linewidth=2, width=0.6)
ax_a.axhline(y=0, color='black', linewidth=1)
ax_a.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax_a.set_title('A. Healthcare Effect Flipped Post-COVID', fontsize=12, fontweight='bold')
ax_a.set_ylim(-0.25, 0.35)

for bar, val in zip(bars_a, corrs):
    y_pos = val + 0.02 if val > 0 else val - 0.04
    ax_a.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:.2f}',
              ha='center', va='bottom' if val > 0 else 'top', fontsize=14, fontweight='bold')

# Add arrow showing flip
ax_a.annotate('', xy=(1, corr_post - 0.03), xytext=(0, corr_pre + 0.03),
              arrowprops=dict(arrowstyle='->', color='#27ae60', lw=3))
ax_a.text(0.5, 0.05, 'FLIPPED', ha='center', fontsize=10, color='#27ae60', fontweight='bold')

# ============================================================================
# Panel B: Who Drives It - Segment Sensitivity (top-middle)
# ============================================================================
ax_b = fig.add_subplot(gs[0, 1])

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

post_data.loc[senior_mask, 'segment'] = 'Senior-Focused'
post_data.loc[family_mask & ~senior_mask, 'segment'] = 'Family-Oriented'
post_data.loc[young_mask & ~senior_mask & ~family_mask, 'segment'] = 'Young Professional'

# Calculate healthcare sensitivity by segment
segment_corrs = []
for seg in ['Young Professional', 'Family-Oriented', 'Senior-Focused']:
    subset = post_data[post_data['segment'] == seg]
    if len(subset) >= 100:
        corr = subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        segment_corrs.append({'Segment': seg, 'Correlation': corr})

seg_df = pd.DataFrame(segment_corrs)
colors_b = {'Young Professional': '#3498db', 'Family-Oriented': '#27ae60', 'Senior-Focused': '#9b59b6'}
bar_colors_b = [colors_b[s] for s in seg_df['Segment']]

bars_b = ax_b.bar(range(len(seg_df)), seg_df['Correlation'], color=bar_colors_b, edgecolor='white', linewidth=2)
ax_b.set_xticks(range(len(seg_df)))
ax_b.set_xticklabels(['Young\nProfessional', 'Family\nOriented', 'Senior\nFocused'], fontsize=9)
ax_b.axhline(y=0, color='black', linewidth=1)
ax_b.set_ylabel('Healthcare Sensitivity', fontsize=10)
ax_b.set_title('B. Who Values Healthcare Most?', fontsize=12, fontweight='bold')
ax_b.set_ylim(-0.15, 0.45)

for i, (bar, row) in enumerate(zip(bars_b, seg_df.itertuples())):
    y_pos = row.Correlation + 0.02 if row.Correlation > 0 else row.Correlation - 0.04
    ax_b.text(bar.get_x() + bar.get_width()/2, y_pos, f'{row.Correlation:.2f}',
              ha='center', va='bottom' if row.Correlation > 0 else 'top', fontsize=12, fontweight='bold')

# Highlight winner
ax_b.annotate('Young professionals\ndrive the effect!', xy=(0, 0.35), xytext=(1.2, 0.38),
              fontsize=9, color='#2c3e50', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#2c3e50'),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'))

# ============================================================================
# Panel C: Confounder Check (top-right)
# ============================================================================
ax_c = fig.add_subplot(gs[0, 2])

# Simple correlation
simple_corr = post_data[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]

# Partial correlation (controlling for urban indicators)
urban_controls = ['aarp_met_prox_trans', 'msa_norm_dist', 'aarp_met_trans_walk']
urban_controls = [c for c in urban_controls if c in post_data.columns]

X = post_data[urban_controls].dropna()
y_hospital = post_data.loc[X.index, 'aarp_met_health_hospital']
y_target = post_data.loc[X.index, 'target']

X_mat = np.column_stack([np.ones(len(X)), X.values])
hospital_resid = y_hospital - X_mat @ np.linalg.lstsq(X_mat, y_hospital, rcond=None)[0]
target_resid = y_target - X_mat @ np.linalg.lstsq(X_mat, y_target, rcond=None)[0]
partial_corr = np.corrcoef(hospital_resid, target_resid)[0, 1]

labels_c = ['Simple\nCorrelation', 'After Controlling\nfor Urbanization']
vals_c = [simple_corr, partial_corr]
colors_c = ['#e74c3c', '#3498db']

bars_c = ax_c.bar(labels_c, vals_c, color=colors_c, edgecolor='white', linewidth=2, width=0.5)
ax_c.set_ylabel('Correlation with RevPAR', fontsize=10)
ax_c.set_title('C. Is It Just Urbanization Proxy?', fontsize=12, fontweight='bold')
ax_c.set_ylim(0, 0.35)

for bar, val in zip(bars_c, vals_c):
    ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
              ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add percentage retained
pct_retained = partial_corr / simple_corr * 100
ax_c.annotate(f'{pct_retained:.0f}% retained\n→ Effect is REAL', xy=(1, partial_corr), xytext=(0.3, 0.28),
              fontsize=10, color='#27ae60', fontweight='bold',
              arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2),
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#d5f5e3', edgecolor='#27ae60'))

# ============================================================================
# Panel D: Geographic Variation (bottom-left)
# ============================================================================
ax_d = fig.add_subplot(gs[1, 0])

state_data = []
for state in train['state'].unique():
    pre_subset = train[(train['state'] == state) & pre_mask]
    post_subset = train[(train['state'] == state) & post_mask]

    if len(pre_subset) >= 500 and len(post_subset) >= 500:
        corr_pre = pre_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        corr_post = post_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        state_data.append({
            'State': state,
            'Shift': corr_post - corr_pre
        })

state_df = pd.DataFrame(state_data).sort_values('Shift', ascending=True)
colors_d = ['#27ae60' if s > 0 else '#e74c3c' for s in state_df['Shift']]

bars_d = ax_d.barh(range(len(state_df)), state_df['Shift'], color=colors_d, edgecolor='white', linewidth=1.5)
ax_d.set_yticks(range(len(state_df)))
ax_d.set_yticklabels(state_df['State'], fontsize=10)
ax_d.axvline(x=0, color='black', linewidth=1)
ax_d.set_xlabel('Post-COVID Shift in Healthcare Effect', fontsize=10)
ax_d.set_title('D. Geographic Variation', fontsize=12, fontweight='bold')

for i, (bar, row) in enumerate(zip(bars_d, state_df.itertuples())):
    x_pos = row.Shift + 0.02 if row.Shift > 0 else row.Shift - 0.02
    ax_d.text(x_pos, i, f'{row.Shift:+.2f}', va='center',
              ha='left' if row.Shift > 0 else 'right', fontsize=9, fontweight='bold')

# ============================================================================
# Panel E: Investment Sweet Spot (bottom-middle + bottom-right, spanning 2 columns)
# ============================================================================
ax_e = fig.add_subplot(gs[1, 1:])

# Calculate RevPAR growth for different segments
# Create investment-relevant segments
post_data['class_group'] = post_data['type_main'].apply(lambda x: x[0] if pd.notna(x) else 'Other')
post_data['location_type'] = post_data['msa_ring'].apply(
    lambda x: 'Urban' if x in ['downtown', 'outercore'] else 'Suburban'
)
post_data['healthcare_level'] = pd.qcut(post_data['aarp_met_health_hospital'], 3, labels=['Low', 'Medium', 'High'])
post_data['rent_level'] = pd.qcut(post_data['rent_percentile'], 3, labels=['Value', 'Mid', 'Premium'])

# Calculate average RevPAR for each combination
investment_segments = []

for cls in ['A', 'B', 'C']:
    for loc in ['Urban', 'Suburban']:
        for health in ['High', 'Medium', 'Low']:
            for rent in ['Value', 'Mid', 'Premium']:
                subset = post_data[
                    (post_data['class_group'] == cls) &
                    (post_data['location_type'] == loc) &
                    (post_data['healthcare_level'] == health) &
                    (post_data['rent_level'] == rent)
                ]
                if len(subset) >= 30:
                    investment_segments.append({
                        'Class': cls,
                        'Location': loc,
                        'Healthcare': health,
                        'Rent': rent,
                        'RevPAR': subset['target'].mean() * 100,
                        'N': len(subset)
                    })

inv_df = pd.DataFrame(investment_segments)

# Find the sweet spot: Class B, Urban, High Healthcare, Mid Rent
sweet_spot = inv_df[
    (inv_df['Class'] == 'B') &
    (inv_df['Location'] == 'Urban') &
    (inv_df['Healthcare'] == 'High')
]

# Create a simplified visualization: compare key segments
comparison_segments = [
    ('Class A\nUrban\nHigh Health\nPremium Rent', 'A', 'Urban', 'High', 'Premium'),
    ('Class B\nUrban\nHigh Health\nMid Rent', 'B', 'Urban', 'High', 'Mid'),
    ('Class B\nUrban\nHigh Health\nValue Rent', 'B', 'Urban', 'High', 'Value'),
    ('Class C\nSuburban\nLow Health\nValue Rent', 'C', 'Suburban', 'Low', 'Value'),
]

comparison_data = []
for label, cls, loc, health, rent in comparison_segments:
    subset = inv_df[
        (inv_df['Class'] == cls) &
        (inv_df['Location'] == loc) &
        (inv_df['Healthcare'] == health) &
        (inv_df['Rent'] == rent)
    ]
    if len(subset) > 0:
        comparison_data.append({
            'Label': label,
            'RevPAR': subset['RevPAR'].values[0],
            'N': subset['N'].values[0]
        })

if len(comparison_data) < 4:
    # Fallback: use broader segments
    comparison_data = []
    for label, query in [
        ('Class A\nPremium', (inv_df['Class'] == 'A') & (inv_df['Rent'] == 'Premium')),
        ('Class B\nUrban\nHigh Health', (inv_df['Class'] == 'B') & (inv_df['Location'] == 'Urban') & (inv_df['Healthcare'] == 'High')),
        ('Class B\nSuburban', (inv_df['Class'] == 'B') & (inv_df['Location'] == 'Suburban')),
        ('Class C\nValue', (inv_df['Class'] == 'C') & (inv_df['Rent'] == 'Value')),
    ]:
        subset = inv_df[query]
        if len(subset) > 0:
            comparison_data.append({
                'Label': label,
                'RevPAR': subset['RevPAR'].mean(),
                'N': subset['N'].sum()
            })

comp_df = pd.DataFrame(comparison_data)

# Identify the sweet spot (highest RevPAR)
if len(comp_df) > 0:
    best_idx = comp_df['RevPAR'].idxmax()
    colors_e = ['#f39c12' if i == best_idx else '#95a5a6' for i in range(len(comp_df))]

    bars_e = ax_e.bar(range(len(comp_df)), comp_df['RevPAR'], color=colors_e, edgecolor='white', linewidth=2)
    ax_e.set_xticks(range(len(comp_df)))
    ax_e.set_xticklabels(comp_df['Label'], fontsize=9)
    ax_e.axhline(y=0, color='black', linewidth=1)
    ax_e.set_ylabel('Average RevPAR Growth (%)', fontsize=10)
    ax_e.set_title('E. Investment Sweet Spot: Class B + Urban + High Healthcare + Moderate Rent',
                   fontsize=12, fontweight='bold')

    for i, (bar, row) in enumerate(zip(bars_e, comp_df.itertuples())):
        y_pos = row.RevPAR + 0.3 if row.RevPAR > 0 else row.RevPAR - 0.8
        color = '#f39c12' if i == best_idx else '#2c3e50'
        ax_e.text(bar.get_x() + bar.get_width()/2, y_pos,
                  f'{row.RevPAR:.1f}%\n(n={row.N})',
                  ha='center', va='bottom' if row.RevPAR > 0 else 'top',
                  fontsize=10, fontweight='bold', color=color)

    # Highlight sweet spot
    if best_idx is not None:
        ax_e.annotate('SWEET SPOT', xy=(best_idx, comp_df.iloc[best_idx]['RevPAR']),
                      xytext=(best_idx + 0.8, comp_df.iloc[best_idx]['RevPAR'] + 2),
                      fontsize=12, color='#f39c12', fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))

# Main title
fig.suptitle('Rice Datathon 2026: Healthcare & RevPAR Growth — Key Findings',
             fontsize=18, fontweight='bold', y=0.98)

plt.savefig(f'{OUTPUT_DIR}/fig_summary_5panels.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig_summary_5panels.png")

print("\nDone!")
