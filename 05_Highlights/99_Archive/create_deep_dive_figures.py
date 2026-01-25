#!/usr/bin/env python3
"""
Rice Datathon 2026 - Deep Dive Visualizations
1. Confounder Analysis: Is hospital proximity just a proxy for urbanization?
2. Geographic Breakdown: Why raw vs model show opposite signs?
3. Renter Segment Profiling: Senior vs Family vs Young Professional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

# Load data
print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')
post_mask = train['is_post_covid'] == 1
post_data = train[post_mask].copy()

print(f"Post-COVID samples: {len(post_data):,}")

# ============================================================================
# FIGURE 4: Confounder Analysis - Is Hospital a Proxy for Urbanization?
# ============================================================================

print("\n=== FIGURE 4: Confounder Analysis ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Hospital proximity correlates with urban indicators
ax1 = axes[0, 0]
urban_features = {
    'aarp_met_prox_trans': 'Transit Access',
    'aarp_met_env_air': 'Air Quality\n(lower=worse)',
    'msa_norm_dist': 'Distance from\nMetro Center',
    'aarp_met_trans_walk': 'Walkability',
    'rent_percentile': 'Rent Level',
}

correlations = []
for feat, label in urban_features.items():
    if feat in post_data.columns:
        corr = post_data[['aarp_met_health_hospital', feat]].corr().iloc[0, 1]
        correlations.append({'Feature': label, 'Correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('Correlation')
colors = ['#e74c3c' if c > 0 else '#3498db' for c in corr_df['Correlation']]
ax1.barh(corr_df['Feature'], corr_df['Correlation'], color=colors, edgecolor='white')
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('Correlation with Hospital Proximity', fontsize=10)
ax1.set_title('A. Hospital Proximity = Urban Location Proxy', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.5, 0.5)
for i, (feat, corr) in enumerate(zip(corr_df['Feature'], corr_df['Correlation'])):
    ax1.text(corr + 0.02 if corr > 0 else corr - 0.02, i, f'{corr:.2f}',
             va='center', ha='left' if corr > 0 else 'right', fontsize=9)

# Panel B: Stratified analysis - effect disappears in high-equality neighborhoods
ax2 = axes[0, 1]

# Stratify by income equality (aarp_met_opp_income)
if 'aarp_met_opp_income' in post_data.columns:
    post_data['income_tertile'] = pd.qcut(post_data['aarp_met_opp_income'], 3, labels=['Low Equality', 'Medium', 'High Equality'])

    stratified_corrs = []
    for tertile in ['Low Equality', 'Medium', 'High Equality']:
        subset = post_data[post_data['income_tertile'] == tertile]
        corr = subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        n = len(subset)
        stratified_corrs.append({'Neighborhood': tertile, 'Correlation': corr, 'N': n})

    strat_df = pd.DataFrame(stratified_corrs)
    colors2 = ['#e74c3c' if c > 0.1 else '#f39c12' if c > 0.05 else '#3498db' for c in strat_df['Correlation']]
    bars = ax2.bar(strat_df['Neighborhood'], strat_df['Correlation'], color=colors2, edgecolor='white', linewidth=2)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_ylabel('Hospital-RevPAR Correlation', fontsize=10)
    ax2.set_title('B. Effect Disappears in High-Equality Areas', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.05, 0.35)

    for bar, row in zip(bars, strat_df.itertuples()):
        ax2.text(bar.get_x() + bar.get_width()/2, row.Correlation + 0.01,
                 f'{row.Correlation:.3f}\n(n={row.N:,})', ha='center', va='bottom', fontsize=9)

    # Add annotation
    ax2.annotate('If hospital were truly causal,\neffect should be STABLE\nacross neighborhoods',
                 xy=(2, 0.05), xytext=(1.5, 0.20),
                 fontsize=9, color='#2c3e50',
                 arrowprops=dict(arrowstyle='->', color='#2c3e50'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#2c3e50'))

# Panel C: The causal diagram
ax3 = axes[1, 0]
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.axis('off')

# Draw boxes
boxes = {
    'Urban Location': (1, 5, '#3498db'),
    'Hospital\nProximity': (5, 8, '#e74c3c'),
    'RevPAR\nGrowth': (5, 2, '#27ae60'),
    'Transit\nAccess': (9, 6.5, '#9b59b6'),
    'Rent\nLevels': (9, 3.5, '#f39c12'),
}

for label, (x, y, color) in boxes.items():
    rect = plt.Rectangle((x-0.8, y-0.6), 1.6, 1.2, facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
    ax3.add_patch(rect)
    ax3.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows
arrows = [
    ((1.8, 5.4), (4.2, 7.6), 'black', 'Causes'),  # Urban -> Hospital
    ((1.8, 4.6), (4.2, 2.4), 'black', 'Causes'),  # Urban -> RevPAR
    ((5.8, 7.6), (8.2, 6.7), '#95a5a6', ''),      # Hospital -> Transit
    ((5.8, 2.4), (8.2, 3.3), '#95a5a6', ''),      # RevPAR -> Rent
    ((5, 7.4), (5, 2.6), '#e74c3c', 'Spurious?'), # Hospital -> RevPAR (spurious)
]

for (x1, y1), (x2, y2), color, label in arrows:
    ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax3.text(mx-0.3, my, label, fontsize=8, color=color, rotation=90 if x1==x2 else 0)

ax3.set_title('C. Causal Diagram: Urban Location is the Confounder', fontsize=12, fontweight='bold')

# Panel D: Partial correlation controlling for urbanization
ax4 = axes[1, 1]

# Calculate partial correlation
from scipy.stats import pearsonr

# Simple correlation
simple_corr = post_data[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]

# Residualize both variables on urban indicators
urban_controls = ['aarp_met_prox_trans', 'msa_norm_dist', 'aarp_met_trans_walk']
urban_controls = [c for c in urban_controls if c in post_data.columns]

X = post_data[urban_controls].dropna()
y_hospital = post_data.loc[X.index, 'aarp_met_health_hospital']
y_target = post_data.loc[X.index, 'target']

# Residualize using linear regression
from numpy.linalg import lstsq

X_mat = np.column_stack([np.ones(len(X)), X.values])
hospital_resid = y_hospital - X_mat @ lstsq(X_mat, y_hospital, rcond=None)[0]
target_resid = y_target - X_mat @ lstsq(X_mat, y_target, rcond=None)[0]

partial_corr = np.corrcoef(hospital_resid, target_resid)[0, 1]

# Bar chart comparing
comparisons = {
    'Simple\nCorrelation': simple_corr,
    'Partial Correlation\n(controlling for\nurban indicators)': partial_corr,
}

x_pos = list(range(len(comparisons)))
colors4 = ['#e74c3c', '#3498db']
bars4 = ax4.bar(x_pos, list(comparisons.values()), color=colors4, edgecolor='white', linewidth=2, width=0.6)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(list(comparisons.keys()), fontsize=10)
ax4.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax4.set_title('D. Controlling for Urbanization Reduces Effect', fontsize=12, fontweight='bold')
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.set_ylim(0, 0.35)

for bar, val in zip(bars4, comparisons.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add reduction annotation
reduction = (simple_corr - partial_corr) / simple_corr * 100
ax4.annotate(f'{reduction:.0f}% reduction\nwhen controlling\nfor urbanization',
             xy=(1, partial_corr), xytext=(1.3, 0.20),
             fontsize=10, color='#2c3e50', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2c3e50'),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#c0392b'))

fig.suptitle('Is Hospital Proximity Causal or Just a Proxy for Urban Location?',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_confounder_analysis.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig4_confounder_analysis.png")


# ============================================================================
# FIGURE 5: Geographic Breakdown - Why Raw vs Model Show Opposite Signs
# ============================================================================

print("\n=== FIGURE 5: Geographic Market Breakdown ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel A: Raw correlation by market (all data)
ax1 = axes[0, 0]

market_corrs = []
for market in train['mrkt_name'].unique():
    subset = train[train['mrkt_name'] == market]
    if len(subset) >= 200:
        corr = subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        market_corrs.append({'Market': market, 'Correlation': corr, 'N': len(subset)})

market_df = pd.DataFrame(market_corrs).sort_values('Correlation')

# Top 10 and bottom 5
display_markets = pd.concat([market_df.head(8), market_df.tail(4)])
colors1 = ['#e74c3c' if c > 0 else '#3498db' for c in display_markets['Correlation']]

y_pos = range(len(display_markets))
ax1.barh(y_pos, display_markets['Correlation'], color=colors1, edgecolor='white')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(display_markets['Market'], fontsize=9)
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.set_xlabel('Raw Correlation (Hospital → RevPAR)', fontsize=10)
ax1.set_title('A. Raw Data: Most Markets Show NEGATIVE', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.35, 0.15)

# Panel B: Pre vs Post comparison by market
ax2 = axes[0, 1]

pre_mask = train['is_pre_covid'] == 1
# Use actual market names (partial match)
top_market_patterns = ['Dallas', 'Houston', 'Atlanta', 'Phoenix', 'Tampa']

pre_post_data = []
for pattern in top_market_patterns:
    # Find matching market
    matching = [m for m in train['mrkt_name'].unique() if pattern in m]
    if not matching:
        continue
    market = matching[0]
    pre_subset = train[(train['mrkt_name'] == market) & pre_mask]
    post_subset = train[(train['mrkt_name'] == market) & post_mask]

    if len(pre_subset) >= 100 and len(post_subset) >= 100:
        corr_pre = pre_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        corr_post = post_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        pre_post_data.append({'Market': pattern, 'PreCOVID': corr_pre, 'PostCOVID': corr_post})

pp_df = pd.DataFrame(pre_post_data)
print(f"Pre-post comparison markets: {len(pp_df)}")

# Handle empty dataframe
if len(pp_df) == 0:
    print("Warning: No markets met criteria, using top markets by count")
    top_markets_full = train['mrkt_name'].value_counts().head(5).index.tolist()
    for market in top_markets_full:
        pre_subset = train[(train['mrkt_name'] == market) & pre_mask]
        post_subset = train[(train['mrkt_name'] == market) & post_mask]
        if len(pre_subset) >= 50 and len(post_subset) >= 50:
            corr_pre = pre_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
            corr_post = post_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
            short_name = market.split('-')[0].split(',')[0][:12]
            pre_post_data.append({'Market': short_name, 'PreCOVID': corr_pre, 'PostCOVID': corr_post})
    pp_df = pd.DataFrame(pre_post_data)
    print(f"Using {len(pp_df)} markets instead")

x = np.arange(len(pp_df))
width = 0.35

bars1 = ax2.bar(x - width/2, pp_df['PreCOVID'], width, label='Pre-COVID', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, pp_df['PostCOVID'], width, label='Post-COVID', color='#e74c3c', alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(pp_df['Market'], fontsize=10)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_ylabel('Correlation (Hospital → RevPAR)', fontsize=10)
ax2.set_title('B. Pre vs Post COVID by Market', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.set_ylim(-0.35, 0.35)

# Panel C: The paradox explained - interaction term
ax3 = axes[1, 0]

# Compare raw feature vs interaction term
comparison_data = {
    'aarp_met_health_hospital\n(Raw Feature)': post_data[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1],
    'health_hospital_x_post\n(Interaction Term)': train[['health_hospital_x_post', 'target']].corr().iloc[0, 1] if 'health_hospital_x_post' in train.columns else 0,
    'healthcare_access\n(Engineered)': post_data[['healthcare_access', 'target']].corr().iloc[0, 1] if 'healthcare_access' in post_data.columns else 0,
}

x_pos = range(len(comparison_data))
vals = list(comparison_data.values())
colors3 = ['#3498db', '#e74c3c', '#f39c12']
bars3 = ax3.bar(x_pos, vals, color=colors3, edgecolor='white', linewidth=2)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(list(comparison_data.keys()), fontsize=9)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_ylabel('Correlation with Target', fontsize=10)
ax3.set_title('C. Why Model Shows Positive: Interaction Term', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.6, 0.35)

for bar, val in zip(bars3, vals):
    y_pos_text = val + 0.02 if val > 0 else val - 0.05
    ax3.text(bar.get_x() + bar.get_width()/2, y_pos_text, f'{val:.3f}',
             ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

# Add explanation
ax3.annotate('Interaction term captures\nPOST-COVID shift only\n(model sees both periods)',
             xy=(1, -0.55), xytext=(1.8, -0.35),
             fontsize=9, color='#2c3e50',
             arrowprops=dict(arrowstyle='->', color='#2c3e50'),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ecf0f1', edgecolor='#2c3e50'))

# Panel D: State-level breakdown
ax4 = axes[1, 1]

state_data = []
for state in train['state'].unique():
    pre_subset = train[(train['state'] == state) & pre_mask]
    post_subset = train[(train['state'] == state) & post_mask]

    if len(pre_subset) >= 500 and len(post_subset) >= 500:
        corr_pre = pre_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        corr_post = post_subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        state_data.append({
            'State': state,
            'PreCOVID': corr_pre,
            'PostCOVID': corr_post,
            'Shift': corr_post - corr_pre
        })

state_df = pd.DataFrame(state_data).sort_values('Shift', ascending=False)
print(f"State comparison: {len(state_df)} states")

x = np.arange(len(state_df))
width = 0.35

bars1 = ax4.bar(x - width/2, state_df['PreCOVID'], width, label='Pre-COVID', color='#3498db', alpha=0.8)
bars2 = ax4.bar(x + width/2, state_df['PostCOVID'], width, label='Post-COVID', color='#e74c3c', alpha=0.8)

ax4.set_xticks(x)
ax4.set_xticklabels(state_df['State'], fontsize=10)
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.set_ylabel('Correlation (Hospital → RevPAR)', fontsize=10)
ax4.set_title('D. State-Level: Post-COVID Shift Direction', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right')

# Add shift annotations
for i, row in enumerate(state_df.itertuples()):
    shift_color = '#27ae60' if row.Shift > 0 else '#e74c3c'
    max_val = max(row.PreCOVID, row.PostCOVID)
    ax4.annotate(f'{row.Shift:+.2f}', xy=(i, max_val + 0.02),
                 ha='center', fontsize=8, color=shift_color, fontweight='bold')

fig.suptitle('Geographic Puzzle: Why Raw Data and Model Show Opposite Signs',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_geographic_breakdown.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig5_geographic_breakdown.png")


# ============================================================================
# FIGURE 6: Renter Segment Profiling
# ============================================================================

print("\n=== FIGURE 6: Renter Segment Profiling ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Create segments based on property characteristics
# Segment 1: Senior-focused (high healthcare, smaller units, B-class)
# Segment 2: Family-oriented (larger units, suburban, moderate healthcare)
# Segment 3: Young professional (small units, urban, low healthcare priority)

# Use clustering proxies
post_data['segment'] = 'Other'

# Senior-focused: high hospital score, moderate unit size, older property
senior_mask = (
    (post_data['aarp_met_health_hospital'] > post_data['aarp_met_health_hospital'].quantile(0.6)) &
    (post_data['areaperunit'] > 850) & (post_data['areaperunit'] < 1000) &
    (post_data['property_age'] > 25)
)

# Family-oriented: larger units, suburban
family_mask = (
    (post_data['areaperunit'] > 1000) &
    (post_data['msa_ring'].isin(['innersuburb', 'outersuburb']))
)

# Young professional: small units, urban, newer
young_mask = (
    (post_data['areaperunit'] < 850) &
    (post_data['msa_ring'].isin(['downtown', 'outercore'])) &
    (post_data['property_age'] < 25)
)

post_data.loc[senior_mask, 'segment'] = 'Senior-Focused'
post_data.loc[family_mask & ~senior_mask, 'segment'] = 'Family-Oriented'
post_data.loc[young_mask & ~senior_mask & ~family_mask, 'segment'] = 'Young Professional'

# Panel A: Segment sizes and RevPAR growth
ax1 = axes[0, 0]

segment_stats = post_data.groupby('segment').agg({
    'target': ['mean', 'count'],
    'aarp_met_health_hospital': 'mean',
    'areaperunit': 'mean',
    'property_age': 'mean'
}).round(3)

segment_stats.columns = ['RevPAR_Growth', 'Count', 'Hospital_Score', 'Unit_Size', 'Age']
segment_stats = segment_stats.sort_values('RevPAR_Growth', ascending=True)

# Filter to main segments
main_segments = ['Senior-Focused', 'Family-Oriented', 'Young Professional', 'Other']
segment_stats = segment_stats.loc[[s for s in main_segments if s in segment_stats.index]]

colors1 = {'Senior-Focused': '#9b59b6', 'Family-Oriented': '#27ae60',
           'Young Professional': '#3498db', 'Other': '#95a5a6'}
bar_colors = [colors1.get(s, '#95a5a6') for s in segment_stats.index]

bars1 = ax1.barh(range(len(segment_stats)), segment_stats['RevPAR_Growth'] * 100,
                 color=bar_colors, edgecolor='white')
ax1.set_yticks(range(len(segment_stats)))
ax1.set_yticklabels([f"{s}\n(n={int(segment_stats.loc[s, 'Count']):,})" for s in segment_stats.index], fontsize=10)
ax1.set_xlabel('Average RevPAR Growth (%)', fontsize=10)
ax1.set_title('A. RevPAR Growth by Renter Segment', fontsize=12, fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.8)

for i, (seg, row) in enumerate(segment_stats.iterrows()):
    ax1.text(row['RevPAR_Growth'] * 100 + 0.3, i, f"{row['RevPAR_Growth']*100:.1f}%",
             va='center', fontsize=10, fontweight='bold')

# Panel B: Healthcare sensitivity by segment
ax2 = axes[0, 1]

segment_healthcare_corr = []
for seg in ['Senior-Focused', 'Family-Oriented', 'Young Professional', 'Other']:
    subset = post_data[post_data['segment'] == seg]
    if len(subset) >= 100:
        corr = subset[['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        segment_healthcare_corr.append({'Segment': seg, 'Correlation': corr, 'N': len(subset)})

seg_corr_df = pd.DataFrame(segment_healthcare_corr)
bar_colors2 = [colors1.get(s, '#95a5a6') for s in seg_corr_df['Segment']]

bars2 = ax2.bar(range(len(seg_corr_df)), seg_corr_df['Correlation'], color=bar_colors2, edgecolor='white', linewidth=2)
ax2.set_xticks(range(len(seg_corr_df)))
ax2.set_xticklabels(seg_corr_df['Segment'], fontsize=9, rotation=15)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_ylabel('Hospital-RevPAR Correlation', fontsize=10)
ax2.set_title('B. Healthcare Sensitivity by Segment', fontsize=12, fontweight='bold')

for i, row in enumerate(seg_corr_df.itertuples()):
    ax2.text(i, row.Correlation + 0.01, f'{row.Correlation:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel C: Segment characteristics radar-style comparison
ax3 = axes[1, 0]

# Normalize characteristics for comparison
chars = ['Hospital_Score', 'Unit_Size', 'Age']
char_labels = ['Hospital\nAccess', 'Unit Size\n(sqft)', 'Property\nAge (yrs)']

# Get stats for main 3 segments
main_3 = ['Senior-Focused', 'Family-Oriented', 'Young Professional']
char_data = segment_stats.loc[[s for s in main_3 if s in segment_stats.index], chars]

x = np.arange(len(chars))
width = 0.25

for i, seg in enumerate(char_data.index):
    vals = char_data.loc[seg].values
    # Normalize to 0-100 scale for display
    norm_vals = [
        vals[0],  # Hospital score already 0-100
        vals[1] / 12,  # Unit size / 12 to get ~80 range
        vals[2] * 2  # Age * 2 to get similar scale
    ]
    ax3.bar(x + i*width, norm_vals, width, label=seg, color=colors1[seg], alpha=0.8)

ax3.set_xticks(x + width)
ax3.set_xticklabels(char_labels, fontsize=10)
ax3.set_ylabel('Normalized Score', fontsize=10)
ax3.set_title('C. Segment Characteristics Comparison', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)

# Add actual values as text
for i, seg in enumerate(char_data.index):
    for j, (char, val) in enumerate(zip(chars, char_data.loc[seg].values)):
        if char == 'Hospital_Score':
            label = f'{val:.0f}'
        elif char == 'Unit_Size':
            label = f'{val:.0f}'
        else:
            label = f'{val:.0f}y'
        norm_val = [val, val/12, val*2][j]
        ax3.text(j + i*width, norm_val + 2, label, ha='center', fontsize=8)

# Panel D: The senior paradox - high healthcare but low growth
ax4 = axes[1, 1]

# Scatter plot: Hospital access vs RevPAR growth, colored by segment
sample = post_data[post_data['segment'].isin(main_3)].sample(min(3000, len(post_data)), random_state=42)

for seg in main_3:
    subset = sample[sample['segment'] == seg]
    ax4.scatter(subset['aarp_met_health_hospital'], subset['target'] * 100,
                alpha=0.3, s=20, c=colors1[seg], label=seg)

ax4.set_xlabel('Hospital Proximity Score', fontsize=10)
ax4.set_ylabel('RevPAR Growth (%)', fontsize=10)
ax4.set_title('D. The Senior Paradox: High Healthcare ≠ High Growth', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)
ax4.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

# Add trend lines
for seg in main_3:
    subset = sample[sample['segment'] == seg]
    z = np.polyfit(subset['aarp_met_health_hospital'], subset['target'] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(subset['aarp_met_health_hospital'].min(), subset['aarp_met_health_hospital'].max(), 100)
    ax4.plot(x_line, p(x_line), color=colors1[seg], linewidth=2, linestyle='--')

# Add interpretation box
textstr = '\n'.join([
    'Interpretation:',
    '• Seniors value healthcare but have fixed incomes',
    '• Young professionals drive rent growth',
    '• Healthcare is a "floor" not a growth driver'
])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax4.text(0.02, 0.98, textstr, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

fig.suptitle('Renter Segment Analysis: Who Values Healthcare vs. Who Drives Growth',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_renter_segments.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig6_renter_segments.png")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("DEEP DIVE SUMMARY")
print("="*60)

print("\n1. CONFOUNDER ANALYSIS:")
print(f"   Simple correlation: {simple_corr:.3f}")
print(f"   Partial correlation (controlling for urbanization): {partial_corr:.3f}")
print(f"   Reduction: {reduction:.1f}%")

print("\n2. GEOGRAPHIC PATTERNS:")
print("   Most markets show NEGATIVE raw correlation")
print("   Model's positive effect comes from interaction term capturing post-COVID shift")

print("\n3. RENTER SEGMENTS:")
for seg, row in segment_stats.iterrows():
    print(f"   {seg}: RevPAR={row['RevPAR_Growth']*100:.1f}%, n={int(row['Count']):,}")

print("\nDone!")
