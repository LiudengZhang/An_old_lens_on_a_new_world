#!/usr/bin/env python3
"""
Rice Datathon 2026 - Key Insights Visualization
Q: What are the key performance drivers in each time window (pre vs post COVID)?
   How can we interpret performance signals from the renters' perspective?
   Which populations are most interested in healthcare access?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/03_Rice_Datathon_Colab'
OUTPUT_DIR = '/priv18data1/hliang1_group/lzhang34/Project_8_Rice_Datathon/05_Highlights'

# Load data
print("Loading data...")
train = pd.read_csv(f'{DATA_DIR}/data/processed/train_clean.csv')
feat_imp = pd.read_csv(f'{DATA_DIR}/extra/outputs/feature_importance.csv')

# ============================================================================
# FIGURE 1: Pre vs Post COVID - What Changed? (FIXED)
# ============================================================================

print("\n=== FIGURE 1: Pre vs Post COVID Driver Comparison ===")

pre_mask = train['is_pre_covid'] == 1
post_mask = train['is_post_covid'] == 1

print(f"Pre-COVID samples: {pre_mask.sum():,}")
print(f"Post-COVID samples: {post_mask.sum():,}")
print(f"Pre-COVID mean RevPAR growth: {train.loc[pre_mask, 'target'].mean()*100:.1f}%")
print(f"Post-COVID mean RevPAR growth: {train.loc[post_mask, 'target'].mean()*100:.1f}%")

# Key features representing renter concerns
renter_features = {
    'aarp_met_health_hospital': 'Hospital Access',
    'healthcare_access': 'Healthcare Overall',
    'aarp_met_prox_trans': 'Transit Proximity',
    'rent_percentile': 'Rent Position',
    'aarp_met_prox_park': 'Park Access',
    'supply_growth_pct': 'New Supply %',
    'aarp_met_trans_walk': 'Walkability Score',
    'num_grocery_ta': 'Grocery Stores',
    'num_food_ta': 'Food Options',
    'property_age': 'Property Age',
}

# Calculate correlation with target for each period
results = []
for feat, label in renter_features.items():
    if feat in train.columns and pd.api.types.is_numeric_dtype(train[feat]):
        corr_pre = train.loc[pre_mask, [feat, 'target']].corr().iloc[0, 1]
        corr_post = train.loc[post_mask, [feat, 'target']].corr().iloc[0, 1]
        results.append({
            'Feature': label,
            'Pre-COVID': corr_pre,
            'Post-COVID': corr_post,
            'Change': corr_post - corr_pre
        })

df_compare = pd.DataFrame(results).sort_values('Change', ascending=True)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(df_compare))
width = 0.35

bars1 = ax.barh(y_pos - width/2, df_compare['Pre-COVID'], width,
                label='Pre-COVID (2015-2020)', color='#3498db', alpha=0.8)
bars2 = ax.barh(y_pos + width/2, df_compare['Post-COVID'], width,
                label='Post-COVID (2022-2025)', color='#e74c3c', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_compare['Feature'], fontsize=11)
ax.set_xlabel('Correlation with RevPAR Growth', fontsize=12)
ax.set_title('Performance Drivers: Pre vs Post COVID\n(From Renter\'s Neighborhood Perspective)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlim(-0.35, 0.35)
ax.grid(axis='x', alpha=0.3)

# Find Hospital Access position and annotate correctly
hospital_idx = df_compare[df_compare['Feature'] == 'Hospital Access'].index[0]
hospital_y = list(df_compare['Feature']).index('Hospital Access')

ax.annotate('Healthcare flipped from\nNEGATIVE to POSITIVE\npost-COVID',
            xy=(0.28, hospital_y + 0.15),
            xytext=(0.15, hospital_y - 2.5),
            fontsize=9, color='#c0392b', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fadbd8', edgecolor='#c0392b'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_pre_vs_post_drivers.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig1_pre_vs_post_drivers.png")

# ============================================================================
# FIGURE 2: Renter's Perspective - Better Categorization (FIXED)
# ============================================================================

print("\n=== FIGURE 2: Renter's Perspective on Performance Drivers ===")

# Comprehensive category mapping
category_mapping = {
    # Healthcare & Safety
    'health_hospital_x_post': 'Healthcare',
    'aarp_met_health_hospital': 'Healthcare',
    'healthcare_access': 'Healthcare',
    'aarp_score_health': 'Healthcare',
    'aarp_met_health_smoke': 'Healthcare',
    'aarp_met_health_obese': 'Healthcare',
    'aarp_met_health_exercise': 'Healthcare',
    'aarp_met_health_sate': 'Healthcare',
    'aarp_met_health_short': 'Healthcare',

    # Local Amenities (15-min city)
    'num_food_ta': 'Local Amenities',
    'num_grocery_ta': 'Local Amenities',
    'num_grocery_highend_ta': 'Local Amenities',
    'total_amenities_ta': 'Local Amenities',
    'food_total_food': 'Local Amenities',
    'food_count_high': 'Local Amenities',
    'dining_quality_score': 'Local Amenities',
    'walkable_dining': 'Local Amenities',
    'aarp_met_prox_park': 'Local Amenities',
    'aarp_met_prox_market': 'Local Amenities',
    'aarp_met_prox_lib': 'Local Amenities',
    'aarp_met_prox_activity': 'Local Amenities',
    'aarp_score_prox': 'Local Amenities',

    # Transit & Mobility
    'aarp_score_trans': 'Mobility',
    'aarp_met_trans_walk': 'Mobility',
    'aarp_met_trans_access': 'Mobility',
    'aarp_met_trans_delay': 'Mobility',
    'aarp_met_trans_cost': 'Mobility',
    'aarp_met_trans_freq': 'Mobility',
    'aarp_met_trans_fatal': 'Mobility',
    'aarp_met_trans_limit': 'Mobility',
    'aarp_met_prox_trans': 'Mobility',
    'aarp_met_prox_auto': 'Mobility',

    # Housing Quality
    'aarp_score_house': 'Housing Quality',
    'aarp_met_house_access_step': 'Housing Quality',
    'aarp_met_house_burden': 'Housing Quality',
    'aarp_met_house_cost': 'Housing Quality',
    'aarp_met_house_multifam': 'Housing Quality',
    'aarp_met_house_subsidy': 'Housing Quality',

    # Economic Position
    'rent_percentile': 'Pricing Power',
    'ownrent_avg_rent': 'Pricing Power',
    'ownrent_spread_pct': 'Pricing Power',
    'affordability_index': 'Pricing Power',

    # Property Characteristics
    'property_age': 'Property Vintage',
    'age_x_post': 'Property Vintage',
    'age_squared': 'Property Vintage',
    'age_cubed': 'Property Vintage',
    'age_log': 'Property Vintage',
    'yearbuilt': 'Property Vintage',
    'years_since_renov': 'Property Vintage',
    'year_renov': 'Property Vintage',
    'numunits': 'Property Vintage',
    'areaperunit': 'Property Vintage',
    'type_main_te': 'Property Vintage',
    'type_sub': 'Property Vintage',
    'type_main': 'Property Vintage',
    'value_tier': 'Property Vintage',
    'vintage_category': 'Property Vintage',
    'recently_renovated': 'Property Vintage',
    'never_renovated': 'Property Vintage',
    'age_x_class_d': 'Property Vintage',

    # Market/Location
    'mrkt_name_te': 'Market Location',
    'state_te': 'Market Location',
    'state': 'Market Location',
    'mrkt_name': 'Market Location',
    'MARKET': 'Market Location',
    'city': 'Market Location',
    'state_metro': 'Market Location',
    'zip_metro': 'Market Location',
    'msa_norm_dist': 'Market Location',
    'msa_ring': 'Market Location',
    'msa_ring_te': 'Market Location',
    'is_sunbelt': 'Market Location',
    'is_texas': 'Market Location',
    'drv10_area': 'Market Location',
    'drv15_area': 'Market Location',
    'drv30_area': 'Market Location',

    # Supply Competition
    'supply_growth_pct': 'Supply Competition',
    'supply_new_units': 'Supply Competition',
    'supply_baseline_units': 'Supply Competition',

    # Environment
    'aarp_score_env': 'Environment',
    'aarp_met_env_air': 'Environment',
    'aarp_met_env_water': 'Environment',
    'aarp_met_env_road': 'Environment',
    'aarp_met_env_pollute': 'Environment',
    'aarp_met_prox_vacant': 'Environment',
    'aarp_met_prox_sec': 'Environment',

    # Community
    'aarp_score_engage': 'Community',
    'aarp_met_engage_social': 'Community',
    'aarp_met_engage_civic': 'Community',
    'aarp_met_engage_culture': 'Community',
    'aarp_met_engage_vote': 'Community',
    'aarp_met_engage_broad': 'Community',

    # Economic Opportunity
    'aarp_score_opp': 'Economic Opportunity',
    'aarp_met_opp_income': 'Economic Opportunity',
    'aarp_met_opp_jobs': 'Economic Opportunity',
    'aarp_met_opp_age': 'Economic Opportunity',
    'aarp_met_opp_grad': 'Economic Opportunity',
}

# Filter out time window features (meta-features, not renter decisions)
time_features = ['time_window_tag', 'time_window_label', 'is_post_covid', 'is_pre_covid']
feat_imp_filtered = feat_imp[~feat_imp['feature'].isin(time_features)].head(80).copy()
feat_imp_filtered['category'] = feat_imp_filtered['feature'].map(category_mapping).fillna('Other')

# Check what's in "Other"
other_features = feat_imp_filtered[feat_imp_filtered['category'] == 'Other']['feature'].tolist()
if other_features:
    print(f"Uncategorized features: {other_features[:10]}")

category_importance = feat_imp_filtered.groupby('category')['avg_importance'].sum().sort_values(ascending=True)

# Remove "Other" if small, or keep if significant
if 'Other' in category_importance and category_importance['Other'] < 0.01:
    category_importance = category_importance.drop('Other')

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

colors = {
    'Healthcare': '#e74c3c',
    'Local Amenities': '#27ae60',
    'Market Location': '#9b59b6',
    'Pricing Power': '#3498db',
    'Property Vintage': '#f39c12',
    'Mobility': '#1abc9c',
    'Housing Quality': '#2ecc71',
    'Supply Competition': '#95a5a6',
    'Environment': '#16a085',
    'Community': '#e67e22',
    'Economic Opportunity': '#8e44ad',
    'Other': '#bdc3c7'
}

bar_colors = [colors.get(cat, '#bdc3c7') for cat in category_importance.index]

bars = ax.barh(range(len(category_importance)),
               category_importance.values * 100,
               color=bar_colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(range(len(category_importance)))
ax.set_yticklabels(category_importance.index, fontsize=11)
ax.set_xlabel('Aggregated Feature Importance (%)', fontsize=12)
ax.set_title("What Neighborhood Qualities Drive RevPAR?\n(Renter's Decision Framework)",
             fontsize=14, fontweight='bold')

# Add value labels
for i, (cat, val) in enumerate(category_importance.items()):
    ax.text(val * 100 + 0.5, i, f'{val*100:.1f}%', va='center', fontsize=10, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(category_importance.values) * 100 * 1.2)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_renter_perspective.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig2_renter_perspective.png")

# ============================================================================
# FIGURE 3: Healthcare Deep Dive - Which Populations Care Most?
# ============================================================================

print("\n=== FIGURE 3: Healthcare Sensitivity by Population Segment ===")

# We'll analyze correlation between hospital access and RevPAR growth
# across different segments (post-COVID only, since that's where it matters)

post_data = train[post_mask].copy()

# Function to calculate correlation for a segment
def segment_corr(df, segment_col, segment_val, feature='aarp_met_health_hospital'):
    mask = df[segment_col] == segment_val
    if mask.sum() < 100:  # Need enough samples
        return np.nan
    return df.loc[mask, [feature, 'target']].corr().iloc[0, 1]

# Analyze by multiple dimensions
segments = {}

# By Property Class (A, B, C, D)
for cls in ['A', 'A-', 'A+', 'B', 'B+', 'B-', 'C', 'C+', 'C-', 'D']:
    mask = post_data['type_main'] == cls
    if mask.sum() >= 100:
        corr = post_data.loc[mask, ['aarp_met_health_hospital', 'target']].corr().iloc[0, 1]
        # Simplify to A/B/C/D
        simple_cls = cls[0]
        if simple_cls not in segments:
            segments[f'Class {simple_cls}'] = []
        segments[f'Class {simple_cls}'].append(corr)

# Average within class groups
class_corrs = {k: np.mean(v) for k, v in segments.items()}

# By Value Tier
tier_corrs = {}
for tier in ['premium', 'mid', 'value']:
    corr = segment_corr(post_data, 'value_tier', tier)
    if not np.isnan(corr):
        tier_corrs[tier.capitalize()] = corr

# By Location (MSA Ring)
location_corrs = {}
location_labels = {
    'downtown': 'Downtown',
    'outercore': 'Outer Core',
    'innersuburb': 'Inner Suburb',
    'outersuburb': 'Outer Suburb'
}
for loc, label in location_labels.items():
    corr = segment_corr(post_data, 'msa_ring', loc)
    if not np.isnan(corr):
        location_corrs[label] = corr

# By Vintage
vintage_corrs = {}
vintage_labels = {
    'pre_1980': 'Pre-1980',
    'eighties': '1980s',
    'nineties': '1990s',
    'two_thousands': '2000s',
    'modern': 'Modern (2010+)'
}
for vin, label in vintage_labels.items():
    corr = segment_corr(post_data, 'vintage_category', vin)
    if not np.isnan(corr):
        vintage_corrs[label] = corr

# Create multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: By Property Class
ax1 = axes[0, 0]
classes = list(class_corrs.keys())
class_vals = list(class_corrs.values())
colors1 = ['#e74c3c' if v > 0 else '#3498db' for v in class_vals]
bars1 = ax1.bar(classes, class_vals, color=colors1, edgecolor='white', linewidth=1.5)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax1.set_title('By Property Class', fontsize=12, fontweight='bold')
ax1.set_ylim(-0.1, 0.35)
for bar, val in zip(bars1, class_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9)

# Panel 2: By Value Tier
ax2 = axes[0, 1]
tiers = list(tier_corrs.keys())
tier_vals = list(tier_corrs.values())
colors2 = ['#e74c3c' if v > 0 else '#3498db' for v in tier_vals]
bars2 = ax2.bar(tiers, tier_vals, color=colors2, edgecolor='white', linewidth=1.5)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax2.set_title('By Value Tier', fontsize=12, fontweight='bold')
ax2.set_ylim(-0.1, 0.35)
for bar, val in zip(bars2, tier_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9)

# Panel 3: By Location
ax3 = axes[1, 0]
locs = list(location_corrs.keys())
loc_vals = list(location_corrs.values())
colors3 = ['#e74c3c' if v > 0 else '#3498db' for v in loc_vals]
bars3 = ax3.bar(locs, loc_vals, color=colors3, edgecolor='white', linewidth=1.5)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax3.set_title('By Location Type', fontsize=12, fontweight='bold')
ax3.set_ylim(-0.1, 0.35)
ax3.tick_params(axis='x', rotation=15)
for bar, val in zip(bars3, loc_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9)

# Panel 4: By Vintage
ax4 = axes[1, 1]
vins = list(vintage_corrs.keys())
vin_vals = list(vintage_corrs.values())
colors4 = ['#e74c3c' if v > 0 else '#3498db' for v in vin_vals]
bars4 = ax4.bar(vins, vin_vals, color=colors4, edgecolor='white', linewidth=1.5)
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.set_ylabel('Correlation with RevPAR Growth', fontsize=10)
ax4.set_title('By Property Vintage', fontsize=12, fontweight='bold')
ax4.set_ylim(-0.1, 0.35)
ax4.tick_params(axis='x', rotation=20)
for bar, val in zip(bars4, vin_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.2f}',
             ha='center', va='bottom', fontsize=9)

fig.suptitle('Who Values Healthcare Access Most? (Post-COVID Analysis)\n'
             'Correlation between Hospital Proximity and RevPAR Growth by Segment',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_healthcare_by_population.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPUT_DIR}/fig3_healthcare_by_population.png")

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print("\n1. PRE vs POST COVID SHIFT:")
print(f"   - Pre-COVID RevPAR growth: +21.7%")
print(f"   - Post-COVID RevPAR growth: -5.2%")
print("   - Hospital access flipped from NEGATIVE to POSITIVE correlation")

print("\n2. RENTER'S PERSPECTIVE (Feature Importance):")
for cat, val in category_importance.sort_values(ascending=False).head(5).items():
    print(f"   - {cat}: {val*100:.1f}%")

print("\n3. WHO VALUES HEALTHCARE MOST (Post-COVID):")
print("   By Class:", class_corrs)
print("   By Tier:", tier_corrs)
print("   By Location:", location_corrs)
print("   By Vintage:", vintage_corrs)

print("\nDone!")
