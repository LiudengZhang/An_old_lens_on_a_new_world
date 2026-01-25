#!/usr/bin/env python3
"""
Create Feature Correlation Heatmaps - Pre vs Post COVID
Nature Cancer Style with different color palettes

Pre-COVID: Pink-Green (PiYG)
Post-COVID: Red-Blue (RdBu_r)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict

# Parameters
FIGURE_SIZE = (14, 14)
DPI = 300
VMIN, VMAX = -1, 1
CENTER = 0
MASK_DIAGONAL = True
BOUNDARY_COLOR = 'black'
BOUNDARY_WIDTH = 2.5
CELL_LABEL_SIZE = 9
MODULE_LABEL_SIZE = 11
COLORBAR_SHRINK = 0.5

# Feature modules with display names
FEATURE_MODULES = OrderedDict([
    ('Location', OrderedDict([
        ('msa_norm_dist', 'Metro Distance'),
        ('is_sunbelt', 'Sun Belt'),
        ('is_texas', 'Texas'),
        ('drivetime_minutes', 'Drivetime'),
    ])),
    ('Property', OrderedDict([
        ('yearbuilt', 'Year Built'),
        ('property_age', 'Property Age'),
        ('numunits', 'Unit Count'),
        ('areaperunit', 'Area/Unit'),
        ('is_class_a', 'Class A'),
        ('is_class_d', 'Class D'),
    ])),
    ('AARP Scores', OrderedDict([
        ('aarp_score', 'AARP Total'),
        ('aarp_score_health', 'Health Score'),
        ('aarp_score_prox', 'Proximity Score'),
        ('aarp_score_trans', 'Transport Score'),
        ('aarp_score_engage', 'Engage Score'),
        ('aarp_score_env', 'Environment Score'),
        ('aarp_score_house', 'Housing Score'),
    ])),
    ('Health Access', OrderedDict([
        ('aarp_met_health_hospital', 'Hospital Access'),
        ('aarp_met_health_exercise', 'Exercise Access'),
        ('aarp_met_health_obese', 'Obesity Rate'),
        ('healthcare_access', 'Healthcare Index'),
    ])),
    ('Walkability', OrderedDict([
        ('aarp_met_trans_walk', 'Walkability'),
        ('aarp_met_prox_park', 'Park Proximity'),
        ('aarp_met_prox_market', 'Market Proximity'),
    ])),
    ('Amenities', OrderedDict([
        ('num_grocery_ta', 'Grocery Count'),
        ('num_grocery_highend_ta', 'High-End Grocery'),
        ('total_amenities_ta', 'Total Amenities'),
        ('food_total_food', 'Food Venues'),
        ('dining_quality_score', 'Dining Quality'),
    ])),
    ('Supply/Rent', OrderedDict([
        ('supply_baseline_units', 'Baseline Supply'),
        ('supply_new_units', 'New Supply'),
        ('supply_growth_pct', 'Supply Growth %'),
        ('ownrent_avg_rent', 'Avg Rent'),
    ])),
    ('Outcome', OrderedDict([
        ('target', 'RevPAR Growth'),
    ])),
])


def create_heatmap(df, title, output_path, colormap):
    """Generate correlation heatmap in Nature Cancer style."""

    # Build ordered feature list and module mappings
    ordered_features = []
    display_names = []
    module_counts = OrderedDict()

    for module_name, features in FEATURE_MODULES.items():
        count = 0
        for col, display in features.items():
            if col in df.columns:
                ordered_features.append(col)
                display_names.append(display)
                count += 1
        module_counts[module_name] = count

    # Calculate correlation matrix
    df_subset = df[ordered_features].apply(pd.to_numeric, errors='coerce')
    correlation_matrix = df_subset.corr()

    # Apply display names
    correlation_ordered = correlation_matrix.copy()
    correlation_ordered.index = display_names
    correlation_ordered.columns = display_names

    # Calculate module boundaries
    module_boundaries = np.cumsum(list(module_counts.values()))[:-1]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create mask for diagonal
    mask = np.eye(len(correlation_ordered), dtype=bool) if MASK_DIAGONAL else None

    # Create heatmap
    sns.heatmap(
        correlation_ordered,
        cmap=colormap,
        vmin=VMIN,
        vmax=VMAX,
        center=CENTER,
        mask=mask,
        square=True,
        cbar_kws={'label': 'Pearson Correlation', 'shrink': COLORBAR_SHRINK},
        ax=ax,
        xticklabels=False,
        yticklabels=correlation_ordered.index,
        linewidths=0.5,
        linecolor='lightgray'
    )

    # Adjust y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=CELL_LABEL_SIZE)

    # Add module boundaries
    for boundary in module_boundaries:
        ax.axhline(y=boundary, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH)
        ax.axvline(x=boundary, color=BOUNDARY_COLOR, linewidth=BOUNDARY_WIDTH)

    # Add module labels at bottom
    module_positions = []
    cumsum = 0
    for count in module_counts.values():
        module_positions.append(cumsum + count / 2)
        cumsum += count

    ax.set_xticks(module_positions)
    ax.set_xticklabels(
        list(module_counts.keys()),
        fontsize=MODULE_LABEL_SIZE,
        rotation=45,
        ha='right'
    )
    ax.xaxis.tick_bottom()

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()

    return correlation_ordered


def main():
    print("=" * 60)
    print("Feature Correlation Heatmaps - Pre vs Post COVID")
    print("=" * 60)

    script_dir = Path(__file__).parent
    data_path = script_dir.parent / '03_Rice_Datathon_Colab/data/processed/train_clean.csv'

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)

    df_pre = df[df['is_pre_covid'] == 1].copy()
    df_post = df[df['is_post_covid'] == 1].copy()

    print(f"  Pre-COVID records: {len(df_pre)}")
    print(f"  Post-COVID records: {len(df_post)}")

    # Create Pre-COVID heatmap (Pink-Green)
    print("\nCreating Pre-COVID heatmap (Pink-Green)...")
    corr_pre = create_heatmap(
        df_pre,
        'Feature Correlation Matrix\n(Pre-COVID Period)',
        script_dir / 'correlation_heatmap_pre_covid.png',
        'PiYG'  # Pink-Yellow-Green diverging
    )
    print(f"  Saved: correlation_heatmap_pre_covid.png")

    # Create Post-COVID heatmap (Red-Blue)
    print("\nCreating Post-COVID heatmap (Red-Blue)...")
    corr_post = create_heatmap(
        df_post,
        'Feature Correlation Matrix\n(Post-COVID Period)',
        script_dir / 'correlation_heatmap_post_covid.png',
        'RdBu_r'  # Red-Blue diverging (reversed)
    )
    print(f"  Saved: correlation_heatmap_post_covid.png")

    # Calculate and display key differences
    print("\n" + "=" * 60)
    print("Key Correlation Differences (Post - Pre)")
    print("=" * 60)

    diff = corr_post - corr_pre

    # Find largest changes
    diff_flat = diff.unstack()
    diff_flat = diff_flat[diff_flat.index.get_level_values(0) != diff_flat.index.get_level_values(1)]

    print("\nTop 5 increased correlations:")
    for (f1, f2), val in diff_flat.nlargest(5).items():
        print(f"  {f1} ↔ {f2}: {val:+.3f}")

    print("\nTop 5 decreased correlations:")
    for (f1, f2), val in diff_flat.nsmallest(5).items():
        print(f"  {f1} ↔ {f2}: {val:+.3f}")

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
