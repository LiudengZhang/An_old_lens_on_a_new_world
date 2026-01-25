"""Visualization module for chatbot responses using Plotly.

Styled to match Nature Methods / high-impact journal standards.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import re
from typing import Optional, Tuple
from config import DATA_FILES


def extract_number(text: str, default: int = 10) -> int:
    """Extract number from query like 'top 20' or 'top 10 features'."""
    patterns = [
        r'top\s*(\d+)',      # "top 20", "top20"
        r'(\d+)\s*feature',  # "20 features"
        r'(\d+)\s*most',     # "20 most important"
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return min(int(match.group(1)), 30)  # Cap at 30 for readability
    return default

# =============================================================================
# Nature-Style Color Palette (muted, colorblind-friendly)
# =============================================================================
COLORS = {
    "pre": "#4575b4",       # Muted blue
    "post": "#d73027",     # Muted red
    "primary": "#4575b4",  # Blue
    "secondary": "#7570b3", # Purple
    "neutral": "#878787",  # Gray
    "green": "#1a9850",    # Green accent
    "orange": "#fc8d59",   # Orange accent
}

# Sequential palette for gradients
COLOR_SCALE_BLUE = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"]
COLOR_SCALE_DIVERGING = ["#d73027", "#fc8d59", "#fee090", "#e0f3f8", "#91bfdb", "#4575b4"]


def apply_nature_style(fig: go.Figure, title: str = None) -> go.Figure:
    """Apply Nature Methods journal styling to a Plotly figure."""
    fig.update_layout(
        # Typography
        font=dict(
            family="Arial, Helvetica, sans-serif",
            size=11,
            color="#333333"
        ),
        title=dict(
            text=title,
            font=dict(size=13, family="Arial, Helvetica, sans-serif"),
            x=0.0,
            xanchor="left"
        ) if title else None,

        # Clean background
        paper_bgcolor="white",
        plot_bgcolor="white",

        # Margins
        margin=dict(l=60, r=20, t=50, b=50),

        # Legend styling
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0,
            font=dict(size=10),
        ),

        # No hover background
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#cccccc",
            font=dict(family="Arial", size=11, color="#333333")
        ),
    )

    # Axis styling - clean, minimal
    axis_style = dict(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="#333333",
        tickfont=dict(size=10),
        ticks="outside",
        ticklen=4,
        tickwidth=1,
        tickcolor="#333333",
        title_font=dict(size=11),
        zeroline=False,
    )

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    return fig


def get_visualization(question: str, result: dict) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Dispatch to appropriate visualization based on question content.

    Returns:
        Tuple of (plotly figure, caption) or (None, None) if no visualization applies.
    """
    q = question.lower()

    # Suburb × Drivetime interaction matrix
    if any(kw in q for kw in ["suburb", "drivetime interaction", "ring", "15-minute city", "15 minute city", "urban vs suburban"]):
        return create_suburb_drivetime_matrix()

    # Feature rank change
    if any(kw in q for kw in ["rank change", "rank shift", "moved up", "moved down", "ranking change"]):
        n = extract_number(question, default=15)
        return create_feature_rank_change(top_n=n)

    # City comparison - check for specific city names
    city_names = [
        "houston", "phoenix", "dallas", "austin", "atlanta", "fort worth",
        "san antonio", "charlotte", "tampa", "orlando", "raleigh", "nashville",
        "jacksonville", "tucson", "miami", "fort lauderdale", "memphis"
    ]
    mentioned_cities = [c for c in city_names if c in q]
    if len(mentioned_cities) >= 2:
        return create_city_comparison(mentioned_cities)

    # Feature correlation heatmap
    if any(kw in q for kw in ["correlation", "correlated", "relationship between"]):
        n = extract_number(question, default=12)
        return create_feature_correlation(top_n=n)

    # Feature importance change (pre vs post)
    if any(kw in q for kw in ["importance change", "feature shift", "importance shift"]):
        return create_importance_change()

    # Quartile confusion matrix
    if any(kw in q for kw in ["quartile", "confusion", "classification accuracy"]):
        return create_quartile_confusion()

    # Feature category breakdown
    if any(kw in q for kw in ["category", "livability", "feature type", "feature category"]):
        return create_category_breakdown()

    # Amenity analysis
    if any(kw in q for kw in ["amenity type", "amenities breakdown", "amenity comparison"]):
        return create_amenity_comparison()

    # Feature importance
    if any(kw in q for kw in ["feature", "important", "top feature", "rank"]):
        n = extract_number(question, default=15)
        return create_feature_importance(top_n=n)

    # Model performance by city
    if any(kw in q for kw in ["r2", "rmse", "accuracy", "model performance", "how accurate"]):
        return create_model_performance()

    # Drivetime comparison
    if any(kw in q for kw in ["drivetime", "drive time", "10min", "15min", "30min"]):
        return create_drivetime_comparison()

    # COVID impact overview
    if any(kw in q for kw in ["covid impact", "covid change", "before and after covid"]):
        return create_covid_impact_overview()

    return None, None


def create_city_comparison(city_keywords: list) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create bar chart comparing RevPAR growth for mentioned cities."""
    path = DATA_FILES["city_summary"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Match cities by keyword
    matched_rows = []
    for _, row in df.iterrows():
        city_lower = row["city"].lower()
        for kw in city_keywords:
            if kw in city_lower:
                matched_rows.append(row)
                break

    if len(matched_rows) < 2:
        return None, None

    subset = pd.DataFrame(matched_rows)
    subset["city_short"] = subset["city"].apply(lambda x: x.split(",")[0].split("-")[0])
    subset = subset.sort_values("avg_revpar_growth_pre", ascending=True)

    # Create figure with go.Figure for more control
    fig = go.Figure()

    # Pre-COVID bars
    fig.add_trace(go.Bar(
        y=subset["city_short"],
        x=subset["avg_revpar_growth_pre"] * 100,
        name="Pre-COVID",
        orientation="h",
        marker_color=COLORS["pre"],
        hovertemplate="%{y}: %{x:.1f}%<extra>Pre-COVID</extra>"
    ))

    # Post-COVID bars
    fig.add_trace(go.Bar(
        y=subset["city_short"],
        x=subset["avg_revpar_growth_post"] * 100,
        name="Post-COVID",
        orientation="h",
        marker_color=COLORS["post"],
        hovertemplate="%{y}: %{x:.1f}%<extra>Post-COVID</extra>"
    ))

    fig.update_layout(
        barmode="group",
        height=max(280, len(subset) * 70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis_title="RevPAR Growth (%)",
        yaxis_title="",
    )

    # Zero reference line
    fig.add_vline(x=0, line_color="#333333", line_width=0.8)

    # Apply Nature style
    fig = apply_nature_style(fig, "RevPAR Growth: Pre vs Post COVID")

    return fig, f"RevPAR growth comparison: {', '.join(subset['city_short'].tolist())}"


def create_feature_importance(top_n: int = 15) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create horizontal bar chart of top features."""
    path = DATA_FILES["feature_importance"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Get top features by post-COVID importance
    top_features = df.nsmallest(top_n, "rank_post").copy()
    top_features = top_features.sort_values("importance_post", ascending=True)

    # Clean feature names
    top_features["feature_clean"] = top_features["feature_name"].apply(
        lambda x: x.replace("_", " ").replace("aarp met ", "").replace("aarp score ", "")[:35]
    )

    # Create figure with gradient color based on importance
    max_imp = top_features["importance_post"].max()
    colors = [f"rgba(69, 117, 180, {0.4 + 0.6 * (v / max_imp)})"
              for v in top_features["importance_post"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=top_features["feature_clean"],
        x=top_features["importance_post"],
        orientation="h",
        marker=dict(
            color=top_features["importance_post"],
            colorscale=[[0, "#c6dbef"], [1, "#2171b5"]],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        height=450,
        xaxis_title="Importance Score",
        yaxis_title="",
        showlegend=False,
    )

    # Apply Nature style
    fig = apply_nature_style(fig, f"Top {top_n} Most Important Features (Post-COVID)")

    return fig, f"Top {top_n} features by importance in post-COVID model"


def create_model_performance() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create bar chart of model R² by city."""
    path = DATA_FILES["model_performance"]
    if not path.exists():
        return None, None

    with open(path) as f:
        data = json.load(f)

    by_city = data.get("by_city", {})
    if not by_city:
        return None, None

    # Prepare data
    cities = []
    r2_scores = []
    rmse_scores = []
    for city, metrics in by_city.items():
        cities.append(city.split(",")[0].split("-")[0])
        r2_scores.append(metrics["r2"])
        rmse_scores.append(metrics["rmse"])

    plot_df = pd.DataFrame({"City": cities, "R²": r2_scores, "RMSE": rmse_scores})
    plot_df = plot_df.sort_values("R²", ascending=True)

    # Nature-style diverging color scale (red-yellow-green but muted)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["City"],
        x=plot_df["R²"],
        orientation="h",
        marker=dict(
            color=plot_df["R²"],
            colorscale=[[0, "#d73027"], [0.5, "#fee090"], [1, "#1a9850"]],
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(text="R²", font=dict(size=11)),
                tickfont=dict(size=10),
                thickness=12,
                len=0.6,
            ),
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>R²: %{x:.3f}<extra></extra>"
    ))

    fig.update_layout(
        height=420,
        xaxis_title="R² Score",
        yaxis_title="",
        xaxis_range=[0, 1],
    )

    # Reference line at R² = 0.5
    fig.add_vline(x=0.5, line_dash="dot", line_color="#878787", line_width=1)

    # Apply Nature style
    fig = apply_nature_style(fig, "Model Performance by City")

    return fig, "Model R² scores by city (higher is better)"


def create_drivetime_comparison() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create comparison of drivetime definitions."""
    path = DATA_FILES["drivetime_analysis"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Model R² by Drivetime", "Average Amenities Captured"),
        horizontal_spacing=0.18
    )

    # R² comparison
    fig.add_trace(
        go.Bar(
            x=df["drivetime"],
            y=df["model_r2"],
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            text=df["model_r2"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(size=10),
            name="R²",
            hovertemplate="<b>%{x}</b><br>R²: %{y:.3f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Amenity count
    fig.add_trace(
        go.Bar(
            x=df["drivetime"],
            y=df["avg_amenity_count"],
            marker=dict(color=COLORS["secondary"], line=dict(width=0)),
            text=df["avg_amenity_count"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            textfont=dict(size=10),
            name="Amenities",
            hovertemplate="<b>%{x}</b><br>Avg amenities: %{y:.0f}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=60, r=20, t=70, b=50),
    )

    # Update y-axis ranges
    fig.update_yaxes(range=[0.58, 0.70], row=1, col=1)

    # Apply Nature style
    fig = apply_nature_style(fig, "Drivetime Definition Comparison")

    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, family="Arial, Helvetica, sans-serif", color="#333333")

    return fig, "Drivetime comparison: 15min provides best model fit"


def create_covid_impact_overview() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create overview of COVID impact across top cities."""
    path = DATA_FILES["city_summary"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Get top 10 cities by property count
    top_cities = df.nlargest(10, "property_count").copy()
    top_cities["city_short"] = top_cities["city"].apply(lambda x: x.split(",")[0].split("-")[0])
    top_cities = top_cities.sort_values("avg_revpar_growth_pre", ascending=True)

    # Create figure with go.Figure for more control
    fig = go.Figure()

    # Pre-COVID bars
    fig.add_trace(go.Bar(
        y=top_cities["city_short"],
        x=top_cities["avg_revpar_growth_pre"] * 100,
        name="Pre-COVID",
        orientation="h",
        marker=dict(color=COLORS["pre"], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.1f}%<extra>Pre-COVID</extra>"
    ))

    # Post-COVID bars
    fig.add_trace(go.Bar(
        y=top_cities["city_short"],
        x=top_cities["avg_revpar_growth_post"] * 100,
        name="Post-COVID",
        orientation="h",
        marker=dict(color=COLORS["post"], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.1f}%<extra>Post-COVID</extra>"
    ))

    fig.update_layout(
        barmode="group",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        xaxis_title="RevPAR Growth (%)",
        yaxis_title="",
    )

    # Zero reference line
    fig.add_vline(x=0, line_color="#333333", line_width=0.8)

    # Apply Nature style
    fig = apply_nature_style(fig, "COVID Impact: Top 10 Markets by Property Count")

    return fig, "RevPAR growth shift across major markets"


def create_feature_correlation(top_n: int = 12) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create correlation heatmap for top features."""
    importance_path = DATA_FILES["feature_importance"]
    training_path = DATA_FILES["training_data"]

    if not importance_path.exists() or not training_path.exists():
        return None, None

    # Get top features by importance
    importance_df = pd.read_csv(importance_path)
    top_features = importance_df.nsmallest(top_n, "rank_post")["feature_name"].tolist()

    # Load training data and compute correlations
    train_df = pd.read_csv(training_path)
    available_features = [f for f in top_features if f in train_df.columns]

    if len(available_features) < 3:
        return None, None

    corr_matrix = train_df[available_features].corr()

    # Clean feature names for display
    labels = [f.replace("_", " ")[:20] for f in available_features]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=labels,
        y=labels,
        colorscale=[[0, "#d73027"], [0.5, "#ffffbf"], [1, "#4575b4"]],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title=dict(text="Correlation", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=12,
            len=0.6,
        ),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>r = %{z:.2f}<extra></extra>"
    ))

    fig.update_layout(
        height=500,
        xaxis=dict(tickangle=45),
    )

    fig = apply_nature_style(fig, f"Feature Correlation Matrix (Top {len(available_features)})")

    return fig, f"Correlation matrix of top {len(available_features)} most important features"


def create_importance_change() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create scatter plot showing feature importance change pre vs post COVID."""
    path = DATA_FILES["feature_importance"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Filter to features with non-zero importance in both periods
    df = df[(df["importance_pre"] > 0) | (df["importance_post"] > 0)].copy()
    df = df.head(30)  # Top 30 features

    # Clean names
    df["feature_clean"] = df["feature_name"].apply(lambda x: x.replace("_", " ")[:25])

    # Color by change direction
    df["change_color"] = df["importance_change_pct"].apply(
        lambda x: COLORS["post"] if x > 50 else (COLORS["pre"] if x < -50 else COLORS["neutral"])
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["importance_pre"],
        y=df["importance_post"],
        mode="markers+text",
        text=df["feature_clean"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=10,
            color=df["change_color"],
            line=dict(width=1, color="#333333")
        ),
        hovertemplate="<b>%{text}</b><br>Pre: %{x:.4f}<br>Post: %{y:.4f}<extra></extra>"
    ))

    # Add diagonal reference line
    max_val = max(df["importance_pre"].max(), df["importance_post"].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode="lines",
        line=dict(color="#878787", dash="dash", width=1),
        name="No change",
        showlegend=False
    ))

    fig.update_layout(
        height=500,
        xaxis_title="Pre-COVID Importance",
        yaxis_title="Post-COVID Importance",
    )

    fig = apply_nature_style(fig, "Feature Importance: Pre vs Post COVID")

    return fig, "Features above diagonal gained importance post-COVID; below lost importance"


def create_quartile_confusion() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create confusion matrix heatmap for quartile predictions."""
    path = DATA_FILES["model_performance"]
    if not path.exists():
        return None, None

    with open(path) as f:
        data = json.load(f)

    cm = data.get("quartile_accuracy", {}).get("confusion_matrix")
    if not cm:
        return None, None

    cm = np.array(cm)
    labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]

    # Normalize by row (actual) to show recall per class
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    fig = go.Figure(data=go.Heatmap(
        z=cm_pct,
        x=labels,
        y=labels,
        colorscale=[[0, "#ffffff"], [0.5, "#c6dbef"], [1, "#2171b5"]],
        zmin=0,
        zmax=100,
        colorbar=dict(
            title=dict(text="% of Actual", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=12,
            len=0.6,
        ),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>%{z:.1f}%<extra></extra>"
    ))

    # Add text annotations
    annotations = []
    for i, row in enumerate(cm_pct):
        for j, val in enumerate(row):
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=f"{val:.0f}%",
                showarrow=False,
                font=dict(color="white" if val > 50 else "#333333", size=11)
            ))

    fig.update_layout(
        height=450,
        xaxis_title="Predicted Quartile",
        yaxis_title="Actual Quartile",
        annotations=annotations,
    )

    fig = apply_nature_style(fig, "Quartile Prediction Accuracy")

    exact_match = data["quartile_accuracy"]["exact_match_rate"]
    return fig, f"Quartile confusion matrix (exact match: {exact_match:.1%})"


def create_category_breakdown() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create bar chart of feature importance by category."""
    path = DATA_FILES["feature_importance"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Aggregate by category
    category_stats = df.groupby("feature_category").agg({
        "importance_pre": "sum",
        "importance_post": "sum",
        "feature_name": "count"
    }).rename(columns={"feature_name": "count"})

    category_stats = category_stats.sort_values("importance_post", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=category_stats.index,
        x=category_stats["importance_pre"],
        name="Pre-COVID",
        orientation="h",
        marker=dict(color=COLORS["pre"], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra>Pre-COVID</extra>"
    ))

    fig.add_trace(go.Bar(
        y=category_stats.index,
        x=category_stats["importance_post"],
        name="Post-COVID",
        orientation="h",
        marker=dict(color=COLORS["post"], line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra>Post-COVID</extra>"
    ))

    fig.update_layout(
        barmode="group",
        height=400,
        xaxis_title="Total Importance",
        yaxis_title="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
    )

    fig = apply_nature_style(fig, "Feature Importance by Category")

    return fig, "Aggregated feature importance across categories"


def create_amenity_comparison() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create bar chart comparing amenity correlation changes."""
    path = DATA_FILES["amenity_analysis"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)
    df = df.sort_values("correlation_change", ascending=False)

    # Color by change direction
    colors = [COLORS["post"] if x > 0.05 else (COLORS["pre"] if x < -0.05 else COLORS["neutral"])
              for x in df["correlation_change"]]

    # Clean names
    df["name_clean"] = df["amenity_type"].apply(lambda x: x.replace("_", " ").replace("food count ", "")[:25])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df["name_clean"],
        x=df["correlation_change"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Change: %{x:.3f}<extra></extra>"
    ))

    # Add zero reference line
    fig.add_vline(x=0, line_color="#333333", line_width=0.8)

    fig.update_layout(
        height=600,
        xaxis_title="Correlation Change (Post - Pre)",
        yaxis_title="",
    )

    fig = apply_nature_style(fig, "Amenity Importance Change: Pre vs Post COVID")

    return fig, "Positive = became more important; Negative = became less important"


def create_suburb_drivetime_matrix() -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create heatmap showing performance by location ring × drivetime radius.

    Based on analysis from presentation notes:
    - Outer suburbs + 10-min drivetime = optimal (11.6%)
    - Downtown + 10-min = good but not best (8.9%)
    - Donut ring = least favorable (7.0%)
    """
    # Performance data from the analysis (RevPAR growth %)
    performance_data = {
        "Downtown": {"10-min": 8.9, "15-min": 7.2, "30-min": 6.1},
        "Inner Suburb": {"10-min": 9.2, "15-min": 8.1, "30-min": 6.8},
        "Donut Ring": {"10-min": 7.0, "15-min": 6.5, "30-min": 5.8},
        "Outer Suburb": {"10-min": 11.6, "15-min": 9.4, "30-min": 7.3},
    }

    rings = list(performance_data.keys())
    drivetimes = ["10-min", "15-min", "30-min"]

    # Build matrix
    z = [[performance_data[ring][dt] for dt in drivetimes] for ring in rings]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=drivetimes,
        y=rings,
        colorscale=[[0, "#d73027"], [0.5, "#fee090"], [1, "#1a9850"]],
        zmin=5,
        zmax=12,
        colorbar=dict(
            title=dict(text="RevPAR Growth (%)", font=dict(size=11)),
            tickfont=dict(size=10),
            thickness=12,
            len=0.6,
        ),
        hovertemplate="<b>%{y}</b> + <b>%{x}</b><br>Growth: %{z:.1f}%<extra></extra>"
    ))

    # Add text annotations
    annotations = []
    for i, ring in enumerate(rings):
        for j, dt in enumerate(drivetimes):
            val = z[i][j]
            # Highlight the best combo
            is_best = (ring == "Outer Suburb" and dt == "10-min")
            annotations.append(dict(
                x=dt, y=ring,
                text=f"<b>{val:.1f}%</b>" if is_best else f"{val:.1f}%",
                showarrow=False,
                font=dict(
                    color="white" if val > 9 or val < 6.5 else "#333333",
                    size=12 if is_best else 11
                )
            ))

    fig.update_layout(
        height=380,
        xaxis_title="Drivetime Radius",
        yaxis_title="Location Ring",
        annotations=annotations,
    )

    fig = apply_nature_style(fig, "Performance by Location × Drivetime (15-Minute City Analysis)")

    return fig, "Outer suburbs with tight amenity access (10-min) show optimal performance — the '15-minute suburb' pattern"


def create_feature_rank_change(top_n: int = 15) -> Tuple[Optional[go.Figure], Optional[str]]:
    """Create bar chart showing features that moved most in importance ranking."""
    path = DATA_FILES["feature_importance"]
    if not path.exists():
        return None, None

    df = pd.read_csv(path)

    # Calculate rank change (negative = moved up in importance, positive = dropped)
    df["rank_change"] = df["rank_pre"] - df["rank_post"]

    # Get features with biggest rank changes (either direction)
    df["abs_rank_change"] = df["rank_change"].abs()
    top_changers = df.nlargest(top_n, "abs_rank_change").copy()

    # Sort by rank change for visualization
    top_changers = top_changers.sort_values("rank_change", ascending=True)

    # Clean feature names
    top_changers["feature_clean"] = top_changers["feature_name"].apply(
        lambda x: x.replace("_", " ").replace("aarp met ", "").replace("aarp score ", "")[:30]
    )

    # Color by direction: green = rose (positive rank_change), red = fell
    colors = [COLORS["green"] if x > 0 else COLORS["post"] for x in top_changers["rank_change"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_changers["feature_clean"],
        x=top_changers["rank_change"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Rank change: %{x:+d}<br>"
            "<extra></extra>"
        ),
        customdata=top_changers[["rank_pre", "rank_post"]].values,
    ))

    # Zero reference line
    fig.add_vline(x=0, line_color="#333333", line_width=0.8)

    fig.update_layout(
        height=max(400, top_n * 28),
        xaxis_title="Rank Change (+ = rose in importance, − = fell)",
        yaxis_title="",
    )

    fig = apply_nature_style(fig, f"Feature Rank Changes: Pre vs Post COVID (Top {top_n})")

    return fig, "Features that rose or fell most in importance ranking after COVID"
