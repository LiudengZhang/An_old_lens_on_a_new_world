"""BroadVail Datathon Query System - Streamlit App."""
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from query_engine import QueryEngine
from visualizations import get_visualization
from config import DATA_FILES, DEFAULT_API_KEY

# Page config
st.set_page_config(
    page_title="BroadVail Datathon Query System",
    page_icon="🏢",
    layout="centered",
)

st.title("🏢 BroadVail Datathon Query System")
st.caption("AI-powered analysis of apartment market data")

# Create tabs for different features
tab_qa, tab_confidence = st.tabs(["💬 Q&A", "📊 Model Confidence"])

# Sidebar for API key
with st.sidebar:
    st.header("Settings")

    # API key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key, or leave blank to use the shared key"
    )

    # Strip whitespace and use fallbacks
    api_key = api_key.strip() if api_key else ""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        api_key = DEFAULT_API_KEY  # Use built-in default

    st.divider()

    # Data status
    st.header("Data Status")
    for name, path in DATA_FILES.items():
        status = "✅" if path.exists() else "❌"
        st.text(f"{status} {name}")

    st.divider()
    st.caption("Layer 1: Pre-computed findings (fast)")
    st.caption("Layer 2: Structured data (medium)")
    st.caption("Layer 3: Raw data analysis (slow)")

# Main interface
if not api_key:
    st.warning("Please enter an OpenAI API key in the sidebar to continue.")
    st.stop()

# Initialize query engine (no caching to avoid stale key issues)
def get_engine(key: str):
    return QueryEngine(api_key=key)

# Cache engine in session state
if "engine" not in st.session_state or st.session_state.get("cached_key") != api_key:
    try:
        st.session_state.engine = get_engine(api_key)
        st.session_state.cached_key = api_key
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

engine = st.session_state.engine

# =============================================================================
# TAB 1: Q&A Interface
# =============================================================================
with tab_qa:
    # Question input
    question = st.text_input(
        "Ask a question about the apartment market data:",
        placeholder="e.g., What are the top 10 most important features?",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("Ask", type="primary", use_container_width=True)

    # Process question
    if ask_button and question:
        with st.spinner("Processing..."):
            result = engine.query(question)
        st.session_state.last_result = result
        st.session_state.last_question = question

    # Display result
    if "last_result" in st.session_state and st.session_state.last_result:
        result = st.session_state.last_result

        st.divider()

        # Answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Visualization (if applicable)
        last_question = st.session_state.get("last_question", "")
        if last_question:
            fig, caption = get_visualization(last_question, result)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(caption)

        # Supporting data (Layer 1)
        if result.get("supporting_data"):
            with st.expander("Supporting Data"):
                st.json(result["supporting_data"])

        # Metadata
        st.divider()
        cols = st.columns(4)
        with cols[0]:
            st.metric("Layer", result["layer"])
        with cols[1]:
            st.metric("Source", result["source"][:20] + "..." if len(result.get("source", "")) > 20 else result.get("source", ""))
        with cols[2]:
            tokens = result.get("tokens", {})
            total_tokens = tokens.get("input", 0) + tokens.get("output", 0)
            st.metric("Tokens", f"{total_tokens:,}")
        with cols[3]:
            cost = engine.estimate_cost(tokens)
            st.metric("Est. Cost", f"${cost:.4f}")

        if result.get("confidence"):
            st.caption(f"Match confidence: {result['confidence']:.2%}")

    # Example questions
    st.divider()
    st.subheader("Example Questions")
    examples = [
        "How did COVID change apartment preferences?",
        "What are the top 10 most important features?",
        "How does Houston compare to Phoenix?",
        "Which drivetime definition works best?",
        "What is the model's R² score?",
        "Which amenities became more important after COVID?",
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.example_question = example
                st.rerun()

    # Handle example button clicks
    if "example_question" in st.session_state:
        question = st.session_state.example_question
        del st.session_state.example_question
        with st.spinner("Processing..."):
            result = engine.query(question, allow_layer3=False)
        st.session_state.last_result = result
        st.session_state.last_question = question
        st.rerun()

# =============================================================================
# TAB 2: Model Confidence Analysis (Aggregated Data Only)
# =============================================================================
with tab_confidence:
    st.subheader("Prediction Confidence Analysis")
    st.caption("Understand model reliability based on ensemble agreement (aggregated metrics only)")

    # Load predictions data
    @st.cache_data
    def load_predictions():
        pred_path = DATA_FILES.get("predictions")
        if pred_path and pred_path.exists():
            return pd.read_csv(pred_path)
        return None

    predictions_df = load_predictions()

    if predictions_df is None or 'ensemble_std' not in predictions_df.columns:
        st.warning("Predictions data with ensemble confidence not available.")
    else:
        # Summary metrics (aggregated only)
        st.markdown("### Overview")
        col1, col2, col3, col4 = st.columns(4)

        n_high = (predictions_df['prediction_confidence'] == 'high').sum()
        n_medium = (predictions_df['prediction_confidence'] == 'medium').sum()
        n_low = (predictions_df['prediction_confidence'] == 'low').sum()

        with col1:
            st.metric("Total Predictions", f"{len(predictions_df):,}")
        with col2:
            st.metric("High Confidence", f"{n_high:,}", delta=f"{n_high/len(predictions_df)*100:.1f}%")
        with col3:
            st.metric("Medium Confidence", f"{n_medium:,}", delta=f"{n_medium/len(predictions_df)*100:.1f}%")
        with col4:
            st.metric("Low Confidence", f"{n_low:,}", delta=f"{n_low/len(predictions_df)*100:.1f}%")

        st.divider()

        # Visualization selection (Property Lookup removed for data confidentiality)
        viz_option = st.selectbox(
            "Select Visualization",
            ["Prediction Accuracy by Confidence", "Error Distribution", "Confidence by Market"]
        )

        if viz_option == "Prediction Accuracy by Confidence":
            st.markdown("### Prediction vs Actual by Confidence Level")
            st.caption("High-confidence predictions (green) cluster closer to the perfect prediction line")

            # Sample for performance - show aggregated scatter without identifiable info
            sample_size = min(5000, len(predictions_df))
            sample_df = predictions_df.sample(sample_size, random_state=42)

            fig = px.scatter(
                sample_df,
                x='actual_revpar_growth',
                y='predicted_revpar_growth',
                color='prediction_confidence',
                color_discrete_map={'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'},
                opacity=0.5,
                labels={
                    'actual_revpar_growth': 'Actual RevPAR Growth',
                    'predicted_revpar_growth': 'Predicted RevPAR Growth',
                    'prediction_confidence': 'Confidence'
                }
            )

            # Add perfect prediction line
            fig.add_trace(go.Scatter(
                x=[-0.6, 0.8], y=[-0.6, 0.8],
                mode='lines', name='Perfect Prediction',
                line=dict(color='black', dash='dash')
            ))

            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Error Distribution":
            st.markdown("### Error Distribution by Confidence Level")
            st.caption("High-confidence predictions have smaller errors")

            # Use 95th percentile as upper bound to show most data while handling outliers
            upper_bound = predictions_df['abs_error'].quantile(0.95)

            fig = go.Figure()
            colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}

            for conf in ['high', 'medium', 'low']:
                subset = predictions_df[predictions_df['prediction_confidence'] == conf]
                fig.add_trace(go.Histogram(
                    x=subset['abs_error'],
                    name=f"{conf.capitalize()} (mean={subset['abs_error'].mean():.3f})",
                    marker_color=colors[conf],
                    opacity=0.6,
                    nbinsx=50
                ))

            fig.update_layout(
                barmode='overlay',
                xaxis_title='Absolute Error',
                yaxis_title='Count',
                height=400,
                xaxis=dict(range=[0, upper_bound])
            )
            st.plotly_chart(fig, use_container_width=True)

            n_outliers = (predictions_df['abs_error'] > upper_bound).sum()
            st.caption(f"Note: {n_outliers} outliers ({n_outliers/len(predictions_df)*100:.1f}%) beyond {upper_bound:.2f} are clipped for clarity.")

            # Summary table (aggregated stats only)
            st.markdown("### Error Statistics by Confidence")
            summary = predictions_df.groupby('prediction_confidence').agg({
                'abs_error': ['mean', 'median', 'std'],
                'ensemble_std': 'mean'
            }).round(4)
            summary.columns = ['Mean Error', 'Median Error', 'Std Error', 'Mean Ensemble Std']
            summary['Count'] = predictions_df.groupby('prediction_confidence').size()
            st.dataframe(summary)

        elif viz_option == "Confidence by Market":
            st.markdown("### Prediction Confidence by Market")
            st.caption("Some markets have more reliable predictions than others")

            # Get top markets
            top_n = st.slider("Number of markets to show", 5, 20, 10)
            top_markets = predictions_df['city'].value_counts().head(top_n).index

            market_conf = predictions_df[predictions_df['city'].isin(top_markets)].groupby(
                ['city', 'prediction_confidence']
            ).size().unstack(fill_value=0)

            # Ensure column order
            for col in ['high', 'medium', 'low']:
                if col not in market_conf.columns:
                    market_conf[col] = 0
            market_conf = market_conf[['high', 'medium', 'low']]

            # Convert to percentages
            market_conf_pct = market_conf.div(market_conf.sum(axis=1), axis=0) * 100
            market_conf_pct = market_conf_pct.sort_values('high', ascending=True)

            fig = go.Figure()
            colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}

            for conf in ['high', 'medium', 'low']:
                fig.add_trace(go.Bar(
                    y=market_conf_pct.index,
                    x=market_conf_pct[conf],
                    name=conf.capitalize(),
                    orientation='h',
                    marker_color=colors[conf]
                ))

            fig.update_layout(
                barmode='stack',
                xaxis_title='Percentage of Properties',
                height=max(400, top_n * 30),
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confidence thresholds reference
        st.divider()
        st.markdown("### Confidence Thresholds")
        st.info("""
        **How confidence is calculated:**
        - Ensemble of 7 models (LightGBM, XGBoost, CatBoost, HistGB, Ridge, KNN, ExtraTrees)
        - **Ensemble Std** = Standard deviation of predictions across models
        - Lower std = models agree = higher confidence
        - Confidence assigned by percentile rank (bottom 33% = high, middle 33% = medium, top 34% = low)

        | Confidence | Ensemble Std | Interpretation |
        |------------|--------------|----------------|
        | **High** | < 0.022 (33rd pctl) | Models strongly agree - trust prediction |
        | **Medium** | 0.022 - 0.033 | Moderate agreement - use with caution |
        | **Low** | > 0.033 (66th pctl) | Models disagree - investigate further |
        """)

        st.caption("Note: Property-level data is not displayed due to data confidentiality requirements.")
