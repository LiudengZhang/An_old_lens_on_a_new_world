"""Configuration for BroadVail Datathon Query System."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Data file paths
DATA_FILES = {
    "key_findings": DATA_DIR / "key_findings.json",
    "feature_importance": DATA_DIR / "feature_importance.csv",
    "city_summary": DATA_DIR / "city_summary.csv",
    "submarket_summary": DATA_DIR / "submarket_summary.csv",
    "predictions": DATA_DIR / "predictions.csv",
    "model_performance": DATA_DIR / "model_performance.json",
    "drivetime_analysis": DATA_DIR / "drivetime_analysis.csv",
    "amenity_analysis": DATA_DIR / "amenity_analysis.csv",
    "data_dictionary": DATA_DIR / "data_dictionary.csv",
    "training_data": DATA_DIR / "training_data.csv",
}

# OpenAI settings - Use environment variable or Streamlit secrets
import os
try:
    import streamlit as st
    DEFAULT_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
except:
    DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# Layer 1 settings
SIMILARITY_THRESHOLD = 0.60  # Minimum similarity to match a pre-computed finding

# Layer 3 settings
LAYER3_WARNING = """
⚠️ This query requires analyzing raw data, which uses more API tokens.
Estimated cost: ~$0.01-0.05 per query.
Do you want to proceed?
"""

# Token pricing (per 1M tokens, as of 2024)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "text-embedding-3-small": {"input": 0.02, "output": 0},
}
