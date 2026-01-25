"""BroadVail Datathon Query System - Streamlit App."""
import streamlit as st
import os
from query_engine import QueryEngine
from config import LAYER3_WARNING, DATA_FILES, DEFAULT_API_KEY

# Page config
st.set_page_config(
    page_title="BroadVail Datathon Query System",
    page_icon="🏢",
    layout="centered",
)

st.title("🏢 BroadVail Datathon Query System")
st.caption("AI-powered analysis of apartment market data")

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

# Session state for Layer 3 confirmation
if "pending_layer3" not in st.session_state:
    st.session_state.pending_layer3 = None

# Question input
question = st.text_input(
    "Ask a question about the apartment market data:",
    placeholder="e.g., What are the top 10 most important features?",
)

col1, col2 = st.columns([1, 4])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)

# Handle Layer 3 confirmation
if st.session_state.pending_layer3:
    st.warning(LAYER3_WARNING)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, proceed"):
            with st.spinner("Analyzing raw data..."):
                result = engine.query(st.session_state.pending_layer3, allow_layer3=True)
            st.session_state.pending_layer3 = None
            st.session_state.last_result = result
            st.rerun()
    with col2:
        if st.button("Cancel"):
            st.session_state.pending_layer3 = None
            st.rerun()

# Process question
if ask_button and question:
    with st.spinner("Processing..."):
        result = engine.query(question, allow_layer3=False)

    if result.get("needs_layer3"):
        st.session_state.pending_layer3 = question
        st.rerun()
    else:
        st.session_state.last_result = result

# Display result
if "last_result" in st.session_state and st.session_state.last_result:
    result = st.session_state.last_result

    st.divider()

    # Answer
    st.subheader("Answer")
    st.write(result["answer"])

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
    st.rerun()
