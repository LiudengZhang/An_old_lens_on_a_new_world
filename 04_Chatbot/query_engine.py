"""Query engine with 3-layer architecture."""
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Optional, Tuple
from config import DATA_FILES, SIMILARITY_THRESHOLD, DEFAULT_MODEL, EMBEDDING_MODEL


DATA_CONFIDENTIALITY_MESSAGE = (
    "I'm unable to provide property-level or raw data due to data confidentiality requirements. "
    "The dataset used in this analysis is provided by BroadVail Capital Partners for the 2026 Rice Datathon "
    "and must be treated as confidential.\n\n"
    "However, I can help you with:\n"
    "- Aggregated city-level insights\n"
    "- Model performance metrics\n"
    "- Feature importance analysis\n"
    "- General trends and findings from our analysis\n\n"
    "Please feel free to ask about these topics instead!"
)

SENSITIVE_KEYWORDS = [
    "property_id", "property id", "specific property", "individual property",
    "raw data", "training data", "download data", "export data", "give me the data",
    "show me all", "list all properties", "all records", "full dataset",
    "p0", "p1", "p2",  # property ID patterns
]


class QueryEngine:
    """3-layer query engine for BroadVail Datathon data."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.findings = self._load_findings()
        self.findings_embeddings = None
        self.token_usage = {"input": 0, "output": 0}

    def _is_sensitive_query(self, question: str) -> bool:
        """Check if the question asks for confidential property-level data."""
        question_lower = question.lower()
        return any(kw in question_lower for kw in SENSITIVE_KEYWORDS)

    def _load_findings(self) -> list:
        """Load pre-computed findings from JSON."""
        path = DATA_FILES["key_findings"]
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def _get_embedding(self, text: str) -> list:
        """Get embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        self.token_usage["input"] += response.usage.total_tokens
        return response.data[0].embedding

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _build_findings_embeddings(self):
        """Build embeddings for all findings (lazy loading)."""
        if self.findings_embeddings is not None:
            return

        self.findings_embeddings = []
        for finding in self.findings:
            patterns = finding.get("question_patterns", [])
            combined_text = " ".join(patterns)
            embedding = self._get_embedding(combined_text)
            self.findings_embeddings.append(embedding)

    def layer1_query(self, question: str) -> Optional[dict]:
        """Layer 1: Match against pre-computed findings."""
        if not self.findings:
            return None

        self._build_findings_embeddings()
        question_embedding = self._get_embedding(question)

        best_score = 0
        best_finding = None

        for i, finding_embedding in enumerate(self.findings_embeddings):
            score = self._cosine_similarity(question_embedding, finding_embedding)
            if score > best_score:
                best_score = score
                best_finding = self.findings[i]

        if best_score >= SIMILARITY_THRESHOLD:
            return {
                "layer": 1,
                "answer": best_finding["answer"],
                "supporting_data": best_finding.get("supporting_data", {}),
                "source": best_finding.get("source", "pre-computed finding"),
                "confidence": best_score,
            }
        return None

    def layer2_query(self, question: str) -> dict:
        """Layer 2: Query structured data files."""
        # Determine which data files are relevant
        context, has_context = self._gather_layer2_context(question)

        # If no relevant context found, signal need for Layer 3
        if not has_context:
            return {
                "layer": 2,
                "answer": "The available structured data files do not contain information relevant to this question.",
                "source": "structured data query",
                "has_context": False,
            }

        prompt = f"""You are a data analyst assistant for the BroadVail Datathon.
Answer the user's question based ONLY on the provided data context.
Be specific and cite numbers from the data.
If the data doesn't contain enough information, say so.

DATA CONTEXT:
{context}

USER QUESTION: {question}

Provide a clear, professional answer:"""

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        self.token_usage["input"] += response.usage.prompt_tokens
        self.token_usage["output"] += response.usage.completion_tokens

        return {
            "layer": 2,
            "answer": response.choices[0].message.content,
            "source": "structured data query",
            "has_context": True,
        }

    def _gather_layer2_context(self, question: str) -> Tuple[str, bool]:
        """Gather relevant data context for Layer 2.

        Returns:
            Tuple of (context_string, has_relevant_context)
        """
        context_parts = []
        question_lower = question.lower()

        # Model performance
        if any(kw in question_lower for kw in ["model", "performance", "rmse", "r2", "accuracy"]):
            path = DATA_FILES["model_performance"]
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                context_parts.append(f"MODEL PERFORMANCE:\n{json.dumps(data, indent=2)}")

        # Feature importance
        if any(kw in question_lower for kw in ["feature", "important", "top", "rank"]):
            path = DATA_FILES["feature_importance"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"FEATURE IMPORTANCE (top 20):\n{df.head(20).to_string()}")

        # City summary (include common city names from the data)
        city_keywords = [
            "city", "cities", "compare", "market",
            "houston", "dallas", "austin", "san antonio", "fort worth",  # Texas
            "phoenix", "tucson", "scottsdale",  # Arizona
            "atlanta", "georgia",  # Georgia
            "charlotte", "raleigh", "durham",  # Carolinas
            "tampa", "orlando", "jacksonville", "miami", "fort lauderdale", "florida",  # Florida
            "nashville", "memphis", "tennessee"  # Tennessee
        ]
        if any(kw in question_lower for kw in city_keywords):
            path = DATA_FILES["city_summary"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"CITY SUMMARY:\n{df.to_string()}")

        # Submarket summary
        if any(kw in question_lower for kw in ["submarket", "neighborhood", "area"]):
            path = DATA_FILES["submarket_summary"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"SUBMARKET SUMMARY (sample):\n{df.head(30).to_string()}")

        # Drivetime analysis
        if any(kw in question_lower for kw in ["drivetime", "drive time", "10min", "15min", "30min", "minutes"]):
            path = DATA_FILES["drivetime_analysis"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"DRIVETIME ANALYSIS:\n{df.to_string()}")

        # Suburb × Drivetime interaction (15-minute city analysis)
        # Note: use word boundaries to avoid false matches like "comparing" -> "ring"
        suburb_keywords = ["suburb", "downtown", "15-minute city", "15 minute city", "location type", "outer suburb", "urban core", "donut ring"]
        if any(kw in question_lower for kw in suburb_keywords) or re.search(r'\bring\b', question_lower):
            # Provide the pre-computed analysis from presentation notes
            suburb_data = """LOCATION × DRIVETIME PERFORMANCE (RevPAR Growth %):
                        10-min    15-min    30-min
Downtown           8.9%      7.2%      6.1%
Inner Suburb       9.2%      8.1%      6.8%
Donut Ring         7.0%      6.5%      5.8%
Outer Suburb      11.6%      9.4%      7.3%

KEY INSIGHT: Outer suburbs with tight amenity access (10-min drivetime) = optimal performance.
This is the '15-minute suburb' pattern - the 15-minute city concept applies more strongly in suburbs."""
            context_parts.append(suburb_data)

        # Feature rank changes
        if any(kw in question_lower for kw in ["rank change", "ranking", "moved up", "moved down", "rose", "fell"]):
            path = DATA_FILES["feature_importance"]
            if path.exists():
                df = pd.read_csv(path)
                df["rank_change"] = df["rank_pre"] - df["rank_post"]
                # Top risers and fallers
                risers = df.nlargest(10, "rank_change")[["feature_name", "rank_pre", "rank_post", "rank_change"]]
                fallers = df.nsmallest(10, "rank_change")[["feature_name", "rank_pre", "rank_post", "rank_change"]]
                context_parts.append(f"FEATURES THAT ROSE IN IMPORTANCE:\n{risers.to_string()}")
                context_parts.append(f"FEATURES THAT FELL IN IMPORTANCE:\n{fallers.to_string()}")

        # Amenity analysis
        if any(kw in question_lower for kw in ["amenity", "amenities", "restaurant", "grocery", "park", "transit"]):
            path = DATA_FILES["amenity_analysis"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"AMENITY ANALYSIS:\n{df.to_string()}")

        # Feature correlation analysis
        if any(kw in question_lower for kw in ["correlation", "correlated", "relationship between"]):
            importance_path = DATA_FILES["feature_importance"]
            training_path = DATA_FILES["training_data"]
            if importance_path.exists() and training_path.exists():
                # Get top features
                match = re.search(r'top\s*(\d+)', question_lower)
                n = min(int(match.group(1)), 25) if match else 15

                importance_df = pd.read_csv(importance_path)
                top_features = importance_df.nsmallest(n, "rank_post")["feature_name"].tolist()

                training_df = pd.read_csv(training_path)
                available = [f for f in top_features if f in training_df.columns]

                if len(available) >= 3:
                    corr_matrix = training_df[available].corr()
                    # Find top correlated pairs
                    pairs = []
                    for i, f1 in enumerate(available):
                        for j, f2 in enumerate(available):
                            if i < j:
                                r = corr_matrix.loc[f1, f2]
                                pairs.append((f1, f2, round(r, 3)))
                    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

                    corr_text = f"FEATURE CORRELATIONS (top {len(available)} features):\n"
                    corr_text += "Highly correlated pairs (|r| > 0.5):\n"
                    for f1, f2, r in pairs[:15]:
                        if abs(r) > 0.5:
                            corr_text += f"  {f1} <-> {f2}: {r}\n"
                    corr_text += f"\nFull correlation matrix available with {len(available)} features."
                    context_parts.append(corr_text)

        # Predictions - only provide aggregated stats, not property-level data
        # Property-level data is confidential per competition rules

        # COVID-related (use specific keywords to avoid false positives)
        if any(kw in question_lower for kw in ["covid", "pre-covid", "post-covid", "pandemic", "before covid", "after covid"]):
            # Add multiple relevant files
            for key in ["feature_importance", "city_summary"]:
                path = DATA_FILES[key]
                if path.exists():
                    if str(path).endswith(".json"):
                        with open(path) as f:
                            data = json.load(f)
                        context_parts.append(f"{key.upper()}:\n{json.dumps(data, indent=2)}")
                    else:
                        df = pd.read_csv(path)
                        context_parts.append(f"{key.upper()}:\n{df.head(20).to_string()}")

        # Track whether we found specific context for the query
        had_specific_context = len(context_parts) > 0

        # If no specific context matched, provide general overview
        if not context_parts:
            for key in ["model_performance", "feature_importance", "city_summary"]:
                path = DATA_FILES[key]
                if path.exists():
                    if str(path).endswith(".json"):
                        with open(path) as f:
                            data = json.load(f)
                        context_parts.append(f"{key.upper()}:\n{json.dumps(data, indent=2)}")
                    else:
                        df = pd.read_csv(path)
                        context_parts.append(f"{key.upper()} (top 10):\n{df.head(10).to_string()}")

        context_str = "\n\n".join(context_parts) if context_parts else ""
        # Return context and whether we found relevant data
        # (fallback general overview still counts as having some context)
        return context_str, len(context_parts) > 0

    def layer3_query(self, question: str) -> dict:
        """Layer 3: Disabled - raw data access restricted for confidentiality."""
        return {
            "layer": 3,
            "answer": DATA_CONFIDENTIALITY_MESSAGE,
            "source": "data policy",
        }

    def query(self, question: str, allow_layer3: bool = False) -> dict:
        """Main query method - tries layers in order."""
        # Reset token usage for this query
        self.token_usage = {"input": 0, "output": 0}

        # Check for sensitive queries first (data confidentiality)
        if self._is_sensitive_query(question):
            return {
                "layer": 0,
                "answer": DATA_CONFIDENTIALITY_MESSAGE,
                "source": "data policy",
                "tokens": self.token_usage.copy(),
            }

        # Try Layer 1 first
        result = self.layer1_query(question)
        if result:
            result["tokens"] = self.token_usage.copy()
            return result

        # Try Layer 2
        result = self.layer2_query(question)
        if result.get("has_context", True):
            result["tokens"] = self.token_usage.copy()
            return result

        # Layer 3 is disabled for data confidentiality
        # Return a helpful message instead
        return {
            "layer": 2,
            "answer": (
                "I don't have specific pre-computed findings for this question, "
                "and raw data analysis is not available due to data confidentiality requirements.\n\n"
                "Try asking about:\n"
                "- COVID impact on apartment preferences\n"
                "- Feature importance and top predictors\n"
                "- City-level performance comparisons\n"
                "- Drivetime analysis (10/15/30 minute)\n"
                "- Model performance metrics"
            ),
            "source": "data policy",
            "tokens": self.token_usage.copy(),
        }

    def estimate_cost(self, tokens: dict, model: str = DEFAULT_MODEL) -> float:
        """Estimate cost based on token usage."""
        from config import PRICING
        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
        input_cost = (tokens.get("input", 0) / 1_000_000) * pricing["input"]
        output_cost = (tokens.get("output", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost
