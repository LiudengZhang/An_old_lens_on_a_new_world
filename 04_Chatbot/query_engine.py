"""Query engine with 3-layer architecture."""
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from typing import Optional, Tuple
from config import DATA_FILES, SIMILARITY_THRESHOLD, DEFAULT_MODEL, EMBEDDING_MODEL


class QueryEngine:
    """3-layer query engine for BroadVail Datathon data."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.findings = self._load_findings()
        self.findings_embeddings = None
        self.token_usage = {"input": 0, "output": 0}

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
        context = self._gather_layer2_context(question)

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
        }

    def _gather_layer2_context(self, question: str) -> str:
        """Gather relevant data context for Layer 2."""
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

        # City summary
        if any(kw in question_lower for kw in ["city", "cities", "houston", "phoenix", "atlanta", "compare"]):
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

        # Amenity analysis
        if any(kw in question_lower for kw in ["amenity", "amenities", "restaurant", "grocery", "park", "transit"]):
            path = DATA_FILES["amenity_analysis"]
            if path.exists():
                df = pd.read_csv(path)
                context_parts.append(f"AMENITY ANALYSIS:\n{df.to_string()}")

        # Predictions (for specific property queries)
        if any(kw in question_lower for kw in ["property", "prediction", "predicted", "actual", "p0"]):
            path = DATA_FILES["predictions"]
            if path.exists():
                df = pd.read_csv(path)
                # If specific property mentioned, filter
                context_parts.append(f"PREDICTIONS (sample):\n{df.head(20).to_string()}")

        # COVID-related
        if any(kw in question_lower for kw in ["covid", "pre", "post", "pandemic", "change"]):
            # Add multiple relevant files
            for key in ["feature_importance", "city_summary"]:
                path = DATA_FILES[key]
                if path.exists():
                    if key.endswith(".json"):
                        with open(path) as f:
                            data = json.load(f)
                        context_parts.append(f"{key.upper()}:\n{json.dumps(data, indent=2)}")
                    else:
                        df = pd.read_csv(path)
                        context_parts.append(f"{key.upper()}:\n{df.head(20).to_string()}")

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

        return "\n\n".join(context_parts) if context_parts else "No relevant data files found."

    def layer3_query(self, question: str) -> dict:
        """Layer 3: Analyze raw training data."""
        path = DATA_FILES["training_data"]
        if not path.exists():
            return {
                "layer": 3,
                "answer": "Raw training data file not found.",
                "source": "error",
            }

        # Load sample of raw data
        df = pd.read_csv(path, nrows=500)  # Limit rows to control token usage

        prompt = f"""You are a data analyst assistant for the BroadVail Datathon.
Analyze the raw training data to answer the user's question.
The data contains apartment property information with RevPAR growth metrics.

DATA COLUMNS: {list(df.columns)}

DATA SAMPLE (first 500 rows):
{df.to_string()}

USER QUESTION: {question}

Provide a detailed analytical answer with specific numbers:"""

        response = self.client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        self.token_usage["input"] += response.usage.prompt_tokens
        self.token_usage["output"] += response.usage.completion_tokens

        return {
            "layer": 3,
            "answer": response.choices[0].message.content,
            "source": "raw data analysis",
        }

    def query(self, question: str, allow_layer3: bool = False) -> dict:
        """Main query method - tries layers in order."""
        # Reset token usage for this query
        self.token_usage = {"input": 0, "output": 0}

        # Try Layer 1 first
        result = self.layer1_query(question)
        if result:
            result["tokens"] = self.token_usage.copy()
            return result

        # Try Layer 2
        result = self.layer2_query(question)
        if "No relevant data files found" not in result["answer"]:
            result["tokens"] = self.token_usage.copy()
            return result

        # Layer 3 requires explicit permission
        if allow_layer3:
            result = self.layer3_query(question)
            result["tokens"] = self.token_usage.copy()
            return result

        return {
            "layer": 2,
            "answer": result["answer"],
            "source": result["source"],
            "tokens": self.token_usage.copy(),
            "needs_layer3": True,
        }

    def estimate_cost(self, tokens: dict, model: str = DEFAULT_MODEL) -> float:
        """Estimate cost based on token usage."""
        from config import PRICING
        pricing = PRICING.get(model, PRICING["gpt-4o-mini"])
        input_cost = (tokens.get("input", 0) / 1_000_000) * pricing["input"]
        output_cost = (tokens.get("output", 0) / 1_000_000) * pricing["output"]
        return input_cost + output_cost
