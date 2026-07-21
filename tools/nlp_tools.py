"""
tools/nlp_tools.py

NLP-powered LangGraph tools — semantic search and sentiment analysis.

Architecture:
  - sentence-transformers (all-MiniLM-L6-v2) for query embeddings.
    The model is loaded ONCE at module import via a singleton.
    AMD Ryzen AI 7 350: OMP_NUM_THREADS=4 controls PyTorch CPU parallelism
    (set in .env or before running the app).
  - VADER sentiment analyser for review/description sentiment scoring.
  - Redis cache (5-min TTL) prevents re-encoding the same query twice.
  - numpy cosine similarity runs in-process — no network call needed.

NLP Model sizing:
  all-MiniLM-L6-v2 → 384-dim, ~80MB, fast on CPU (~5ms per query encode).
  Sufficient for semantic search across 60–5000 products with numpy.
"""

import os
import threading
import numpy as np
from typing import Optional

from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ── Singleton model loader ────────────────────────────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    model_name = os.getenv("NLP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                    _model = SentenceTransformer(model_name)
                    print(f"[NLP] Model loaded: {model_name}")
                except ImportError:
                    print("[NLP] WARNING: sentence-transformers not installed.")
                    _model = None
    return _model


# ── Sentiment analyser singleton ──────────────────────────────────────────────

_vader = None
_vader_lock = threading.Lock()


def _get_vader():
    global _vader
    if _vader is None:
        with _vader_lock:
            if _vader is None:
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    _vader = SentimentIntensityAnalyzer()
                except ImportError:
                    print("[NLP] WARNING: vaderSentiment not installed.")
                    _vader = None
    return _vader


# ── Cosine similarity helpers ─────────────────────────────────────────────────

def _embed_query(query: str) -> Optional[np.ndarray]:
    """
    Encode a single query string to a numpy embedding vector.
    Checks Redis cache first (5-min TTL) to avoid re-encoding duplicates.
    """
    from memory.redis_memory import get_cached_embedding, set_cached_embedding

    cached = get_cached_embedding(query)
    if cached:
        return np.array(cached, dtype=np.float32)

    model = _get_model()
    if model is None:
        return None

    vec = model.encode(query, normalize_embeddings=True)
    set_cached_embedding(query, vec.tolist())
    return vec


def _cosine_similarities(query_vec: np.ndarray, product_vectors: list[dict]) -> list[tuple[float, dict]]:
    """
    Compute cosine similarity between query_vec and all stored product embeddings.
    Returns list of (score, product_vector_doc) sorted descending.
    """
    if not product_vectors or query_vec is None:
        return []

    matrix = np.array([pv["embedding"] for pv in product_vectors], dtype=np.float32)
    # query_vec is already normalised; normalise matrix rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix = matrix / norms

    scores = matrix @ query_vec  # cosine similarity = dot product of normalised vecs
    indexed = sorted(zip(scores.tolist(), product_vectors), key=lambda x: x[0], reverse=True)
    return indexed


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def semantic_search_products(query: str, top_n: int = 8) -> dict:
    """
    Semantic product search using sentence-transformer embeddings + cosine similarity.

    Unlike keyword search, this understands INTENT:
      "I'm thirsty"      → finds water bottles, sports drinks
      "carry my laptop"  → finds laptop bags, backpacks
      "rainy day"        → finds umbrellas, raincoats, rain boots
      "gift for fitness" → finds dumbbells, yoga mats, protein powder

    Falls back to MongoDB text search if the NLP model is unavailable.

    Returns:
        {
          "query"          : str,
          "method"         : "semantic" | "text_fallback",
          "found"          : int,
          "products"       : list[dict]   — prices in INR (raw float),
          "top_score"      : float        — best cosine similarity score
        }
    """
    from db.mongo_client import get_all_product_vectors, get_product_by_id, get_popular_products
    from memory.redis_memory import increment_product_view

    query_vec = _embed_query(query)

    if query_vec is None:
        # Fallback: regular text search
        from db.mongo_client import search_products_text
        results = search_products_text(query, limit=top_n)
        return {
            "query":      query,
            "method":     "text_fallback",
            "found":      len(results),
            "top_score":  None,
            "products":   results,
        }

    # Load all precomputed product vectors
    product_vectors = get_all_product_vectors()
    if not product_vectors:
        from db.mongo_client import search_products_text
        results = search_products_text(query, limit=top_n)
        return {
            "query": query, "method": "text_fallback",
            "found": len(results), "top_score": None, "products": results,
        }

    threshold = float(os.getenv("NLP_SIMILARITY_THRESHOLD", "0.30"))
    ranked = _cosine_similarities(query_vec, product_vectors)

    # Filter by threshold and take top_n
    top_ids = [
        pv["product_id"]
        for score, pv in ranked
        if score >= threshold
    ][:top_n]

    products = []
    for pid in top_ids:
        doc = get_product_by_id(pid)
        if doc and doc.get("stock", 0) > 0:
            products.append({
                "product_id":   doc["product_id"],
                "title":        doc["title"],
                "category":     doc.get("category", ""),
                "brand":        doc.get("brand", ""),
                "price":        float(doc["price"]),
                "rating":       doc.get("rating", 0),
                "rating_count": doc.get("rating_count", 0),
                "stock":        doc.get("stock", 0),
            })
            increment_product_view(pid)

    top_score = ranked[0][0] if ranked else 0.0

    if not products:
        # Nothing above threshold — return popular fallback
        fallback = get_popular_products(5)
        return {
            "query":      query,
            "method":     "semantic_fallback",
            "found":      0,
            "top_score":  top_score,
            "products":   [_clean(p) for p in fallback],
            "fallback":   True,
            "fallback_reason": (
                f"No semantically similar products found for '{query}'. "
                "Showing popular items instead."
            ),
        }

    return {
        "query":     query,
        "method":    "semantic",
        "found":     len(products),
        "top_score": round(top_score, 4),
        "products":  products,
    }


@tool
def analyze_sentiment(text: str, product_title: str = "") -> dict:
    """
    Perform sentiment analysis on product review text or description.

    Use when user asks:
      "Are the reviews good for this?"
      "Is it worth buying?"
      "What do people think about X?"

    Uses VADER (Valence Aware Dictionary and sEntiment Reasoner):
      - Designed for social media / short texts
      - Returns compound score −1.0 (very negative) to +1.0 (very positive)
      - Breakdowns: positive, negative, neutral proportions

    Returns:
        {
          "sentiment_label"    : "Positive" | "Negative" | "Neutral",
          "compound_score"     : float,   — overall score
          "positive_fraction"  : float,
          "negative_fraction"  : float,
          "neutral_fraction"   : float,
          "confidence"         : "High" | "Medium" | "Low",
          "summary"            : str      — human-readable one-liner
        }
    """
    vader = _get_vader()

    if vader is None:
        return {
            "sentiment_label":   "Unknown",
            "compound_score":    0.0,
            "summary":           "Sentiment analysis unavailable (vaderSentiment not installed).",
        }

    scores = vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    # Confidence based on how far compound is from 0
    abs_c = abs(compound)
    if abs_c >= 0.5:
        confidence = "High"
    elif abs_c >= 0.2:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Human-readable summary
    pct_pos = int(scores["pos"] * 100)
    pct_neg = int(scores["neg"] * 100)
    item_ref = f'"{product_title}"' if product_title else "this item"

    if label == "Positive":
        summary = (
            f"Reviews for {item_ref} are predominantly {label.lower()} "
            f"({pct_pos}% positive sentiment). "
            f"Overall score: {compound:.2f}/1.0."
        )
    elif label == "Negative":
        summary = (
            f"Reviews for {item_ref} show mixed or negative feedback "
            f"({pct_neg}% negative sentiment). "
            f"Consider checking individual reviews before purchasing."
        )
    else:
        summary = (
            f"Reviews for {item_ref} are largely neutral. "
            "Customer experiences appear to vary."
        )

    return {
        "sentiment_label":   label,
        "compound_score":    round(compound, 4),
        "positive_fraction": round(scores["pos"], 4),
        "negative_fraction": round(scores["neg"], 4),
        "neutral_fraction":  round(scores["neu"], 4),
        "confidence":        confidence,
        "summary":           summary,
    }


@tool
def get_semantic_suggestions(query: str, n_suggestions: int = 5) -> dict:
    """
    Generate semantically related search terms for query expansion.

    Use when user's query is unusual or you want to offer alternative terms
    to help the user refine their search.

    Returns:
        {
          "original_query": str,
          "suggestions"   : list[str],   — related search terms
          "categories"    : list[str]    — matching product categories
        }
    """
    # Predefined intent → keyword expansion map
    intent_map = {
        "thirsty":           ["water bottle", "sports drink", "beverage"],
        "laptop":            ["laptop bag", "laptop sleeve", "backpack", "laptop stand"],
        "rain":              ["raincoat", "umbrella", "rain jacket", "waterproof bag"],
        "fitness":           ["yoga mat", "dumbbells", "resistance bands", "protein powder", "fitness tracker"],
        "gift":              ["gift set", "gift hamper", "birthday gift", "personal care kit"],
        "music":             ["earbuds", "headphones", "bluetooth speaker", "wireless earphones"],
        "travel":            ["trolley bag", "travel backpack", "neck pillow", "passport holder"],
        "study":             ["notebook", "stationery", "desk lamp", "headphones", "backpack"],
        "cooking":           ["mixer grinder", "electric kettle", "pressure cooker", "storage containers"],
        "skincare":          ["face wash", "moisturizer", "sunscreen", "serum", "face mask"],
        "running":           ["running shoes", "sports t-shirt", "earbuds", "fitness tracker"],
    }

    query_lower = query.lower()
    suggestions = []
    for keyword, terms in intent_map.items():
        if keyword in query_lower:
            suggestions.extend(terms)

    if not suggestions:
        # Generate generic suggestions using category names
        from db.mongo_client import get_distinct_categories
        categories = get_distinct_categories()[:8]
        suggestions = [f"{query} in {cat}" for cat in categories[:5]]

    # Get matching categories via semantic similarity
    from db.mongo_client import get_distinct_categories
    categories = get_distinct_categories()

    return {
        "original_query": query,
        "suggestions":    list(dict.fromkeys(suggestions))[:n_suggestions],
        "categories":     categories[:8],
        "tip": "Use any suggestion as a more specific search term.",
    }


# ── Internal helper ────────────────────────────────────────────────────────────

def _clean(doc: dict) -> dict:
    """Strip MongoDB internal fields and normalise price."""
    doc.pop("_id", None)
    doc.pop("embedding", None)
    if "price" in doc:
        doc["price"] = float(doc["price"])
    return doc
