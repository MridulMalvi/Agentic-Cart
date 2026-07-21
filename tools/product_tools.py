"""
tools/product_tools.py  (Refactored — MongoDB backend + NLP integration)

All product-facing tools. Returns only DB data — LLM never fabricates.

Changes from original:
  - Switched from MySQL → MongoDB for all product queries.
  - search_products now also records trending counters in Redis.
  - get_reviews returns VADER sentiment scores alongside rating data.
  - clarify_product_query uses MongoDB distinct() for live category/brand lists.
  - semantic_search_products is exposed here as a re-export for the shopping LLM
    (actual implementation lives in tools/nlp_tools.py to keep it reusable).

Currency: ALL prices returned as raw float (INR). LLM displays as ₹<price>.
"""

from langchain_core.tools import tool
from db.mongo_client import (
    search_products_text,
    search_products_filtered,
    get_popular_products,
    get_product_by_id,
    get_distinct_categories,
    get_distinct_brands,
)
from memory.redis_memory import increment_product_view
from tools.nlp_tools import analyze_sentiment


# ── Shared helpers ────────────────────────────────────────────────────────────

def _clean(doc: dict) -> dict:
    """Strip MongoDB metadata, normalise price to float."""
    if doc is None:
        return {}
    doc.pop("_id", None)
    doc.pop("embedding", None)
    if "price" in doc:
        doc["price"] = float(doc["price"])
    return doc


def _popular_fallback(limit: int = 5) -> list[dict]:
    return [_clean(p) for p in get_popular_products(limit)]


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def search_products(query: str, top_n: int = 8) -> dict:
    """
    Search for products by keyword, category, or brand.

    The LLM must translate the user's real-world request into a product keyword
    before calling this tool. Examples from the system prompt:
      "I'm thirsty"          → query="water bottle"
      "carry my laptop"      → query="laptop bag"
      "gift for fitness"     → query="fitness"
      "rainy day outdoors"   → query="raincoat"

    Uses MongoDB full-text search (title/category/brand/description).
    Falls back to popular products automatically when found=0.

    Returns:
        {
          "query"          : str,
          "found"          : int,
          "fallback"       : bool,
          "fallback_reason": str | None,
          "products"       : list[dict]   — prices in INR (raw float)
        }
    """
    rows = search_products_text(query, limit=top_n)

    if rows:
        products = [_clean(r) for r in rows]
        for p in products:
            increment_product_view(p.get("product_id", 0))
        return {
            "query":           query,
            "found":           len(products),
            "fallback":        False,
            "fallback_reason": None,
            "products":        products,
        }

    fallback = _popular_fallback(5)
    return {
        "query":           query,
        "found":           0,
        "fallback":        True,
        "fallback_reason": (
            f"No products matched '{query}'. "
            "Showing our most popular items instead."
        ),
        "products": fallback,
    }


@tool
def filter_products(
    query: str,
    max_price: float = None,
    min_rating: float = None,
    category: str = None,
    top_n: int = 8,
) -> dict:
    """
    Search products with optional filters.

    Use when the user adds constraints:
      'under ₹500'         → max_price=500
      '4+ stars'           → min_rating=4.0
      'only electronics'   → category='electronics'

    Returns:
        {
          "query"          : str,
          "filters"        : dict,
          "found"          : int,
          "fallback"       : bool,
          "fallback_reason": str | None,
          "products"       : list[dict]   — prices in INR (raw float)
        }
    """
    rows = search_products_filtered(
        query=query,
        max_price=max_price,
        min_rating=min_rating,
        category=category,
        limit=top_n,
    )

    if rows:
        products = [_clean(r) for r in rows]
        return {
            "query":           query,
            "filters":         {"max_price": max_price, "min_rating": min_rating, "category": category},
            "found":           len(products),
            "fallback":        False,
            "fallback_reason": None,
            "products":        products,
        }

    fallback = _popular_fallback(5)
    return {
        "query":           query,
        "filters":         {"max_price": max_price, "min_rating": min_rating, "category": category},
        "found":           0,
        "fallback":        True,
        "fallback_reason": (
            "No products matched those filters. "
            "Try relaxing the price or rating. Here are our top picks:"
        ),
        "products": fallback,
    }


@tool
def get_product_details(product_id: int) -> dict:
    """
    Get full details for a specific product by its product_id.

    Use when user asks: 'tell me more', 'what are the specs',
    'describe this product', 'more info about it'.

    Always use the product_id from the context list in the system prompt.
    Never search again for a product already shown.

    Returns:
        {
          "found"       : bool,
          "product_id"  : int,
          "title"       : str,
          "description" : str,
          "category"    : str,
          "brand"       : str,
          "price"       : float   — INR (raw float),
          "rating"      : float,
          "rating_count": int,
          "stock"       : int
        }
    """
    doc = get_product_by_id(product_id)
    if not doc:
        return {"found": False, "product_id": product_id}

    doc = _clean(doc)
    increment_product_view(product_id)
    return {
        "found":        True,
        "product_id":   doc.get("product_id"),
        "title":        doc.get("title", ""),
        "description":  doc.get("description", ""),
        "category":     doc.get("category", ""),
        "brand":        doc.get("brand", ""),
        "price":        doc.get("price", 0.0),
        "rating":       doc.get("rating", 0),
        "rating_count": doc.get("rating_count", 0),
        "stock":        doc.get("stock", 0),
    }


@tool
def get_reviews(query: str) -> dict:
    """
    Retrieve rating, review data, and sentiment analysis for products
    matching a name or category.

    Use when user asks: 'how are the reviews', 'is it good quality',
    'what do people say about X', 'ratings for X'.

    Now includes VADER sentiment analysis on product descriptions as a
    proxy for review sentiment.

    Returns:
        {
          "query"   : str,
          "found"   : int,
          "products": list[dict]   — includes rating, rating_count, description, sentiment
        }
    """
    rows = search_products_text(query, limit=4)

    enriched = []
    for r in rows:
        r = _clean(r)
        description = r.get("description", "")
        # VADER sentiment on product description as review proxy
        if description:
            sentiment_result = analyze_sentiment.invoke({
                "text": description,
                "product_title": r.get("title", ""),
            })
        else:
            sentiment_result = {
                "sentiment_label": "Unknown",
                "compound_score": 0.0,
                "summary": "No review data available.",
            }
        enriched.append({
            "product_id":      r.get("product_id"),
            "title":           r.get("title", ""),
            "category":        r.get("category", ""),
            "brand":           r.get("brand", ""),
            "price":           r.get("price", 0.0),
            "rating":          r.get("rating", 0),
            "rating_count":    r.get("rating_count", 0),
            "description":     description,
            "sentiment_label": sentiment_result.get("sentiment_label", "Unknown"),
            "sentiment_score": sentiment_result.get("compound_score", 0.0),
            "sentiment_summary": sentiment_result.get("summary", ""),
        })

    return {
        "query":    query,
        "found":    len(enriched),
        "products": enriched,
    }


@tool
def clarify_product_query(vague_input: str) -> dict:
    """
    Use when the user's request is too vague to search meaningfully.

    Trigger examples: 'something nice', 'a gift', 'blue thing',
    'buy me something', 'I need that product'.

    Fetches REAL categories and brands from MongoDB so the LLM can ask
    ONE specific, data-driven follow-up question instead of guessing.

    Returns:
        {
          "vague_input"         : str,
          "available_categories": list[str],
          "available_brands"    : list[str],
          "suggested_questions" : list[str],
          "instruction"         : str
        }
    """
    cat_list   = get_distinct_categories()[:30]
    brand_list = get_distinct_brands()[:20]

    return {
        "vague_input":          vague_input,
        "available_categories": cat_list,
        "available_brands":     brand_list,
        "suggested_questions": [
            "Which category interests you? Options: " + ", ".join(cat_list[:7]),
            "Do you have a budget in mind? (e.g. under ₹500, under ₹2,000)",
            "Any preferred brand? We carry: " + ", ".join(brand_list[:5]),
            "Is this for personal use or a gift?",
        ],
        "instruction": (
            "The user's request is too vague to search. "
            "Use available_categories and available_brands to ask "
            "ONE specific, friendly clarifying question. "
            "Do NOT search yet — wait for the user to narrow it down."
        ),
    }