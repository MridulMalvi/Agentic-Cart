"""
api.py — Flask API backend for the AI Shopping Assistant

Runs as a separate process on port 5001 (Streamlit stays on 8501).
Streamlit dashboard polls these endpoints for analytics panels and
recommendation cards without blocking the main chat thread.

Endpoints:
  GET  /api/health                → service status check
  POST /api/recommendations       → CrewAI recommendation crew
  POST /api/sentiment             → CrewAI sentiment agent
  POST /api/semantic-search       → NLP semantic product search
  GET  /api/trending              → Top trending product IDs from Redis
  GET  /api/analytics/overview    → Order + product stats from MongoDB
  GET  /api/analytics/categories  → Sales by category

Security: CORS enabled for localhost:8501 only (Streamlit origin).
"""

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:8501",
    "http://127.0.0.1:8501",
])


# ── Lazy imports (heavy modules load on first request, not at startup) ─────────

def _get_crew():
    from crew import ShoppingCrewAI
    return ShoppingCrewAI()


_crew_instance = None


def get_crew():
    global _crew_instance
    if _crew_instance is None:
        _crew_instance = _get_crew()
    return _crew_instance


# ── Health ─────────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Quick health check — verifies DB connectivity."""
    status = {"api": "ok", "mongodb": False, "redis": False, "mysql": False}

    try:
        from db.mongo_client import ping as mongo_ping
        status["mongodb"] = mongo_ping()
    except Exception as exc:
        status["mongodb_error"] = str(exc)

    try:
        from memory.redis_memory import ping as redis_ping
        status["redis"] = redis_ping()
    except Exception as exc:
        status["redis_error"] = str(exc)

    try:
        from db.db_client import execute_query
        execute_query("SELECT 1", fetch=True)
        status["mysql"] = True
    except Exception as exc:
        status["mysql_error"] = str(exc)

    http_code = 200 if all([status["mongodb"], status["redis"], status["mysql"]]) else 503
    return jsonify(status), http_code


# ── Recommendations ───────────────────────────────────────────────────────────

@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    """
    Generate product recommendations based on cart contents.
    Body: { "user_id": int, "cart_items": [...] }
    """
    body = request.get_json(silent=True) or {}
    user_id    = body.get("user_id", 0)
    cart_items = body.get("cart_items", [])

    crew = get_crew()
    result = crew.run_recommendations(
        user_id=user_id,
        cart_items=cart_items,
        top_n=6,
    )
    return jsonify(result)


# ── Sentiment ─────────────────────────────────────────────────────────────────

@app.route("/api/sentiment", methods=["POST"])
def sentiment():
    """
    Run VADER sentiment analysis on product text.
    Body: { "text": str, "product_title": str }
    """
    body = request.get_json(silent=True) or {}
    text          = body.get("text", "")
    product_title = body.get("product_title", "")

    if not text:
        return jsonify({"error": "text is required"}), 400

    crew = get_crew()
    result = crew.run_sentiment(text=text, product_title=product_title)
    return jsonify(result)


# ── Semantic search ───────────────────────────────────────────────────────────

@app.route("/api/semantic-search", methods=["POST"])
def semantic_search():
    """
    NLP semantic product search via sentence-transformers.
    Body: { "query": str, "top_n": int }
    """
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "")
    top_n = int(body.get("top_n", 6))

    if not query:
        return jsonify({"error": "query is required"}), 400

    crew = get_crew()
    result = crew.run_semantic_search(query=query, top_n=top_n)
    return jsonify(result)


# ── Trending ──────────────────────────────────────────────────────────────────

@app.route("/api/trending", methods=["GET"])
def trending():
    """
    Return trending product IDs from Redis sorted set + full product docs.
    Query: ?top_n=10
    """
    top_n = int(request.args.get("top_n", 10))

    try:
        from memory.redis_memory import get_trending_product_ids
        from db.mongo_client import get_product_by_id

        product_ids = get_trending_product_ids(top_n)
        products = []
        for pid in product_ids:
            doc = get_product_by_id(pid)
            if doc:
                doc.pop("_id", None)
                doc.pop("embedding", None)
                if "price" in doc:
                    doc["price"] = float(doc["price"])
                products.append(doc)

        return jsonify({
            "trending":       products,
            "count":          len(products),
            "source":         "redis_sorted_set",
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "trending": []}), 500


# ── Analytics: overview ───────────────────────────────────────────────────────

@app.route("/api/analytics/overview", methods=["GET"])
def analytics_overview():
    """
    High-level stats for the dashboard analytics panel.
    Returns total orders, revenue, unique customers from MySQL.
    """
    try:
        from db.db_client import execute_query
        from db.mongo_client import products_col

        order_stats = execute_query(
            "SELECT COUNT(*) AS total_orders, "
            "       IFNULL(SUM(total_amount), 0) AS total_revenue, "
            "       COUNT(DISTINCT user_id) AS unique_customers "
            "FROM orders",
        )
        product_count = products_col().count_documents({"stock": {"$gt": 0}})

        stats = order_stats[0] if order_stats else {}
        return jsonify({
            "total_orders":      int(stats.get("total_orders", 0)),
            "total_revenue":     float(stats.get("total_revenue", 0)),
            "unique_customers":  int(stats.get("unique_customers", 0)),
            "products_in_stock": product_count,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Analytics: category breakdown ─────────────────────────────────────────────

@app.route("/api/analytics/categories", methods=["GET"])
def analytics_categories():
    """
    Product count and average rating by category (from MongoDB).
    Used by the dashboard donut/bar chart.
    """
    try:
        from db.mongo_client import get_db
        db = get_db()
        pipeline = [
            {"$match": {"stock": {"$gt": 0}}},
            {"$group": {
                "_id":          "$category",
                "count":        {"$sum": 1},
                "avg_rating":   {"$avg": "$rating"},
                "avg_price":    {"$avg": "$price"},
                "total_stock":  {"$sum": "$stock"},
            }},
            {"$sort": {"count": -1}},
        ]
        results = list(db["products"].aggregate(pipeline))
        categories = [
            {
                "category":   r["_id"] or "Uncategorised",
                "count":      r["count"],
                "avg_rating": round(r["avg_rating"] or 0, 2),
                "avg_price":  round(r["avg_price"] or 0, 2),
                "total_stock": r["total_stock"],
            }
            for r in results
        ]
        return jsonify({"categories": categories})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host  = os.getenv("FLASK_HOST", "0.0.0.0")
    port  = int(os.getenv("FLASK_PORT", 5001))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    print(f"\n[Flask API] Starting on http://{host}:{port}")
    print("[Flask API] Endpoints:")
    print("  GET  /api/health")
    print("  POST /api/recommendations")
    print("  POST /api/sentiment")
    print("  POST /api/semantic-search")
    print("  GET  /api/trending")
    print("  GET  /api/analytics/overview")
    print("  GET  /api/analytics/categories\n")

    app.run(host=host, port=port, debug=debug, use_reloader=False)
