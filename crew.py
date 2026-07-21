"""
crew.py — CrewAI-powered analytics and recommendation crew

Three specialised agents running as background tasks via ProcessPoolExecutor,
exploiting all 8 Zen 5 cores of the AMD Ryzen AI 7 350.

Agents:
  ProductSearchAgent    — semantic vector search across product catalogue
  RecommendationAgent   — collaborative-filter-style recommendations from order history
  SentimentAgent        — deep VADER + TextBlob sentiment on product descriptions

Architecture:
  - Each agent is a crewai.Agent with its own tool set.
  - Crews are created fresh per request (stateless, thread-safe).
  - ProcessPoolExecutor (max_workers=6) for CPU-bound NLP.
  - Results returned as JSON-serialisable dicts.
  - Exposed via Flask API (api.py) — Streamlit dashboard polls these endpoints.

Usage:
    from crew import ShoppingCrewAI
    crew = ShoppingCrewAI()
    result = crew.run_recommendations(user_id=1, cart_items=[...])
    result = crew.run_sentiment(product_id=5, text="...")
"""

import os
import json
import concurrent.futures
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Limit CPU threads for each worker to avoid over-subscription
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_NLP_WORKERS = int(os.getenv("NLP_WORKER_PROCESSES", "4"))
_executor = concurrent.futures.ProcessPoolExecutor(max_workers=_NLP_WORKERS)


# ── CrewAI imports (graceful fallback if not installed) ───────────────────────

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import tool as crewai_tool
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False
    print("[CrewAI] WARNING: crewai not installed. Crew endpoints will use fallback logic.")


# ── Standalone worker functions (must be top-level for pickling in ProcessPool) ─

def _worker_semantic_search(query: str, top_n: int) -> dict:
    """Runs in a separate process — safe for ProcessPoolExecutor."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.nlp_tools import semantic_search_products
    result = semantic_search_products.invoke({"query": query, "top_n": top_n})
    return result


def _worker_sentiment(text: str, product_title: str) -> dict:
    """Runs in a separate process."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.nlp_tools import analyze_sentiment
    return analyze_sentiment.invoke({"text": text, "product_title": product_title})


def _worker_recommendations(user_id: int, cart_categories: list, top_n: int) -> dict:
    """
    Simple content-based recommendations: find products in the same categories
    as the cart items that are NOT already in the cart.
    Runs in a separate process.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from db.mongo_client import search_products_filtered, get_popular_products

    if not cart_categories:
        # Cold start — return popular products
        products = get_popular_products(top_n)
        return {
            "method":   "popular",
            "products": [_strip(p) for p in products],
            "reason":   "Based on our top-rated items",
        }

    results = []
    for category in cart_categories[:3]:
        hits = search_products_filtered(query="", category=category, limit=3)
        results.extend(hits)

    # Deduplicate by product_id
    seen = set()
    unique = []
    for p in results:
        pid = p.get("product_id")
        if pid not in seen:
            seen.add(pid)
            unique.append(_strip(p))

    return {
        "method":   "category_based",
        "products": unique[:top_n],
        "reason":   f"Because you're interested in {', '.join(cart_categories[:2])}",
    }


def _strip(doc: dict) -> dict:
    doc.pop("_id", None)
    doc.pop("embedding", None)
    if "price" in doc:
        doc["price"] = float(doc["price"])
    return doc


# ── CrewAI tool wrappers (only used when crewai IS installed) ─────────────────

def _make_crewai_tools():
    """Return CrewAI-decorated tool functions."""
    if not _CREWAI_AVAILABLE:
        return [], [], []

    @crewai_tool("SemanticProductSearch")
    def semantic_search_tool(query: str) -> str:
        """Search products semantically using NLP embeddings."""
        from tools.nlp_tools import semantic_search_products
        result = semantic_search_products.invoke({"query": query, "top_n": 6})
        return json.dumps(result, indent=2)

    @crewai_tool("SentimentAnalysis")
    def sentiment_tool(text: str) -> str:
        """Analyse sentiment of product description or review text."""
        from tools.nlp_tools import analyze_sentiment
        result = analyze_sentiment.invoke({"text": text, "product_title": ""})
        return json.dumps(result, indent=2)

    @crewai_tool("ProductRecommender")
    def recommend_tool(categories_json: str) -> str:
        """Recommend products based on a JSON list of category strings."""
        try:
            categories = json.loads(categories_json)
        except Exception:
            categories = []
        result = _worker_recommendations(0, categories, 6)
        return json.dumps(result, indent=2)

    return [semantic_search_tool], [sentiment_tool], [recommend_tool]


# ── ShoppingCrewAI ────────────────────────────────────────────────────────────

class ShoppingCrewAI:
    """
    Orchestrates three specialised CrewAI agents for the shopping assistant.
    Falls back to direct Python execution when CrewAI is not installed.
    """

    def __init__(self):
        self.available = _CREWAI_AVAILABLE
        if self.available:
            search_tools, sentiment_tools, rec_tools = _make_crewai_tools()
            if os.getenv("GOOGLE_API_KEY"):
                llm_config = "gemini/" + os.getenv("LLM_MODEL", "gemini-2.0-flash")
            else:
                llm_config = {
                    "provider": "mistral",
                    "config": {
                        "model":       os.getenv("LLM_MODEL", "mistral-large-latest"),
                        "api_key":     os.getenv("MISTRAL_API_KEY", ""),
                        "temperature": 0.2,
                    },
                }
            self.product_agent = Agent(
                role        = "Product Search Specialist",
                goal        = "Find the most relevant products using semantic NLP search",
                backstory   = (
                    "You are an expert at understanding user shopping intent "
                    "and translating it into precise product searches."
                ),
                tools       = search_tools,
                llm         = llm_config,
                verbose     = False,
                allow_delegation = False,
            )
            self.sentiment_agent = Agent(
                role        = "Review Sentiment Analyst",
                goal        = "Analyse product descriptions and review text for sentiment",
                backstory   = (
                    "You are a sentiment analysis expert specialising in e-commerce "
                    "product reviews and descriptions."
                ),
                tools       = sentiment_tools,
                llm         = llm_config,
                verbose     = False,
                allow_delegation = False,
            )
            self.recommendation_agent = Agent(
                role        = "Product Recommendation Engine",
                goal        = "Recommend complementary products based on cart contents",
                backstory   = (
                    "You understand shopping patterns and can suggest products "
                    "that complement what the user already wants to buy."
                ),
                tools       = rec_tools,
                llm         = llm_config,
                verbose     = False,
                allow_delegation = False,
            )

    def run_semantic_search(self, query: str, top_n: int = 6) -> dict:
        """
        Run NLP semantic search in a worker process (non-blocking).
        Falls back gracefully if model not available.
        """
        try:
            future = _executor.submit(_worker_semantic_search, query, top_n)
            return future.result(timeout=30)
        except Exception as exc:
            return {"error": str(exc), "products": [], "found": 0}

    def run_sentiment(self, text: str, product_title: str = "") -> dict:
        """Run VADER sentiment analysis in a worker process."""
        try:
            future = _executor.submit(_worker_sentiment, text, product_title)
            return future.result(timeout=15)
        except Exception as exc:
            return {"error": str(exc), "sentiment_label": "Unknown", "compound_score": 0.0}

    def run_recommendations(
        self,
        user_id: int,
        cart_items: list,
        top_n: int = 6,
    ) -> dict:
        """
        Generate product recommendations based on cart categories.
        Runs in a worker process to keep the API thread free.
        """
        cart_categories = list({
            item.get("category", "").strip()
            for item in cart_items
            if item.get("category")
        })
        try:
            future = _executor.submit(
                _worker_recommendations, user_id, cart_categories, top_n
            )
            return future.result(timeout=20)
        except Exception as exc:
            return {"error": str(exc), "products": [], "method": "error"}

    def run_crew_sentiment_analysis(self, product_id: int, text: str) -> dict:
        """
        Full CrewAI crew run for sentiment — uses Agent reasoning.
        Only called when CrewAI is installed and user wants detailed analysis.
        Falls back to direct Python if not available.
        """
        if not self.available:
            return self.run_sentiment(text)

        task = Task(
            description=(
                f"Analyse the sentiment of this product description:\n\n{text}\n\n"
                "Provide: sentiment label (Positive/Negative/Neutral), "
                "compound score, and a 2-sentence customer-facing summary."
            ),
            expected_output="JSON with sentiment_label, compound_score, summary",
            agent=self.sentiment_agent,
        )
        crew = Crew(
            agents  = [self.sentiment_agent],
            tasks   = [task],
            process = Process.sequential,
            verbose = False,
        )
        try:
            result_str = crew.kickoff()
            # Try to parse JSON from crew output
            try:
                return json.loads(str(result_str))
            except Exception:
                return {"summary": str(result_str), "sentiment_label": "Unknown"}
        except Exception as exc:
            return self.run_sentiment(text)
