"""
db/mongo_client.py — MongoDB client layer with transparent JSON file fallback

Collections:
  products       — full product documents + embedding field
  carts          — one document per user: {user_id, items: [...], updated_at}
  orders_log     — denormalised order snapshots for analytics
  product_vectors— lightweight {product_id, title, embedding} for vector search
"""

import os
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_client: Optional[object] = None
_lock = threading.Lock()
_use_fallback = False
_json_store_path = Path(__file__).resolve().parent / "mongo_store.json"

# Memory/File fallback store structure
_fallback_data = {
    "products": {},         # product_id_str -> dict
    "carts": {},            # user_id_str -> list of dicts
    "orders_log": [],       # list of dicts
    "product_vectors": {}   # product_id_str -> dict
}


def _load_fallback_store():
    global _fallback_data
    if _json_store_path.exists():
        try:
            with open(_json_store_path, "r", encoding="utf-8") as f:
                _fallback_data = json.load(f)
        except Exception:
            pass


def _save_fallback_store():
    try:
        with open(_json_store_path, "w", encoding="utf-8") as f:
            json.dump(_fallback_data, f, indent=2, default=str)
    except Exception:
        pass


_load_fallback_store()


def ping() -> bool:
    global _client, _use_fallback
    if _use_fallback:
        return True
    try:
        from pymongo import MongoClient
        if _client is None:
            uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
            _client = MongoClient(
                uri,
                maxPoolSize=10,
                connectTimeoutMS=2000,
                serverSelectionTimeoutMS=2000,
            )
        _client.admin.command("ping")
        return True
    except Exception:
        _use_fallback = True
        print(f"[MongoDB] Server offline. Using local JSON store fallback ({_json_store_path}).")
        return True


def get_client():
    if ping() and not _use_fallback:
        return _client
    return None


def get_db():
    client = get_client()
    if client and not _use_fallback:
        db_name = os.getenv("MONGO_DATABASE", "shopping_assistant")
        return client[db_name]
    return None


def products_col():
    db = get_db()
    return db["products"] if db else None

def carts_col():
    db = get_db()
    return db["carts"] if db else None

def orders_log_col():
    db = get_db()
    return db["orders_log"] if db else None

def product_vectors_col():
    db = get_db()
    return db["product_vectors"] if db else None


def ensure_indexes() -> None:
    if get_db() and not _use_fallback:
        try:
            from pymongo import ASCENDING, DESCENDING, TEXT
            db = get_db()
            db["products"].create_index(
                [("title", TEXT), ("category", TEXT), ("brand", TEXT), ("description", TEXT)],
                name="products_text_idx",
                default_language="english",
                weights={"title": 10, "category": 5, "brand": 3, "description": 1},
            )
            db["products"].create_index([("product_id", ASCENDING)], unique=True)
            db["carts"].create_index([("user_id", ASCENDING)], unique=True)
            db["orders_log"].create_index([("order_id", ASCENDING)], unique=True)
            db["product_vectors"].create_index([("product_id", ASCENDING)], unique=True)
        except Exception:
            pass
    print("[MongoDB] Indexes ensured.")


def upsert_product(doc: dict) -> None:
    if get_db() and not _use_fallback:
        products_col().update_one({"product_id": doc["product_id"]}, {"$set": doc}, upsert=True)
    else:
        pid = str(doc["product_id"])
        _fallback_data["products"][pid] = dict(doc)
        _save_fallback_store()


def get_product_by_id(product_id: int) -> Optional[dict]:
    if get_db() and not _use_fallback:
        doc = products_col().find_one({"product_id": product_id}, {"_id": 0})
        return doc
    else:
        pid = str(product_id)
        doc = _fallback_data["products"].get(pid)
        return dict(doc) if doc else None


def search_products_text(query: str, limit: int = 8) -> list[dict]:
    if get_db() and not _use_fallback:
        from pymongo import DESCENDING
        cursor = products_col().find(
            {"$text": {"$search": query}, "stock": {"$gt": 0}},
            {"score": {"$meta": "textScore"}, "_id": 0},
        ).sort([("score", {"$meta": "textScore"}), ("rating", DESCENDING)]).limit(limit)
        results = list(cursor)
        if results:
            return results
        pattern = {"$regex": query, "$options": "i"}
        return list(products_col().find({
            "$or": [{"title": pattern}, {"category": pattern}, {"brand": pattern}, {"description": pattern}],
            "stock": {"$gt": 0}
        }, {"_id": 0}).sort("rating", DESCENDING).limit(limit))
    else:
        q = query.lower()
        matched = []
        for p in _fallback_data["products"].values():
            if p.get("stock", 0) <= 0:
                continue
            title = p.get("title", "").lower()
            cat = p.get("category", "").lower()
            brand = p.get("brand", "").lower()
            desc = p.get("description", "").lower()
            if q in title or q in cat or q in brand or q in desc:
                matched.append(dict(p))
        matched.sort(key=lambda x: x.get("rating", 0), reverse=True)
        return matched[:limit]


def search_products_filtered(
    query: str,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    category: Optional[str] = None,
    limit: int = 8,
) -> list[dict]:
    if get_db() and not _use_fallback:
        from pymongo import DESCENDING
        match: dict = {"stock": {"$gt": 0}}
        if query:
            match["$or"] = [
                {"title": {"$regex": query, "$options": "i"}},
                {"category": {"$regex": query, "$options": "i"}},
                {"brand": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}},
            ]
        if max_price is not None:
            match["price"] = {"$lte": max_price}
        if min_rating is not None:
            match.setdefault("rating", {})["$gte"] = min_rating
        if category:
            match["category"] = {"$regex": category, "$options": "i"}
        return list(products_col().find(match, {"_id": 0}).sort([("rating", DESCENDING)]).limit(limit))
    else:
        q = query.lower() if query else ""
        cat_q = category.lower() if category else ""
        matched = []
        for p in _fallback_data["products"].values():
            if p.get("stock", 0) <= 0:
                continue
            if max_price is not None and p.get("price", 0) > max_price:
                continue
            if min_rating is not None and p.get("rating", 0) < min_rating:
                continue
            if cat_q and cat_q not in p.get("category", "").lower():
                continue
            if q:
                t = p.get("title", "").lower() + p.get("category", "").lower() + p.get("brand", "").lower() + p.get("description", "").lower()
                if q not in t:
                    continue
            matched.append(dict(p))
        matched.sort(key=lambda x: (x.get("rating", 0), x.get("rating_count", 0)), reverse=True)
        return matched[:limit]


def get_popular_products(limit: int = 5) -> list[dict]:
    if get_db() and not _use_fallback:
        from pymongo import DESCENDING
        return list(products_col().find({"stock": {"$gt": 0}}, {"_id": 0}).sort([("rating", DESCENDING)]).limit(limit))
    else:
        items = [dict(p) for p in _fallback_data["products"].values() if p.get("stock", 0) > 0]
        items.sort(key=lambda x: (x.get("rating", 0), x.get("rating_count", 0)), reverse=True)
        return items[:limit]


def get_distinct_categories() -> list[str]:
    if get_db() and not _use_fallback:
        return [c for c in products_col().distinct("category") if c]
    else:
        cats = {p.get("category") for p in _fallback_data["products"].values() if p.get("category")}
        return sorted(list(cats))


def get_distinct_brands() -> list[str]:
    if get_db() and not _use_fallback:
        return [b for b in products_col().distinct("brand") if b]
    else:
        brands = {p.get("brand") for p in _fallback_data["products"].values() if p.get("brand")}
        return sorted(list(brands))


def decrement_stock(product_id: int) -> bool:
    if get_db() and not _use_fallback:
        res = products_col().update_one({"product_id": product_id, "stock": {"$gt": 0}}, {"$inc": {"stock": -1}})
        return res.modified_count == 1
    else:
        pid = str(product_id)
        p = _fallback_data["products"].get(pid)
        if p and p.get("stock", 0) > 0:
            p["stock"] -= 1
            _save_fallback_store()
            return True
        return False


def get_cart(user_id: int) -> list:
    if get_db() and not _use_fallback:
        doc = carts_col().find_one({"user_id": user_id}, {"_id": 0, "items": 1})
        return doc["items"] if doc else []
    else:
        uid = str(user_id)
        return list(_fallback_data["carts"].get(uid, []))


def save_cart(user_id: int, items: list) -> None:
    if get_db() and not _use_fallback:
        carts_col().update_one(
            {"user_id": user_id},
            {"$set": {"items": items, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
    else:
        uid = str(user_id)
        _fallback_data["carts"][uid] = list(items)
        _save_fallback_store()


def add_item_to_cart(user_id: int, item: dict) -> dict:
    cart = get_cart(user_id)
    if any(i.get("product_id") == item.get("product_id") for i in cart):
        return {"added": False, "reason": "duplicate", "title": item["title"], "cart": cart}

    cart.append(item)
    save_cart(user_id, cart)
    return {"added": True, "title": item["title"], "cart": cart}


def remove_item_from_cart(user_id: int, product_name: str) -> dict:
    cart = get_cart(user_id)
    lower = product_name.lower()
    match = next((p for p in cart if lower in p["title"].lower()), None)
    if not match:
        return {"removed": False, "query": product_name, "cart": cart}

    cart = [i for i in cart if i.get("product_id") != match.get("product_id")]
    save_cart(user_id, cart)
    return {"removed": True, "title": match["title"], "query": product_name, "cart": cart}


def clear_cart(user_id: int) -> None:
    save_cart(user_id, [])


def log_order(order: dict) -> None:
    if get_db() and not _use_fallback:
        try:
            orders_log_col().insert_one({**order, "logged_at": datetime.now(timezone.utc)})
        except Exception:
            pass
    else:
        _fallback_data["orders_log"].append(dict(order))
        _save_fallback_store()


def upsert_product_vector(product_id: int, title: str, embedding: list) -> None:
    if get_db() and not _use_fallback:
        product_vectors_col().update_one(
            {"product_id": product_id},
            {"$set": {"product_id": product_id, "title": title, "embedding": embedding}},
            upsert=True,
        )
    else:
        pid = str(product_id)
        _fallback_data["product_vectors"][pid] = {
            "product_id": product_id,
            "title": title,
            "embedding": list(embedding)
        }
        _save_fallback_store()


def get_all_product_vectors() -> list[dict]:
    if get_db() and not _use_fallback:
        return list(product_vectors_col().find({}, {"_id": 0}))
    else:
        return list(_fallback_data["product_vectors"].values())
