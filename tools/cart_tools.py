"""
tools/cart_tools.py  (Refactored — MongoDB cart storage)

All cart and order operations. Prices in INR (₹). Hard login guard on every tool.

Changes from original:
  - Cart storage moved from Redis to MongoDB (db/mongo_client.py).
    MongoDB provides persistence across server restarts and supports
    richer queries (e.g. aggregation for analytics).
  - Stock decrement still happens in MySQL (source of truth for inventory).
  - Email still sent in daemon thread (non-blocking).
  - buy_now never touches the existing cart (unchanged).
  - place_order removes ONLY the ordered items from MongoDB cart.
  - All cart writes are atomic at the document level via MongoDB
    findAndModify semantics (no WATCH/pipeline needed for single-doc ops).
"""

import uuid
import json
import threading
from datetime import datetime

from langchain_core.tools import tool
from db.db_client import execute_query, execute_transaction
from db.mongo_client import (
    get_cart,
    save_cart,
    add_item_to_cart,
    remove_item_from_cart as mongo_remove,
    clear_cart,
    log_order,
)
from memory.redis_memory import increment_product_view
from services.email_service import send_order_confirmation


# ── Login guard ───────────────────────────────────────────────────────────────

def _require_login(user_id) -> dict | None:
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        uid = 0
    if uid <= 0:
        return {
            "blocked": True,
            "reason":  "not_authenticated",
            "message": "You are not logged in. Please login first.",
        }
    return None


# ── Product resolver (still uses MongoDB) ─────────────────────────────────────

def _resolve_product(query: str) -> list:
    """
    Find in-stock products matching query via MongoDB text search.
    Returns up to 5 results.
    """
    from db.mongo_client import search_products_filtered
    return search_products_filtered(query=query, limit=5)


# ── User info ─────────────────────────────────────────────────────────────────

def _fetch_user_info(user_id: int) -> tuple[str, str]:
    rows = execute_query(
        "SELECT email, full_name FROM users WHERE user_id = %s",
        (user_id,),
    )
    if rows:
        return rows[0]["email"], rows[0].get("full_name") or rows[0]["email"]
    return "", ""


# ── Async email ───────────────────────────────────────────────────────────────

def _send_email_async(**kwargs) -> None:
    """Fire-and-forget email in a daemon thread. Never blocks the order response."""
    threading.Thread(
        target=send_order_confirmation,
        kwargs=kwargs,
        daemon=True,
    ).start()


# ── Item dict ─────────────────────────────────────────────────────────────────

def _to_item_dict(p: dict) -> dict:
    p.pop("_id", None)
    return {
        "product_id": p["product_id"],
        "title":      p["title"],
        "price":      float(p["price"]),
        "category":   p.get("category", ""),
        "rating":     p.get("rating", 0),
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def add_to_cart(product_name: str, user_id: int) -> dict:
    """
    Add a product to the user's cart (stored in MongoDB).

    REQUIRES: user must be logged in.

    Call when user says: 'add X to cart', 'put X in my cart', 'I want X'.

    Behaviour:
      - Single match  → add immediately.
      - Multiple matches → return all so LLM asks user to pick one.
      - No match → return not_found.

    Stock is NOT decremented here — only on order placement.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    matches = _resolve_product(product_name)

    if not matches:
        return {
            "added":   False,
            "reason":  "not_found",
            "query":   product_name,
            "message": f"No product matching '{product_name}' found in our catalogue.",
            "cart":    get_cart(user_id),
        }

    if len(matches) > 1:
        return {
            "added":   False,
            "reason":  "multiple_matches",
            "query":   product_name,
            "message": "Multiple products matched. Ask the user to pick one.",
            "matches": [_to_item_dict(p) for p in matches],
            "cart":    get_cart(user_id),
        }

    item   = _to_item_dict(matches[0])
    result = add_item_to_cart(user_id, item)
    cart   = result["cart"]

    return {
        "added":      result["added"],
        "reason":     result.get("reason", ""),
        "product":    item,
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "cart":       cart,
    }


@tool
def remove_from_cart(product_name: str, user_id: int) -> dict:
    """
    Remove a product from the user's cart.

    REQUIRES: user must be logged in.

    Call when user says: 'remove X', 'delete X from cart', 'drop X'.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    result = mongo_remove(user_id, product_name)
    cart   = result["cart"]

    return {
        "removed":    result["removed"],
        "query":      product_name,
        "title":      result.get("title"),
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "cart":       cart,
    }


@tool
def view_cart(user_id: int) -> dict:
    """
    Retrieve the user's current cart contents.

    REQUIRES: user must be logged in.

    Call when user says: 'show my cart', "what's in my cart", 'cart summary'.
    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    cart = get_cart(user_id)
    return {
        "cart":       cart,
        "cart_size":  len(cart),
        "cart_total": round(sum(i["price"] for i in cart), 2),
        "is_empty":   len(cart) == 0,
    }


@tool
def place_order(user_id: int, product_name: str = "") -> dict:
    """
    Place an order from the user's cart.

    REQUIRES: user must be logged in. Cart must not be empty.

    Behaviour:
      - product_name provided → order ONLY that item; remaining cart items stay.
      - product_name empty    → order ALL cart items; cart is cleared.

    For each ordered item (ATOMIC in MySQL — both succeed or both roll back):
      1. UPDATE products SET stock = stock - 1 WHERE product_id = %s AND stock > 0
      2. INSERT INTO orders ...

    Then:
      3. Remove ordered items from MongoDB cart.
      4. Log order snapshot to MongoDB (analytics).
      5. Send confirmation email in background thread.

    Returns raw dict — LLM formats. Prices in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    cart = get_cart(user_id)
    if not cart:
        return {
            "success": False,
            "reason":  "empty_cart",
            "message": "Your cart is empty. Add products first.",
        }

    # ── Determine which items to order ────────────────────────────────────
    if product_name:
        pn_lower    = product_name.lower()
        order_items = [i for i in cart if pn_lower in i["title"].lower()]
        if not order_items:
            return {
                "success": False,
                "reason":  "not_in_cart",
                "message": f"'{product_name}' is not in your cart.",
                "cart":    cart,
            }
    else:
        order_items = list(cart)

    total     = round(sum(i["price"] for i in order_items), 2)
    order_id  = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    placed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Atomic MySQL: decrement stock + insert order ───────────────────────
    try:
        statements = []
        for item in order_items:
            statements.append((
                "UPDATE products "
                "SET stock = stock - 1 "
                "WHERE product_id = %s AND stock > 0",
                (item["product_id"],),
            ))
        statements.append((
            "INSERT INTO orders (order_id, user_id, total_amount, items_json) "
            "VALUES (%s, %s, %s, %s)",
            (order_id, user_id, total, json.dumps(order_items)),
        ))
        execute_transaction(statements)
    except Exception as exc:
        return {"success": False, "reason": "db_error", "error": str(exc)}

    # ── Remove ONLY ordered items from MongoDB cart ────────────────────────
    ordered_ids = {str(i["product_id"]) for i in order_items}
    remaining   = [i for i in cart if str(i["product_id"]) not in ordered_ids]
    save_cart(user_id, remaining)

    # ── Log order snapshot to MongoDB for analytics ────────────────────────
    log_order({
        "order_id":  order_id,
        "user_id":   user_id,
        "total":     total,
        "items":     order_items,
        "placed_at": placed_at,
    })

    # ── Email in background (non-blocking) ────────────────────────────────
    email, full_name = _fetch_user_info(user_id)
    if email:
        _send_email_async(
            to_email  = email,
            full_name = full_name,
            order_id  = order_id,
            placed_at = placed_at,
            items     = order_items,
            total     = total,
        )

    return {
        "success":         True,
        "order_id":        order_id,
        "placed_at":       placed_at,
        "ordered_items":   order_items,
        "items_count":     len(order_items),
        "total":           total,
        "remaining_cart":  remaining,
        "remaining_count": len(remaining),
    }


@tool
def buy_now(product_name: str, user_id: int) -> dict:
    """
    INSTANT PURCHASE: find product → order it immediately.

    REQUIRES: user must be logged in.

    Orders ONLY the requested product. Existing cart is untouched.
    Atomic MySQL transaction: stock decrement + order INSERT.
    Email sent in background thread.

    Use when user says: 'buy X now', 'order X right now', 'get me X immediately'.

    Multiple matches → return options so LLM asks user to pick.
    Returns raw dict — LLM formats. Price in INR (raw float).
    """
    block = _require_login(user_id)
    if block:
        return block

    matches = _resolve_product(product_name)

    if not matches:
        return {"success": False, "reason": "not_found", "query": product_name}

    if len(matches) > 1:
        return {
            "success": False,
            "reason":  "multiple_matches",
            "query":   product_name,
            "matches": [_to_item_dict(p) for p in matches],
        }

    item        = _to_item_dict(matches[0])
    order_items = [item]
    total       = round(item["price"], 2)
    order_id    = f"ORD-{str(uuid.uuid4())[:8].upper()}"
    placed_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Atomic: decrement stock + insert order ────────────────────────────
    try:
        execute_transaction([
            (
                "UPDATE products "
                "SET stock = stock - 1 "
                "WHERE product_id = %s AND stock > 0",
                (item["product_id"],),
            ),
            (
                "INSERT INTO orders (order_id, user_id, total_amount, items_json) "
                "VALUES (%s, %s, %s, %s)",
                (order_id, user_id, total, json.dumps(order_items)),
            ),
        ])
    except Exception as exc:
        return {"success": False, "reason": "db_error", "error": str(exc)}

    # Cart intentionally untouched — log to MongoDB
    log_order({
        "order_id":  order_id,
        "user_id":   user_id,
        "total":     total,
        "items":     order_items,
        "placed_at": placed_at,
    })

    # ── Email in background ───────────────────────────────────────────────
    email, full_name = _fetch_user_info(user_id)
    if email:
        _send_email_async(
            to_email  = email,
            full_name = full_name,
            order_id  = order_id,
            placed_at = placed_at,
            items     = order_items,
            total     = total,
        )

    return {
        "success":          True,
        "instant_purchase": True,
        "order_id":         order_id,
        "placed_at":        placed_at,
        "product":          item,
        "items_count":      1,
        "total":            total,
    }