"""
memory/redis_memory.py — Redis memory with in-memory dictionary fallback

Redis is responsible for:
  1. session:{user_id}          — auth session (24h TTL)
  2. trending:all               — sorted set of product_id → click count
  3. search_cache:{query_hash}  — cached NLP embedding lookups (5-min TTL)

If Redis is unreachable, automatically falls back to an in-memory dictionary store.
"""

import os
import json
import time
import hashlib
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_SESSION_TTL      = 86_400  # 24 hours
_SEARCH_CACHE_TTL = 300     # 5 minutes

_client = None
_use_memory_fallback = False
_mem_store = {}        # key -> (value, expiry_timestamp)
_mem_trending = {}     # product_id_str -> count


def _get_client():
    global _client, _use_memory_fallback
    if _use_memory_fallback:
        return None
    if _client is not None:
        try:
            if _client.ping():
                return _client
        except Exception:
            pass
    try:
        import redis
        kwargs = dict(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        pw = os.getenv("REDIS_PASSWORD", "")
        if pw:
            kwargs["password"] = pw
        c = redis.Redis(**kwargs)
        if c.ping():
            _client = c
            return _client
    except Exception:
        pass

    _use_memory_fallback = True
    print("[Redis] Unavailable. Using in-memory fallback store.")
    return None


def ping() -> bool:
    client = _get_client()
    if client is not None:
        try:
            return client.ping()
        except Exception:
            return False
    return True  # Fallback mode is always active and working


def set_session(user_id: int, email: str, full_name: str) -> None:
    client = _get_client()
    data = json.dumps({
        "user_id": user_id,
        "email": email,
        "full_name": full_name,
        "logged_in": True,
    })
    key = f"session:{user_id}"
    if client:
        client.set(key, data, ex=_SESSION_TTL)
    else:
        _mem_store[key] = (data, time.time() + _SESSION_TTL)


def get_session(user_id: int) -> Optional[dict]:
    client = _get_client()
    key = f"session:{user_id}"
    if client:
        raw = client.get(key)
    else:
        item = _mem_store.get(key)
        if item:
            val, exp = item
            if exp > time.time():
                raw = val
            else:
                del _mem_store[key]
                raw = None
        else:
            raw = None
    return json.loads(raw) if raw else None


def clear_session(user_id: int) -> None:
    client = _get_client()
    key = f"session:{user_id}"
    if client:
        client.delete(key)
    else:
        _mem_store.pop(key, None)


def refresh_session(user_id: int) -> None:
    client = _get_client()
    key = f"session:{user_id}"
    if client:
        client.expire(key, _SESSION_TTL)
    else:
        item = _mem_store.get(key)
        if item:
            _mem_store[key] = (item[0], time.time() + _SESSION_TTL)


def increment_product_view(product_id: int) -> None:
    client = _get_client()
    pid_str = str(product_id)
    if client:
        client.zincrby("trending:all", 1, pid_str)
    else:
        _mem_trending[pid_str] = _mem_trending.get(pid_str, 0) + 1


def get_trending_product_ids(top_n: int = 10) -> list[int]:
    client = _get_client()
    if client:
        ids = client.zrevrange("trending:all", 0, top_n - 1)
        return [int(i) for i in ids]
    else:
        sorted_items = sorted(_mem_trending.items(), key=lambda x: x[1], reverse=True)
        return [int(k) for k, _ in sorted_items[:top_n]]


def reset_trending() -> None:
    client = _get_client()
    if client:
        client.delete("trending:all")
    else:
        _mem_trending.clear()


def _cache_key(query: str) -> str:
    return "search_cache:" + hashlib.md5(query.lower().strip().encode()).hexdigest()


def get_cached_embedding(query: str) -> Optional[list]:
    client = _get_client()
    key = _cache_key(query)
    if client:
        raw = client.get(key)
    else:
        item = _mem_store.get(key)
        if item:
            val, exp = item
            if exp > time.time():
                raw = val
            else:
                del _mem_store[key]
                raw = None
        else:
            raw = None
    return json.loads(raw) if raw else None


def set_cached_embedding(query: str, embedding: list) -> None:
    client = _get_client()
    key = _cache_key(query)
    data = json.dumps(embedding)
    if client:
        client.set(key, data, ex=_SEARCH_CACHE_TTL)
    else:
        _mem_store[key] = (data, time.time() + _SEARCH_CACHE_TTL)


# Stubs
def get_cart(user_id: int) -> list:
    from db.mongo_client import get_cart as mongo_get_cart
    return mongo_get_cart(user_id)


def save_cart(user_id: int, cart: list) -> None:
    from db.mongo_client import save_cart as mongo_save_cart
    mongo_save_cart(user_id, cart)