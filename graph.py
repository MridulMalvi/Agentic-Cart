"""
graph.py — LangGraph + Mistral AI Shopping Assistant  (Refactored)

Graph topology (pure state-driven, zero intent routing):

  ┌─────────────┐
  │  auth_gate  │  ← Pure Python. Sets routing_decision.
  └──────┬──────┘
         │
    ┌────┴─────┐
    ▼          ▼
login_llm  shopping_llm  ← Two LLM nodes, different tool bindings.
    │          │
    └────┬─────┘
         ▼
       tools    ← Executes tool calls. Syncs auth. Updates last_products.
         │
    ┌────┴─────┐
    ▼          ▼
login_llm  shopping_llm  ← route_from_tools picks based on is_logged_in.

New in this refactor:
  - Semantic search tool (semantic_search_products) added to SHOPPING_TOOLS.
  - NLP tools (analyze_sentiment, get_semantic_suggestions) added.
  - register_user added to LOGIN_TOOLS.
  - Stream-mode support: use run_graph_stream() for token-by-token output.
  - ShoppingState gains nlp_last_method field to track search method used.
  - System prompts updated to instruct LLM on when to use semantic search.
  - LLM provider configurable via LLM_MODEL env var.
"""

import os
import json
import re
from typing import Annotated, Sequence, Generator
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
def _is_valid_key(key: str | None) -> bool:
    if not key:
        return False
    k = key.strip().lower()
    return k != "" and "your_" not in k and "here" not in k and "placeholder" not in k

def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    mistral_api_key = os.getenv("MISTRAL_API_KEY")

    if _is_valid_key(google_api_key) and not (_is_valid_key(mistral_api_key) and os.getenv("LLM_PROVIDER") == "mistral"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        _model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=_model_name,
            google_api_key=google_api_key,
            temperature=0.3,
            streaming=True,
        )
    elif _is_valid_key(mistral_api_key):
        from langchain_mistralai import ChatMistralAI
        _model_name = os.getenv("LLM_MODEL", "mistral-large-latest")
        return ChatMistralAI(
            model=_model_name,
            api_key=mistral_api_key,
            temperature=0.3,
            streaming=True,
        )
    return None

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools.auth_tools    import login_user, register_user
from tools.product_tools import (
    search_products,
    filter_products,
    get_reviews,
    get_product_details,
    clarify_product_query,
)
from tools.nlp_tools import (
    semantic_search_products,
    analyze_sentiment,
    get_semantic_suggestions,
)
from tools.cart_tools import (
    add_to_cart,
    remove_from_cart,
    view_cart,
    place_order,
    buy_now,
)

load_dotenv()

# ── Tool sets ─────────────────────────────────────────────────────────────────

LOGIN_TOOLS = [login_user, register_user]

SHOPPING_TOOLS = [
    # Product discovery
    search_products,
    semantic_search_products,   # NEW: NLP-powered intent search
    filter_products,
    get_reviews,
    get_product_details,
    clarify_product_query,
    # NLP
    analyze_sentiment,           # NEW: VADER sentiment on descriptions
    get_semantic_suggestions,    # NEW: query expansion hints
    # Cart & orders
    add_to_cart,
    remove_from_cart,
    view_cart,
    place_order,
    buy_now,
]

ALL_TOOLS = LOGIN_TOOLS + SHOPPING_TOOLS

# Singleton ToolNode — created once at module load
_tool_node = ToolNode(ALL_TOOLS)


# ── State ─────────────────────────────────────────────────────────────────────

class ShoppingState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], add_messages]
    user_id:           int | None
    is_logged_in:      bool
    user_email:        str | None
    routing_decision:  str    # "needs_login" | "go_shopping"
    last_products:     list   # context memory — products shown this session
    nlp_last_method:   str    # "semantic" | "text" | "" — last search method


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_login_attempt(text: str) -> bool:
    """
    Returns True when the message contains an email address plus something
    that looks like a password — regardless of exact phrasing.
    """
    has_email = bool(
        re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    )
    if not has_email:
        return False
    t = text.lower()
    if any(kw in t for kw in ["password", "pass ", "pass:", "pwd", " and "]):
        return True
    leftover = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "", text
    ).strip()
    return len(leftover) >= 4


def _is_registration_attempt(text: str) -> bool:
    """Returns True when the user is explicitly asking to sign up."""
    t = text.lower()
    return any(kw in t for kw in [
        "sign up", "signup", "register", "create account", "new account", "join",
    ])


def _normalize_tool_calls(response: AIMessage) -> AIMessage:
    """
    Normalize Mistral's tool_calls from additional_kwargs into the
    response.tool_calls list that LangGraph ToolNode reads.

    CRITICAL: After writing normalized tool_calls, CLEAR additional_kwargs["tool_calls"]
    to prevent the ToolNode from seeing stale pending calls after they've already run.
    """
    raw = getattr(response, "additional_kwargs", {}).get("tool_calls")
    if not raw:
        return response

    normalized = []
    for tc in raw:
        args = tc["function"]["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        normalized.append({
            "id":   tc["id"],
            "name": tc["function"]["name"],
            "args": args,
        })

    response.tool_calls = normalized

    if hasattr(response, "additional_kwargs") and "tool_calls" in response.additional_kwargs:
        response.additional_kwargs.pop("tool_calls", None)

    return response


def _has_pending_tool_calls(state: ShoppingState) -> bool:
    """
    Returns True only if the LAST message is an AIMessage with pending tool_calls.
    Single authoritative check used by ALL routers.
    """
    messages = list(state["messages"])
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return False
    if isinstance(last, AIMessage):
        tc = getattr(last, "tool_calls", None)
        return bool(tc)
    return False


def _extract_products_from_messages(messages: list) -> list:
    """
    Scan ToolMessages for product data.
    Returns the LAST non-empty product list found.
    """
    product_tools = {
        "search_products", "filter_products", "semantic_search_products",
        "get_reviews", "get_product_details", "add_to_cart", "buy_now",
    }
    last_products = []
    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name not in product_tools:
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if not isinstance(data, dict):
                continue
            if "products" in data and isinstance(data["products"], list) and data["products"]:
                last_products = data["products"]
            elif data.get("found") and "product_id" in data:
                last_products = [data]
        except Exception:
            pass
    return last_products


def _format_last_products_context(last_products: list) -> str:
    if not last_products:
        return "  (none — no products shown yet)"
    lines = []
    for i, p in enumerate(last_products, 1):
        lines.append(
            f"  [{i}] product_id={p.get('product_id')}  |  "
            f"{p.get('title', 'Unknown')}  |  "
            f"₹{p.get('price', '-')}  |  "
            f"⭐{p.get('rating', '-')}  |  "
            f"brand: {p.get('brand', '-')}  |  "
            f"stock: {p.get('stock', '-')}"
        )
    return "\n".join(lines)


# ── System prompts ────────────────────────────────────────────────────────────

def _login_system_prompt() -> str:
    return """You are the authentication assistant for an Indian shopping platform.

YOUR ONLY JOB: Help the user log in OR register.

RULES:
1. If the user says "sign up", "register", "create account" AND provides email + password:
   → call register_user(email=..., password=..., full_name=...) immediately.

2. If the user provides both email AND password (ANY format, not mentioning registration):
   → call login_user(email=..., password=...) immediately.

3. AFTER calling either tool — DO NOT call any other tool. STOP.
   Just respond warmly. The system will handle the rest.

4. If credentials are missing → respond warmly and ask for them.

Login prompt format:
  "Of course! Please log in first:
   👉  your@email.com  YourPassword
   Or to create a new account: just say 'sign up with email@x.com Password123'"
"""


def _shopping_system_prompt(state: ShoppingState) -> str:
    products_block = _format_last_products_context(state.get("last_products") or [])
    uid = state["user_id"]
    nlp_note = (
        "\n  LAST SEARCH: Used semantic (NLP) search."
        if state.get("nlp_last_method") == "semantic"
        else ""
    )
    return f"""You are a warm, knowledgeable AI shopping assistant for an Indian e-commerce platform.

AUTHENTICATED SESSION:
  user_id = {uid}
  email   = {state['user_email']}

PRODUCTS SHOWN IN THIS CONVERSATION (context memory):
{products_block}{nlp_note}

Always pass user_id={uid} to ALL cart and order tools.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RULE 1 — CONTEXT MEMORY:
  When user says "this", "that", "it", "the first one", "add it":
  • Use product [1] from PRODUCTS SHOWN above (or the one named/numbered).
  • NEVER search again for a product already in the context list.

RULE 2 — SEMANTIC vs KEYWORD SEARCH:
  • Use semantic_search_products for INTENT-based or vague queries:
      "I'm thirsty", "carry my laptop", "rainy day", "gift for fitness lover"
  • Use search_products for EXPLICIT product queries:
      "water bottle", "laptop bag", "umbrella"
  • Use filter_products when user adds price/rating/category constraints.

RULE 3 — VAGUE QUERIES:
  If too vague ("something nice", "a gift", "blue thing"):
  • Call clarify_product_query. Ask ONE specific follow-up.
  • Do NOT search until user gives a clear keyword.

RULE 4 — REVIEWS AND SENTIMENT:
  When user asks "how are the reviews", "is it worth buying":
  • Call get_reviews — it includes VADER sentiment scores.
  • Present the sentiment_label and sentiment_summary to the user.

RULE 5 — INSTANT PURCHASE:
  "buy X now", "order X right now" → buy_now(product_name=..., user_id={uid}).

RULE 6 — ONE TOOL PER TURN:
  Call only the tools needed for THIS specific request.
  After getting tool results, STOP calling tools and format the response.

RULE 7 — STRICT PRODUCT DATA:
  ONLY show products from tool results — NEVER fabricate names, prices, ratings.

RULE 8 — CURRENCY:
  ALL prices as ₹<amount>. NEVER use $ or USD.

RULE 9 — CART & ORDERS:
  Pass user_id={uid} to all cart/order tools.
  After order → show Order ID and ₹ total.

RULE 10 — RESPONSE FORMAT:
  Warm, concise. Product list: name · ₹price · ⭐rating · one-line highlight.
"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def auth_gate_node(state: ShoppingState) -> dict:
    """Pure Python gate — zero LLM cost. Sets routing_decision only."""
    if state.get("is_logged_in") and state.get("user_id"):
        return {"routing_decision": "go_shopping"}
    last_human = next(
        (m for m in reversed(list(state["messages"])) if isinstance(m, HumanMessage)),
        None,
    )
    if last_human and (
        _is_login_attempt(last_human.content) or
        _is_registration_attempt(last_human.content)
    ):
        return {"routing_decision": "needs_login"}
    return {"routing_decision": "needs_login"}


_KEY_PROMPT_MSG = (
    "🔑 **API Key Required**\n\n"
    "Please enter your **Google Gemini API Key** in the sidebar on the left (or add `GOOGLE_API_KEY=...` to your `.env` file) to start chatting!\n\n"
    "👉 Get a free key instantly at **[aistudio.google.com](https://aistudio.google.com/)**."
)

def login_llm_node(state: ShoppingState) -> dict:
    """LLM with ONLY login_user + register_user bound."""
    llm = get_llm()
    if llm is None:
        return {"messages": [AIMessage(content=_KEY_PROMPT_MSG)]}
    bound_llm = llm.bind_tools(LOGIN_TOOLS)
    system_msg = SystemMessage(content=_login_system_prompt())
    messages   = [system_msg] + list(state["messages"])
    response   = bound_llm.invoke(messages)
    return {"messages": [_normalize_tool_calls(response)]}


def shopping_llm_node(state: ShoppingState) -> dict:
    """LLM with all shopping tools. login tools excluded."""
    llm = get_llm()
    if llm is None:
        return {"messages": [AIMessage(content=_KEY_PROMPT_MSG)]}
    bound_llm = llm.bind_tools(SHOPPING_TOOLS)
    system_msg = SystemMessage(content=_shopping_system_prompt(state))
    messages   = [system_msg] + list(state["messages"])
    response   = bound_llm.invoke(messages)
    return {"messages": [_normalize_tool_calls(response)]}


def tool_node_handler(state: ShoppingState) -> dict:
    """Execute tool calls. Preserve all state. Sync auth + last_products + nlp_last_method."""
    result = _tool_node.invoke(state)

    updated_state = {
        "user_id":          state.get("user_id"),
        "is_logged_in":     state.get("is_logged_in", False),
        "user_email":       state.get("user_email"),
        "routing_decision": state.get("routing_decision", ""),
        "last_products":    state.get("last_products") or [],
        "nlp_last_method":  state.get("nlp_last_method", ""),
        "messages":         result.get("messages", []),
    }

    new_messages = result.get("messages", [])

    # ── Sync login state ──────────────────────────────────────────────
    for msg in new_messages:
        if not isinstance(msg, ToolMessage) or msg.name not in ("login_user", "register_user"):
            continue
        try:
            data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            if isinstance(data, dict) and data.get("success"):
                updated_state["user_id"]      = data["user_id"]
                updated_state["is_logged_in"] = True
                updated_state["user_email"]   = data["email"]
        except (json.JSONDecodeError, TypeError):
            try:
                from db.db_client import execute_query
                last_ai = next(
                    (m for m in reversed(list(state["messages"]))
                     if not isinstance(m, (HumanMessage, ToolMessage))),
                    None,
                )
                if last_ai:
                    for tc in getattr(last_ai, "tool_calls", []):
                        if tc["name"] in ("login_user", "register_user"):
                            email = tc["args"].get("email", "").strip().lower()
                            rows  = execute_query(
                                "SELECT user_id, email FROM users WHERE email = %s",
                                (email,),
                            )
                            if rows:
                                updated_state["user_id"]      = rows[0]["user_id"]
                                updated_state["is_logged_in"] = True
                                updated_state["user_email"]   = rows[0]["email"]
            except Exception:
                pass

    # ── Update last_products ──────────────────────────────────────────
    fresh = _extract_products_from_messages(new_messages)
    if fresh:
        updated_state["last_products"] = fresh

    # ── Track NLP search method ───────────────────────────────────────
    for msg in new_messages:
        if isinstance(msg, ToolMessage) and msg.name == "semantic_search_products":
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                updated_state["nlp_last_method"] = data.get("method", "semantic")
            except Exception:
                updated_state["nlp_last_method"] = "semantic"

    return updated_state


# ── Routers ───────────────────────────────────────────────────────────────────

def route_from_gate(state: ShoppingState) -> str:
    return state.get("routing_decision", "needs_login")


def route_from_login_llm(state: ShoppingState) -> str:
    if _has_pending_tool_calls(state):
        return "tools"
    return END


def route_from_shopping_llm(state: ShoppingState) -> str:
    if _has_pending_tool_calls(state):
        return "tools"
    return END


def route_from_tools(state: ShoppingState) -> str:
    if state.get("is_logged_in") and state.get("user_id"):
        return "shopping_llm"
    return "login_llm"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(ShoppingState)

    graph.add_node("auth_gate",    auth_gate_node)
    graph.add_node("login_llm",    login_llm_node)
    graph.add_node("shopping_llm", shopping_llm_node)
    graph.add_node("tools",        tool_node_handler)

    graph.set_entry_point("auth_gate")

    graph.add_conditional_edges(
        "auth_gate", route_from_gate,
        {"needs_login": "login_llm", "go_shopping": "shopping_llm"},
    )
    graph.add_conditional_edges(
        "login_llm", route_from_login_llm,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "shopping_llm", route_from_shopping_llm,
        {"tools": "tools", END: END},
    )
    graph.add_conditional_edges(
        "tools", route_from_tools,
        {"shopping_llm": "shopping_llm", "login_llm": "login_llm"},
    )

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ── Public runner — blocking ───────────────────────────────────────────────────

def run_graph(user_message: str, session_state: dict):
    """
    Entry point called from Streamlit on every user message (blocking).
    Returns (response_text, updated_session_dict).
    """
    graph   = get_graph()
    history = session_state.get("history") or []

    initial_state = ShoppingState(
        messages         = history + [HumanMessage(content=user_message)],
        user_id          = session_state.get("user_id"),
        is_logged_in     = session_state.get("is_logged_in", False),
        user_email       = session_state.get("user_email"),
        routing_decision = "",
        last_products    = session_state.get("last_products") or [],
        nlp_last_method  = session_state.get("nlp_last_method", ""),
    )

    final_state = graph.invoke(
        initial_state,
        config={"recursion_limit": 25},
    )

    # ── Final text response ───────────────────────────────────────────────
    ai_messages = [
        m for m in final_state["messages"]
        if not isinstance(m, (HumanMessage, ToolMessage))
        and not getattr(m, "tool_calls", None)
    ]
    response = (
        ai_messages[-1].content
        if ai_messages
        else "Something went wrong. Please try again."
    )

    # ── Persist last_products for next turn ───────────────────────────────
    tool_msgs      = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    fresh_products = _extract_products_from_messages(tool_msgs)
    last_products  = fresh_products if fresh_products else (
        final_state.get("last_products") or []
    )

    # ── Clean history ─────────────────────────────────────────────────────
    clean_history = [
        m for m in final_state["messages"]
        if not isinstance(m, ToolMessage)
        and not getattr(m, "tool_calls", None)
    ]

    updated_session = {
        "user_id":         final_state.get("user_id"),
        "is_logged_in":    final_state.get("is_logged_in", False),
        "user_email":      final_state.get("user_email"),
        "history":         clean_history[-20:],
        "last_products":   last_products,
        "nlp_last_method": final_state.get("nlp_last_method", ""),
    }

    return response, updated_session


# ── Public runner — streaming ─────────────────────────────────────────────────

def run_graph_stream(user_message: str, session_state: dict) -> Generator[str, None, dict]:
    """
    Streaming entry point for Streamlit st.write_stream().
    Yields text tokens as they arrive from the LLM.
    After all tokens, sends the updated_session dict as the final yield.

    Usage in Streamlit:
        stream = run_graph_stream(user_input, session)
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
    """
    graph   = get_graph()
    history = session_state.get("history") or []

    initial_state = ShoppingState(
        messages         = history + [HumanMessage(content=user_message)],
        user_id          = session_state.get("user_id"),
        is_logged_in     = session_state.get("is_logged_in", False),
        user_email       = session_state.get("user_email"),
        routing_decision = "",
        last_products    = session_state.get("last_products") or [],
        nlp_last_method  = session_state.get("nlp_last_method", ""),
    )

    final_state = None
    accumulated  = ""

    for event in graph.stream(initial_state, config={"recursion_limit": 25}):
        for node_name, node_output in event.items():
            if node_name in ("login_llm", "shopping_llm"):
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                        chunk = msg.content or ""
                        if chunk and chunk != accumulated:
                            delta = chunk[len(accumulated):]
                            accumulated = chunk
                            if delta:
                                yield delta
        final_state = event.get(list(event.keys())[-1]) if event else None

    # Rebuild session from the last state snapshot available
    if final_state is None:
        final_state = {}

    messages_list = final_state.get("messages", []) or []
    tool_msgs      = [m for m in messages_list if isinstance(m, ToolMessage)]
    fresh_products = _extract_products_from_messages(tool_msgs)
    last_products  = fresh_products if fresh_products else (
        final_state.get("last_products") or []
    )
    clean_history = [
        m for m in messages_list
        if not isinstance(m, ToolMessage)
        and not getattr(m, "tool_calls", None)
    ]

    updated_session = {
        "user_id":         final_state.get("user_id"),
        "is_logged_in":    final_state.get("is_logged_in", False),
        "user_email":      final_state.get("user_email"),
        "history":         clean_history[-20:],
        "last_products":   last_products,
        "nlp_last_method": final_state.get("nlp_last_method", ""),
    }

    # Yield session as last item — caller unpacks it
    yield updated_session