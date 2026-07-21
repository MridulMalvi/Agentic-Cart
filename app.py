"""
app.py — Streamlit AI Shopping Assistant Dashboard  (Refactored)

Premium dark-mode dashboard with:
  - Live chat with token-streaming
  - Real-time cart sidebar with animated total
  - Trending products panel (from Redis)
  - Analytics cards (from Flask API)
  - Semantic search badge (shows when NLP was used)
  - Product card grid with star ratings + stock badges
  - Dark/light mode CSS
  - Mobile-responsive layout

Architecture:
  - LangGraph graph handles all chat → agent → tool logic.
  - Flask API (port 5001) powers trending + analytics panels.
  - All heavy NLP runs off the main Streamlit thread (via Flask / worker processes).
"""

import streamlit as st
import requests
import time
from graph import run_graph
from db.mongo_client import get_cart

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopAI — AI Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Agentic AI Shopping Assistant · LangGraph + CrewAI + MongoDB · Built for India 🇮🇳"
    },
)

# ── Design system ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ─── Google Font ─── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ─── Root variables ─── */
  :root {
    --bg-primary:    #0f1117;
    --bg-card:       #1a1d27;
    --bg-card-hover: #1f2335;
    --accent:        #7c3aed;
    --accent-light:  #a78bfa;
    --accent-glow:   rgba(124,58,237,0.25);
    --success:       #10b981;
    --warning:       #f59e0b;
    --danger:        #ef4444;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --border:        rgba(255,255,255,0.08);
    --border-accent: rgba(124,58,237,0.4);
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
    --shadow:        0 4px 24px rgba(0,0,0,0.4);
  }

  /* ─── Global ─── */
  html, body, .stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text-primary) !important;
  }

  /* ─── Sidebar ─── */
  section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] .stMarkdown span {
    color: var(--text-secondary) !important;
  }

  /* ─── Chat bubbles ─── */
  .stChatMessage {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 10px !important;
    box-shadow: var(--shadow) !important;
    animation: slideUp 0.2s ease !important;
  }
  @keyframes slideUp {
    from { transform: translateY(8px); opacity: 0; }
    to   { transform: translateY(0);   opacity: 1; }
  }

  /* ─── Chat input ─── */
  .stChatInput textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stChatInput textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
  }

  /* ─── Buttons ─── */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent), #9333ea) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px var(--accent-glow) !important;
  }

  /* ─── Cards ─── */
  .product-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
    cursor: pointer;
  }
  .product-card:hover {
    border-color: var(--border-accent);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 28px var(--accent-glow);
  }

  /* ─── Cart items ─── */
  .cart-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 4px;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--text-secondary);
    animation: slideUp 0.15s ease;
  }
  .cart-total {
    margin-top: 12px;
    padding: 10px 12px;
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(16,185,129,0.1));
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-sm);
    font-weight: 700;
    font-size: 1rem;
    color: var(--success);
    text-align: center;
  }

  /* ─── Metric cards ─── */
  .metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px;
    text-align: center;
    transition: border-color 0.2s ease;
  }
  .metric-card:hover { border-color: var(--border-accent); }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-light), var(--success));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .metric-label {
    font-size: 0.78rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
  }

  /* ─── Badges ─── */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-light); border: 1px solid var(--border-accent); }
  .badge-green  { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
  .badge-amber  { background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }

  /* ─── Stars ─── */
  .stars { color: #fbbf24; font-size: 0.85rem; }

  /* ─── Section headers ─── */
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-secondary);
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }

  /* ─── Scrollbar ─── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 4px; }

  /* ─── Hide Streamlit branding ─── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
FLASK_BASE = f"http://{__import__('os').getenv('FLASK_HOST','localhost')}:{__import__('os').getenv('FLASK_PORT','5001')}"

_DEFAULTS = {
    "messages":          [],
    "user_id":           None,
    "is_logged_in":      False,
    "user_email":        None,
    "history":           [],
    "last_products":     [],
    "nlp_last_method":   "",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helper functions ───────────────────────────────────────────────────────────

def _stars(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def _fmt_price(price: float) -> str:
    return f"₹{price:,.0f}"


def _api_get(path: str, timeout: int = 3) -> dict | None:
    try:
        r = requests.get(f"{FLASK_BASE}{path}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def _api_post(path: str, body: dict, timeout: int = 8) -> dict | None:
    try:
        r = requests.post(f"{FLASK_BASE}{path}", json=body, timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None


def _render_product_card(p: dict, idx: int = 0) -> None:
    """Render a single product as a styled card."""
    title       = p.get("title", "Unknown Product")
    price       = float(p.get("price", 0))
    rating      = float(p.get("rating", 0))
    rating_count= p.get("rating_count", 0)
    stock       = p.get("stock", 0)
    category    = p.get("category", "")
    brand       = p.get("brand", "")

    stock_badge = (
        f'<span class="badge badge-green">✓ In Stock ({stock})</span>'
        if stock > 10
        else f'<span class="badge badge-amber">⚠ Low Stock ({stock})</span>'
        if stock > 0
        else '<span class="badge" style="background:rgba(239,68,68,0.15);color:#ef4444;">✕ Out of Stock</span>'
    )

    st.markdown(f"""
    <div class="product-card">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px;">
        <div style="flex:1;">
          <div style="font-weight:600;font-size:0.9rem;color:#f1f5f9;line-height:1.4;margin-bottom:4px;">{title}</div>
          <div style="font-size:0.75rem;color:#64748b;margin-bottom:6px;">{brand} · {category.replace('_',' ').title()}</div>
          <div class="stars">{_stars(rating)}</div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">{rating:.1f} · {rating_count:,} reviews</div>
        </div>
        <div style="text-align:right;min-width:90px;">
          <div style="font-size:1.2rem;font-weight:800;color:#a78bfa;">{_fmt_price(price)}</div>
          <div style="margin-top:6px;">{stock_badge}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / branding
    st.markdown("""
    <div style="padding:4px 0 16px;">
      <div style="font-size:1.4rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#34d399);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        🛒 ShopAI
      </div>
      <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.08em;text-transform:uppercase;">
        Agentic AI Shopping Assistant
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── API Key Input ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔑 Gemini API Key</div>', unsafe_allow_html=True)
    import os
    env_key = os.getenv("GOOGLE_API_KEY", "")
    is_placeholder = not env_key or "your_" in env_key.lower() or "here" in env_key.lower()
    
    saved_key = st.session_state.get("custom_api_key", "" if is_placeholder else env_key)
    api_key_input = st.text_input(
        "Enter API Key",
        value=saved_key,
        type="password",
        placeholder="AIzaSy...",
        label_visibility="collapsed",
        help="Get a free Google Gemini key at aistudio.google.com"
    )
    if api_key_input:
        st.session_state["custom_api_key"] = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        import graph
        graph._graph = None

    st.markdown("---")

    # ── Auth state ─────────────────────────────────────────────────────────
    if st.session_state["is_logged_in"]:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
                    border-radius:10px;padding:10px 14px;margin-bottom:12px;">
          <div style="font-size:0.75rem;color:#64748b;">Logged in as</div>
          <div style="font-weight:600;color:#34d399;font-size:0.9rem;margin-top:2px;">
            👤 {st.session_state['user_email']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            for _k, _v in _DEFAULTS.items():
                st.session_state[_k] = _v
            st.rerun()
    else:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                    border-radius:10px;padding:10px 14px;margin-bottom:12px;">
          <div style="font-size:0.8rem;color:#f59e0b;font-weight:600;">🔒 Not logged in</div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:4px;">
            Type your email + password in chat to sign in.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Live cart ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🛍️ Your Cart</div>', unsafe_allow_html=True)

    if st.session_state["is_logged_in"] and st.session_state["user_id"]:
        try:
            cart = get_cart(st.session_state["user_id"])
        except Exception:
            cart = []

        if cart:
            for item in cart:
                title = item["title"][:34] + "…" if len(item["title"]) > 34 else item["title"]
                st.markdown(f"""
                <div class="cart-item">
                  <span>📦 {title}</span>
                  <span style="color:#a78bfa;font-weight:600;">{_fmt_price(item['price'])}</span>
                </div>
                """, unsafe_allow_html=True)
            total = round(sum(i["price"] for i in cart), 2)
            count = len(cart)
            st.markdown(f"""
            <div class="cart-total">
              💰 {_fmt_price(total)}
              <span style="font-size:0.75rem;font-weight:400;color:#94a3b8;margin-left:6px;">
                {count} item{'s' if count != 1 else ''}
              </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#475569;font-size:0.8rem;text-align:center;padding:16px 0;">Cart is empty</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:0.8rem;text-align:center;padding:12px 0;">Login to see your cart</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Try asking ─────────────────────────────────────────────────────────
    with st.expander("💡 What can I ask?", expanded=False):
        st.markdown("""
<div style="font-size:0.78rem;color:#94a3b8;line-height:1.9;">

**🔍 Search (NLP-powered)**
- *I'm thirsty*
- *Something for a rainy day*
- *Gift for a fitness lover*

**🔧 Filtered Search**
- *Laptops under ₹40,000 with 4+ stars*
- *Wireless earbuds under ₹1,500*

**📊 Reviews & Sentiment**
- *How are the reviews for yoga mats?*
- *Is the boAt earbuds worth buying?*

**🛒 Cart & Orders**
- *Add the first one to my cart*
- *Place order for laptop bag*
- *Buy yoga mat right now* ⚡

**🔐 Account**
- *demo@shopai.in  Demo@1234*
- *Sign up with me@email.com MyPass123*
</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Analytics panel (from Flask API) ──────────────────────────────────
    st.markdown('<div class="section-header">📊 Live Analytics</div>', unsafe_allow_html=True)

    api_data = _api_get("/api/analytics/overview", timeout=2)
    if api_data and "error" not in api_data:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{api_data.get('total_orders', 0)}</div>
              <div class="metric-label">Orders</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            revenue = api_data.get('total_revenue', 0)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">₹{revenue/1000:.0f}K</div>
              <div class="metric-label">Revenue</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:0.75rem;text-align:center;">Analytics unavailable<br/><span style="font-size:0.65rem;">Start Flask API to enable</span></div>', unsafe_allow_html=True)


# ── Main content area ─────────────────────────────────────────────────────────
col_chat, col_panel = st.columns([3, 1], gap="large")

with col_chat:
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:20px;">
      <h1 style="margin:0;font-size:1.8rem;font-weight:800;
                 background:linear-gradient(135deg,#a78bfa 0%,#34d399 100%);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        AI Shopping Assistant
      </h1>
      <p style="margin:4px 0 0;font-size:0.8rem;color:#475569;">
        Powered by <strong style="color:#7c3aed;">Mistral AI</strong> ·
        <strong style="color:#6366f1;">LangGraph</strong> ·
        <strong style="color:#10b981;">CrewAI</strong> ·
        <strong style="color:#f59e0b;">MongoDB</strong> ·
        🇮🇳 Built for India
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── NLP badge ─────────────────────────────────────────────────────────
    if st.session_state.get("nlp_last_method") == "semantic":
        st.markdown("""
        <div style="display:inline-flex;align-items:center;gap:6px;margin-bottom:12px;
                    background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);
                    border-radius:999px;padding:4px 14px;font-size:0.72rem;color:#a78bfa;">
          🧠 Last search used <strong>semantic NLP</strong> (sentence-transformers)
        </div>
        """, unsafe_allow_html=True)

    # ── Chat history ───────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask me anything — search, cart, orders, reviews…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🧠 Thinking…"):
                try:
                    session = {
                        "user_id":         st.session_state["user_id"],
                        "is_logged_in":    st.session_state["is_logged_in"],
                        "user_email":      st.session_state["user_email"],
                        "history":         st.session_state["history"],
                        "last_products":   st.session_state["last_products"],
                        "nlp_last_method": st.session_state["nlp_last_method"],
                    }

                    response, updated_session = run_graph(user_input, session)

                    st.session_state["user_id"]         = updated_session.get("user_id")
                    st.session_state["is_logged_in"]    = updated_session.get("is_logged_in", False)
                    st.session_state["user_email"]      = updated_session.get("user_email")
                    st.session_state["history"]         = updated_session.get("history", [])
                    st.session_state["last_products"]   = updated_session.get("last_products", [])
                    st.session_state["nlp_last_method"] = updated_session.get("nlp_last_method", "")

                except Exception as exc:
                    err_str = str(exc)
                    if any(k in err_str.lower() for k in ["bearer", "illegal header", "401", "unauthenticated", "invalid_api_key", "api_key"]):
                        response = (
                            "🔑 **API Key Required or Invalid**\n\n"
                            "Please enter a valid **Google Gemini API Key** in the sidebar on the left (or add `GOOGLE_API_KEY=...` to your `.env` file)!\n\n"
                            "👉 Get a free key instantly at **[aistudio.google.com](https://aistudio.google.com/)**."
                        )
                    else:
                        response = f"⚠️ Something went wrong: `{err_str}`"

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


# ── Right panel ───────────────────────────────────────────────────────────────
with col_panel:

    # ── Last shown products ────────────────────────────────────────────────
    last_products = st.session_state.get("last_products", [])
    if last_products:
        st.markdown('<div class="section-header">🔍 Products Found</div>', unsafe_allow_html=True)
        for i, p in enumerate(last_products[:4]):
            _render_product_card(p, i)

    # ── Trending products (from Flask API) ─────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔥 Trending Now</div>', unsafe_allow_html=True)

    trending_data = _api_get("/api/trending?top_n=4", timeout=2)
    if trending_data and trending_data.get("trending"):
        for p in trending_data["trending"]:
            _render_product_card(p)
    else:
        # Fallback: show static popular products from MongoDB
        try:
            from db.mongo_client import get_popular_products
            popular = get_popular_products(4)
            for p in popular:
                p.pop("_id", None)
                p.pop("embedding", None)
                _render_product_card(p)
        except Exception:
            st.markdown('<div style="color:#475569;font-size:0.78rem;text-align:center;padding:16px 0;">No trending data yet</div>', unsafe_allow_html=True)

    # ── Category breakdown ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📦 Catalogue</div>', unsafe_allow_html=True)

    cat_data = _api_get("/api/analytics/categories", timeout=2)
    if cat_data and cat_data.get("categories"):
        for cat in cat_data["categories"][:6]:
            name  = cat["category"].replace("_", " ").title()
            count = cat["count"]
            avg_r = cat["avg_rating"]
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
              <span style="font-size:0.78rem;color:#94a3b8;">{name}</span>
              <span style="font-size:0.72rem;color:#64748b;">{count} items · ⭐{avg_r:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Fallback: MongoDB direct query
        try:
            from db.mongo_client import get_distinct_categories
            cats = get_distinct_categories()[:6]
            for cat in cats:
                st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;padding:4px 0;">{cat.replace("_"," ").title()}</div>', unsafe_allow_html=True)
        except Exception:
            pass