# 🛒 AI Shopping Assistant — Local Setup Guide

> **Hardware Target**: AMD Ryzen AI 7 350 · Windows 11  
> **Stack**: LangGraph · CrewAI · Mistral AI · Streamlit · Flask · MongoDB · MySQL · Redis · sentence-transformers

---

## Prerequisites

Install these before anything else:

| Tool | Version | Download |
|------|---------|----------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| MySQL | 8.0+ | [dev.mysql.com/downloads](https://dev.mysql.com/downloads/) |
| MongoDB | 7.0+ | [mongodb.com/try/download](https://www.mongodb.com/try/download/community) |
| Redis | 7.x (via WSL or Memurai) | See [Windows Redis section](#redis-on-windows) below |
| Git | any | [git-scm.com](https://git-scm.com/) |

---

## Step 1 — Clone and enter the project

```powershell
cd C:\Users\mridu\Downloads\Agentic-AI-Shopping-Assistant-main\Agentic-AI-Shopping-Assistant-main
```

---

## Step 2 — Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If you see a PowerShell execution-policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Step 3 — Install PyTorch (CPU-only, AMD compatible)

Install this **before** `requirements.txt` so pip doesn't pull in a CUDA build:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> This installs ~800MB of CPU-optimised PyTorch. The AMD Ryzen AI 7 350 runs
> sentence-transformers efficiently on its 8 Zen 5 cores with this build.

---

## Step 4 — Install all dependencies

```powershell
pip install -r requirements.txt
```

> **If crewai install fails**: try `pip install crewai --no-deps` and install
> individual missing deps. crewai sometimes has platform-specific issues on Windows.

---

## Step 5 — Configure environment variables

```powershell
copy .env.template .env
```

Open `.env` in your editor and fill in:

```env
# REQUIRED — get from console.mistral.ai
MISTRAL_API_KEY=your_key_here

# REQUIRED — your MySQL root password
MYSQL_PASSWORD=your_mysql_password

# Optional (for order confirmation emails)
EMAIL_SENDER=your@gmail.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# AMD Ryzen AI 7 350 — optimal thread settings
OMP_NUM_THREADS=4
NLP_WORKER_PROCESSES=6
```

---

## Step 6 — Redis on Windows

Redis doesn't have an official Windows binary.  
**Option A** (recommended) — [Memurai](https://www.memurai.com/) (Redis-compatible for Windows):
```powershell
# After installing Memurai from their site:
# Redis will run as a Windows service automatically on port 6379
```

**Option B** — WSL2 + Redis:
```bash
# Inside WSL terminal:
sudo apt update && sudo apt install redis-server -y
sudo service redis-server start
```

Verify Redis is running:
```powershell
# PowerShell — if you have redis-cli:
redis-cli ping
# Should return: PONG
```

---

## Step 7 — Start MySQL and MongoDB

**MySQL** (should be running as a Windows service after installation):
```powershell
# Verify MySQL is accessible:
mysql -u root -p -e "SELECT 1;"
```

**MongoDB** (should be running as a Windows service):
```powershell
# Verify MongoDB is accessible:
mongosh --eval "db.runCommand({ connectionStatus: 1 })"
```

---

## Step 8 — Initialise databases

This creates all tables, seeds 60 products, and computes NLP embeddings:

```powershell
python scripts\init_db.py
```

**Expected output:**
```
[MySQL] Schema ready.
[MySQL] Test user created → email=demo@shopai.in  password=Demo@1234
[MongoDB] Indexes ensured.
[MongoDB] Seeded 60 products.
[NLP] Model loaded: all-MiniLM-L6-v2
[NLP] Encoding 60 products...   ████████ 100%
[NLP] Stored 60 embeddings in MongoDB.
Setup complete!
  Test credentials:
    Email   : demo@shopai.in
    Password: Demo@1234
```

> The NLP model (~80MB) is downloaded on first run from HuggingFace.
> On AMD Ryzen AI 7 350, encoding 60 products takes ~3–5 seconds.

---

## Step 9 — Run the application

### Option A: Single command launcher (recommended)
```powershell
python scripts\start_dev.py
```

This starts:
- **Flask API** on `http://localhost:5001`
- **Streamlit dashboard** on `http://localhost:8501`

### Option B: Start each manually (two terminal windows)

**Terminal 1 — Flask API:**
```powershell
.\venv\Scripts\Activate.ps1
python api.py
```

**Terminal 2 — Streamlit:**
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

---

## Step 10 — Open the app

Navigate to: **http://localhost:8501**

---

## Verification Commands

```powershell
# Test Flask API health
curl http://localhost:5001/api/health

# Test MongoDB connection
python -c "from db.mongo_client import ping; print('MongoDB:', ping())"

# Test Redis connection
python -c "from memory.redis_memory import ping; print('Redis:', ping())"

# Test MySQL connection
python -c "from db.db_client import execute_query; print('MySQL:', execute_query('SELECT 1'))"

# Test LangGraph
python -c "from graph import get_graph; get_graph(); print('LangGraph: OK')"

# Test NLP model
python -c "from tools.nlp_tools import _get_model; m = _get_model(); print('NLP model:', m)"
```

---

## AMD Ryzen AI 7 350 — Performance Tips

The `.env.template` already contains optimal settings for the Ryzen AI 7 350:

| Setting | Value | Reason |
|---------|-------|--------|
| `OMP_NUM_THREADS=4` | Half logical cores | Prevents over-subscription with Streamlit |
| `MKL_NUM_THREADS=4` | Same | PyTorch uses MKL on x86 |
| `NLP_WORKER_PROCESSES=6` | 6 of 8 cores | Leaves 2 cores for OS + Redis + MySQL |
| `TOKENIZERS_PARALLELISM=false` | Disabled | Avoids HuggingFace tokenizer fork warnings |

**For best NLP throughput** on long product catalogues (>1000 products):
```powershell
# Set before running — these tell NumPy/OpenBLAS to use all cores
$env:OMP_NUM_THREADS = "8"
$env:OPENBLAS_NUM_THREADS = "8"
python scripts\init_db.py
```

---

## Project Structure (after refactor)

```
├── app.py                  # Streamlit dashboard (dark-mode, product cards, analytics)
├── graph.py                # LangGraph graph — auth_gate, login_llm, shopping_llm, tools
├── crew.py                 # CrewAI — ProductSearch, Sentiment, Recommendation agents
├── api.py                  # Flask API — /recommendations, /sentiment, /trending, /analytics
│
├── tools/
│   ├── auth_tools.py       # login_user + register_user (bcrypt + MySQL)
│   ├── product_tools.py    # search, filter, details, reviews (MongoDB + VADER sentiment)
│   ├── cart_tools.py       # add, remove, view, place_order, buy_now (MongoDB cart)
│   └── nlp_tools.py        # semantic_search_products, analyze_sentiment (sentence-transformers)
│
├── db/
│   ├── db_client.py        # MySQL pool — users + orders (relational source of truth)
│   └── mongo_client.py     # MongoDB — products, carts, embeddings, analytics log
│
├── memory/
│   └── redis_memory.py     # Session + trending counters + NLP embedding cache
│
├── services/
│   └── email_service.py    # HTML order confirmation emails (Gmail SMTP, daemon thread)
│
├── scripts/
│   ├── init_db.py          # One-shot DB init + seed + NLP embedding computation
│   └── start_dev.py        # Single-command launcher (Flask + Streamlit)
│
├── requirements.txt        # All pinned dependencies
├── .env.template           # Environment variable template
└── SETUP.md                # This file
```

---

## Tech Stack Summary

| Layer | Technology | Role |
|-------|-----------|------|
| Agent framework | LangGraph 1.1 | Conversational state machine (auth → shopping → tools → END) |
| Multi-agent crew | CrewAI 0.80+ | NLP analytics via ProcessPoolExecutor |
| LLM | Mistral AI (mistral-large-latest) | Tool-calling, reasoning, formatting |
| UI | Streamlit 1.45 | Dark-mode dashboard with product cards |
| API backend | Flask 3.1 | Recommendations, sentiment, analytics endpoints |
| Product DB | MongoDB 7 | Flexible documents + pre-computed embeddings |
| User/Order DB | MySQL 8 | Relational integrity, atomic transactions |
| Session cache | Redis 7 | Auth sessions (24h), trending counters, NLP cache |
| NLP / Semantic | sentence-transformers | all-MiniLM-L6-v2, 384-dim, CPU/AMD |
| Sentiment | VADER + TextBlob | Review sentiment scoring |
| Auth | bcrypt 4 | Password hashing (cost factor 12) |
| Email | Gmail SMTP | Order confirmations (daemon thread, non-blocking) |

---

## Common Issues

**`ModuleNotFoundError: No module named 'crewai'`**
```powershell
pip install crewai crewai-tools
```

**`MongoServerError: Command failed`**
- Ensure MongoDB service is running: `Get-Service MongoDB` in PowerShell

**`redis.exceptions.ConnectionError`**
- Start Memurai or WSL Redis. Verify on port 6379.

**`mysql.connector.errors.DatabaseError`**
- Check `MYSQL_PASSWORD` in `.env`. Verify MySQL is running on port 3306.

**`huggingface_hub` download hanging**
- The first run downloads `all-MiniLM-L6-v2` (~80MB). Just wait.
- If it fails: `pip install --upgrade huggingface_hub`

**Streamlit `StreamlitAPIException` on Rerun**
- This is a known Streamlit 1.45 issue with `st.rerun()` in some edge cases.
- Solution: `pip install streamlit==1.45.1`

---

## License

MIT — free to use, fork, and adapt for your own portfolio.
