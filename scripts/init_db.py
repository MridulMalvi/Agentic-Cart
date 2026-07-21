"""
scripts/init_db.py

One-shot database initialisation script.

What it does (in order):
  1. MySQL: create database + tables (users, products, orders)
  2. MongoDB: ensure indexes on all collections
  3. Seed 60 sample Indian e-commerce products into MongoDB
  4. Create a test user with bcrypt-hashed password in MySQL
  5. Compute sentence-transformer embeddings for all products → store in MongoDB

Run once before starting the application:
    python scripts/init_db.py

Safe to re-run: MySQL uses IF NOT EXISTS; MongoDB upserts are idempotent.
"""

import os
import sys
import json
import bcrypt
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import mysql.connector
from db.mongo_client import (
    get_db, ensure_indexes, upsert_product, upsert_product_vector,
    get_db as mongo_db,
)


# ── MySQL setup ───────────────────────────────────────────────────────────────

MYSQL_DDL = """
CREATE DATABASE IF NOT EXISTS {db};
USE {db};

CREATE TABLE IF NOT EXISTS users (
    user_id       INT PRIMARY KEY AUTO_INCREMENT,
    email         VARCHAR(255) UNIQUE NOT NULL,
    full_name     VARCHAR(255),
    password_hash VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    order_id     VARCHAR(20) PRIMARY KEY,
    user_id      INT NOT NULL,
    total_amount DECIMAL(10,2),
    items_json   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
"""


def init_mysql():
    print("\n[DB] Initialising relational schema...")
    from db.db_client import execute_query
    # Run user and order creation DDL via db_client execute_query
    execute_query("""
    CREATE TABLE IF NOT EXISTS users (
        user_id       INTEGER PRIMARY KEY AUTO_INCREMENT,
        email         VARCHAR(255) UNIQUE NOT NULL,
        full_name     VARCHAR(255),
        password_hash VARCHAR(255) NOT NULL,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """, fetch=False)
    execute_query("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id     VARCHAR(20) PRIMARY KEY,
        user_id      INT NOT NULL,
        total_amount DECIMAL(10,2),
        items_json   TEXT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """, fetch=False)
    print("[DB] Relational schema ready.")


def create_test_user():
    """Create a demo user (idempotent — skips if email already exists)."""
    print("[MySQL] Creating test user...")
    from db.db_client import execute_query

    email     = "demo@shopai.in"
    password  = "Demo@1234"
    full_name = "Demo User"

    existing = execute_query("SELECT user_id FROM users WHERE email = %s", (email,))
    if existing:
        print(f"[MySQL] Test user already exists: {email}")
        return

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    execute_query(
        "INSERT INTO users (email, full_name, password_hash) VALUES (%s, %s, %s)",
        (email, full_name, pw_hash),
        fetch=False,
    )
    print(f"[MySQL] Test user created -> email={email}  password={password}")


# ── Product seed data ─────────────────────────────────────────────────────────

PRODUCTS = [
    # ── Electronics ───────────────────────────────────────────────────────────
    {"product_id": 1,  "title": "boAt Airdopes 141 True Wireless Earbuds",       "category": "electronics", "brand": "boAt",        "price": 999.0,   "rating": 4.1, "rating_count": 185420, "stock": 245, "description": "True wireless earbuds with 42 hours total playback, IPX4 water resistance, and touch controls. Ideal for workouts and daily commute."},
    {"product_id": 2,  "title": "Samsung Galaxy Buds2 Pro Wireless Earphones",   "category": "electronics", "brand": "Samsung",     "price": 9999.0,  "rating": 4.4, "rating_count": 22100,  "stock": 89,  "description": "Premium ANC earbuds with 360 Audio, Hi-Fi sound, and 29-hour battery. Seamlessly pairs with Galaxy devices."},
    {"product_id": 3,  "title": "Redmi 13C Smartphone 4GB RAM 128GB",             "category": "electronics", "brand": "Redmi",       "price": 8999.0,  "rating": 4.2, "rating_count": 67340,  "stock": 310, "description": "6.74-inch display, 50MP triple camera, 5000mAh battery. Powered by MediaTek Helio G85."},
    {"product_id": 4,  "title": "Lenovo IdeaPad Slim 3 Laptop 15.6 inch",        "category": "electronics", "brand": "Lenovo",      "price": 38990.0, "rating": 4.3, "rating_count": 14520,  "stock": 55,  "description": "AMD Ryzen 5 7520U, 8GB RAM, 512GB SSD, Windows 11 Home. Thin, light, and built for productivity."},
    {"product_id": 5,  "title": "HP 15s Ryzen 5 Laptop 8GB 512GB SSD",           "category": "electronics", "brand": "HP",          "price": 42490.0, "rating": 4.4, "rating_count": 9870,   "stock": 40,  "description": "15.6 inch FHD display, AMD Ryzen 5 5500U, integrated Radeon graphics, dual speakers. Backlit keyboard."},
    {"product_id": 6,  "title": "TP-Link Archer C6 AC1200 Wi-Fi Router",         "category": "electronics", "brand": "TP-Link",     "price": 1799.0,  "rating": 4.3, "rating_count": 78200,  "stock": 185, "description": "Dual-band 1200Mbps router with 4 antennas, MU-MIMO, and easy setup. Covers up to 2000 sq ft."},
    {"product_id": 7,  "title": "Portronics Kronos Y2 Smart Watch",               "category": "electronics", "brand": "Portronics",  "price": 1299.0,  "rating": 3.9, "rating_count": 31200,  "stock": 200, "description": "1.69-inch HD display smartwatch, health tracking, 7-day battery, SpO2 monitor, multiple sports modes."},
    {"product_id": 8,  "title": "JBL Flip 6 Portable Bluetooth Speaker",         "category": "electronics", "brand": "JBL",         "price": 7499.0,  "rating": 4.6, "rating_count": 44100,  "stock": 72,  "description": "IP67 waterproof speaker with 12 hours playtime, bold JBL Original Pro Sound, and PartyBoost."},
    {"product_id": 9,  "title": "Anker 65W GaN USB-C Charger",                   "category": "electronics", "brand": "Anker",       "price": 2499.0,  "rating": 4.5, "rating_count": 19300,  "stock": 155, "description": "Compact GaN charger with 2 USB-C + 1 USB-A ports. Charges MacBook, iPhone, and Android fast."},
    {"product_id": 10, "title": "Amazon Fire TV Stick 4K Max",                    "category": "electronics", "brand": "Amazon",      "price": 5999.0,  "rating": 4.5, "rating_count": 88900,  "stock": 130, "description": "Streaming with Wi-Fi 6E, Ambient Experience, Alexa Voice Remote. Supports 4K Ultra HD, HDR10+."},

    # ── Home & Kitchen ─────────────────────────────────────────────────────────
    {"product_id": 11, "title": "Milton Thermosteel Flip Lid Water Bottle 1L",   "category": "home",        "brand": "Milton",      "price": 449.0,   "rating": 4.3, "rating_count": 92400,  "stock": 520, "description": "Double-walled stainless steel bottle keeps beverages hot for 24 hours and cold for 48 hours. Leak-proof flip lid."},
    {"product_id": 12, "title": "Cello H2O Stainless Steel Water Bottle 750ml",  "category": "home",        "brand": "Cello",       "price": 299.0,   "rating": 4.1, "rating_count": 34100,  "stock": 480, "description": "Food-grade stainless steel, BPA-free, wide mouth, easy to clean. Perfect for school, office, and gym."},
    {"product_id": 13, "title": "Pigeon by Stovekraft Amaze Plus Electric Kettle","category": "home",       "brand": "Pigeon",      "price": 649.0,   "rating": 4.4, "rating_count": 108000, "stock": 290, "description": "1.5L, 1500W, boil-dry protection, food-grade stainless steel interior. Boils 1L in under 4 minutes."},
    {"product_id": 14, "title": "Prestige PKGSS 1.0 L Electric Kettle",          "category": "home",        "brand": "Prestige",    "price": 799.0,   "rating": 4.2, "rating_count": 51200,  "stock": 210, "description": "Stainless steel body, 360-degree rotation base, cool-touch handle, auto-shutoff and boil-dry protection."},
    {"product_id": 15, "title": "Nilkamal Plastic Storage Box Set of 3",          "category": "home",        "brand": "Nilkamal",    "price": 549.0,   "rating": 4.0, "rating_count": 22600,  "stock": 350, "description": "Stackable, lightweight, airtight containers. Ideal for dry foods, toys, stationery. BPA-free plastic."},
    {"product_id": 16, "title": "Solimo Microfibre Cleaning Cloth Pack of 6",     "category": "home",        "brand": "Solimo",      "price": 299.0,   "rating": 4.2, "rating_count": 47800,  "stock": 600, "description": "Super-absorbent, lint-free microfibre cloths. Safe for glass, chrome, and stainless steel surfaces."},
    {"product_id": 17, "title": "Usha Instafresh 35L Mini Refrigerator",         "category": "home",        "brand": "Usha",        "price": 7999.0,  "rating": 3.8, "rating_count": 5600,   "stock": 28,  "description": "Compact personal refrigerator, direct cool, energy efficient, ideal for dorms and small offices."},
    {"product_id": 18, "title": "Wonderchef Nutri-Blend 400W Mixer Grinder",     "category": "home",        "brand": "Wonderchef",  "price": 1999.0,  "rating": 4.4, "rating_count": 63500,  "stock": 145, "description": "400W motor, 3 stainless steel jars, 22000 RPM, stainless steel blades. Comes with spatula and recipe book."},

    # ── Bags & Travel ──────────────────────────────────────────────────────────
    {"product_id": 19, "title": "Safari 67cm Trolley Suitcase Hard Case",        "category": "bags",        "brand": "Safari",      "price": 3299.0,  "rating": 4.2, "rating_count": 14800,  "stock": 62,  "description": "ABS hardshell trolley with 360° spinner wheels, TSA-approved lock, expandable design. 67 litres."},
    {"product_id": 20, "title": "American Tourister Linex 55cm Cabin Bag",       "category": "bags",        "brand": "American Tourister", "price": 3999.0, "rating": 4.4, "rating_count": 21200, "stock": 48, "description": "Polycarbonate cabin size bag, TSA lock, 4 spinner wheels, USB loop access. 37.5 litres."},
    {"product_id": 21, "title": "Wildcraft Daypack 30L Laptop Backpack",         "category": "bags",        "brand": "Wildcraft",   "price": 1799.0,  "rating": 4.1, "rating_count": 18400,  "stock": 125, "description": "Dedicated 15.6-inch laptop compartment, rain cover, ergonomic padded straps, reflective strip for safety."},
    {"product_id": 22, "title": "Tommy Hilfiger 25L Canvas Backpack",            "category": "bags",        "brand": "Tommy Hilfiger", "price": 4599.0, "rating": 4.5, "rating_count": 7300, "stock": 55, "description": "Classic canvas backpack with signature branding, laptop sleeve, multiple organiser pockets. Premium quality."},
    {"product_id": 23, "title": "FabSeasons Transparent Rain Cover Backpack",    "category": "bags",        "brand": "FabSeasons",  "price": 249.0,   "rating": 4.0, "rating_count": 29100,  "stock": 800, "description": "Waterproof transparent PVC rain cover fits bags up to 30L. Lightweight and foldable."},
    {"product_id": 24, "title": "Gear Stratosphere 45L Trekking Rucksack",       "category": "bags",        "brand": "Gear",        "price": 2299.0,  "rating": 4.2, "rating_count": 9800,   "stock": 70,  "description": "45L capacity rucksack with hip belt, rain cover, hydration port, and chest strap. For trekking and hiking."},

    # ── Sports & Fitness ──────────────────────────────────────────────────────
    {"product_id": 25, "title": "Cockatoo CFR-01 Folding Fitness Cycle",         "category": "sports",      "brand": "Cockatoo",    "price": 5999.0,  "rating": 4.0, "rating_count": 12400,  "stock": 35,  "description": "8-level resistance, 16-inch alloy cranks, LCD display, padded seat. Folds flat for storage."},
    {"product_id": 26, "title": "Boldfit Yoga Mat 6mm Anti-Slip",                "category": "sports",      "brand": "Boldfit",     "price": 499.0,   "rating": 4.3, "rating_count": 54200,  "stock": 430, "description": "6mm thick TPE yoga mat with carrying strap. Non-slip, sweat-resistant, eco-friendly material."},
    {"product_id": 27, "title": "Strauss Rubber Resistance Bands Set of 5",      "category": "sports",      "brand": "Strauss",     "price": 399.0,   "rating": 4.4, "rating_count": 38700,  "stock": 580, "description": "5 resistance levels (extra light to extra heavy). For stretching, rehab, pilates, and home workouts."},
    {"product_id": 28, "title": "Nivia Storm Football Size 5",                   "category": "sports",      "brand": "Nivia",       "price": 499.0,   "rating": 4.1, "rating_count": 19800,  "stock": 200, "description": "32-panel machine stitched football, PVC outer. Good for training and recreational play."},
    {"product_id": 29, "title": "Cullmann Ultralight Tripod for Camera/Phone",   "category": "sports",      "brand": "Cullmann",    "price": 1299.0,  "rating": 4.2, "rating_count": 7300,   "stock": 90,  "description": "1.3m lightweight aluminum tripod with ball head, universal phone mount, carry bag included."},
    {"product_id": 30, "title": "MuscleBlaze Whey Protein 2kg Chocolate",       "category": "sports",      "brand": "MuscleBlaze", "price": 2999.0,  "rating": 4.5, "rating_count": 71400,  "stock": 175, "description": "25g protein per serving, 5.5g BCAAs, 3rd party INFORMED-SPORT certified. Chocolate flavour."},

    # ── Clothing & Fashion ─────────────────────────────────────────────────────
    {"product_id": 31, "title": "Levi's Men's 511 Slim Fit Jeans",              "category": "clothing",    "brand": "Levi's",      "price": 2499.0,  "rating": 4.4, "rating_count": 33100,  "stock": 145, "description": "Slim fit jeans, mid-rise waistband, stretch denim for comfort. Classic 5-pocket style."},
    {"product_id": 32, "title": "H&M Women's Floral Midi Dress",                "category": "clothing",    "brand": "H&M",         "price": 1299.0,  "rating": 4.2, "rating_count": 18700,  "stock": 110, "description": "Floral print, v-neck, flutter sleeves, midi length. Light, flowy fabric perfect for summer."},
    {"product_id": 33, "title": "Van Heusen Men's Formal Shirt",                "category": "clothing",    "brand": "Van Heusen",  "price": 999.0,   "rating": 4.3, "rating_count": 41200,  "stock": 200, "description": "Regular fit formal shirt in pure cotton. Available in multiple solid colours. Iron-free finish."},
    {"product_id": 34, "title": "Puma Men's Running T-Shirt Dri-Cell",          "category": "clothing",    "brand": "Puma",        "price": 699.0,   "rating": 4.3, "rating_count": 22600,  "stock": 265, "description": "Dri-CELL moisture-wicking technology, flatlock seams, reflective logo. Lightweight for running."},
    {"product_id": 35, "title": "Bata Men's Casual Slip-On Shoes",              "category": "clothing",    "brand": "Bata",        "price": 1299.0,  "rating": 4.1, "rating_count": 15400,  "stock": 180, "description": "Casual canvas slip-on with flexible rubber sole, breathable fabric lining, and cushioned insole."},
    {"product_id": 36, "title": "Reebok Classic Leather Sneakers Women",        "category": "clothing",    "brand": "Reebok",      "price": 4499.0,  "rating": 4.5, "rating_count": 9700,   "stock": 65,  "description": "Iconic Reebok Classic Leather in updated colourways. Soft leather upper, die-cut EVA midsole."},
    {"product_id": 37, "title": "Jockey Men's Briefs Pack of 3",                "category": "clothing",    "brand": "Jockey",      "price": 399.0,   "rating": 4.4, "rating_count": 89300,  "stock": 500, "description": "Super combed cotton briefs. Elasticated waistband, moisture-wicking, tagless comfort."},
    {"product_id": 38, "title": "Fabindia Women's Kurta Cotton Printed",        "category": "clothing",    "brand": "Fabindia",    "price": 1599.0,  "rating": 4.3, "rating_count": 12100,  "stock": 90,  "description": "Cotton kurta with block print, straight fit, side slits. Handcrafted by Indian artisans."},

    # ── Books & Stationery ─────────────────────────────────────────────────────
    {"product_id": 39, "title": "Atomic Habits by James Clear",                  "category": "books",       "brand": "Penguin",     "price": 399.0,   "rating": 4.7, "rating_count": 134200, "stock": 900, "description": "The #1 bestselling guide to building good habits and breaking bad ones. A proven framework for self-improvement."},
    {"product_id": 40, "title": "The Psychology of Money by Morgan Housel",      "category": "books",       "brand": "Jaico",       "price": 299.0,   "rating": 4.7, "rating_count": 98400,  "stock": 850, "description": "Timeless lessons on wealth, greed, and happiness. One of the best books on personal finance."},
    {"product_id": 41, "title": "DOMS Super Dark Pencil Set of 10",              "category": "stationery",  "brand": "DOMS",        "price": 99.0,    "rating": 4.5, "rating_count": 62100,  "stock": 1200, "description": "Super dark, smooth writing HB pencils. Break-resistant leads, hexagonal barrel, ideal for sketching."},
    {"product_id": 42, "title": "Classmate 6 Subject Spiral Notebook A4",       "category": "stationery",  "brand": "Classmate",   "price": 159.0,   "rating": 4.3, "rating_count": 47200,  "stock": 950, "description": "A4 size, 300 pages, 6-subject dividers, micro-perforated sheets, hard cover."},

    # ── Personal Care & Beauty ─────────────────────────────────────────────────
    {"product_id": 43, "title": "Himalaya Neem Face Wash 150ml",                "category": "personal_care","brand": "Himalaya",    "price": 149.0,   "rating": 4.4, "rating_count": 148300, "stock": 700, "description": "Neem and turmeric purifying face wash. Controls acne, removes excess oil. Dermatologically tested."},
    {"product_id": 44, "title": "Biotique Bio Coconut Whitening Cream 50g",     "category": "personal_care","brand": "Biotique",    "price": 199.0,   "rating": 4.1, "rating_count": 31200,  "stock": 400, "description": "Natural whitening cream with coconut, dandelion, and lodhra. Reduces dark spots, moisturises skin."},
    {"product_id": 45, "title": "Mamaearth Onion Hair Oil 250ml",               "category": "personal_care","brand": "Mamaearth",   "price": 349.0,   "rating": 4.3, "rating_count": 79800,  "stock": 330, "description": "Onion and redensyl hair oil. Reduces hair fall, promotes regrowth, nourishes scalp. Toxin-free."},
    {"product_id": 46, "title": "Neutrogena Ultra Sheer Sunscreen SPF 50+",    "category": "personal_care","brand": "Neutrogena",  "price": 499.0,   "rating": 4.5, "rating_count": 55100,  "stock": 280, "description": "Lightweight SPF 50+ broad-spectrum sunscreen. Non-greasy, water-resistant for 80 minutes. Dermatologist recommended."},
    {"product_id": 47, "title": "Park Avenue Beer Shampoo for Men 650ml",      "category": "personal_care","brand": "Park Avenue", "price": 299.0,   "rating": 4.2, "rating_count": 19700,  "stock": 360, "description": "Beer-enriched shampoo that strengthens hair from roots, adds volume and shine. No parabens."},

    # ── Outdoor & Rain ─────────────────────────────────────────────────────────
    {"product_id": 48, "title": "Lal Haveli Umbrella Windproof Double Layer",   "category": "outdoor",     "brand": "Lal Haveli",  "price": 599.0,   "rating": 4.0, "rating_count": 12300,  "stock": 185, "description": "Double-layer windproof umbrella, 8-rib construction, UV protection, auto open/close button."},
    {"product_id": 49, "title": "Columbia Watertight II Rain Jacket",           "category": "outdoor",     "brand": "Columbia",    "price": 5999.0,  "rating": 4.5, "rating_count": 8700,   "stock": 40,  "description": "Packable, waterproof, seam-sealed rain jacket. Adjustable hood, zippered pockets, available in 6 colours."},
    {"product_id": 50, "title": "Wildcraft Raincoat Poncho Unisex",             "category": "outdoor",     "brand": "Wildcraft",   "price": 799.0,   "rating": 4.1, "rating_count": 19200,  "stock": 220, "description": "Lightweight PVC poncho with hoodie, covers rider + backpack. One size fits most. Tear-resistant."},

    # ── Food & Beverages ──────────────────────────────────────────────────────
    {"product_id": 51, "title": "Tata Tea Gold 500g",                           "category": "food",        "brand": "Tata Tea",    "price": 249.0,   "rating": 4.5, "rating_count": 62400,  "stock": 850, "description": "Blend of long and short leaf tea, with live leaf tea granules. Bold, refreshing taste."},
    {"product_id": 52, "title": "Nescafe Classic Instant Coffee 100g",          "category": "food",        "brand": "Nescafe",     "price": 299.0,   "rating": 4.6, "rating_count": 91300,  "stock": 720, "description": "100% coffee, rich aroma, smooth taste. Made from high-quality robusta beans. Dissolves instantly."},
    {"product_id": 53, "title": "Bournvita Health Drink Chocolate 500g",        "category": "food",        "brand": "Cadbury",     "price": 299.0,   "rating": 4.4, "rating_count": 48700,  "stock": 650, "description": "Vitamin D & calcium fortified drink mix. 5 power nutrients, classic chocolate flavour for kids and adults."},
    {"product_id": 54, "title": "Pringles Original Potato Crisps 134g",        "category": "food",        "brand": "Pringles",    "price": 199.0,   "rating": 4.5, "rating_count": 38100,  "stock": 500, "description": "Crispy, stackable potato crisps in iconic canister. The original seasoning, satisfyingly crunchy."},

    # ── Gift & Miscellaneous ──────────────────────────────────────────────────
    {"product_id": 55, "title": "Archies Happy Birthday Gift Set",              "category": "gifts",       "brand": "Archies",     "price": 599.0,   "rating": 4.2, "rating_count": 8100,   "stock": 120, "description": "Birthday gift hamper with greeting card, showpiece, and chocolates. Beautifully gift-wrapped."},
    {"product_id": 56, "title": "Craftsvilla Handmade Macrame Wall Hanging",   "category": "gifts",       "brand": "Craftsvilla", "price": 799.0,   "rating": 4.4, "rating_count": 5400,   "stock": 75,  "description": "Boho-style handwoven macrame wall art, 24 inches, natural cotton cord. Unique home decor gift."},
    {"product_id": 57, "title": "Just Herbs Rosehip Face Serum 30ml",          "category": "gifts",       "brand": "Just Herbs",  "price": 699.0,   "rating": 4.3, "rating_count": 9200,   "stock": 88,  "description": "Rosehip, niacinamide, and hyaluronic acid serum. Reduces pigmentation, brightens, and hydrates skin."},
    {"product_id": 58, "title": "Fitbit Inspire 3 Health Fitness Tracker",    "category": "gifts",       "brand": "Fitbit",      "price": 8999.0,  "rating": 4.4, "rating_count": 14300,  "stock": 45,  "description": "10-day battery, 24/7 heart rate, SpO2, stress management. 40+ exercise modes. Slim, stylish design."},
    {"product_id": 59, "title": "Lego Creator 3-in-1 Mini Police Car",         "category": "gifts",       "brand": "Lego",        "price": 1299.0,  "rating": 4.7, "rating_count": 3200,   "stock": 55,  "description": "3-in-1 buildable toy. Build a police car, helicopter, or boat. 218 pieces. Ages 6+."},
    {"product_id": 60, "title": "Fujifilm Instax Mini 11 Instant Camera",      "category": "gifts",       "brand": "Fujifilm",    "price": 5999.0,  "rating": 4.6, "rating_count": 24100,  "stock": 38,  "description": "Instant print camera with automatic exposure, selfie mode, and flash. Credit-card-sized prints."},
]


# ── Embedding computation ─────────────────────────────────────────────────────

def compute_and_store_embeddings():
    """
    Compute sentence-transformer embeddings for all products and store in MongoDB.
    Uses all-MiniLM-L6-v2 (384-dim, runs on CPU, ~80MB download on first call).
    """
    print("\n[NLP] Loading sentence-transformer model (first run downloads ~80MB)...")
    try:
        from sentence_transformers import SentenceTransformer
        import os
        model_name = os.getenv("NLP_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)

        texts = [
            f"{p['title']}. {p.get('description', '')} Category: {p['category']}. Brand: {p['brand']}."
            for p in PRODUCTS
        ]
        print(f"[NLP] Encoding {len(texts)} products...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

        for product, embedding in zip(PRODUCTS, embeddings):
            upsert_product_vector(
                product_id=product["product_id"],
                title=product["title"],
                embedding=embedding.tolist(),
            )
        print(f"[NLP] Stored {len(PRODUCTS)} embeddings in MongoDB.")
    except ImportError:
        print("[NLP] WARNING: sentence-transformers not installed. Skipping embeddings.")
        print("       Run: pip install sentence-transformers")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Shopping Assistant — Database Initialiser")
    print("=" * 60)

    # 1. MySQL
    init_mysql()
    create_test_user()

    # 2. MongoDB indexes
    print("\n[MongoDB] Ensuring indexes...")
    ensure_indexes()

    # 3. Seed products into MongoDB
    print(f"\n[MongoDB] Seeding {len(PRODUCTS)} products...")
    for product in PRODUCTS:
        upsert_product(product)
    print(f"[MongoDB] Seeded {len(PRODUCTS)} products.")

    # 4. Compute and store embeddings
    compute_and_store_embeddings()

    print("\n" + "=" * 60)
    print("  Setup complete!")
    print()
    print("  Test credentials:")
    print("    Email   : demo@shopai.in")
    print("    Password: Demo@1234")
    print()
    print("  Start the app:")
    print("    python scripts/start_dev.py")
    print("  OR manually:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
