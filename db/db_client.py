"""
db/db_client.py

MySQL connection pool with automatic SQLite fallback when MySQL is unreachable.

Key design points:
  - Tries MySQL connection pool first.
  - If MySQL is unreachable, transparently falls back to local SQLite database (db/app.sqlite).
  - execute_query       : single SELECT / INSERT / UPDATE / DELETE.
  - execute_transaction : multiple statements in ONE atomic transaction.
"""

import os
import sqlite3
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_pool = None
_pool_lock = threading.Lock()
_use_sqlite = False
_sqlite_db_path = Path(__file__).resolve().parent / "app.sqlite"


def _init_sqlite_tables(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        email         TEXT UNIQUE NOT NULL,
        full_name     TEXT,
        password_hash TEXT NOT NULL,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id     TEXT PRIMARY KEY,
        user_id      INTEGER NOT NULL,
        total_amount REAL,
        items_json   TEXT,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)
    conn.commit()


def get_sqlite_conn():
    conn = sqlite3.connect(str(_sqlite_db_path))
    conn.row_factory = sqlite3.Row
    _init_sqlite_tables(conn)
    return conn


def _check_mysql():
    global _pool, _use_sqlite
    if _use_sqlite:
        return False
    if _pool is not None:
        return True
    with _pool_lock:
        if _pool is not None:
            return True
        try:
            import mysql.connector
            from mysql.connector import pooling
            _pool = pooling.MySQLConnectionPool(
                pool_name="shop_pool",
                pool_size=10,
                pool_reset_session=True,
                connection_timeout=2,
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "shopping_assistant"),
                autocommit=False,
            )
            # Test connection
            c = _pool.get_connection()
            c.close()
            return True
        except Exception as exc:
            print(f"[MySQL] Unavailable ({exc}). Falling back to SQLite ({_sqlite_db_path}).")
            _use_sqlite = True
            return False


def execute_query(sql: str, params: tuple = None, fetch: bool = True):
    if _check_mysql():
        conn = _pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params or ())
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
        except Exception as exc:
            conn.rollback()
            raise exc
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()
    else:
        # SQLite fallback
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        sql_sqlite = sql.replace("%s", "?").replace("AUTO_INCREMENT", "AUTOINCREMENT").replace("INT PRIMARY KEY", "INTEGER PRIMARY KEY")
        try:
            cursor.execute(sql_sqlite, params or ())
            if fetch:
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
            else:
                conn.commit()
                result = cursor.rowcount
            return result
        except Exception as exc:
            conn.rollback()
            raise exc
        finally:
            conn.close()


def execute_transaction(statements: list[tuple]) -> bool:
    if _check_mysql():
        conn = _pool.get_connection()
        cursor = None
        try:
            cursor = conn.cursor(dictionary=True)
            conn.start_transaction()
            for sql, params in statements:
                cursor.execute(sql, params or ())
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            print(f"[DB TRANSACTION ERROR] {exc}")
            raise exc
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()
    else:
        # SQLite fallback
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        try:
            conn.execute("BEGIN TRANSACTION")
            for sql, params in statements:
                sql_sqlite = sql.replace("%s", "?").replace("AUTO_INCREMENT", "AUTOINCREMENT").replace("INT PRIMARY KEY", "INTEGER PRIMARY KEY")
                cursor.execute(sql_sqlite, params or ())
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            print(f"[DB TRANSACTION ERROR] {exc}")
            raise exc
        finally:
            conn.close()