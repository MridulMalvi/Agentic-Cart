"""
tools/auth_tools.py  (Refactored — added register_user)

login_user  — Authenticates via bcrypt + MySQL. Stores session in Redis.
register_user — Registers a new user with bcrypt-hashed password in MySQL.

Security notes:
  - Auth failures return a single generic message to prevent user-enumeration.
  - Passwords are bcrypt-hashed with cost factor 12 (default).
  - Email is normalised to lowercase before any DB operation.
  - register_user enforces minimum password length before touching the DB.
"""

import bcrypt
from langchain_core.tools import tool
from db.db_client import execute_query
from memory.redis_memory import set_session

_AUTH_FAIL_MSG  = "Invalid email or password. Please check your credentials and try again."
_MIN_PASS_LEN   = 8


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user with email and password.

    Call ONLY when the user explicitly provides both email AND password.
    On success, user_id is written into LangGraph ShoppingState by tool_node_handler.

    Returns:
        {
          "success"  : bool,
          "user_id"  : int   (only on success),
          "email"    : str   (only on success),
          "full_name": str   (only on success),
          "message"  : str
        }
    """
    if not email or not password:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    rows = execute_query(
        "SELECT user_id, email, full_name, password_hash "
        "FROM users WHERE email = %s",
        (email.strip().lower(),),
    )

    # Generic failure — no hint whether email or password was wrong
    if not rows:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    user = rows[0]

    try:
        valid = bcrypt.checkpw(
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8"),
        )
    except Exception:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    if not valid:
        return {"success": False, "message": _AUTH_FAIL_MSG}

    set_session(user["user_id"], user["email"], user.get("full_name", ""))

    return {
        "success":   True,
        "user_id":   user["user_id"],
        "email":     user["email"],
        "full_name": user.get("full_name") or user["email"],
        "message":   "Login successful. Welcome back!",
    }


@tool
def register_user(email: str, password: str, full_name: str = "") -> dict:
    """
    Register a new user account.

    Call when user explicitly asks to sign up / create an account.
    Validates input, hashes password with bcrypt, inserts into MySQL.

    Rules:
      - email must contain '@'
      - password must be at least 8 characters
      - email is normalised to lowercase
      - duplicate email → clear error (registration is an explicit intent,
        unlike login where we avoid enumeration)

    Returns:
        {
          "success"  : bool,
          "user_id"  : int   (only on success),
          "email"    : str,
          "full_name": str,
          "message"  : str
        }
    """
    if not email or "@" not in email:
        return {"success": False, "message": "Please provide a valid email address."}

    if not password or len(password) < _MIN_PASS_LEN:
        return {
            "success": False,
            "message": f"Password must be at least {_MIN_PASS_LEN} characters.",
        }

    norm_email = email.strip().lower()

    # Check if email already exists
    existing = execute_query(
        "SELECT user_id FROM users WHERE email = %s",
        (norm_email,),
    )
    if existing:
        return {
            "success": False,
            "message": (
                f"An account with {norm_email} already exists. "
                "Please log in instead."
            ),
        }

    # Hash password
    try:
        pw_hash = bcrypt.hashpw(
            password.encode("utf-8"),
            bcrypt.gensalt(rounds=12),
        ).decode("utf-8")
    except Exception as exc:
        return {"success": False, "message": f"Registration failed: {exc}"}

    # Insert user
    try:
        execute_query(
            "INSERT INTO users (email, full_name, password_hash) VALUES (%s, %s, %s)",
            (norm_email, full_name.strip() or norm_email, pw_hash),
            fetch=False,
        )
    except Exception as exc:
        return {"success": False, "message": f"Registration failed: {exc}"}

    # Fetch the auto-generated user_id
    rows = execute_query(
        "SELECT user_id FROM users WHERE email = %s",
        (norm_email,),
    )
    user_id = rows[0]["user_id"] if rows else None

    if user_id:
        set_session(user_id, norm_email, full_name.strip() or norm_email)

    return {
        "success":   True,
        "user_id":   user_id,
        "email":     norm_email,
        "full_name": full_name.strip() or norm_email,
        "message":   "Account created successfully! You are now logged in. 🎉",
    }