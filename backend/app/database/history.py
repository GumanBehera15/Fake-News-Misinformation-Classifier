import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from app.core.config import DATABASE_PATH

@contextmanager
def connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def initialize_database() -> None:
    with connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preview TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )""")

def add_history(text: str, prediction: str, confidence: float) -> None:
    preview = text[:300] + ("…" if len(text) > 300 else "")
    with connection() as conn:
        conn.execute(
            "INSERT INTO analyses (preview, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
            (preview, prediction, confidence, datetime.now(timezone.utc).isoformat()),
        )

def list_history(limit: int = 50) -> list[dict]:
    with connection() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]

def clear_history() -> None:
    with connection() as conn:
        conn.execute("DELETE FROM analyses")

