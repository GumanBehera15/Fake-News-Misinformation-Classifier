from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", BACKEND_DIR / "models"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BACKEND_DIR / "fake_news.db"))
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "10000"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

