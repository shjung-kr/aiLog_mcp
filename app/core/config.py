import os
import re
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _build_database_url() -> URL | str:
    raw = os.getenv("DATABASE_URL", "sqlite:///./ailog.db")
    password = os.getenv("DB_PASSWORD", "")

    if not password:
        return raw

    # SQLite doesn't use passwords
    if raw.startswith("sqlite"):
        return raw

    # Parse the URL and replace the password component so special characters
    # (e.g. @) are handled correctly without manual percent-encoding.
    parsed = make_url(raw)
    return parsed.set(password=password)


class Settings:
    app_name = "aiLog API"
    database_url = _build_database_url()
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    episode_idle_seconds = int(os.getenv("EPISODE_IDLE_SECONDS", "30"))


settings = Settings()
