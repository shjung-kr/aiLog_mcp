import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class Settings:
    app_name = "aiLog API"
    database_url = os.getenv("DATABASE_URL", "sqlite:///./ailog.db")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    episode_idle_seconds = int(os.getenv("EPISODE_IDLE_SECONDS", "30"))


settings = Settings()
