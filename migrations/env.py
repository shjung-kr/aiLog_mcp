from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from alembic import context

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import all models so autogenerate can detect schema changes.
from app.db.base import Base  # noqa: E402
import app.db.models.episode  # noqa: F401, E402
import app.db.models.episode_rawlog  # noqa: F401, E402
import app.db.models.gist  # noqa: F401, E402
import app.db.models.long_term_memory  # noqa: F401, E402
import app.db.models.rawlog  # noqa: F401, E402
import app.db.models.search_log  # noqa: F401, E402
import app.db.models.session  # noqa: F401, E402
import app.db.models.turn  # noqa: F401, E402

# Use app's URL builder so DB_PASSWORD with special chars is handled correctly.
from app.core.config import _build_database_url  # noqa: E402

target_metadata = Base.metadata
_db_url = _build_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(_db_url, poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
