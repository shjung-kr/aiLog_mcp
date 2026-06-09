from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

_url = settings.database_url
_is_sqlite = (
    _url.startswith("sqlite")
    if isinstance(_url, str)
    else _url.drivername.startswith("sqlite")
)
connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_engine(_url, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
