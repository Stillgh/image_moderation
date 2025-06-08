from functools import lru_cache
from sqlmodel import create_engine, Session
from config.db_config import get_settings


def create_db_engine():
    settings = get_settings()
    return create_engine(
        url=settings.DATABASE_URL_psycopg,
        echo=False,
        pool_size=5,
        max_overflow=10
    )


@lru_cache(maxsize=1)
def get_engine():
    return create_db_engine()


def get_session():
    engine = get_engine()
    with Session(engine) as session:
        yield session
