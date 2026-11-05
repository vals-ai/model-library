from typing import TypedDict

from sqlalchemy import MetaData
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

metadata = MetaData()


class DatabaseSecret(TypedDict):
    username: str
    password: str
    host: str
    port: int


def build_database_url(secret: DatabaseSecret, database_name: str) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username=secret["username"],
        password=secret["password"],
        host=secret["host"],
        port=secret["port"],
        database=database_name,
    )


def create_engine(database_url: URL, pool_size: int) -> AsyncEngine:
    return create_async_engine(database_url, pool_size=pool_size, max_overflow=0)
