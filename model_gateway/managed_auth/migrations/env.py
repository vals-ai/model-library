import asyncio
import json
import os
from logging.config import fileConfig
from typing import cast

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from model_gateway.managed_auth import tables
from model_gateway.managed_auth.database import (
    DatabaseSecret,
    build_database_url,
    metadata,
)

_ = tables  # Register managed tables with metadata for Alembic.

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

DATABASE_URL = build_database_url(
    cast(
        DatabaseSecret,
        json.loads(os.environ["MODEL_GATEWAY_AUTH_DATABASE_SECRET"]),
    ),
    os.environ["MODEL_GATEWAY_AUTH_DATABASE_NAME"],
)
_MIGRATION_ADVISORY_LOCK_ID = 0x4D47415554484D47


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL.render_as_string(hide_password=False),
        target_metadata=metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(DATABASE_URL, poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("SELECT pg_advisory_lock(:lock_id)"),
                {"lock_id": _MIGRATION_ADVISORY_LOCK_ID},
            )
            await connection.commit()
            try:
                await connection.run_sync(run_migrations)
            finally:
                await connection.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"),
                    {"lock_id": _MIGRATION_ADVISORY_LOCK_ID},
                )
                await connection.commit()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
