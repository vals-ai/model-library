from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from model_gateway.managed_auth import tables as t


async def list_key_spend_sources(engine: AsyncEngine) -> list[UUID]:
    async with engine.connect() as connection:
        rows = (
            (
                await connection.execute(
                    select(t.keys.c.id)
                    .where(
                        t.keys.c.key_limit_period.is_not(None),
                        t.keys.c.key_limit_usd.is_not(None),
                    )
                    .order_by(t.keys.c.id)
                )
            )
            .scalars()
            .all()
        )
    return list(rows)
