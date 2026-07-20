from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_library.retriers.token.benchmark_queue import benchmark_queue
    from model_library.retriers.token.token import TokenRetrier
    from model_library.retriers.token.utils import (
        AsyncRedisClient,
        cleanup_all_keys,
        set_redis_client,
    )

__all__ = [
    "AsyncRedisClient",
    "TokenRetrier",
    "benchmark_queue",
    "cleanup_all_keys",
    "set_redis_client",
]


def __getattr__(name: str) -> object:
    if name == "benchmark_queue":
        from model_library.retriers.token.benchmark_queue import benchmark_queue

        value = benchmark_queue
    elif name == "TokenRetrier":
        from model_library.retriers.token.token import TokenRetrier

        value = TokenRetrier
    elif name in {"AsyncRedisClient", "cleanup_all_keys", "set_redis_client"}:
        from model_library.retriers.token import utils

        value = getattr(utils, name)
    else:
        raise AttributeError(name)

    globals()[name] = value
    return value
