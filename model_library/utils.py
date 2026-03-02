import logging
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pprint import pformat

from pydantic.main import BaseModel

MAX_LLM_LOG_LENGTH = 100
logger = logging.getLogger("llm")


class PrettyModel(BaseModel):
    """BaseModel with pformat __repr__ and __str__"""

    def __repr__(self) -> str:
        attrs = vars(self).copy()
        for name in self.__class__.model_computed_fields:
            attrs[name] = getattr(self, name)
        return f"{self.__class__.__name__}(\n{pformat(attrs, indent=2, sort_dicts=False)}\n)"

    __str__ = __repr__


def create_run_dir(benchmark: str, model_name: str, base: Path = Path("logs")) -> Path:
    """Create a timestamped run directory for benchmark logs

    Returns: Path like logs/<benchmark>/<model>/<timestamp>_<uuid>/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        base
        / benchmark
        / model_name.replace("/", "_")
        / f"{timestamp}_{uuid.uuid4().hex[:6]}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@contextmanager
def create_file_logger(
    name: str,
    log_file: str | Path,
    level: int = logging.INFO,
    console: bool = False,
) -> Iterator[logging.Logger]:
    """Context manager that creates a logger writing to a file

    Usage:
        with create_file_logger("agent", "logs/run.log") as logger:
            agent = Agent(llm=llm, tools=tools, logger=logger)
            result = await agent.run(input)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handlers: list[logging.Handler] = []

    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        handlers.append(console_handler)

    for h in handlers:
        logger.addHandler(h)

    try:
        yield logger
    finally:
        for h in handlers:
            logger.removeHandler(h)
            h.close()


def setup_history_dir(logger: logging.Logger) -> Path | None:
    """Create directory structure for history serialization

    Detects FileHandler on the logger, converts:
      .../question.log → .../question/agent.log + .../question/histories/
    """
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handler = handler
            break

    if file_handler is None:
        logger.warning("serialize_histories enabled but no FileHandler found, skipping")
        return None

    log_path = Path(file_handler.baseFilename)
    output_dir = log_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    histories_dir = output_dir / "histories"
    histories_dir.mkdir(exist_ok=True)

    # Redirect handler to write inside the new directory
    new_log_path = output_dir / "agent.log"
    file_handler.close()
    if log_path.exists() and log_path.stat().st_size == 0:
        log_path.unlink()
    file_handler.baseFilename = str(new_log_path)
    file_handler.stream = file_handler._open()

    return histories_dir


def truncate_str(s: str | None, max_len: int = MAX_LLM_LOG_LENGTH) -> str:
    if not s:
        return ""
    if len(s) <= max_len:
        return s
    half = (max_len - 1) // 2
    return s[:half] + " […] " + s[-half:]


def get_logger(name: str | None = None):
    if not name:
        return logger
    return logging.getLogger(f"{logger.name}.{name}")


def deep_model_dump(obj: object) -> object:
    if isinstance(obj, BaseModel):
        return deep_model_dump(obj.model_dump(exclude_unset=True, exclude_none=True))

    if isinstance(obj, Mapping):
        return {k: deep_model_dump(v) for k, v in obj.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [deep_model_dump(v) for v in obj]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]

    return obj


def default_httpx_client():
    return httpx.AsyncClient(
        timeout=httpx.Timeout(None),
        limits=httpx.Limits(
            max_connections=2000, max_keepalive_connections=300
        ),  # TODO: increase, but make sure prod enough sockets to not hit file descriptor limit
    )


def create_openai_client_with_defaults(
    api_key: str, base_url: str | None = None
) -> AsyncOpenAI:
    """
    OpenAI defaults:
    DEFAULT_TIMEOUT = httpx.Timeout(timeout=600, connect=5.0)
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_CONNECTION_LIMITS = httpx.Limits(max_connections=1000, max_keepalive_connections=100)
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=default_httpx_client(),
        max_retries=3,
    )


def create_anthropic_client_with_defaults(
    api_key: str, base_url: str | None = None, default_headers: dict[str, str] = {}
) -> AsyncAnthropic:
    return AsyncAnthropic(
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
        http_client=default_httpx_client(),
        max_retries=3,
    )


def get_context_window_for_model(model_name: str) -> int | None:
    """
    Get the context window for a model by looking up its configuration from the registry.

    Args:
        model_name: The name of the model in the registry (e.g., "openai/gpt-4o-mini-2024-07-18" or "azure/gpt-4o-mini-2024-07-18")

    Returns:
        Context window size in tokens (or `None` if not found)
    """
    # import here to avoid circular imports
    from model_library.register_models import get_model_registry

    model_config = get_model_registry().get(model_name, None)
    if (
        model_config
        and model_config.properties
        and model_config.properties.context_window
    ):
        return model_config.properties.context_window
    else:
        logger.warning(
            f"Model {model_name} not found in registry or missing context_window"
        )
        return None
