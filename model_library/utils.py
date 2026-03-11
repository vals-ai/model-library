import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat

import httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
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


@contextmanager
def run_logging(
    logger: logging.Logger,
    log_dir: Path,
    question_id: str,
) -> Iterator[Path]:
    """Manage file logging for an agent run.

    Always yields a question-scoped directory for output files (result.json, histories/).
    If the logger already has a FileHandler, reuses it (no new handler created).
    Otherwise creates log_dir/<question_id>/agent.log.

    The output directory is always scoped: <base>/<question_id>/
    """
    # Use existing FileHandler if present (walk up the logger hierarchy, skip root)
    current: logging.Logger | None = logger
    while current and current is not logging.root:
        for h in current.handlers:
            if isinstance(h, logging.FileHandler):
                output_dir = Path(h.baseFilename).parent / question_id
                output_dir.mkdir(parents=True, exist_ok=True)
                yield output_dir
                return
        current = current.parent if current.propagate else None

    # Create our own FileHandler
    output_dir = log_dir / question_id
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "agent.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)
    try:
        yield output_dir
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


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
