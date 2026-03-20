import logging

from rich.console import Console
from rich.logging import RichHandler

_llm_logger = logging.getLogger("llm")
_agent_logger = logging.getLogger("agent")


def set_logging(
    enable: bool = True,
    level: int = logging.INFO,
    handler: logging.Handler | None = None,
):
    """
    Sets up logging for the model library

    Args:
        enable (bool): Enable or disable logging.
        handler (logging.Handler, optional): A custom logging handler. Defaults to RichHandler.
    """
    if not enable:
        _llm_logger.setLevel(logging.CRITICAL)
        _agent_logger.setLevel(logging.CRITICAL)
        return

    _llm_logger.setLevel(level)
    _agent_logger.setLevel(level)

    if handler is not None:
        for existing_handler in list(_llm_logger.handlers):
            _llm_logger.removeHandler(existing_handler)
        for existing_handler in list(_agent_logger.handlers):
            _agent_logger.removeHandler(existing_handler)
    elif _llm_logger.hasHandlers():
        for h in _llm_logger.handlers:
            h.setLevel(level)
        for h in _agent_logger.handlers:
            h.setLevel(level)
        return
    else:
        console = Console()
        handler = RichHandler(
            console=console, markup=False, show_time=False, rich_tracebacks=True
        )

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _llm_logger.addHandler(handler)
    _agent_logger.addHandler(handler)
