import logging
import re

from rich.console import Console
from rich.logging import RichHandler

_llm_logger = logging.getLogger("llm")
_agent_logger = logging.getLogger("agent")

# Matches child loggers of agent.<name> (e.g. agent.submit<gpt-4o>.<run=abc>)
# but not agent.<name> itself — those are agent milestones we want to show.
_AGENT_CHILD_RE = re.compile(r"^agent\.[^.]+\..+")


class _AgentChildFilter(logging.Filter):
    """Suppress DEBUG/INFO from agent child loggers (LLM query details).

    WARNING+ still reaches the console; all levels still go to file handlers
    (which don't have this filter).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _AGENT_CHILD_RE.match(record.name):
            return record.levelno >= logging.WARNING
        return True


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
        return
    else:
        console = Console()
        handler = RichHandler(console=console, markup=False, show_time=False)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    handler.addFilter(_AgentChildFilter())
    _llm_logger.addHandler(handler)
    _agent_logger.addHandler(handler)
