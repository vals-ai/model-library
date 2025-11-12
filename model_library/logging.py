import logging

from rich.console import Console
from rich.logging import RichHandler

_llm_logger = logging.getLogger("llm")


def set_logging(enable: bool = True, handler: logging.Handler | None = None):
    """
    Sets up logging for the model library

    Args:
        enable (bool): Enable or disable logging.
        handler (logging.Handler, optional): A custom logging handler. Defaults to RichHandler.
    """
    if enable:
        _llm_logger.setLevel(logging.INFO)
    else:
        _llm_logger.setLevel(logging.CRITICAL)

    if not enable or _llm_logger.hasHandlers():
        return

    if handler is None:
        console = Console()
        handler = RichHandler(console=console, markup=True, show_time=False)

    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
    _llm_logger.addHandler(handler)
