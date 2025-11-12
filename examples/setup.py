import logging
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

# console logger (if you want to print specific colors)
console_logger = logging.getLogger("console")
console_logger.setLevel(logging.INFO)
console_logger.handlers.clear()
console_logger.addHandler(
    RichHandler(console=Console(), markup=True, show_time=False, rich_tracebacks=True)
)

# all other logs
logging.root.setLevel(logging.WARN)  # set to DEBUG for more info (ex. network requests)


def console_log(output: Any, level: int = logging.INFO, color: str = "red"):
    if isinstance(output, str):
        console_logger.log(level, f"[{color}]{output}[/{color}]")
    else:
        console_logger.log(level, output)


def setup(disable_logging: bool = False, load_gcp: bool = True):
    if disable_logging:
        llm_logger = logging.getLogger("llm")
        llm_logger.setLevel(logging.ERROR)
