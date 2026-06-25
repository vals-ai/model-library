import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from model_library import set_logging
from model_library.base import LLM

_ENV_FILE = Path(".env")

# console logger (if you want to print specific colors)
console_logger = logging.getLogger("console")
console_logger.setLevel(logging.INFO)
console_logger.handlers.clear()
console_logger.addHandler(
    RichHandler(console=Console(), markup=True, show_time=False, rich_tracebacks=True)
)

# all other logs
# logging.root.setLevel(logging.WARN)  # set to DEBUG for more info (ex. network requests)


def console_log(output: Any, level: int = logging.INFO, color: str = "red"):
    if isinstance(output, str):
        console_logger.log(level, f"[{color}]{output}[/{color}]")
    else:
        console_logger.log(level, output)


async def sync_model_metadata(model: LLM) -> None:
    if model.gateway_mode:
        await model.ensure_metadata_loaded()


def setup(disable_logging: bool = False, load_gcp: bool = True):
    """Load environment variables for local examples."""
    if disable_logging:
        set_logging(level=logging.ERROR)
    load_dotenv(override=True, dotenv_path=_ENV_FILE)
    if not load_gcp:
        for key in ("GCP_CREDS", "GCP_PROJECT_ID", "GCP_REGION", "GS_URI"):
            os.environ.pop(key, None)
