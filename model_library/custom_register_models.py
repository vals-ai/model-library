import urllib.request
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import yaml

from model_library.register_models import (
    ModelRegistry,
    get_model_registry,
    parse_yaml_blocks,
)
from model_library.utils import get_logger

logger = get_logger("custom_register_models")

DEFAULT_URL_TIMEOUT_SECONDS = 30


def _default_config_files() -> list[str]:
    config_dir = Path(__file__).parent / "config"
    return sorted(
        path.name
        for path in config_dir.glob("*.yaml")
        if path.name != "all_models.json"
    )


def load_custom_model_configs(
    source: str | Path,
    registry: ModelRegistry | None = None,
) -> None:
    """Load model configs from a single YAML file and merge into the registry, overriding defaults.

    source can be a URL (http:// or https://) or a local file path.
    The YAML format is the same as the bundled provider configs.
    """
    if registry is None:
        registry = get_model_registry()

    source_str = str(source)
    logger.info(f"Loading model configs from {source_str}")

    if urlparse(source_str).scheme in {"http", "https"}:
        with urllib.request.urlopen(
            source_str, timeout=DEFAULT_URL_TIMEOUT_SECONDS
        ) as response:
            content = response.read().decode("utf-8")
    else:
        with open(source, "r") as f:
            content = f.read()

    try:
        model_blocks = cast(
            dict[str, dict[str, dict[str, Any]]] | None, yaml.safe_load(content)
        )
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML: {e}")
        raise e

    if model_blocks:
        parse_yaml_blocks(model_blocks, registry)


def load_latest_vals_model_configs(branch: str = "main") -> None:
    """Fetch the latest model configs from the public model-library GitHub repo.

    Pulls each bundled provider YAML from the given branch and merges them into
    the registry, overriding any existing models with the same key.
    """
    base = f"https://raw.githubusercontent.com/vals-ai/model-library/{branch}/model_library/config"
    for filename in _default_config_files():
        load_custom_model_configs(f"{base}/{filename}")
