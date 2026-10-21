"""Parity between the seeded pools and the positional selection they replaced."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import model_gateway.model_helpers as model_helpers
from collections.abc import Iterator

from model_gateway.key_pools import key_setting_provider, keys_for, provider_key_settings
from model_gateway.model_helpers import ManagedApiKey
from model_gateway.rate_limit_monitor.manager import (
    _source_specs,  # pyright: ignore[reportPrivateUsage]
)
from model_gateway.rate_limit_monitor.types import MonitorFacts, MonitorSourceFacts
from model_gateway.types import QueryRequest
from model_library.settings import ModelLibrarySettings
@pytest.mark.parametrize(
    ("setting", "provider"),
    [
        ("ANTHROPIC_API_KEY_2", "anthropic"),
        ("XAI_API_KEY", "grok"),
        ("MERCURY_API_KEY_2", "inception"),
        ("GCP_CREDS", None),
    ],
)
def test_key_setting_provider_names_every_pooled_provider(
    setting: str, provider: str | None
) -> None:
    assert key_setting_provider(setting, provider_key_settings()) == provider
