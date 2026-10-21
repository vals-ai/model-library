"""Every registry provider takes its Gateway key from the seeded pools or is listed native."""

import os
from typing import Any
from unittest.mock import patch

import pytest

import model_gateway.key_pools as key_pools
import model_gateway.model_helpers as model_helpers
from model_gateway.types import QueryRequest
from pydantic import SecretStr

from model_library.base import LLM, LLMConfig
from model_library.registry_utils import get_model_registry, get_registry_model
from model_library.settings import ModelLibrarySettings
