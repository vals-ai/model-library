import logging

import pytest

from model_library.logging import _llm_logger, set_logging


@pytest.fixture(autouse=True)
def restore_llm_logger_state():
    original_handlers = list(_llm_logger.handlers)
    original_level = _llm_logger.level

    yield

    for handler in list(_llm_logger.handlers):
        _llm_logger.removeHandler(handler)
    for handler in original_handlers:
        _llm_logger.addHandler(handler)
    _llm_logger.setLevel(original_level)


def test_custom_handler_replaces_default():
    custom = logging.StreamHandler()
    set_logging(handler=custom)
    assert custom in _llm_logger.handlers
    assert len(_llm_logger.handlers) == 1


def test_disable_sets_critical():
    set_logging(enable=False)
    assert _llm_logger.level == logging.CRITICAL


def test_set_level():
    set_logging(level=logging.DEBUG)
    assert _llm_logger.level == logging.DEBUG


def test_existing_handlers_skip_local_handler(monkeypatch):
    _llm_logger.handlers.clear()
    monkeypatch.setattr(_llm_logger, "hasHandlers", lambda: True)

    set_logging()

    assert _llm_logger.handlers == []
