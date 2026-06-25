"""Shared query correlation ID resolution."""

from __future__ import annotations

import uuid

import model_library


def _clean_id(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _explicit_id(value: object | None, name: str) -> str | None:
    text = _clean_id(value)
    if value is not None and text is None:
        raise ValueError(f"{name} must not be blank")
    return text


def resolve_query_ids(
    *,
    run_id: object | None,
    question_id: object | None,
    query_id: object | None,
) -> tuple[str, str, str]:
    """Resolve run, question, and query IDs for LLM query paths."""
    settings = model_library.model_library_settings

    return (
        _explicit_id(run_id, "run_id")
        or _clean_id(settings.get("RUN_ID", None))
        or uuid.uuid4().hex[:8],
        _explicit_id(question_id, "question_id")
        or _clean_id(settings.get("QUESTION_ID", None))
        or _clean_id(settings.get("TASK_ID", None))
        or uuid.uuid4().hex[:14],
        _clean_id(query_id) or uuid.uuid4().hex[:14],
    )
