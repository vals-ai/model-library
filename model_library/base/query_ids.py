"""Shared query ID and prompt cache key helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, NamedTuple

from model_library.base.input import InputItem, SystemInput, ToolDefinition

PromptCacheKeyMode = Literal["hash", "id"]


class ResolvedQueryIds(NamedTuple):
    run_id: str | None
    question_id: str | None
    query_id: str
    attempt_id: str | None

    def as_attributes(self) -> dict[str, str]:
        return {name: value for name, value in self._asdict().items() if value}


_current_query_ids: ContextVar[ResolvedQueryIds | None] = ContextVar(
    "model_library_current_query_ids",
    default=None,
)


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
    attempt_id: object | None,
) -> ResolvedQueryIds:
    """Resolve run, question, query, and attempt IDs for LLM query paths."""
    return ResolvedQueryIds(
        _explicit_id(run_id, "run_id"),
        _explicit_id(question_id, "question_id"),
        _clean_id(query_id) or uuid.uuid4().hex[:14],
        _explicit_id(attempt_id, "attempt_id"),
    )


@contextmanager
def scoped_query_ids(ids: ResolvedQueryIds) -> Generator[None, None, None]:
    token = _current_query_ids.set(ids)
    try:
        yield
    finally:
        _current_query_ids.reset(token)


def get_current_query_ids() -> ResolvedQueryIds | None:
    return _current_query_ids.get()


def get_current_query_id_attributes() -> dict[str, str]:
    query_ids = get_current_query_ids()
    if query_ids is None:
        return {}
    return query_ids.as_attributes()


def prompt_cache_key_from_prompt_prefix(
    *,
    model_name: str,
    input: Sequence[InputItem],
    tools: Sequence[ToolDefinition],
) -> str:
    system = [item.text for item in input if isinstance(item, SystemInput)]
    tool_bodies = [tool.model_dump(mode="json") for tool in tools]
    prompt_prefix = (
        [system, tool_bodies]
        if system or tool_bodies
        else [item.model_dump(mode="json") for item in input[:1]]
    )
    serialized_prefix = json.dumps(
        prompt_prefix,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _hashed_prompt_cache_key(model_name=model_name, value=serialized_prefix)


def prompt_identity(
    *,
    model_name: str,
    run_id: str | None,
    question_id: str | None,
    input: Sequence[InputItem],
    tools: Sequence[ToolDefinition],
) -> tuple[str, str]:
    """The caller's run and question IDs, with the prompt-prefix hash for missing ones."""
    if run_id is not None and question_id is not None:
        return run_id, question_id
    prefix = prompt_cache_key_from_prompt_prefix(
        model_name=model_name, input=input, tools=tools
    )
    return run_id or prefix, question_id or prefix


def prompt_cache_key_from_query_ids(
    *,
    model_name: str,
    input: Sequence[InputItem],
    tools: Sequence[ToolDefinition],
) -> str:
    query_ids = get_current_query_ids()
    run_id, question_id = prompt_identity(
        model_name=model_name,
        run_id=query_ids and query_ids.run_id,
        question_id=query_ids and query_ids.question_id,
        input=input,
        tools=tools,
    )
    return _hashed_prompt_cache_key(
        model_name=model_name, value=f"{run_id}:{question_id}"
    )


def _hashed_prompt_cache_key(*, model_name: str, value: str) -> str:
    return hashlib.sha256(f"{model_name}:{value}".encode("utf-8")).hexdigest()[:32]
