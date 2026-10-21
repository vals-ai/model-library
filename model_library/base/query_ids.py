"""Shared query ID and prompt cache key helpers."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, NamedTuple, TypeVar

import model_library
from model_library.base.input import RawResponse

PromptCacheKeyMode = Literal["hash", "id"]


class ResolvedQueryIds(NamedTuple):
    run_id: str
    question_id: str
    query_id: str
    attempt_id: str

    def as_attributes(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "question_id": self.question_id,
            "query_id": self.query_id,
            "attempt_id": self.attempt_id,
        }


InputT = TypeVar("InputT")
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
    settings = model_library.model_library_settings

    return ResolvedQueryIds(
        _explicit_id(run_id, "run_id")
        or _clean_id(settings.get("RUN_ID", None))
        or uuid.uuid4().hex[:8],
        _explicit_id(question_id, "question_id")
        or _clean_id(settings.get("QUESTION_ID", None))
        or _clean_id(settings.get("TASK_ID", None))
        or uuid.uuid4().hex[:14],
        _clean_id(query_id) or uuid.uuid4().hex[:14],
        _explicit_id(attempt_id, "attempt_id")
        or _clean_id(settings.get("ATTEMPT_ID", None))
        or uuid.uuid4().hex[:14],
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


def prompt_cache_key_prefix(input: Sequence[InputT]) -> Sequence[InputT]:
    for index, item in enumerate(input):
        if isinstance(item, RawResponse):
            return input[:index]
    return input


async def resolve_prompt_cache_key(
    *,
    mode: PromptCacheKeyMode | None,
    model_name: str,
    input: Sequence[InputT],
    parse_prompt_prefix: Callable[[Sequence[InputT]], Awaitable[object]],
    run_id: object | None = None,
    question_id: object | None = None,
) -> str | None:
    match mode:
        case None:
            return None
        case "hash":
            prefix = prompt_cache_key_prefix(input)
            prompt_prefix: object = await parse_prompt_prefix(prefix) if prefix else []
            return prompt_cache_key_from_prompt_prefix(
                model_name=model_name,
                prompt_prefix=prompt_prefix,
            )
        case "id":
            return prompt_cache_key_from_query_ids(
                model_name=model_name,
                run_id=run_id,
                question_id=question_id,
            )


def prompt_cache_key_from_prompt_prefix(
    *,
    model_name: str,
    prompt_prefix: object,
) -> str:
    serialized_prefix = json.dumps(
        prompt_prefix,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _hashed_prompt_cache_key(model_name=model_name, value=serialized_prefix)


def prompt_cache_key_from_query_ids(
    *,
    model_name: str,
    run_id: object | None = None,
    question_id: object | None = None,
) -> str:
    if run_id is None or question_id is None:
        query_ids = get_current_query_ids()
        if query_ids is None:
            raise ValueError("prompt_cache_key='id' requires resolved query IDs")
        run_id = query_ids.run_id
        question_id = query_ids.question_id

    resolved_run_id = _explicit_id(run_id, "run_id")
    resolved_question_id = _explicit_id(question_id, "question_id")
    if resolved_run_id is None or resolved_question_id is None:
        raise ValueError("prompt_cache_key='id' requires resolved query IDs")

    return _hashed_prompt_cache_key(
        model_name=model_name,
        value=f"{resolved_run_id}:{resolved_question_id}",
    )


def _hashed_prompt_cache_key(*, model_name: str, value: str) -> str:
    return hashlib.sha256(f"{model_name}:{value}".encode("utf-8")).hexdigest()[:32]
