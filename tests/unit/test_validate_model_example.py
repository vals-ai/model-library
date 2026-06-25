from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from model_library.base import TextInput
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata

from examples.validate_model import _build_cases


class _FakeModel(SimpleNamespace):
    async def query(self, *_args: Any, **_kwargs: Any) -> QueryResult:
        return _query_result("ok")

    async def get_rate_limit(self) -> None:
        return None


def _query_result(text: str) -> QueryResult:
    return QueryResult(
        output_text=text,
        metadata=QueryResultMetadata(
            in_tokens=10,
            out_tokens=5,
            cost=QueryResultCost(input=0.001, output=0.001),
        ),
        tool_calls=[],
        history=[TextInput(text="prompt")],
    )


def _fake_model() -> _FakeModel:
    return _FakeModel(
        supports_images=True,
        supports_files=True,
        supports_tools=False,
        supports_reasoning=False,
        reasoning=False,
        supports_prompt_caching=False,
        supports_batch=False,
        supports_temperature=True,
        supports_output_schema=False,
        provider="test-provider",
        model_name="test-model",
        model_key="test-provider/test-model",
        rate_limit=None,
        pricing=None,
    )


def test_validate_model_media_and_file_cases_require_semantic_answers() -> None:
    cases = {
        (case.section, case.name): case
        for case in _build_cases(_fake_model())  # pyright: ignore[reportArgumentType]
    }

    semantic_expectations = {
        ("Images", "base64 transport"): "red",
        ("Images", "upload-id transport"): "red",
        ("Images", "URL transport"): "cat",
        ("Files", "base64 transport"): "pineapple",
        ("Files", "upload-id transport"): "pineapple",
        ("Files", "URL transport"): "sample",
    }

    for case_key, expected in semantic_expectations.items():
        predicate = cases[case_key].predicate

        passed, detail = predicate(_query_result("unrelated but non-empty"))
        assert not passed
        assert detail is not None
        assert expected in detail

        passed, detail = predicate(_query_result(f"The answer mentions {expected}."))
        assert passed, detail
