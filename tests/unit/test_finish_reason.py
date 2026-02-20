import typing

from model_library.base import FinishReason, FinishReasonInfo
from model_library.providers.ai21labs import map_ai21_finish_reason
from model_library.providers.amazon import map_amazon_finish_reason
from model_library.providers.anthropic import map_anthropic_finish_reason
from model_library.providers.google.google import map_google_finish_reason
from model_library.providers.mistral import map_mistral_finish_reason
from model_library.providers.openai import (
    map_openai_completions_finish_reason,
    map_openai_responses_finish_reason,
)
from model_library.providers.xai import map_xai_finish_reason


def _assert_finish_reason_info(
    info: FinishReasonInfo,
    *,
    expected_reason: FinishReason,
    expected_raw: str | None,
) -> None:
    assert isinstance(info, FinishReasonInfo)
    assert info.reason is expected_reason
    assert info.raw == expected_raw


async def test_anthropic_finish_reason_all_values():
    from anthropic.types import Message

    stop_reason_type = Message.__annotations__["stop_reason"]
    stop_reasons = [
        x for x in typing.get_args(typing.get_args(stop_reason_type)[0]) if x is not None
    ]

    expected: dict[str, FinishReason] = {
        "end_turn": FinishReason.STOP,
        "max_tokens": FinishReason.MAX_TOKENS,
        "stop_sequence": FinishReason.STOP_SEQUENCE,
        "tool_use": FinishReason.TOOL_CALLS,
        "pause_turn": FinishReason.STOP,
        "refusal": FinishReason.CONTENT_FILTER,
    }

    assert set(stop_reasons) == set(expected)

    info = map_anthropic_finish_reason("model_context_window_exceeded")
    _assert_finish_reason_info(
        info,
        expected_reason=FinishReason.MAX_TOKENS,
        expected_raw="model_context_window_exceeded",
    )

    for stop_reason in stop_reasons:
        info = map_anthropic_finish_reason(stop_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=expected[stop_reason],
            expected_raw=stop_reason,
        )


async def test_openai_completions_finish_reason_all_values():
    from openai.types.chat.chat_completion import Choice

    finish_reasons = list(typing.get_args(Choice.__annotations__["finish_reason"]))

    expected: dict[str, FinishReason] = {
        "stop": FinishReason.STOP,
        "length": FinishReason.MAX_TOKENS,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
        "function_call": FinishReason.TOOL_CALLS,
    }

    assert set(finish_reasons) == set(expected)

    for finish_reason in finish_reasons:
        info = map_openai_completions_finish_reason(finish_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=expected[finish_reason],
            expected_raw=finish_reason,
        )


async def test_openai_responses_finish_reason_all_values():
    from openai.types.responses.response import Response

    status_type = Response.__annotations__["status"]
    statuses = [
        x for x in typing.get_args(typing.get_args(status_type)[0]) if x is not None
    ]

    for status in statuses:
        match status:
            case "completed":
                for has_tool_calls in (False, True):
                    info = map_openai_responses_finish_reason(
                        status, None, has_tool_calls
                    )
                    _assert_finish_reason_info(
                        info,
                        expected_reason=(
                            FinishReason.TOOL_CALLS
                            if has_tool_calls
                            else FinishReason.STOP
                        ),
                        expected_raw="completed",
                    )
            case "incomplete":
                for incomplete_reason, expected_reason in (
                    ("max_output_tokens", FinishReason.MAX_TOKENS),
                    ("content_filter", FinishReason.CONTENT_FILTER),
                ):
                    info = map_openai_responses_finish_reason(
                        status, incomplete_reason, False
                    )
                    _assert_finish_reason_info(
                        info,
                        expected_reason=expected_reason,
                        expected_raw=f"incomplete:{incomplete_reason}",
                    )
                info = map_openai_responses_finish_reason(status, None, False)
                _assert_finish_reason_info(
                    info,
                    expected_reason=FinishReason.UNKNOWN,
                    expected_raw="incomplete",
                )
            case "failed":
                info = map_openai_responses_finish_reason(status, None, False)
                _assert_finish_reason_info(
                    info,
                    expected_reason=FinishReason.ERROR,
                    expected_raw=status,
                )
            case "in_progress" | "cancelled" | "queued":
                info = map_openai_responses_finish_reason(status, None, False)
                _assert_finish_reason_info(
                    info,
                    expected_reason=FinishReason.UNKNOWN,
                    expected_raw=status,
                )


async def test_google_finish_reason_all_values():
    from google.genai.types import FinishReason as GoogleFinishReason

    handled: dict[GoogleFinishReason, FinishReason] = {
        GoogleFinishReason.STOP: FinishReason.STOP,
        GoogleFinishReason.MAX_TOKENS: FinishReason.MAX_TOKENS,
        GoogleFinishReason.SAFETY: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.RECITATION: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.LANGUAGE: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.BLOCKLIST: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.PROHIBITED_CONTENT: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.SPII: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.IMAGE_SAFETY: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.IMAGE_PROHIBITED_CONTENT: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.IMAGE_RECITATION: FinishReason.CONTENT_FILTER,
        GoogleFinishReason.MALFORMED_FUNCTION_CALL: FinishReason.MALFORMED_TOOL_CALL,
        GoogleFinishReason.UNEXPECTED_TOOL_CALL: FinishReason.MALFORMED_TOOL_CALL,
    }

    unknown: set[GoogleFinishReason] = {
        GoogleFinishReason.FINISH_REASON_UNSPECIFIED,
        GoogleFinishReason.OTHER,
        GoogleFinishReason.NO_IMAGE,
        GoogleFinishReason.IMAGE_OTHER,
    }

    assert set(GoogleFinishReason) == set(handled) | unknown

    for finish_reason in handled:
        info = map_google_finish_reason(finish_reason, False)
        _assert_finish_reason_info(
            info,
            expected_reason=handled[finish_reason],
            expected_raw=finish_reason.name,
        )

    for finish_reason in unknown:
        info = map_google_finish_reason(finish_reason, False)
        _assert_finish_reason_info(
            info,
            expected_reason=FinishReason.UNKNOWN,
            expected_raw=finish_reason.name,
        )

    info = map_google_finish_reason(None, False)
    _assert_finish_reason_info(
        info,
        expected_reason=FinishReason.UNKNOWN,
        expected_raw=None,
    )

    info = map_google_finish_reason(GoogleFinishReason.STOP, True)
    _assert_finish_reason_info(
        info,
        expected_reason=FinishReason.TOOL_CALLS,
        expected_raw=GoogleFinishReason.STOP.name,
    )


async def test_amazon_finish_reason_all_values():
    stop_reasons = [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "content_filtered",
        "guardrail_intervened",
    ]

    expected: dict[str, FinishReason] = {
        "end_turn": FinishReason.STOP,
        "max_tokens": FinishReason.MAX_TOKENS,
        "stop_sequence": FinishReason.STOP_SEQUENCE,
        "tool_use": FinishReason.TOOL_CALLS,
        "content_filtered": FinishReason.CONTENT_FILTER,
        "guardrail_intervened": FinishReason.GUARDRAIL,
    }

    for stop_reason in stop_reasons:
        info = map_amazon_finish_reason(stop_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=expected[stop_reason],
            expected_raw=stop_reason,
        )


async def test_xai_finish_reason_all_values():
    from xai_sdk.proto.v6 import sample_pb2

    finish_reasons = [
        sample_pb2.FinishReason.Name(x)
        for x in sample_pb2.FinishReason.values()
        if sample_pb2.FinishReason.Name(x)
    ]

    handled: dict[str, FinishReason] = {
        "REASON_MAX_LEN": FinishReason.MAX_TOKENS,
        "REASON_MAX_CONTEXT": FinishReason.MAX_TOKENS,
        "REASON_STOP": FinishReason.STOP,
        "REASON_TOOL_CALLS": FinishReason.TOOL_CALLS,
        "REASON_TIME_LIMIT": FinishReason.MAX_TOKENS,
    }

    unknown = {"REASON_INVALID"}

    assert set(finish_reasons) == set(handled) | unknown

    for finish_reason in handled:
        info = map_xai_finish_reason(finish_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=handled[finish_reason],
            expected_raw=finish_reason,
        )

    for finish_reason in unknown:
        info = map_xai_finish_reason(finish_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=FinishReason.UNKNOWN,
            expected_raw=finish_reason,
        )


async def test_mistral_finish_reason_all_values():
    from mistralai.models import FinishReason as MistralFinishReason

    union_args = typing.get_args(MistralFinishReason)
    literal_values = list(typing.get_args(union_args[0]))

    expected: dict[str, FinishReason] = {
        "stop": FinishReason.STOP,
        "length": FinishReason.MAX_TOKENS,
        "model_length": FinishReason.MAX_TOKENS,
        "error": FinishReason.ERROR,
        "tool_calls": FinishReason.TOOL_CALLS,
    }

    assert set(literal_values) == set(expected)

    for finish_reason in literal_values:
        info = map_mistral_finish_reason(finish_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=expected[finish_reason],
            expected_raw=finish_reason,
        )


async def test_ai21_finish_reason_all_values():
    finish_reasons = ["stop", "length", "content_filter"]

    expected: dict[str, FinishReason] = {
        "stop": FinishReason.STOP,
        "length": FinishReason.MAX_TOKENS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }

    for finish_reason in finish_reasons:
        info = map_ai21_finish_reason(finish_reason)
        _assert_finish_reason_info(
            info,
            expected_reason=expected[finish_reason],
            expected_raw=finish_reason,
        )

    info = map_ai21_finish_reason("stop", has_tool_calls=True)
    _assert_finish_reason_info(
        info,
        expected_reason=FinishReason.TOOL_CALLS,
        expected_raw="stop",
    )
