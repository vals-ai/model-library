from __future__ import annotations

import json
import logging

import pytest
from rich.pretty import pretty_repr

from model_library.base.input import (
    FileWithBase64,
    FileWithBytes,
    FileWithId,
    FileWithUrl,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolInput,
    ToolResult,
)
from model_library.base.query_logging import log_query_started
from model_library.base.utils import get_pretty_input_types
from model_library.utils import MAX_LLM_LOG_LENGTH, truncate_str


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_truncate_str_honors_maximum_length() -> None:
    text = "private-prefix-" + "x" * 200 + "-private-suffix"
    marker = " […] "
    remaining = MAX_LLM_LOG_LENGTH - len(marker)
    prefix_length = (remaining + 1) // 2
    suffix_length = remaining - prefix_length

    preview = truncate_str(text)

    assert preview == text[:prefix_length] + marker + text[-suffix_length:]
    assert len(preview) == MAX_LLM_LOG_LENGTH


def test_text_and_url_sanitization_choose_preview_or_length() -> None:
    long_text = "private-prefix-" + "x" * 200 + "-private-suffix"
    items = [
        (SystemInput(text=long_text), "text"),
        (TextInput(text=long_text), "text"),
        (
            FileWithUrl(
                type="file",
                name="document.pdf",
                mime="application/pdf",
                url=long_text,
            ),
            "url",
        ),
    ]

    for item, field in items:
        length_summary = item.sanitize_content()
        preview_summary = item.sanitize_content(show_content=True)

        assert field not in length_summary
        assert length_summary[f"{field}_length"] == len(long_text)
        assert preview_summary[field] == truncate_str(long_text)
        assert f"{field}_length" not in preview_summary


def test_binary_provider_file_and_raw_sanitization_never_preview_content() -> None:
    base64 = "private-base64-payload"
    bytes_data = b"private-bytes-payload"
    raw_input = {"messages": ["private raw input"]}
    raw_response = ["private raw response"]
    base64_file = FileWithBase64(
        type="image",
        name="image.png",
        mime="image/png",
        base64=base64,
    )
    bytes_file = FileWithBytes(
        type="file",
        name="audio.wav",
        mime="audio/wav",
        data=bytes_data,
    )
    provider_file = FileWithId(
        type="file",
        name="provider-document.pdf",
        mime="application/pdf",
        file_id="provider-file-1",
    )

    expected_base64 = {
        "kind": "file_base",
        "type": "image",
        "name": "image.png",
        "mime": "image/png",
        "append_type": "base64",
        "base64_length": len(base64),
    }
    expected_bytes = {
        "kind": "file_base",
        "type": "file",
        "name": "audio.wav",
        "mime": "audio/wav",
        "append_type": "bytes",
        "data_length": len(bytes_data),
    }
    assert base64_file.sanitize_content() == expected_base64
    assert base64_file.sanitize_content(show_content=True) == expected_base64
    assert bytes_file.sanitize_content() == expected_bytes
    assert bytes_file.sanitize_content(show_content=True) == expected_bytes
    assert provider_file.sanitize_content() == provider_file.model_dump(mode="json")
    assert provider_file.sanitize_content(
        show_content=True
    ) == provider_file.model_dump(mode="json")

    expected_raw_input = {
        "kind": "raw_input",
        "input_length": len(_compact_json(raw_input)),
    }
    expected_raw_response = {
        "kind": "raw_response",
        "response_length": len(_compact_json(raw_response)),
    }
    raw_input_item = RawInput(input=raw_input)
    raw_response_item = RawResponse(response=raw_response)
    assert raw_input_item.sanitize_content() == expected_raw_input
    assert raw_input_item.sanitize_content(show_content=True) == expected_raw_input
    assert raw_response_item.sanitize_content() == expected_raw_response
    assert (
        raw_response_item.sanitize_content(show_content=True) == expected_raw_response
    )


def test_tool_sanitization_choose_preview_or_length() -> None:
    args = {"query": "private tool argument"}
    result = {"content": "private tool result"}
    tool_call = ToolCall(
        id="tool-call-1",
        call_id="provider-call-1",
        name="search",
        args=args,
        sequence=2,
    )
    tool_result = ToolResult(tool_call=tool_call, result=result)
    tool_body = ToolBody(
        name="search",
        description="private tool description",
        properties={"query": {"type": "string"}},
        required=["query"],
    )
    tool_definition = ToolDefinition(name="search", body=tool_body)

    call_length_summary = tool_call.sanitize_content()
    call_preview_summary = tool_call.sanitize_content(show_content=True)
    assert call_length_summary["args_length"] == len(_compact_json(args))
    assert "args" not in call_length_summary
    assert "parsed_args" not in call_length_summary
    assert call_preview_summary["args"] == truncate_str(_compact_json(args))
    assert "args_length" not in call_preview_summary
    assert "parsed_args" not in call_preview_summary

    string_args = '{"query":"private string argument"}'
    string_tool_call = ToolCall(id="tool-call-2", name="search", args=string_args)
    assert string_tool_call.sanitize_content()["args_length"] == len(string_args)
    assert string_tool_call.sanitize_content(show_content=True)["args"] == truncate_str(
        string_args
    )

    result_length_summary = tool_result.sanitize_content()
    result_preview_summary = tool_result.sanitize_content(show_content=True)
    assert result_length_summary["tool_call"] == call_length_summary
    assert result_length_summary["result_length"] == len(_compact_json(result))
    assert "result" not in result_length_summary
    assert result_preview_summary["tool_call"] == call_preview_summary
    assert result_preview_summary["result"] == truncate_str(_compact_json(result))
    assert "result_length" not in result_preview_summary

    assert tool_body.sanitize_content() == tool_body.model_dump(mode="json")
    body_json = _compact_json(tool_body.model_dump(mode="json"))
    tool_length_summary = {
        "name": "search",
        "body_length": len(body_json),
    }
    assert tool_definition.sanitize_content() == tool_length_summary
    tool_preview = {
        "name": "search",
        "body": truncate_str(body_json),
    }
    assert tool_definition.sanitize_content(show_content=True) == tool_preview

    mapping_body = {"description": "private mapping definition"}
    mapping_definition = ToolDefinition(name="notify", body=mapping_body)
    mapping_json = _compact_json(mapping_body)
    mapping_length_summary = {
        "name": "notify",
        "body_length": len(mapping_json),
    }
    mapping_preview = {
        "name": "notify",
        "body": truncate_str(mapping_json),
    }
    assert mapping_definition.sanitize_content() == mapping_length_summary
    assert mapping_definition.sanitize_content(show_content=True) == mapping_preview

    tool_input = ToolInput(tools=[tool_definition, mapping_definition])
    assert tool_input.sanitize_content() == {
        "tools": [tool_length_summary, mapping_length_summary]
    }
    assert tool_input.sanitize_content(show_content=True) == {
        "tools": [tool_preview, mapping_preview]
    }


def test_rich_repr_and_query_logging_use_content_preview() -> None:
    text = "private-prefix-" + "x" * 200 + "-private-suffix"

    item = TextInput(text=text)
    rendered_values = [
        pretty_repr(item, max_string=None),
        get_pretty_input_types([item], verbose=False),
        get_pretty_input_types([item], verbose=True),
    ]

    for rendered in rendered_values:
        assert truncate_str(text) in rendered
        assert text not in rendered
        assert "text_length" not in rendered


def test_query_started_log_uses_sanitized_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    prompt = "private-prompt-prefix-" + "x" * 200 + "-private-prompt-suffix"
    tool_output = "private-tool-prefix-" + "y" * 200 + "-private-tool-suffix"
    text_input = TextInput(text=prompt)
    tool_result = ToolResult(
        tool_call=ToolCall(
            id="tool-call-1",
            name="search",
            args={"query": prompt},
        ),
        result=tool_output,
    )
    tool_definition = ToolDefinition(name="search", body={"description": prompt})
    logger = logging.getLogger("test.sanitize-content")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_query_started(
            logger,
            input=[text_input, tool_result],
            all_input=[text_input, tool_result],
            history=[],
            tools=[tool_definition],
            kwargs={},
            info_enabled=True,
            debug_enabled=False,
        )

    rendered = caplog.records[-1].getMessage()
    assert truncate_str(prompt) in rendered
    assert truncate_str(tool_output) in rendered
    assert prompt not in rendered
    assert tool_output not in rendered
