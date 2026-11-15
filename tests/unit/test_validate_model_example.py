from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

from model_library.base import LLM, TextInput
from model_library.base.output import QueryResult, QueryResultCost, QueryResultMetadata
from model_library.rate_limits import (
    RateLimit,
    RateLimitCapacity,
    RequestRateLimit,
    TokenRateLimit,
)
from examples.validate_model import (
    ConversationProbeResult,
    ProbeValue,
    StructuredAnswer,
    ValidationCase,
    ValidationReport,
    _build_cases,
    _print_report,
    _rate_limit_valid,
    _run_case,
)


class _FakeModel(SimpleNamespace):
    is_router = False

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


def _fake_model(
    *,
    supports_audio: bool = True,
    supports_files: bool = True,
    supports_images: bool = True,
) -> _FakeModel:
    return _FakeModel(
        supports_images=supports_images,
        supports_files=supports_files,
        supports_audio=supports_audio,
        supports_tools=False,
        supports_reasoning=False,
        reasoning=False,
        supports_prompt_caching=False,
        supports_temperature=True,
        supports_output_schema=False,
        provider="test-provider",
        model_name="test-model",
        model_key="test-provider/test-model",
        pricing=None,
    )


def _fake_llm(
    *,
    supports_audio: bool = True,
    supports_files: bool = True,
    supports_images: bool = True,
) -> LLM:
    return cast(
        LLM,
        _fake_model(
            supports_audio=supports_audio,
            supports_files=supports_files,
            supports_images=supports_images,
        ),
    )


def test_validate_model_rate_limit_detail_names_algorithms() -> None:
    valid, detail = _rate_limit_valid(
        RateLimit(
            unix_timestamp=1_700_000_000,
            requests=(RequestRateLimit(limit=30_000, remaining=29_999),),
            tokens=TokenRateLimit(
                total=RateLimitCapacity(limit=180_000_000, remaining=180_000_000)
            ),
        )
    )

    assert valid
    assert detail == (
        "requests=sliding_window=29999/30000; TPM=token_bucket=180000000/180000000"
    )
def test_validate_model_media_and_file_cases_require_semantic_answers() -> None:
    cases = {(case.section, case.name): case for case in _build_cases(_fake_llm())}

    semantic_expectations = {
        ("Images", "base64 transport"): "red",
        ("Images", "upload-id transport"): "red",
        ("Images", "URL transport"): "cat",
        ("Files", "base64 transport"): "pineapple",
        ("Files", "upload-id transport"): "pineapple",
        ("Files", "URL transport"): "sample",
        ("Audio", "bytes transport"): "tone",
    }

    for case_key, expected in semantic_expectations.items():
        predicate = cases[case_key].predicate

        passed, detail = predicate(_query_result("unrelated but non-empty"))
        assert not passed
        assert detail is not None
        assert expected in detail

        passed, detail = predicate(_query_result(f"The answer mentions {expected}."))
        assert passed, detail


def _passes(_result: ProbeValue) -> tuple[bool, str | None]:
    return True, None


async def _successful_file_probe() -> QueryResult:
    return _query_result("pineapple")


async def _unsupported_file_probe() -> QueryResult:
    raise NotImplementedError("files unsupported")


async def _erroring_file_probe() -> QueryResult:
    raise RuntimeError("provider rejected the request")


def test_validate_model_runs_all_capability_cases_when_not_declared() -> None:
    cases = _build_cases(
        _fake_llm(
            supports_audio=False,
            supports_files=False,
            supports_images=False,
        )
    )

    undeclared = [case for case in cases if case.expected == "not_declared"]
    sections = {case.section for case in undeclared}
    assert {"Images", "Files", "Audio", "Agent"} <= sections
    for case in undeclared:
        assert case.declared is False


def _turns(*turns: QueryResult) -> ConversationProbeResult:
    return ConversationProbeResult(turns=list(turns))


def _reasoned(text: str, reasoning_tokens: int) -> QueryResult:
    result = _query_result(text)
    result.metadata.reasoning_tokens = reasoning_tokens
    return result


def test_validate_model_flags_undeclared_reasoning() -> None:
    cases = {(case.section, case.name): case for case in _build_cases(_fake_llm())}
    case = cases[("Reasoning", "undeclared reasoning")]
    assert case.declared is False
    assert case.runner is not None

    passed, detail = case.predicate(_turns(_query_result("352")))
    assert passed, detail

    reasoning_result = _query_result("352")
    reasoning_result.metadata.reasoning_tokens = 7
    passed, detail = case.predicate(_turns(_query_result("42"), reasoning_result))
    assert not passed
    assert detail is not None
    assert "reasoning_model is false" in detail


def test_validate_model_reasoning_fails_on_turn_one_and_warns_on_later_turns() -> None:
    model = _fake_llm()
    model.reasoning = True
    cases = {(case.section, case.name): case for case in _build_cases(model)}
    evidence = cases[("Reasoning", "evidence")]
    later = cases[("Reasoning", "later turns")]
    assert evidence.runner is later.runner
    assert (evidence.severity, later.severity) == ("fail", "warn")

    dropped = _turns(_reasoned("157", 5), _reasoned("352", 0))
    assert evidence.predicate(dropped)[0]
    passed, detail = later.predicate(dropped)
    assert not passed
    assert detail is not None and "turn 2: no reasoning" in detail
    passed, detail = evidence.predicate(
        _turns(_reasoned("157", 0), _reasoned("352", 5))
    )
    assert not passed
    assert detail is not None and "turn 1: no reasoning" in detail
    passed, detail = evidence.predicate(_turns(_reasoned("157", 5), _reasoned("2", 5)))
    assert not passed
    assert detail is not None and "final answer is not 352" in detail


def test_validate_model_core_fails_when_served_model_differs() -> None:
    model = _fake_llm()
    key_model = _fake_llm()
    cases = {
        (case.section, case.name): case
        for case in _build_cases(model, key_models={"LAB_KEY_2": key_model})
    }

    for case_key in [("Core", "text query"), ("Keys", "LAB_KEY_2")]:
        predicate = cases[case_key].predicate
        result = _query_result("ok")
        assert predicate(result)[0]
        result.metadata.served_model = "test-model-2026-01-01"
        assert predicate(result)[0]
        for served in ["test-model-mini", "other-model"]:
            result.metadata.served_model = served
            passed, detail = predicate(result)
            assert not passed
            assert detail == f"requested test-model but provider served {served}"


def test_validate_model_structured_output_checks_parsed_answer() -> None:
    cases = {(case.section, case.name): case for case in _build_cases(_fake_llm())}
    case = cases[("Structured Output", "output schema")]
    assert case.severity == "warn"

    result = _query_result("{}")
    result.output_parsed = StructuredAnswer(city="Tokyo", temperature_c=21)
    assert case.predicate(result)[0]
    result.output_parsed = StructuredAnswer(city="Osaka", temperature_c=21)
    passed, detail = case.predicate(result)
    assert not passed
    assert detail is not None and "wrong structured answer" in detail


def test_validate_model_fails_when_undeclared_file_example_works() -> None:
    result = asyncio.run(
        _run_case(
            ValidationCase(
                section="Files",
                name="base64 transport",
                expected="not_declared",
                severity="fail",
                runner=_successful_file_probe,
                predicate=_passes,
                declared=False,
                skip_reason="supports.files is false",
            )
        )
    )

    assert result.status == "fail"
    assert result.observed == "worked"
    assert result.reason_code == "config_mismatch"
    assert result.detail == "supports.files is false, but example worked"


def test_validate_model_passes_when_undeclared_file_example_is_unsupported() -> None:
    result = asyncio.run(
        _run_case(
            ValidationCase(
                section="Files",
                name="base64 transport",
                expected="not_declared",
                severity="fail",
                runner=_unsupported_file_probe,
                predicate=_passes,
                declared=False,
                skip_reason="supports.files is false",
            )
        )
    )

    assert result.status == "pass"
    assert result.observed == "not_supported"
    assert result.reason_code == "ok"
    assert result.detail == "supports.files is false and example reported unsupported"


def test_validate_model_passes_when_undeclared_file_example_errors() -> None:
    result = asyncio.run(
        _run_case(
            ValidationCase(
                section="Files",
                name="base64 transport",
                expected="not_declared",
                severity="fail",
                runner=_erroring_file_probe,
                predicate=_passes,
                declared=False,
                skip_reason="supports.files is false",
            )
        )
    )

    assert result.status == "pass"
    assert result.observed == "not_supported"
    assert result.reason_code == "unsupported_ambiguous"
    assert result.error_type == "RuntimeError"


def test_validate_model_passes_when_undeclared_file_example_gives_wrong_answer() -> (
    None
):
    def _fails(_result: ProbeValue) -> tuple[bool, str | None]:
        return False, "answer did not mention 'pineapple'"

    result = asyncio.run(
        _run_case(
            ValidationCase(
                section="Files",
                name="base64 transport",
                expected="not_declared",
                severity="fail",
                runner=_successful_file_probe,
                predicate=_fails,
                declared=False,
                skip_reason="supports.files is false",
            )
        )
    )

    assert result.status == "pass"
    assert result.observed == "not_supported"
    assert result.reason_code == "unsupported_ambiguous"


def test_validate_model_prints_undeclared_file_match_detail(capsys: Any) -> None:
    model = _fake_llm(supports_files=False)
    cases = {(case.section, case.name): case for case in _build_cases(model)}
    case = replace(
        cases[("Files", "base64 transport")],
        runner=_unsupported_file_probe,
    )
    result = asyncio.run(_run_case(case))
    report = ValidationReport(
        model={},
        verdict="PASS",
        summary={"pass": 1, "fail": 0, "warn": 0, "skip": 0},
        results=[result],
    )

    _print_report(report, model)

    output = capsys.readouterr().out
    assert "Files pass" in output
    assert "supports.files is false and example reported unsupported" in output


def test_validate_model_surfaces_ambiguous_undeclared_probes(capsys: Any) -> None:
    model = _fake_llm(supports_images=False)
    result = asyncio.run(
        _run_case(
            ValidationCase(
                section="Images",
                name="base64 transport",
                expected="not_declared",
                severity="fail",
                runner=_erroring_file_probe,
                predicate=_passes,
                declared=False,
                skip_reason="supports_images is false",
            )
        )
    )
    report = ValidationReport(
        model={},
        verdict="PASS",
        summary={"pass": 1, "fail": 0, "warn": 0, "skip": 0},
        results=[result],
    )

    _print_report(report, model)

    output = " ".join(capsys.readouterr().out.split())
    assert "Images pass" in output
    assert "supports_images is false and example errored" in output
    assert "some undeclared capability probes were inconclusive" in output
    assert "All validation checks passed" not in output
