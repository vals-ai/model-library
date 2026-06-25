# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, Field
from rich.console import Console

from model_library.agent import AgentResult
from model_library.base import LLM, SystemInput, TextInput
from model_library.base.delegate_only import DelegateOnlyException
from model_library.base.output import QueryResult, QueryResultMetadata, RateLimit
from model_library.exceptions import (
    BadInputError,
    GatewayMethodNotSupported,
    ToolCallingNotSupportedError,
)

from model_library.registry_utils import get_registry_model

from examples.quickstart import basic_agent
from examples.inputs import file_base64, file_id, file_url
from examples.inputs import image_base64, image_id, image_url
from examples.setup import setup, sync_model_metadata

Status = Literal["pass", "fail", "warn", "skip"]
Observed = Literal[
    "worked",
    "no_result",
    "errored",
    "semantic_mismatch",
    "not_run",
    "not_supported",
]
ReasonCode = Literal[
    "ok",
    "unsupported_explicit",
    "unsupported_ambiguous",
    "provider_error",
    "timeout",
    "no_result",
    "diagnostic_missing",
    "prerequisite_failed",
    "semantic_mismatch",
]
Expected = Literal[
    "required", "required_if_implemented", "optional", "diagnostic", "not_declared"
]
Severity = Literal["fail", "warn"]


@dataclass(frozen=True, slots=True)
class CacheProbeResult:
    first: QueryResult
    second: QueryResult


@dataclass(frozen=True, slots=True)
class PricingProbeResult:
    pricing: dict[str, object]


ProbeValue = (
    QueryResult | AgentResult | RateLimit | CacheProbeResult | PricingProbeResult | None
)
Probe = Callable[[], Awaitable[ProbeValue]]
Predicate = Callable[[ProbeValue], tuple[bool, str | None]]

QUIET_QUERY_LOGGER = logging.getLogger("llm.validate_model.quiet")
QUIET_QUERY_LOGGER.setLevel(logging.WARNING)
QUIET_AGENT_LOGGER = logging.getLogger("agent.validate_model.quiet")
QUIET_AGENT_LOGGER.propagate = False


@dataclass(frozen=True, slots=True)
class ValidationCase:
    section: str
    name: str
    expected: Expected
    severity: Severity
    runner: Probe | None
    predicate: Predicate
    declared: bool | None = None
    skip_reason: str | None = None
    skip_explicit_unsupported: bool = False


class ValidationResult(BaseModel):
    section: str
    name: str
    expected: Expected
    declared: bool | None
    status: Status
    observed: Observed
    reason_code: ReasonCode
    duration: float
    detail: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: QueryResultMetadata | None = None
    rate_limit: dict[str, object] | None = None


class ValidationReport(BaseModel):
    model: dict[str, object]
    verdict: Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]
    summary: dict[str, int]
    results: list[ValidationResult]
    usage: dict[str, object] = Field(default_factory=dict)
    rate_limit: dict[str, object] = Field(default_factory=dict)


def _text(result: ProbeValue) -> str:
    if isinstance(result, QueryResult):
        query_result = cast(QueryResult, result)  # pyright: ignore[reportUnnecessaryCast]
        return query_result.output_text_str
    if isinstance(result, AgentResult):
        agent_result = cast(AgentResult, result)  # pyright: ignore[reportUnnecessaryCast]
        return agent_result.final_answer
    return ""


def _detail(errors: list[str]) -> str | None:
    return "; ".join(errors) if errors else None


def _query_result_errors(result: ProbeValue, *, require_text: bool = True) -> list[str]:
    if not isinstance(result, QueryResult):
        return ["probe did not return a QueryResult"]

    query_result = cast(QueryResult, result)  # pyright: ignore[reportUnnecessaryCast]
    errors: list[str] = []
    if require_text and not query_result.output_text_str.strip():
        errors.append("no output text returned")
    if query_result.metadata.total_input_tokens <= 0:
        errors.append("no input tokens reported")
    if query_result.metadata.total_output_tokens <= 0:
        errors.append("no output tokens reported")
    return errors


def _text_contains(expected: str) -> Predicate:
    def predicate(result: ProbeValue) -> tuple[bool, str | None]:
        errors = _query_result_errors(result)
        output = _text(result).lower()
        if expected.lower() not in output:
            errors.append(
                f"answer did not mention {expected!r}: {_short(_text(result))}"
            )
        return not errors, _detail(errors)

    return predicate


def _non_empty_text(result: ProbeValue) -> tuple[bool, str | None]:
    errors = _query_result_errors(result)
    return not errors, _detail(errors)


def _agent_tool_use_valid(result: ProbeValue) -> tuple[bool, str | None]:
    if not isinstance(result, AgentResult):
        return False, "agent probe did not return an AgentResult"

    agent_result = cast(AgentResult, result)  # pyright: ignore[reportUnnecessaryCast]
    errors: list[str] = []
    if agent_result.final_error is not None:
        errors.append(
            f"agent ended with {agent_result.final_error.type}: {agent_result.final_error.message}"
        )
    if agent_result.tool_calls_count < 1:
        errors.append("agent did not record any tool calls")
    if not agent_result.final_answer.strip():
        errors.append("agent did not produce a final answer")

    metadata = agent_result.final_aggregated_metadata
    if metadata.total_input_tokens <= 0:
        errors.append("agent did not report input tokens")
    if metadata.total_output_tokens <= 0:
        errors.append("agent did not report output tokens")
    return not errors, _detail(errors)


def _format_rate_limit(rate_limit: RateLimit) -> str:
    return (
        f"RPM={rate_limit.request_remaining}/{rate_limit.request_limit}; "
        f"TPM={rate_limit.token_remaining_total}/{rate_limit.token_limit_total}"
    )


def _format_configured_rate_limit(rate_limit: RateLimit) -> str:
    return f"RPM={rate_limit.request_limit}; TPM={rate_limit.token_limit_total}"


def _format_pricing(pricing: dict[str, object]) -> str:
    return json.dumps(pricing, indent=2)


def _configured_rate_limit(model: LLM) -> RateLimit | None:
    return None


def _rate_limit_valid(result: ProbeValue) -> tuple[bool, str | None]:
    if not isinstance(result, RateLimit):
        return False, "no live rate limit returned"
    rate_limit = cast(RateLimit, result)  # pyright: ignore[reportUnnecessaryCast]
    detail = _format_rate_limit(rate_limit)
    if rate_limit.request_limit is None and rate_limit.token_limit_total <= 0:
        return False, f"rate limit returned but no limits were set; {detail}"
    return True, detail


def _configured_rate_limit_valid(result: ProbeValue) -> tuple[bool, str | None]:
    if not isinstance(result, RateLimit):
        return False, "no configured rate limit"
    rate_limit = cast(RateLimit, result)  # pyright: ignore[reportUnnecessaryCast]
    detail = _format_configured_rate_limit(rate_limit)
    if rate_limit.request_limit is None and rate_limit.token_limit_total <= 0:
        return False, f"configured rate limit exists but no limits were set; {detail}"
    return True, detail


def _pricing_valid(result: ProbeValue) -> tuple[bool, str | None]:
    if not isinstance(result, PricingProbeResult):
        return False, "no configured pricing"
    if not result.pricing:
        return False, "no configured pricing"
    return True, _format_pricing(result.pricing)


def _reasoning_evidence_seen(result: ProbeValue) -> tuple[bool, str | None]:
    errors = _query_result_errors(result)
    detail: str | None = None
    if isinstance(result, QueryResult):
        query_result = cast(QueryResult, result)  # pyright: ignore[reportUnnecessaryCast]
        reasoning = query_result.reasoning or ""
        reasoning_tokens = query_result.metadata.reasoning_tokens or 0
        reasoning_chars = len(reasoning.strip())
        detail = (
            f"reasoning_tokens={reasoning_tokens}; reasoning_chars={reasoning_chars}"
        )
        if not reasoning and reasoning_tokens <= 0:
            errors.append("no reasoning content or reasoning tokens returned")
    return not errors, detail if not errors else _detail(errors)


def _cache_read_detail(
    *,
    total_input_tokens: int,
    first_cached_tokens: int | None,
    second_cached_tokens: int | None,
    first_cache_write_tokens: int | None,
    second_cache_write_tokens: int | None,
) -> str:
    total_cached_tokens = (first_cached_tokens or 0) + (second_cached_tokens or 0)
    total_cache_write_tokens = (first_cache_write_tokens or 0) + (
        second_cache_write_tokens or 0
    )
    return (
        f"total_input_tokens={total_input_tokens}\n"
        "cached_tokens: "
        f"first={first_cached_tokens or 0}; "
        f"second={second_cached_tokens or 0}; "
        f"total={total_cached_tokens}\n"
        "cache_write_tokens: "
        f"first={first_cache_write_tokens or 0}; "
        f"second={second_cache_write_tokens or 0}; "
        f"total={total_cache_write_tokens}"
    )


def _cache_read_seen(result: ProbeValue) -> tuple[bool, str | None]:
    if isinstance(result, CacheProbeResult):
        errors = _query_result_errors(result.second)
        first_metadata = result.first.metadata
        second_metadata = result.second.metadata
        total_metadata = first_metadata + second_metadata
        cached_tokens = second_metadata.cache_read_tokens or 0
        detail = _cache_read_detail(
            total_input_tokens=total_metadata.total_input_tokens,
            first_cached_tokens=first_metadata.cache_read_tokens,
            second_cached_tokens=second_metadata.cache_read_tokens,
            first_cache_write_tokens=first_metadata.cache_write_tokens,
            second_cache_write_tokens=second_metadata.cache_write_tokens,
        )
        if cached_tokens <= 0:
            errors.append(detail)
        return not errors, detail if not errors else _detail(errors)

    errors = _query_result_errors(result)
    if not isinstance(result, QueryResult):
        return False, _detail(errors)

    query_result = cast(QueryResult, result)  # pyright: ignore[reportUnnecessaryCast]
    metadata = query_result.metadata
    cached_tokens = metadata.cache_read_tokens or 0
    detail = _cache_read_detail(
        total_input_tokens=metadata.total_input_tokens,
        first_cached_tokens=None,
        second_cached_tokens=metadata.cache_read_tokens,
        first_cache_write_tokens=None,
        second_cache_write_tokens=metadata.cache_write_tokens,
    )
    if cached_tokens <= 0:
        errors.append(detail)
    return not errors, detail if not errors else _detail(errors)


def _short(value: str, max_length: int = 160) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3]}..."


def _metadata_from_probe(result: ProbeValue) -> QueryResultMetadata | None:
    if isinstance(result, CacheProbeResult):
        return result.first.metadata + result.second.metadata
    if isinstance(result, QueryResult):
        query_result = cast(QueryResult, result)  # pyright: ignore[reportUnnecessaryCast]
        return query_result.metadata
    if isinstance(result, AgentResult):
        agent_result = cast(AgentResult, result)  # pyright: ignore[reportUnnecessaryCast]
        return agent_result.final_aggregated_metadata
    return None


def _rate_limit_from_probe(result: ProbeValue) -> dict[str, object] | None:
    if not isinstance(result, RateLimit):
        return None
    rate_limit = cast(RateLimit, result)  # pyright: ignore[reportUnnecessaryCast]
    return cast(dict[str, object], rate_limit.model_dump(mode="json", exclude={"raw"}))


def _is_explicit_unsupported(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            NotImplementedError,
            DelegateOnlyException,
            GatewayMethodNotSupported,
            ToolCallingNotSupportedError,
        ),
    ):
        return True
    if isinstance(exc, BadInputError) and "does not support" in str(exc).lower():
        return True
    return False


async def _run_case(case: ValidationCase) -> ValidationResult:
    if case.runner is None:
        return ValidationResult(
            section=case.section,
            name=case.name,
            expected=case.expected,
            declared=case.declared,
            status="skip",
            observed="not_supported",
            reason_code="prerequisite_failed",
            duration=0,
            detail=case.skip_reason or "not declared supported",
        )

    t0 = time.monotonic()
    try:
        result = await case.runner()
        duration = time.monotonic() - t0
        if result is None:
            detail = "example returned None"
            if case.section == "Rate Limit":
                _, predicate_detail = case.predicate(result)
                detail = predicate_detail or detail
            return ValidationResult(
                section=case.section,
                name=case.name,
                expected=case.expected,
                declared=case.declared,
                status="fail" if case.severity == "fail" else "warn",
                observed="no_result",
                reason_code="no_result",
                duration=duration,
                detail=detail,
            )

        metadata = _metadata_from_probe(result)
        rate_limit = _rate_limit_from_probe(result)
        passed, detail = case.predicate(result)
        if passed:
            return ValidationResult(
                section=case.section,
                name=case.name,
                expected=case.expected,
                declared=case.declared,
                status="pass",
                observed="worked",
                reason_code="ok",
                duration=duration,
                detail=detail,
                metadata=metadata,
                rate_limit=rate_limit,
            )

        return ValidationResult(
            section=case.section,
            name=case.name,
            expected=case.expected,
            declared=case.declared,
            status="fail" if case.severity == "fail" else "warn",
            observed="semantic_mismatch",
            reason_code="semantic_mismatch",
            duration=duration,
            detail=detail,
            metadata=metadata,
            rate_limit=rate_limit,
        )
    except Exception as e:
        duration = time.monotonic() - t0
        if _is_explicit_unsupported(e):
            return ValidationResult(
                section=case.section,
                name=case.name,
                expected=case.expected,
                declared=case.declared,
                status=(
                    "skip"
                    if case.skip_explicit_unsupported or case.severity == "warn"
                    else "fail"
                ),
                observed="not_supported",
                reason_code="unsupported_explicit",
                duration=duration,
                error_type=type(e).__name__,
                error_message=str(e),
            )

        return ValidationResult(
            section=case.section,
            name=case.name,
            expected=case.expected,
            declared=case.declared,
            status="fail" if case.severity == "fail" else "warn",
            observed="errored",
            reason_code="provider_error",
            duration=duration,
            error_type=type(e).__name__,
            error_message=str(e),
        )


def _capability_case(
    *,
    section: str,
    name: str,
    declared: bool,
    severity: Severity,
    runner: Probe,
    predicate: Predicate,
    capability_name: str | None = None,
    skip_explicit_unsupported: bool = False,
) -> ValidationCase:
    if declared:
        return ValidationCase(
            section=section,
            name=name,
            expected=(
                "required_if_implemented"
                if skip_explicit_unsupported
                else "required"
                if severity == "fail"
                else "optional"
            ),
            severity=severity,
            runner=runner,
            predicate=predicate,
            declared=declared,
            skip_explicit_unsupported=skip_explicit_unsupported,
        )
    return ValidationCase(
        section=section,
        name=name,
        expected="not_declared",
        severity=severity,
        runner=None,
        predicate=predicate,
        declared=declared,
        skip_reason=f"{capability_name or f'supports_{section.lower()}'} is false",
    )


def _build_cases(model: LLM) -> list[ValidationCase]:
    async def core_probe() -> QueryResult:
        return await model.query(
            [TextInput(text="Reply with exactly: ok")], logger=QUIET_QUERY_LOGGER
        )

    async def reasoning_probe() -> QueryResult:
        return await model.query(
            [
                TextInput(
                    text=(
                        "Use your reasoning capability to solve this, then answer "
                        "with only the number: the smallest positive integer n "
                        "where n mod 5 = 2, n mod 7 = 3, and n mod 9 = 4."
                    )
                )
            ]
        )

    async def caching_probe() -> CacheProbeResult:
        cache_prefix = (
            "Reusable validation context for prompt cache measurement. " * 800
        ).strip()
        prompt = [
            SystemInput(text=cache_prefix),
            TextInput(text="Reply with exactly: cached."),
        ]
        first = await model.query(prompt, logger=QUIET_QUERY_LOGGER)
        second = await model.query(prompt, logger=QUIET_QUERY_LOGGER)
        return CacheProbeResult(first=first, second=second)

    async def rate_limit_probe() -> RateLimit | None:
        return await model.get_rate_limit()

    async def configured_rate_limit_probe() -> RateLimit | None:
        return _configured_rate_limit(model)

    async def pricing_probe() -> PricingProbeResult:
        return PricingProbeResult(pricing=_pricing_summary(model))

    def semantic_transport_case(
        *,
        section: str,
        name: str,
        declared: bool,
        runner: Probe,
        expected_text: str,
        skip_explicit_unsupported: bool = False,
    ) -> ValidationCase:
        return _capability_case(
            section=section,
            name=name,
            declared=declared,
            severity="fail",
            runner=runner,
            predicate=_text_contains(expected_text),
            skip_explicit_unsupported=skip_explicit_unsupported,
        )

    cases = [
        ValidationCase(
            section="Core",
            name="text query",
            expected="required",
            severity="fail",
            runner=core_probe,
            predicate=_non_empty_text,
        ),
        semantic_transport_case(
            section="Images",
            name="base64 transport",
            declared=model.supports_images,
            runner=lambda: image_base64(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="red",
        ),
        semantic_transport_case(
            section="Images",
            name="upload-id transport",
            declared=model.supports_images,
            runner=lambda: image_id(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="red",
            skip_explicit_unsupported=True,
        ),
        semantic_transport_case(
            section="Images",
            name="URL transport",
            declared=model.supports_images,
            runner=lambda: image_url(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="cat",
            skip_explicit_unsupported=True,
        ),
        semantic_transport_case(
            section="Files",
            name="base64 transport",
            declared=model.supports_files,
            runner=lambda: file_base64(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="pineapple",
        ),
        semantic_transport_case(
            section="Files",
            name="upload-id transport",
            declared=model.supports_files,
            runner=lambda: file_id(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="pineapple",
            skip_explicit_unsupported=True,
        ),
        semantic_transport_case(
            section="Files",
            name="URL transport",
            declared=model.supports_files,
            runner=lambda: file_url(
                model,
                quiet=True,
                raise_errors=True,
                logger=QUIET_QUERY_LOGGER,
            ),
            expected_text="sample",
            skip_explicit_unsupported=True,
        ),
        _capability_case(
            section="Agent",
            name="bounded tool use",
            declared=model.supports_tools,
            severity="fail",
            runner=lambda: basic_agent(
                model,
                quiet=True,
                raise_errors=True,
                max_turns=5,
                max_seconds=90,
                logger=QUIET_AGENT_LOGGER,
            ),
            predicate=_agent_tool_use_valid,
            capability_name="supports_tools",
        ),
    ]

    if model.reasoning:
        cases.append(
            ValidationCase(
                section="Reasoning",
                name="evidence",
                expected="required",
                severity="fail",
                runner=reasoning_probe,
                predicate=_reasoning_evidence_seen,
                declared=model.reasoning,
            )
        )

    cases.extend(
        [
            ValidationCase(
                section="Caching",
                name="prompt cache read",
                expected="diagnostic",
                severity="warn",
                runner=caching_probe,
                predicate=_cache_read_seen,
            ),
            ValidationCase(
                section="Rate Limit",
                name="configured limit",
                expected="diagnostic",
                severity="warn",
                runner=configured_rate_limit_probe,
                predicate=_configured_rate_limit_valid,
            ),
            ValidationCase(
                section="Rate Limit",
                name="header check",
                expected="diagnostic",
                severity="warn",
                runner=rate_limit_probe,
                predicate=_rate_limit_valid,
            ),
            ValidationCase(
                section="Pricing",
                name="configured prices",
                expected="diagnostic",
                severity="warn",
                runner=pricing_probe,
                predicate=_pricing_valid,
            ),
        ]
    )
    return cases


async def _run_cases(cases: list[ValidationCase]) -> list[ValidationResult]:
    return list(await asyncio.gather(*(_run_case(case) for case in cases)))


def _summary(results: list[ValidationResult]) -> dict[str, int]:
    return {
        "pass": sum(1 for result in results if result.status == "pass"),
        "fail": sum(1 for result in results if result.status == "fail"),
        "warn": sum(1 for result in results if result.status == "warn"),
        "skip": sum(1 for result in results if result.status == "skip"),
    }


def _pricing_summary(model: LLM) -> dict[str, object]:
    metadata = model.metadata
    if metadata is None or metadata.costs_per_million_token is None:
        return {}
    return metadata.costs_per_million_token.model_dump(mode="json")


def _usage_summary(results: list[ValidationResult], model: LLM) -> dict[str, object]:
    metadata_values = [result.metadata for result in results if result.metadata]
    if not metadata_values:
        return {"price_per_million_tokens": _pricing_summary(model)}

    total = QueryResultMetadata()
    for metadata in metadata_values:
        total = total + metadata

    return {
        "probes_with_metadata": len(metadata_values),
        "tokens": {
            "input": total.in_tokens,
            "output": total.out_tokens,
            "reasoning": total.reasoning_tokens or 0,
            "cache_read": total.cache_read_tokens or 0,
            "cache_write": total.cache_write_tokens or 0,
            "total_input": total.total_input_tokens,
            "total_output": total.total_output_tokens,
        },
        "duration_seconds": total.duration_seconds,
        "price_per_million_tokens": _pricing_summary(model),
    }


def _rate_limit_summary(results: list[ValidationResult]) -> dict[str, object]:
    for result in results:
        if (
            result.section == "Rate Limit"
            and result.name == "header check"
            and result.rate_limit is not None
        ):
            return result.rate_limit
    return {}


def _verdict(
    results: list[ValidationResult],
) -> Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]:
    if any(result.status == "fail" for result in results):
        return "FAIL"
    if any(result.status == "warn" for result in results):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _dump_base_model(value: BaseModel) -> object:
    return value.model_dump(mode="json")


def _safe_dump(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return _dump_base_model(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return {str(k): _safe_dump(v) for k, v in mapping.items()}
    if isinstance(value, list):
        values = cast(list[object], value)
        return [_safe_dump(item) for item in values]
    if isinstance(value, tuple):
        values = cast(tuple[object, ...], value)
        return [_safe_dump(item) for item in values]
    return str(value)


def _model_info(model_key: str, model: LLM) -> dict[str, object]:
    metadata = model.metadata
    registry_key = cast(object, getattr(model, "registry_key", None))
    provider_config = cast(object, model.provider_config)
    return {
        "requested_key": model_key,
        "registry_key": registry_key,
        "provider": model.provider,
        "model_name": model.model_name,
        "max_tokens": model.max_tokens,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "top_k": model.top_k,
        "reasoning": model.reasoning,
        "reasoning_effort": model.reasoning_effort,
        "compute_effort": model.compute_effort,
        "native": model.native,
        "custom_endpoint": bool(model.custom_endpoint),
        "supports": {
            "images": model.supports_images,
            "files": model.supports_files,
            "videos": model.supports_videos,
            "batch": model.supports_batch,
            "temperature": model.supports_temperature,
            "tools": model.supports_tools,
            "output_schema": model.supports_output_schema,
        },
        "metadata": _safe_dump(metadata) if metadata is not None else None,
        "provider_config": _safe_dump(provider_config),
    }


def _print_report(report: ValidationReport, model: LLM) -> None:
    console = Console()
    console.print()
    console.print(model)

    section_counts: dict[str, int] = {}
    for result in report.results:
        section_counts[result.section] = section_counts.get(result.section, 0) + 1

    prev_section = ""
    for result in report.results:
        timing = f" [dim]({result.duration:.1f}s)[/dim]" if result.duration > 0 else ""
        message = result.detail or result.error_message or result.reason_code
        inline_reason = " ".join(message.split())
        inline_message = (
            f" - {inline_reason}" if result.status in {"fail", "warn"} else ""
        )
        single_row_section = section_counts[result.section] == 1
        if single_row_section:
            console.print()
            console.print(
                f"[bold]{result.section}[/bold] {_status_text(result.status)}"
                f"{timing}{inline_message}"
            )
            detail_prefix = "  "
        else:
            if result.section != prev_section:
                console.print()
                console.print(f"[bold]{result.section}[/bold]")
                prev_section = result.section
            console.print(
                f"  {_status_text(result.status)} {result.name}{timing}{inline_message}"
            )
            detail_prefix = "    "
        if (
            result.status not in {"fail", "warn"}
            and result.section in {"Reasoning", "Caching", "Rate Limit", "Pricing"}
            and result.detail
        ):
            console.print(
                "\n".join(
                    f"{detail_prefix}{line}" for line in result.detail.splitlines()
                )
            )

    problem_rows = [
        result for result in report.results if result.status in {"fail", "warn"}
    ]
    if problem_rows:
        console.print()
        console.print("[bold yellow]Warnings and failures:[/bold yellow]")
        for result in problem_rows:
            message = result.detail or result.error_message or result.reason_code
            console.print(
                f"  [{result.status.upper()}] {result.section} / {result.name}: "
                f"{' '.join(message.split())}"
            )
    else:
        console.print()
        if any(result.status == "skip" for result in report.results):
            console.print("[bold green]No warnings or failures.[/bold green]")
        else:
            console.print("[bold green]All validation checks passed.[/bold green]")


def _status_text(status: Status) -> str:
    if status == "pass":
        return "[green]pass[/green]"
    if status == "fail":
        return "[red]fail[/red]"
    if status == "warn":
        return "[yellow]warn[/yellow]"
    return "[dim]not supported[/dim]"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Validate model capabilities")
    parser.add_argument(
        "model", type=str, help="Model key (e.g. deepseek/deepseek-v4-pro)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of a human-readable table",
    )
    args = parser.parse_args()

    try:
        model = get_registry_model(args.model)
    except Exception as e:
        if args.json:
            print(
                json.dumps(
                    {
                        "verdict": "ERROR",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
            )
        else:
            Console().print(
                f"[bold red]Validation error:[/bold red] {type(e).__name__}: {e}"
            )
        return 2 if "not found in registry" in str(e) else 3

    try:
        await sync_model_metadata(model)
        results = await _run_cases(_build_cases(model))
    except Exception as e:
        if args.json:
            print(
                json.dumps(
                    {
                        "verdict": "ERROR",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                )
            )
        else:
            Console().print(
                f"[bold red]Validation error:[/bold red] {type(e).__name__}: {e}"
            )
        return 3

    report = ValidationReport(
        model=_model_info(args.model, model),
        verdict=_verdict(results),
        summary=_summary(results),
        results=results,
        usage=_usage_summary(results, model),
        rate_limit=_rate_limit_summary(results),
    )

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        _print_report(report, model)

    return 1 if report.verdict == "FAIL" else 0


if __name__ == "__main__":
    setup(disable_logging="--json" in sys.argv)
    raise SystemExit(asyncio.run(main()))
