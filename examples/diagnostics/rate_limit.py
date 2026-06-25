# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
import random
import string
import time
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from model_library import set_logging
from model_library.base import LLM, LLMConfig
from model_library.base.input import SystemInput, TextInput
from model_library.base.output import QueryResult
from model_library.exceptions import MaxContextWindowExceededError, RateLimitException
from model_library.registry_utils import get_registry_model

from examples.setup import console_log, setup

set_logging(False)

rich_console = Console()


TARGET_TOKENS = 40_000
# "word " is 1 token in cl100k_base and DeepSeek's tokenizer.
# ~22 tokens overhead from system message + random prefix + framing.
PROMPT_OVERHEAD_TOKENS = 22

MAX_BATCH_REQUESTS = 2000  # safety cap: 2000 * 40k = 80M tokens
DEFAULT_MAX_TOKEN_BUDGET = 20_000_000
DEFAULT_MAX_CONCURRENCY = 50
DEFAULT_MAX_UNKNOWN_ERRORS = 3
DEFAULT_MAX_ROUND_SECONDS = 60.0


class ProbeOutcomeKind(StrEnum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    UNKNOWN_ERROR = "unknown_error"
    FATAL_ERROR = "fatal_error"


@dataclass(frozen=True)
class ProbeOutcome:
    kind: ProbeOutcomeKind
    tokens: int = 0
    error_type: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class ProbeConfig:
    rounds: int = 2
    target_tokens: int = TARGET_TOKENS
    max_requests: int = MAX_BATCH_REQUESTS
    max_token_budget: int | None = DEFAULT_MAX_TOKEN_BUDGET
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_unknown_errors: int = DEFAULT_MAX_UNKNOWN_ERRORS
    max_round_seconds: float = DEFAULT_MAX_ROUND_SECONDS
    refill_wait_seconds: int = 60
    overshoot_multiplier: float = 1.2


@dataclass
class ProbeReport:
    status: str
    tpm_estimate: int
    tokens_per_request: int
    calibration_tokens: int
    elapsed_seconds: float
    issued_requests: int
    settled_requests: int
    success_requests: int
    rate_limit_requests: int
    unknown_error_requests: int
    fatal_error_requests: int
    success_tokens: int
    tpm_observed_upper_estimate: int
    unknown_errors: list[str] = field(default_factory=list)


def _random_padding(target_tokens: int = TARGET_TOKENS) -> str:
    """Generate a unique prompt to avoid cache hits."""
    prefix = "".join(random.choices(string.ascii_lowercase, k=20)) + " "
    repeats = max(1, target_tokens - PROMPT_OVERHEAD_TOKENS)
    return prefix + "word " * repeats


def _no_retry(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Passthrough retrier that disables backoff/token retry wrappers."""
    return func


def _make_input(target_tokens: int = TARGET_TOKENS) -> list[SystemInput | TextInput]:
    return [
        SystemInput(
            text="Do not reason. Do not think. Do not explain. Reply exactly: ok."
        ),
        TextInput(text=_random_padding(target_tokens)),
    ]


def _exception_status_code(exception: Exception) -> int | None:
    response = getattr(exception, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status

    status = getattr(exception, "status_code", None)
    if isinstance(status, int):
        return status

    return None


def _is_rate_limit_error(exception: Exception) -> bool:
    if isinstance(exception, RateLimitException):
        return True

    if _exception_status_code(exception) == 429:
        return True

    code = getattr(exception, "code", None)
    if isinstance(code, str) and code.lower() in {
        "rate_limit",
        "rate_limit_exceeded",
        "too_many_requests",
        "throttling",
        "throttled",
    }:
        return True

    message = str(exception).lower()
    return any(
        signal in message
        for signal in (
            "rate limit",
            "rate_limit",
            "too many requests",
            "throttling",
            "throttled",
        )
    )


def _is_fatal_probe_error(exception: Exception) -> bool:
    if isinstance(exception, MaxContextWindowExceededError):
        return True

    status = _exception_status_code(exception)
    if status in {400, 401, 403, 404}:
        return True

    message = str(exception).lower()
    return any(
        signal in message
        for signal in (
            "api key",
            "authentication",
            "authorization",
            "unauthorized",
            "forbidden",
            "context window",
            "maximum context",
            "not found",
        )
    )


def _classify_probe_exception(exception: Exception) -> ProbeOutcome:
    if _is_rate_limit_error(exception):
        return ProbeOutcome(
            kind=ProbeOutcomeKind.RATE_LIMITED,
            error_type=type(exception).__name__,
            message=str(exception),
        )

    if _is_fatal_probe_error(exception):
        return ProbeOutcome(
            kind=ProbeOutcomeKind.FATAL_ERROR,
            error_type=type(exception).__name__,
            message=str(exception),
        )

    return ProbeOutcome(
        kind=ProbeOutcomeKind.UNKNOWN_ERROR,
        error_type=type(exception).__name__,
        message=str(exception),
    )


def _tokens_from_result(result: QueryResult, fallback: int) -> int:
    tokens = result.metadata.total_input_tokens + result.metadata.total_output_tokens
    return tokens or fallback


def _max_requests_for_budget(
    config: ProbeConfig, tokens_per_request: int, *, spent_tokens: int = 0
) -> int:
    if config.max_token_budget is None:
        return config.max_requests
    remaining_budget = max(0, config.max_token_budget - spent_tokens)
    return min(config.max_requests, remaining_budget // tokens_per_request)


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def _format_int(value: int | None) -> str:
    return "unknown" if value is None else f"{value:,}"


def _format_estimate_range(report: ProbeReport) -> str:
    if report.tpm_observed_upper_estimate > report.tpm_estimate:
        return f"~{report.tpm_estimate:,}-~{report.tpm_observed_upper_estimate:,}"
    suffix = "+" if report.status == "lower_bound" else ""
    return f"~{report.tpm_estimate:,}{suffix}"


def _live_round_estimate(
    *,
    success_tokens: int,
    rate_limits: int,
    elapsed_seconds: float,
    max_concurrency: int,
    issued_requests: int,
    tokens_per_request: int,
) -> str:
    if success_tokens <= 0:
        return "estimate: waiting"

    refill_corrected_tpm = int(success_tokens / (1 + elapsed_seconds / 60))
    overshoot_window = min(max_concurrency, issued_requests) * tokens_per_request
    overshoot_adjusted_tpm = max(0, success_tokens - overshoot_window)
    if rate_limits > 0:
        lower = min(refill_corrected_tpm, overshoot_adjusted_tpm)
        return f"estimate: ~{lower:,}-~{success_tokens:,}"

    return f"lower bound: ~{success_tokens:,}+"


def _round_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed:>4.0f}/{task.total:.0f} settled"),
        TextColumn("{task.fields[requests]}"),
        TextColumn("{task.fields[tokens]}"),
        TextColumn("{task.fields[estimate]}"),
        TimeElapsedColumn(),
        console=rich_console,
    )


def _probe_plan_panel(
    config: ProbeConfig,
    first_round_requests: int,
    first_round_tokens: int,
) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("probe token budget", _format_int(config.max_token_budget))
    table.add_row("tokens/request", f"~{config.target_tokens:,}")
    table.add_row("max concurrency", str(config.max_concurrency))
    table.add_row("round timeout", f"{config.max_round_seconds:g}s wall-clock cap")
    table.add_row(
        "rounds", f"{config.rounds} (stress, then midpoint refinement if bounded)"
    )
    table.add_row("first-round requests", str(first_round_requests))
    table.add_row("first-round tokens", f"~{first_round_tokens:,}")
    caption = (
        "Probe mode skips provider rate-limit preflight and estimates TPM from "
        "observed successful tokens and 429s."
    )
    return Panel(table, title="TPM probe plan", subtitle=caption, border_style="cyan")


def _confirm_quota_probe() -> bool:
    try:
        answer = input("Proceed with quota-consuming probe traffic? [y/N] ")
    except EOFError:
        console_log(
            "Refusing to start probe: no confirmation received.", color="yellow"
        )
        return False

    if answer.strip().lower() in {"y", "yes"}:
        return True

    console_log("Probe cancelled; no quota-consuming traffic was sent.", color="yellow")
    return False


def _round_summary_table(
    *,
    round_num: int,
    round_mode: str,
    issued: int,
    settled: int,
    successes: int,
    rate_limits: int,
    unknown_errors: int,
    fatal_errors: int,
    success_tokens: int,
    refill_corrected_tpm: int,
    overshoot_adjusted_tpm: int,
    observed_upper: int,
    tpm_estimate: int,
    status: str,
    selected_estimate: str,
) -> Table:
    table = Table(title=f"Round {round_num} {round_mode} summary")
    table.add_column("metric", style="bold cyan")
    table.add_column("value", justify="right")
    table.add_row("issued / settled", f"{issued:,} / {settled:,}")
    table.add_row("success / 429", f"{successes:,} / {rate_limits:,}")
    table.add_row("unknown / fatal errors", f"{unknown_errors:,} / {fatal_errors:,}")
    table.add_row("successful tokens", f"~{success_tokens:,}")
    table.add_row("refill-corrected floor", f"~{refill_corrected_tpm:,}")
    table.add_row("concurrency-adjusted floor", f"~{overshoot_adjusted_tpm:,}")
    if observed_upper:
        table.add_row("observed upper signal", f"~{observed_upper:,}")
    table.add_row("tightened estimate", f"~{tpm_estimate:,}")
    table.add_row("selected estimate", selected_estimate)
    table.add_row("status", status)
    return table


def _validate_probe_config(config: ProbeConfig) -> None:
    if config.rounds < 1:
        raise ValueError("rounds must be at least 1")
    if config.max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    if config.target_tokens < 1:
        raise ValueError("target_tokens must be at least 1")
    if config.max_requests < 1:
        raise ValueError("max_requests must be at least 1")
    if config.max_token_budget is not None and config.max_token_budget < 0:
        raise ValueError("max_token_budget must be non-negative")
    if config.max_unknown_errors < 0:
        raise ValueError("max_unknown_errors must be non-negative")


def _skipped_probe_report(config: ProbeConfig) -> ProbeReport:
    return ProbeReport(
        status="skipped",
        tpm_estimate=0,
        tokens_per_request=config.target_tokens,
        calibration_tokens=0,
        elapsed_seconds=0,
        issued_requests=0,
        settled_requests=0,
        success_requests=0,
        rate_limit_requests=0,
        unknown_error_requests=0,
        fatal_error_requests=0,
        success_tokens=0,
        tpm_observed_upper_estimate=0,
    )


async def probe_rate_limit(model: LLM, rounds: int | ProbeConfig) -> ProbeReport:
    """Estimate TPM by sending bounded requests until rate limited.

    WARNING: This will deplete TPM quota for the current rate limit window.
    Providers without a token-per-minute limit should be skipped before calling
    this function.
    """
    config = ProbeConfig(rounds=rounds) if isinstance(rounds, int) else rounds
    _validate_probe_config(config)

    if (
        config.max_token_budget is not None
        and config.max_token_budget < config.target_tokens
    ):
        console_log(
            "Skipping probe: max-token-budget is below one calibration request.",
            color="yellow",
        )
        return _skipped_probe_report(config)

    previous_custom_retrier = model.custom_retrier
    # Disable backoff/token retry wrappers so rate limit errors propagate without
    # retrying through those layers. LLM.query may still perform immediate
    # transient retries before the custom retrier boundary.
    model.custom_retrier = _no_retry

    try:
        console_log("Calibrating token count (consumes one probe-sized request)...")
        calibration_tokens = 0
        cached = 0
        try:
            calibration_result = await model.query(_make_input(config.target_tokens))
            calibration_tokens = _tokens_from_result(
                calibration_result, config.target_tokens
            )
            cached = calibration_result.metadata.cache_read_tokens or 0
        except Exception as e:
            outcome = _classify_probe_exception(e)
            console_log(
                f"Calibration failed: {outcome.kind} {outcome.error_type}: {outcome.message}",
                color="yellow",
            )
            return ProbeReport(
                status="failed",
                tpm_estimate=0,
                tokens_per_request=config.target_tokens,
                calibration_tokens=0,
                elapsed_seconds=0,
                issued_requests=0,
                settled_requests=0,
                success_requests=0,
                rate_limit_requests=1
                if outcome.kind == ProbeOutcomeKind.RATE_LIMITED
                else 0,
                unknown_error_requests=1
                if outcome.kind == ProbeOutcomeKind.UNKNOWN_ERROR
                else 0,
                fatal_error_requests=1
                if outcome.kind == ProbeOutcomeKind.FATAL_ERROR
                else 0,
                success_tokens=0,
                tpm_observed_upper_estimate=0,
                unknown_errors=[f"{outcome.error_type}: {outcome.message}"]
                if outcome.kind != ProbeOutcomeKind.RATE_LIMITED
                else [],
            )

        tokens_per_request = calibration_tokens or config.target_tokens
        console_log(
            f"Calibration: {tokens_per_request} total tokens, "
            f"{cached} cached (target={config.target_tokens})"
        )
        if cached > 0:
            console_log(
                "WARNING: cache hit detected — estimate will be approximate",
                color="yellow",
            )

        report = ProbeReport(
            status="lower_bound",
            tpm_estimate=0,
            tokens_per_request=tokens_per_request,
            calibration_tokens=calibration_tokens,
            elapsed_seconds=0,
            issued_requests=0,
            settled_requests=0,
            success_requests=0,
            rate_limit_requests=0,
            unknown_error_requests=0,
            fatal_error_requests=0,
            success_tokens=0,
            tpm_observed_upper_estimate=0,
        )

        budget_spent_tokens = calibration_tokens
        status_quality = {"failed": 0, "lower_bound": 1, "approximate": 2, "clean": 3}

        async def fire_one() -> ProbeOutcome:
            try:
                result = await model.query(_make_input(config.target_tokens))
                return ProbeOutcome(
                    kind=ProbeOutcomeKind.SUCCESS,
                    tokens=_tokens_from_result(result, tokens_per_request),
                )
            except Exception as e:
                return _classify_probe_exception(e)

        for round_num in range(1, config.rounds + 1):
            max_to_issue = _max_requests_for_budget(
                config,
                tokens_per_request,
                spent_tokens=budget_spent_tokens,
            )
            round_mode = "stress"
            refinement_target_tokens = 0
            if report.tpm_observed_upper_estimate > report.tpm_estimate:
                round_mode = "refine"
                refinement_target_tokens = (
                    report.tpm_estimate + report.tpm_observed_upper_estimate
                ) // 2
                max_to_issue = min(
                    max_to_issue,
                    max(1, _ceil_div(refinement_target_tokens, tokens_per_request)),
                )
            elif report.tpm_estimate > 0:
                overshoot_tokens = int(
                    report.tpm_estimate * config.overshoot_multiplier
                )
                max_to_issue = min(
                    max_to_issue, max(1, overshoot_tokens // tokens_per_request)
                )

            if max_to_issue == 0:
                console_log(
                    "  Token budget exhausted; no more probe requests will be issued."
                )
                break

            if round_num > 1:
                wait_seconds = config.refill_wait_seconds
                console_log(
                    f"Waiting {wait_seconds}s for bucket refill before refinement..."
                )
                await asyncio.sleep(wait_seconds)
                console_log("Bucket refill complete; starting next probe round")

            round_successes = 0
            round_rate_limits = 0
            round_unknown_errors = 0
            round_fatal_errors = 0
            round_success_tokens = 0
            round_issued = 0
            round_settled = 0
            round_unknown_messages: list[str] = []
            stop_scheduling = False
            round_timed_out = False
            t0 = time.monotonic()
            round_deadline = t0 + config.max_round_seconds

            round_goal = (
                f"midpoint target ~{refinement_target_tokens:,} tokens"
                if round_mode == "refine"
                else "find the first 429-derived range"
            )
            rich_console.print(
                Panel(
                    f"Issuing up to {max_to_issue:,} requests at max concurrency "
                    f"{config.max_concurrency}.\nGoal: {round_goal}.",
                    title=f"Round {round_num}: {round_mode}",
                    border_style="magenta" if round_mode == "refine" else "cyan",
                )
            )

            active: set[asyncio.Task[ProbeOutcome]] = set()

            def record_outcome(outcome: ProbeOutcome) -> None:
                nonlocal round_settled
                nonlocal round_successes
                nonlocal round_rate_limits
                nonlocal round_unknown_errors
                nonlocal round_fatal_errors
                nonlocal round_success_tokens
                nonlocal stop_scheduling

                round_settled += 1
                if outcome.kind == ProbeOutcomeKind.SUCCESS:
                    round_successes += 1
                    round_success_tokens += outcome.tokens
                elif outcome.kind == ProbeOutcomeKind.RATE_LIMITED:
                    round_rate_limits += 1
                    stop_scheduling = True
                elif outcome.kind == ProbeOutcomeKind.UNKNOWN_ERROR:
                    round_unknown_errors += 1
                    stop_scheduling = round_unknown_errors > config.max_unknown_errors
                    round_unknown_messages.append(
                        f"{outcome.error_type}: {outcome.message}"
                    )
                else:
                    round_fatal_errors += 1
                    stop_scheduling = True
                    round_unknown_messages.append(
                        f"{outcome.error_type}: {outcome.message}"
                    )

            with _round_progress() as progress:
                progress_task = progress.add_task(
                    f"Round {round_num} {round_mode}",
                    total=max_to_issue,
                    requests=f"issued 0/{max_to_issue:,} | 429 0",
                    tokens="tokens ~0",
                    estimate="estimate: waiting",
                )

                try:
                    while active or (
                        round_issued < max_to_issue and not stop_scheduling
                    ):
                        remaining_seconds = round_deadline - time.monotonic()
                        if remaining_seconds <= 0:
                            round_timed_out = True
                            stop_scheduling = True
                            for active_task in active:
                                active_task.cancel()
                            if active:
                                cleanup_results = await asyncio.gather(
                                    *active, return_exceptions=True
                                )
                                active.clear()
                                for cleanup_result in cleanup_results:
                                    if isinstance(cleanup_result, ProbeOutcome):
                                        record_outcome(cleanup_result)
                            progress.update(
                                progress_task,
                                completed=round_settled,
                                requests=(
                                    f"issued {round_issued:,}/{max_to_issue:,} | "
                                    f"429 {round_rate_limits:,} | time cap"
                                ),
                                tokens=f"tokens ~{round_success_tokens:,}",
                                estimate=_live_round_estimate(
                                    success_tokens=round_success_tokens,
                                    rate_limits=round_rate_limits,
                                    elapsed_seconds=time.monotonic() - t0,
                                    max_concurrency=config.max_concurrency,
                                    issued_requests=round_issued,
                                    tokens_per_request=tokens_per_request,
                                ),
                            )
                            break

                        while (
                            len(active) < config.max_concurrency
                            and round_issued < max_to_issue
                            and not stop_scheduling
                            and time.monotonic() < round_deadline
                        ):
                            active.add(asyncio.create_task(fire_one()))
                            round_issued += 1
                            budget_spent_tokens += tokens_per_request
                            progress.update(
                                progress_task,
                                requests=(
                                    f"issued {round_issued:,}/{max_to_issue:,} | "
                                    f"429 {round_rate_limits:,}"
                                ),
                            )

                        if not active:
                            break

                        done, active = await asyncio.wait(
                            active,
                            timeout=max(0.0, round_deadline - time.monotonic()),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if not done:
                            round_timed_out = True
                            stop_scheduling = True
                            for active_task in active:
                                active_task.cancel()
                            if active:
                                cleanup_results = await asyncio.gather(
                                    *active, return_exceptions=True
                                )
                                active.clear()
                                for cleanup_result in cleanup_results:
                                    if isinstance(cleanup_result, ProbeOutcome):
                                        record_outcome(cleanup_result)
                            progress.update(
                                progress_task,
                                completed=round_settled,
                                requests=(
                                    f"issued {round_issued:,}/{max_to_issue:,} | "
                                    f"429 {round_rate_limits:,} | time cap"
                                ),
                                tokens=f"tokens ~{round_success_tokens:,}",
                                estimate=_live_round_estimate(
                                    success_tokens=round_success_tokens,
                                    rate_limits=round_rate_limits,
                                    elapsed_seconds=time.monotonic() - t0,
                                    max_concurrency=config.max_concurrency,
                                    issued_requests=round_issued,
                                    tokens_per_request=tokens_per_request,
                                ),
                            )
                            break
                        for task in done:
                            record_outcome(task.result())

                            live_elapsed = time.monotonic() - t0
                            progress.update(
                                progress_task,
                                completed=round_settled,
                                requests=(
                                    f"issued {round_issued:,}/{max_to_issue:,} | "
                                    f"429 {round_rate_limits:,}"
                                ),
                                tokens=f"tokens ~{round_success_tokens:,}",
                                estimate=_live_round_estimate(
                                    success_tokens=round_success_tokens,
                                    rate_limits=round_rate_limits,
                                    elapsed_seconds=live_elapsed,
                                    max_concurrency=config.max_concurrency,
                                    issued_requests=round_issued,
                                    tokens_per_request=tokens_per_request,
                                ),
                            )
                finally:
                    if active:
                        for active_task in active:
                            active_task.cancel()
                        await asyncio.gather(*active, return_exceptions=True)
                        active.clear()

            elapsed = time.monotonic() - t0
            if round_timed_out:
                console_log(
                    f"Round {round_num}: hit {config.max_round_seconds:g}s "
                    "wall-clock cap; summarizing completed requests.",
                    color="yellow",
                )
            round_refill_corrected_tpm = int(round_success_tokens / (1 + elapsed / 60))
            round_overshoot_window = (
                min(config.max_concurrency, round_issued) * tokens_per_request
            )
            round_overshoot_adjusted_tpm = max(
                0, round_success_tokens - round_overshoot_window
            )
            round_hit_limit = round_rate_limits > 0
            round_tpm_estimate = (
                min(round_refill_corrected_tpm, round_overshoot_adjusted_tpm)
                if round_hit_limit
                else round_success_tokens
            )
            round_observed_upper_estimate = (
                round_success_tokens if round_hit_limit else 0
            )
            if round_fatal_errors > 0:
                round_status = "failed"
            elif round_rate_limits == 0:
                round_status = "lower_bound"
            elif round_unknown_errors > 0 or cached > 0:
                round_status = "approximate"
            else:
                round_status = "clean"

            report.issued_requests += round_issued
            report.settled_requests += round_settled
            report.success_requests += round_successes
            report.rate_limit_requests += round_rate_limits
            report.unknown_error_requests += round_unknown_errors
            report.fatal_error_requests += round_fatal_errors
            report.success_tokens += round_success_tokens
            report.elapsed_seconds += elapsed
            if round_success_tokens > 0 and round_status != "failed":
                round_quality = status_quality[round_status]
                report_quality = status_quality[report.status]
                has_tighter_upper = round_observed_upper_estimate > 0 and (
                    report.tpm_observed_upper_estimate == 0
                    or round_observed_upper_estimate
                    < report.tpm_observed_upper_estimate
                )
                if has_tighter_upper:
                    report.tpm_observed_upper_estimate = round_observed_upper_estimate

                keeps_existing_upper_range = (
                    report.tpm_observed_upper_estimate == 0
                    or round_observed_upper_estimate > 0
                    or round_tpm_estimate < report.tpm_observed_upper_estimate
                )
                should_select_round = keeps_existing_upper_range and (
                    round_tpm_estimate > report.tpm_estimate
                    or (
                        round_tpm_estimate == report.tpm_estimate
                        and (
                            round_quality > report_quality
                            or (round_quality == report_quality and has_tighter_upper)
                        )
                    )
                )
                if should_select_round:
                    report.tpm_estimate = round_tpm_estimate
                    report.status = round_status
            report.unknown_errors.extend(round_unknown_messages)

            if report.fatal_error_requests > 0:
                report.status = "failed"

            selected_range = _format_estimate_range(report)
            rich_console.print(
                _round_summary_table(
                    round_num=round_num,
                    round_mode=round_mode,
                    issued=round_issued,
                    settled=round_settled,
                    successes=round_successes,
                    rate_limits=round_rate_limits,
                    unknown_errors=round_unknown_errors,
                    fatal_errors=round_fatal_errors,
                    success_tokens=round_success_tokens,
                    refill_corrected_tpm=round_refill_corrected_tpm,
                    overshoot_adjusted_tpm=round_overshoot_adjusted_tpm,
                    observed_upper=round_observed_upper_estimate,
                    tpm_estimate=round_tpm_estimate,
                    status=round_status,
                    selected_estimate=f"{selected_range} ({report.status})",
                )
            )

            if round_fatal_errors > 0 or round_rate_limits == 0:
                break

        summary_estimate = _format_estimate_range(report)
        observed_upper = (
            f", observed_upper_signal=~{report.tpm_observed_upper_estimate:,}"
            if report.tpm_observed_upper_estimate
            else ""
        )
        console_log(
            f"Probe summary: estimated TPM {summary_estimate}, "
            f"status={report.status}, successes={report.success_requests}, "
            f"rate_limited={report.rate_limit_requests}, "
            f"unknown_errors={report.unknown_error_requests}, "
            f"fatal_errors={report.fatal_error_requests}"
            f"{observed_upper}"
        )

    finally:
        model.custom_retrier = previous_custom_retrier

    return report


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Get rate limit for a model")
    parser.add_argument(
        "model",
        type=str,
        help="Model endpoint",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe the token rate limit with bounded requests until 429",
    )
    parser.add_argument(
        "--confirm-probe",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--probe-rounds",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tokens-per-request",
        type=int,
        default=TARGET_TOKENS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-probe-requests",
        type=int,
        default=MAX_BATCH_REQUESTS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-token-budget",
        type=int,
        default=DEFAULT_MAX_TOKEN_BUDGET,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-unknown-errors",
        type=int,
        default=DEFAULT_MAX_UNKNOWN_ERRORS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-round-seconds",
        type=float,
        default=DEFAULT_MAX_ROUND_SECONDS,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    model = get_registry_model(
        args.model,
        override_config=LLMConfig(max_tokens=16) if args.probe else None,
    )

    key_hash = (
        model.delegate._client_registry_key[1]  # pyright: ignore[reportPrivateUsage]
        if model.delegate
        else model._client_registry_key[1]  # pyright: ignore[reportPrivateUsage]
    )
    console_log(f"Key hash: {key_hash}")

    if args.probe:
        console_log(
            "Skipping provider rate-limit preflight in probe mode; estimating from "
            "probe traffic instead.",
            color="yellow",
        )

        config = ProbeConfig(
            rounds=args.probe_rounds,
            target_tokens=args.tokens_per_request,
            max_requests=args.max_probe_requests,
            max_token_budget=args.max_token_budget,
            max_concurrency=args.max_concurrency,
            max_unknown_errors=args.max_unknown_errors,
            max_round_seconds=args.max_round_seconds,
        )
        first_round_requests = _max_requests_for_budget(
            config,
            config.target_tokens,
            spent_tokens=config.target_tokens,
        )
        first_round_tokens = (first_round_requests + 1) * config.target_tokens
        rich_console.print(
            _probe_plan_panel(config, first_round_requests, first_round_tokens)
        )
        if not args.confirm_probe and not _confirm_quota_probe():
            return

        console_log("Starting rate limit probe...")
        report = await probe_rate_limit(model, config)
        if report.status == "failed":
            raise SystemExit(1)
        return

    rate_limit = await model.get_rate_limit()
    console_log(f"Rate limit: {rate_limit}")


if __name__ == "__main__":
    setup()
    asyncio.run(main())
