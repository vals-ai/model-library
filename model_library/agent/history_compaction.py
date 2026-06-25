import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from model_library.agent.config import (
    DEFAULT_COMPACTION_PROMPT,
    DEFAULT_SUMMARY_PREFIX,
    HistoryCompaction,
    truncate_oldest,
)
from model_library.agent.metadata import (
    CompactionSummary,
    SerializableException,
)
from model_library.base.base import LLM
from model_library.base.input import (
    InputItem,
    SystemInput,
    TextInput,
    ToolDefinition,
)
from model_library.base.output import QueryResultMetadata
from model_library.exceptions import MaxContextWindowExceededError
from model_library.registry_utils import get_model_input_context_window

if TYPE_CHECKING:
    from model_library.agent.metadata import AgentTurn

MIN_COMPACTION_THRESHOLD_HEADROOM_TOKENS = 1024


async def estimate_next_input_tokens(
    llm: LLM,
    previous_turn: "AgentTurn",
) -> int:
    """Estimate the next turn's input-token size from prior-turn signals.

    Provider-reported ``total_input_tokens + total_output_tokens`` from the
    previous turn is ground truth; the only thing we need to estimate is
    the delta — tool results appended since. Tokenize those via
    ``llm.get_encoding()`` rather than a bytes/4 heuristic (encoding
    lookup is cached + nearly free, tiktoken is fast, and we're only
    tokenizing the *new* content, not the whole history).

    Note that ``llm.get_encoding`` falls back to ``cl100k_base`` /
    ``o200k_base`` for non-OpenAI models, so the count for Claude /
    Gemini / Mistral remains an approximation — but a closer one than
    ``bytes/4``, especially for dense content (JSON, code, base64).
    """
    metadata = previous_turn.query_result.metadata
    estimate = metadata.total_input_tokens + metadata.total_output_tokens
    if previous_turn.tool_call_records:
        encoding = await llm.get_encoding()
        for record in previous_turn.tool_call_records:
            estimate += len(
                encoding.encode(str(record.tool_output.output), disallowed_special=())
            )
    return estimate


class EmptyCompactionSummaryError(RuntimeError):
    """The compaction LLM returned no usable text after stripping whitespace."""


class CompactionHook(Protocol):
    """Hook deciding when and how to compact history.

    The agent calls this hook every turn (with ``trigger='each_turn'``)
    *and* on ``MaxContextWindowExceededError`` (with ``trigger='max_context'``).
    The hook owns the entire compaction policy: when to fire, how to size,
    what algorithm to use, when to give up.

    Return ``(history, None)`` to opt out for this call. Return
    ``(new_history, CompactionSummary(...))`` to record an attempt; success
    is determined by ``summary.error is None``. The agent appends the
    summary to ``AgentResult.compactions`` and uses it for ATIF export, but
    otherwise treats the hook as a black box.

    ``state`` is a hook-private scratchpad. It's a fresh dict for each
    ``Agent.run()`` call, so per-run state can't leak across runs and
    concurrent runs don't race. Use it for failure counters, accumulated
    estimates, anything else the hook needs to remember between calls.

    ``tools`` is the set of tool definitions active for the upcoming LLM
    query. Hooks can pass it through to preserve provider-side validation
    and caching behavior, or ignore/filter it for custom strategies.

    ``previous_turn`` is the prior successful agent turn — its
    ``query_result.metadata`` and ``tool_call_records`` are the raw
    signals the hook can use to compute its own estimate (cheap
    byte-rate, tokenizer-accurate, custom). It is ``None`` on the first
    call (turn 1), when the previous turn errored, and on the
    ``max_context`` trigger.
    """

    async def __call__(
        self,
        history: list[InputItem],
        *,
        state: dict[str, Any],
        trigger: Literal["each_turn", "max_context"],
        turn_number: int,
        compaction_number: int,
        tools: list[ToolDefinition],
        previous_turn: "AgentTurn | None",
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary | None]: ...


async def apply_compaction(
    hook: CompactionHook | None,
    history: list[InputItem],
    *,
    trigger: Literal["each_turn", "max_context"],
    state: dict[str, Any],
    compactions: list[CompactionSummary],
    turn_number: int,
    tools: list[ToolDefinition],
    previous_turn: "AgentTurn | None",
    output_dir: Path,
    question_id: str,
    run_id: str | None,
    logger: logging.Logger,
) -> tuple[list[InputItem], CompactionSummary | None]:
    """Invoke ``hook`` (if configured) and append any returned summary to
    ``compactions``. The hook owns the entire policy — this wrapper just
    threads the agent's compactions list through and is a no-op when no
    hook is configured.
    """
    if hook is None:
        return history, None
    new_history, summary = await hook(
        history,
        state=state,
        trigger=trigger,
        turn_number=turn_number,
        compaction_number=len(compactions) + 1,
        tools=tools,
        previous_turn=previous_turn,
        output_dir=output_dir,
        question_id=question_id,
        run_id=run_id,
        logger=logger,
    )
    if summary is not None:
        compactions.append(summary)
    return new_history, summary


def resolve_threshold_tokens(llm: LLM, cfg: HistoryCompaction) -> int | None:
    """Resolve the input-token threshold from the config + model context window.

    Returns ``None`` for raw (non-registry-backed) LLMs where the context
    window is unknown. Custom hooks can use this if they want to honor the
    config's threshold; the default ``llm_summary_compactor`` does.
    """
    registry_key = llm._registry_key  # pyright: ignore[reportPrivateUsage]
    input_context_window = llm.input_context_window
    ctx = input_context_window if isinstance(input_context_window, int) else None
    if ctx is None and registry_key:
        ctx = get_model_input_context_window(registry_key)
    if ctx is None:
        return None
    if cfg.threshold_tokens is not None:
        threshold = cfg.threshold_tokens
    else:
        assert cfg.threshold_percentage is not None
        threshold = int(ctx * cfg.threshold_percentage)
    # Keep explicit thresholds from sitting exactly on the model's input limit.
    # This is request headroom; summary output still uses llm.max_tokens or the
    # provider default.
    return min(threshold, ctx - MIN_COMPACTION_THRESHOLD_HEADROOM_TOKENS)


def llm_summary_compactor(
    llm: LLM,
    cfg: HistoryCompaction,
    *,
    prompt: str = DEFAULT_COMPACTION_PROMPT,
    summary_prefix: str = DEFAULT_SUMMARY_PREFIX,
) -> CompactionHook | None:
    """Default :class:`CompactionHook` implementation: send the history to
    ``llm`` with ``prompt`` appended as a trailing user message, and replace
    the agent history with the returned summary.

    Returns ``None`` when the LLM isn't registry-backed (no context window
    information is available), so the caller can disable compaction with a
    warning rather than fail at agent construction.

    Strategy-specific knobs (``prompt``, ``summary_prefix``) are factory
    arguments rather than fields on :class:`HistoryCompaction` so the
    config stays strategy-agnostic.

    The returned hook owns its policy:

    - On ``trigger='each_turn'``: maintains a cheap input-token estimate in
      ``state`` and fires when the estimate crosses ``cfg.threshold_tokens``.
      Stops trying after ``cfg.max_failures`` consecutive failures (a
      successful compaction resets the counter).
    - On ``trigger='max_context'``: opt-in via ``cfg.compact_on_max_context``.
      Bypasses the threshold and failure gates — last-resort attempt.
    - On either trigger: compaction-call context overflows truncate and retry
      up to ``cfg.max_compaction_context_retries`` times.
    """
    threshold = resolve_threshold_tokens(llm, cfg)
    if threshold is None:
        return None

    async def _do_compact(
        history: list[InputItem],
        *,
        turn_number: int,
        compaction_number: int,
        input_token_estimate: int | None,
        context_retry_limit: int,
        tools: list[ToolDefinition],
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary]:
        """Run the summarize → replace step. Internal."""
        attempt = history
        summary = ""
        metadata: QueryResultMetadata | None = None
        context_retries = 0
        try:
            while not summary:
                try:
                    response = await llm.query(
                        input=[*attempt, TextInput(text=prompt)],
                        tools=[],
                        question_id=question_id,
                        run_id=run_id,
                        logger=logger,
                        in_agent=True,
                    )
                    summary = (response.output_text or "").strip()
                    metadata = response.metadata
                    if not summary:
                        raise EmptyCompactionSummaryError(
                            "Compaction LLM returned empty output"
                        )
                except MaxContextWindowExceededError:
                    context_retries += 1
                    if context_retries > context_retry_limit:
                        raise
                    # History too big; drop oldest exchange and retry.
                    attempt = truncate_oldest(attempt)
        except Exception as e:
            logger.warning(
                "History compaction failed; continuing with original history",
                exc_info=True,
            )
            return history, CompactionSummary(
                turn_number=turn_number,
                input_token_estimate=input_token_estimate,
                threshold_tokens=threshold,
                metadata=metadata,
                error=SerializableException.from_exception(e),
            )

        new_history: list[InputItem] = [
            *(item for item in history if isinstance(item, SystemInput)),
            TextInput(text=summary_prefix + summary),
        ]
        artifacts = output_dir / "compactions" / f"compaction_{compaction_number:03d}"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "previous_history.bin").write_text(LLM.serialize_input(history))
        rel = str(artifacts.relative_to(output_dir))
        logger.info(
            "History compacted: %s -> %s items | summary=%r",
            len(history),
            len(new_history),
            summary[:500],
        )
        return new_history, CompactionSummary(
            turn_number=turn_number,
            input_token_estimate=input_token_estimate,
            threshold_tokens=threshold,
            summary=summary,
            artifacts_subdir=rel,
            metadata=metadata,
        )

    async def hook(
        history: list[InputItem],
        *,
        state: dict[str, Any],
        trigger: Literal["each_turn", "max_context"],
        turn_number: int,
        compaction_number: int,
        tools: list[ToolDefinition],
        previous_turn: "AgentTurn | None",
        output_dir: Path,
        question_id: str,
        run_id: str | None,
        logger: logging.Logger,
    ) -> tuple[list[InputItem], CompactionSummary | None]:
        # Update the estimate from the prior turn's signals.
        if previous_turn is not None:
            state["next_input_estimate"] = await estimate_next_input_tokens(
                llm, previous_turn
            )
        next_input_estimate: int = state.get("next_input_estimate", 0)
        consecutive_failures: int = state.get("consecutive_failures", 0)

        # Decide whether to actually compact.
        if trigger == "each_turn":
            if next_input_estimate < threshold:
                return history, None
            if consecutive_failures >= cfg.max_failures:
                return history, None
            input_token_estimate: int | None = next_input_estimate
            context_retry_limit = cfg.max_compaction_context_retries
            logger.info(
                f"Compacting history: {next_input_estimate:,} >= {threshold:,} tokens"
            )
        else:  # max_context — opt-in, bypasses threshold + failure gates
            if not cfg.compact_on_max_context:
                return history, None
            input_token_estimate = None
            context_retry_limit = cfg.max_compaction_context_retries
            logger.warning("Query exceeded context window; compacting")
            try:
                history = truncate_oldest(history)
            except ValueError:
                pass

        new_history, summary = await _do_compact(
            history,
            turn_number=turn_number,
            compaction_number=compaction_number,
            input_token_estimate=input_token_estimate,
            context_retry_limit=context_retry_limit,
            tools=tools,
            output_dir=output_dir,
            question_id=question_id,
            run_id=run_id,
            logger=logger,
        )
        if summary.success:
            state["next_input_estimate"] = 0
            state["consecutive_failures"] = 0
        elif trigger == "each_turn":
            state["consecutive_failures"] = consecutive_failures + 1
        return new_history, summary

    return hook
