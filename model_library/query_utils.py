"""Utility helpers for common LLM query patterns."""

from collections.abc import Callable
from math import floor

from model_library.base.base import LLM
from model_library.base.input import TextInput
from model_library.base.output import FinishReason, QueryResult
from model_library.exceptions import (
    MaxContextWindowExceededError,
    MaxOutputTokensExceededError,
)
from model_library.registry_utils import get_model_input_context_window


async def query_with_truncation_retry(
    llm: LLM,
    doc_text: str,
    build_prompt: Callable[[str], str],
    *,
    question_id: str | None = None,
    run_id: str | None = None,
    docent_ingest: bool = False,
) -> tuple[QueryResult, dict[str, int]]:
    """Query an LLM with automatic document truncation on context window errors.

    Truncates ``doc_text`` until the query succeeds:
    - First, shortens to fit within the model's input context window using token count
    - Then retries on ``MaxContextWindowExceededError``, shortening by 10% each time
    - Also retries on ``MAX_TOKENS`` finish reason (output truncated), shortening by 5%
    """
    if not llm._registry_key:  # pyright: ignore[reportPrivateUsage]
        raise ValueError("LLM has no registry key — use get_registry_model()")

    context_window = get_model_input_context_window(llm._registry_key)  # pyright: ignore[reportPrivateUsage]

    truncation_record = {
        "initial_context_window_truncation": 0,
        "max_context_window_exceeded_truncation": 0,
        "max_output_tokens_exceeded_truncation": 0,
    }

    prompt = build_prompt(doc_text)

    def shorten(ratio: float) -> None:
        nonlocal doc_text, prompt
        doc_text = doc_text[: floor(len(doc_text) * ratio)]
        prompt = build_prompt(doc_text)

    # truncate to fit input context window
    length = await llm.count_tokens(input=[TextInput(text=prompt)])
    if length > context_window:
        truncation_record["initial_context_window_truncation"] += 1
        shorten(context_window / length)

    # retry until query succeeds
    while True:
        try:
            result = await llm.query(
                prompt,
                question_id=question_id,
                run_id=run_id,
                docent_ingest=docent_ingest,
            )

            if result.finish_reason.reason == FinishReason.MAX_TOKENS:
                raise MaxOutputTokensExceededError(
                    f"finish_reason: {result.finish_reason.raw}"
                )

            if result.finish_reason.reason == FinishReason.UNKNOWN:
                raise ValueError(f"Unknown finish reason: {result.finish_reason.raw}")

            return result, {k: v for k, v in truncation_record.items() if v > 0}

        except MaxContextWindowExceededError:
            truncation_record["max_context_window_exceeded_truncation"] += 1
            shorten(0.90)
        except MaxOutputTokensExceededError:
            truncation_record["max_output_tokens_exceeded_truncation"] += 1
            shorten(0.95)
