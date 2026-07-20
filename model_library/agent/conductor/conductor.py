import json
import logging
import time
import uuid
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.pretty import pretty_repr

from model_library.agent.agent import Agent, AgentStopReason
from model_library.agent.conductor.config import ConductorConfig
from model_library.agent.conductor.metadata import (
    ConductorResult,
    ConductorStopReason,
    ConversationMessage,
)
from model_library.base.input import InputItem, SystemInput, TextInput
from model_library.utils import run_logging

DEFAULT_INITIAL_PROMPT = "Start the conversation. Send your first message in character."


class ConductorAgent:
    """Orchestrates multi-turn conversations between an auditor Agent and a target Agent.

    The auditor asks questions or probes; the target responds. The conductor
    shuttles messages between them, logging each exchange, and stops when the
    auditor signals done (via a done-tool), the maximum number of exchanges is
    reached, or the time limit expires.
    """

    def __init__(
        self,
        auditor: Agent,
        target: Agent,
        *,
        auditor_system_prompt: SystemInput,
        target_system_prompt: SystemInput,
        name: str,
        log_dir: Path = Path("logs"),
        config: ConductorConfig,
    ):
        self._auditor = auditor
        self._target = target
        self._auditor_system_prompt = auditor_system_prompt
        self._target_system_prompt = target_system_prompt
        self._name = name
        self._config = config
        self._log_dir = self._build_log_dir(
            log_dir,
            name,
            auditor.model_name,
            target.model_name,
        )

    def __rich_repr__(self) -> Generator[tuple[str, Any], None, None]:
        yield "name", self._name
        yield "auditor", self._auditor
        yield "target", self._target
        yield "config", self._config

    def __repr__(self) -> str:
        return pretty_repr(self)

    __str__ = __repr__

    @staticmethod
    def _build_log_dir(
        base: Path,
        name: str,
        auditor_model: str,
        target_model: str,
    ) -> Path:
        auditor_safe = auditor_model.replace("/", "_")
        target_safe = target_model.replace("/", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        short_id = uuid.uuid4().hex[:6]

        return base / name / f"{auditor_safe}_{target_safe}" / f"{timestamp}_{short_id}"

    async def run(
        self,
        *,
        question_id: str,
        run_id: str | None = None,
        initial_prompt: str = DEFAULT_INITIAL_PROMPT,
        state: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> ConductorResult:
        """Run the conductor loop.

        Args:
            question_id: Identifier scoping this run's log directory.
            run_id: Optional run identifier forwarded to both agents.
            initial_prompt: The first message sent to the auditor to start the conversation.
            state: Optional shared state dict forwarded to both agents.
            logger: Optional parent logger; a child logger is created from it.
        """
        conductor_logger = (
            (logger or logging.getLogger("conductor"))
            .getChild(f"{self._name}")
            .getChild(f"<question={question_id}>")
        )
        conductor_logger.setLevel(logging.DEBUG)

        with run_logging(conductor_logger, self._log_dir, question_id) as output_dir:
            conductor_logger.debug(repr(self))

            return await self._run(
                initial_prompt=initial_prompt,
                question_id=question_id,
                run_id=run_id,
                state=state if state is not None else {},
                output_dir=output_dir,
                logger=conductor_logger,
            )

    def _write_init_dir(
        self,
        output_dir: Path,
        state: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        """Write exchanges/init/ with config and initial state."""
        init_dir = output_dir / "exchanges" / "init"
        init_dir.mkdir(parents=True, exist_ok=True)

        try:
            (init_dir / "config.json").write_text(
                json.dumps(
                    {
                        "conductor_config": self._config.model_dump(),
                        "auditor": {
                            "name": self._auditor.name,
                            "model": self._auditor.model_name,
                            "system_prompt": self._auditor_system_prompt.text,
                            "config": self._auditor.config.model_dump(),
                            "tools": [
                                td.model_dump() for td in self._auditor.tool_definitions
                            ],
                        },
                        "target": {
                            "name": self._target.name,
                            "model": self._target.model_name,
                            "system_prompt": self._target_system_prompt.text,
                            "config": self._target.config.model_dump(),
                            "tools": [
                                td.model_dump() for td in self._target.tool_definitions
                            ],
                        },
                    },
                    indent=2,
                    default=str,
                )
            )
            (init_dir / "state.json").write_text(
                json.dumps(state, indent=2, default=str)
            )
        except Exception:
            logger.exception("Failed to write init directory")

    async def _run(
        self,
        *,
        initial_prompt: str,
        question_id: str,
        run_id: str | None,
        state: dict[str, Any],
        output_dir: Path,
        logger: logging.Logger,
    ) -> ConductorResult:
        self._write_init_dir(output_dir, state, logger)

        messages: list[ConversationMessage] = []
        auditor_history: list[InputItem] = [self._auditor_system_prompt]
        target_history: list[InputItem] = [self._target_system_prompt]
        stop_reason = ConductorStopReason.MAX_EXCHANGES
        start_time = time.monotonic()

        time_limit = self._config.time_limit

        for exchange_number in range(1, self._config.max_exchanges + 1):
            # Check time limit
            if time_limit is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= time_limit.max_seconds:
                    logger.warning(
                        f"Time limit reached ({elapsed:.1f}s >= {time_limit.max_seconds}s)"
                    )
                    stop_reason = ConductorStopReason.MAX_TIME
                    break

            logger.info(
                f"Exchange {exchange_number}/{self._config.max_exchanges} starting"
            )

            # Build auditor input: history + new message
            new_message = (
                messages[-1].result.final_answer if messages else initial_prompt
            )
            auditor_input = auditor_history + [TextInput(text=new_message)]

            # Run auditor
            try:
                auditor_result = await self._auditor.run(
                    auditor_input,
                    question_id=question_id,
                    run_id=run_id,
                    state=state,
                )
            except Exception as e:
                logger.error(
                    f"Auditor failed on exchange {exchange_number}: {e}",
                    exc_info=True,
                )
                stop_reason = ConductorStopReason.ERROR
                break

            auditor_history = auditor_result.final_history
            messages.append(ConversationMessage(role="auditor", result=auditor_result))

            # Stop before running target if auditor signaled done
            if auditor_result.stop_reason == AgentStopReason.DONE_TOOL:
                logger.info("Stop: auditor signaled done")
                stop_reason = ConductorStopReason.AUDITOR_DONE
                break

            # Stop if auditor produced no usable message
            if not auditor_result.final_answer:
                logger.warning(
                    f"Auditor produced empty response (stop_reason={auditor_result.stop_reason})"
                )
                stop_reason = ConductorStopReason.ERROR
                break

            # Build target input: history + auditor's message
            target_input = target_history + [
                TextInput(text=auditor_result.final_answer)
            ]

            # Run target
            try:
                target_result = await self._target.run(
                    target_input,
                    question_id=question_id,
                    run_id=run_id,
                    state=state,
                )
            except Exception as e:
                logger.error(
                    f"Target failed on exchange {exchange_number}: {e}",
                    exc_info=True,
                )
                stop_reason = ConductorStopReason.ERROR
                break

            target_history = target_result.final_history
            messages.append(ConversationMessage(role="target", result=target_result))

            # Stop if target produced no usable message
            if not target_result.final_answer:
                logger.warning(
                    f"Target produced empty response (stop_reason={target_result.stop_reason})"
                )
                stop_reason = ConductorStopReason.ERROR
                break

        total_duration = time.monotonic() - start_time

        result = ConductorResult(
            messages=messages,
            stop_reason=stop_reason,
            total_duration_seconds=total_duration,
            output_dir=output_dir,
        )

        # Write result.json
        try:
            (output_dir / "result.json").write_text(result.model_dump_json())
        except Exception:
            logger.exception("Failed to write result.json")

        # Write transcript.json
        try:
            transcript: list[dict[str, str]] = []
            for message in messages:
                transcript.append(
                    {"role": message.role, "content": message.result.final_answer}
                )
            (output_dir / "transcript.json").write_text(
                json.dumps(transcript, indent=2)
            )
        except Exception:
            logger.exception("Failed to write transcript.json")

        logger.info(
            f"Conductor finished: {stop_reason}, "
            f"{len(messages)} messages, {total_duration:.1f}s"
        )

        return result
