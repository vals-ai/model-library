from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field, field_validator

from model_library.agent.agent import AgentResult
from model_library.base.output import QueryResultMetadata
from model_library.utils import PrettyModel


class ConductorStopReason(StrEnum):
    AUDITOR_DONE = "auditor_done"
    MAX_EXCHANGES = "max_exchanges"
    MAX_TIME = "max_time"
    ERROR = "error"


class ConversationMessage(PrettyModel):
    role: Literal["auditor", "target"]
    result: AgentResult


class ConductorResult(PrettyModel):
    messages: list[ConversationMessage]
    stop_reason: ConductorStopReason
    total_duration_seconds: float
    output_dir: Path = Field(exclude=True)

    @field_validator("total_duration_seconds", mode="before")
    @classmethod
    def _round_duration(cls, v: float) -> float:
        return round(v, 3)

    @computed_field
    @property
    def auditor_aggregated_metadata(self) -> QueryResultMetadata:
        """Aggregated LLM query metadata across all auditor agent turns"""
        result = QueryResultMetadata()
        for message in self.messages:
            if message.role == "auditor":
                result = result + message.result.final_aggregated_metadata
        return result

    @computed_field
    @property
    def target_aggregated_metadata(self) -> QueryResultMetadata:
        """Aggregated LLM query metadata across all target agent turns"""
        result = QueryResultMetadata()
        for message in self.messages:
            if message.role == "target":
                result = result + message.result.final_aggregated_metadata
        return result

    @computed_field
    @property
    def final_aggregated_metadata(self) -> QueryResultMetadata:
        """Aggregated LLM query metadata across all auditor and target agent turns"""
        return self.auditor_aggregated_metadata + self.target_aggregated_metadata
