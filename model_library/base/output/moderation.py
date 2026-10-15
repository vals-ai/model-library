"""Gateway moderation response models.

Shapes adapted from openai-python 2.54.0, openai/types/moderation.py and
openai/types/moderation_create_response.py (Apache-2.0). This local model
preserves wire aliases without importing SDK types.
See model_library/_licenses/openai-python.txt.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _ModerationModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ModerationCategories(_ModerationModel):
    harassment: bool
    harassment_threatening: bool = Field(alias="harassment/threatening")
    hate: bool
    hate_threatening: bool = Field(alias="hate/threatening")
    illicit: bool | None = None
    illicit_violent: bool | None = Field(default=None, alias="illicit/violent")
    self_harm: bool = Field(alias="self-harm")
    self_harm_instructions: bool = Field(alias="self-harm/instructions")
    self_harm_intent: bool = Field(alias="self-harm/intent")
    sexual: bool
    sexual_minors: bool = Field(alias="sexual/minors")
    violence: bool
    violence_graphic: bool = Field(alias="violence/graphic")


class ModerationCategoryScores(_ModerationModel):
    harassment: float
    harassment_threatening: float = Field(alias="harassment/threatening")
    hate: float
    hate_threatening: float = Field(alias="hate/threatening")
    illicit: float
    illicit_violent: float = Field(alias="illicit/violent")
    self_harm: float = Field(alias="self-harm")
    self_harm_instructions: float = Field(alias="self-harm/instructions")
    self_harm_intent: float = Field(alias="self-harm/intent")
    sexual: float
    sexual_minors: float = Field(alias="sexual/minors")
    violence: float
    violence_graphic: float = Field(alias="violence/graphic")


class ModerationCategoryAppliedInputTypes(_ModerationModel):
    harassment: list[Literal["text"]]
    harassment_threatening: list[Literal["text"]] = Field(
        alias="harassment/threatening"
    )
    hate: list[Literal["text"]]
    hate_threatening: list[Literal["text"]] = Field(alias="hate/threatening")
    illicit: list[Literal["text"]]
    illicit_violent: list[Literal["text"]] = Field(alias="illicit/violent")
    self_harm: list[Literal["text", "image"]] = Field(alias="self-harm")
    self_harm_instructions: list[Literal["text", "image"]] = Field(
        alias="self-harm/instructions"
    )
    self_harm_intent: list[Literal["text", "image"]] = Field(alias="self-harm/intent")
    sexual: list[Literal["text", "image"]]
    sexual_minors: list[Literal["text"]] = Field(alias="sexual/minors")
    violence: list[Literal["text", "image"]]
    violence_graphic: list[Literal["text", "image"]] = Field(alias="violence/graphic")


class ModerationResult(_ModerationModel):
    categories: ModerationCategories
    category_applied_input_types: ModerationCategoryAppliedInputTypes
    category_scores: ModerationCategoryScores
    flagged: bool


class GatewayModerationResult(_ModerationModel):
    id: str
    model: str
    results: list[ModerationResult]
