from pydantic import computed_field

from model_library.utils import ValsModel


class TranscriptionMetadata(ValsModel):
    audio_bytes: int
    request_duration_seconds: float
    audio_duration_seconds: float | None = None
    billable_duration_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    audio_tokens: int | None = None
    text_tokens: int | None = None
    cost_usd: float | None = None
    time_to_first_partial_seconds: float | None = None

    @computed_field
    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class TranscriptionResult(ValsModel):
    text: str
    metadata: TranscriptionMetadata
