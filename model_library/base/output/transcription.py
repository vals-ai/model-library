from model_library.utils import ValsModel


class TranscriptionMetadata(ValsModel):
    audio_bytes: int
    request_duration_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    audio_tokens: int | None = None
    text_tokens: int | None = None
    cost_usd: float | None = None


class TranscriptionResult(ValsModel):
    text: str
    metadata: TranscriptionMetadata
