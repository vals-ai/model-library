# Speech-to-text

Install the dedicated voice SDKs with `pip install 'model-library[voice]'`.

The library supports complete-file speech-to-text through `transcribe_audio()`.
Standalone speech-to-text providers subclass `TranscriptionOnly`; the base
rejects LLM-only operations. Reusable SDK clients share the normal process-wide
client registry, while streaming connections and recognizers remain scoped to
one request. Native adapters implement `_transcribe_audio()` so the public
wrapper retains language normalization, timing, error handling, duration, and
pricing finalization.

```python
from model_library.registry_utils import get_transcription_model

model = get_transcription_model("deepgram/flux-general-en")
result = await model.transcribe_audio(
    name="clip.wav",
    mime="audio/wav",
    audio=audio_bytes,
    language="en",
)
print(result.text)
```

Callers always provide one bounded audio file; provider batch jobs and
caller-controlled unbounded audio streams are not supported. The same call works
through `MODEL_GATEWAY_URL`, which forwards the complete payload to
`POST /audio/transcriptions`.

## Supported models

Read the list from the local registry so it stays current; gateway clients use
`GET /registry?include_excluded_fields=true&include_transcription=true`
(`GET /models` lists chat models only).

```python
from model_library.register_models import get_transcription_registry

transcription_models = sorted(get_transcription_registry())
```

Set provider credentials using the names in [API keys](api-keys.md).

## Audio input

Realtime-audio models require a complete WAV payload: 16 kHz mono PCM16 for
AssemblyAI, ElevenLabs, Inworld, xAI, Cartesia, realtime Mistral, Gemini Live,
Muse Voice Transcribe, and OpenAI Live; 8–48 kHz mono PCM16 for AWS Transcribe;
any mono PCM16 rate for Azure Speech and Deepgram; Google Cloud auto-detects the
container's encoding.
Complete-file models — both Groq models, GPT-4o Transcribe, OpenAI Mini's
streaming-response transport, Mistral Voxtral Mini 2602, Gemini Transcribe,
and Cohere Transcribe — accept the MIME types their own endpoint supports.
Cohere also requires an ISO-639-1 `language`; Muse Voice Transcribe ignores it,
since its API takes only a bias hint and detects the language itself.

ElevenLabs pads clips shorter than one second with silence. The padded duration
is used for `billable_duration_seconds` and cost calculation. Source
`audio_duration_seconds` and `audio_bytes` remain unchanged.

## Response

The common response contains the transcript text and local request metadata.
Provider-specific response fields are not flattened into the common model.

- `request_duration_seconds` is the wall time of the provider adapter call.
- `audio_duration_seconds` is the decoded WAV media length when available.
- `billable_duration_seconds` is the raw provider or audio pricing basis before
  registry minimums and billing increments. Cost calculation does not replace
  it with the rounded charged duration.
- `total_tokens` is derived from `input_tokens` and `output_tokens` when either
  value is available.
- `time_to_first_partial_seconds` measures from the start of the audio send to
  the first nonempty transcript event for audio-streaming transports. For
  OpenAI's complete-file streaming response, it measures from request start to
  the first nonempty transcript delta. It is `null` when a transport returns no
  transcript text.

Audio is sent as fast as the transport accepts it, except for AWS Transcribe,
xAI, Gemini Live, Meta, and Muse Voice Transcribe, which are paced for their
realtime streaming contracts. AWS uses approximately 100 ms PCM chunks.
AssemblyAI and ElevenLabs bound each transport write and finalization operation;
their terminal-response timeout starts only after finalization.
