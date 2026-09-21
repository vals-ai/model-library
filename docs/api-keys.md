# Provider API Keys

## Credential sources

Model provider credentials are read from environment variables or from
`model_library_settings.set(...)` using the same names. Gateway servers keep
these provider keys server-side; gateway clients authenticate with
`MODEL_GATEWAY_API_KEY` and `MODEL_GATEWAY_URL`.

`LLMConfig(custom_api_key=SecretStr(...))` can override the default provider key
for a single model call. `custom_endpoint` is for caller-supplied,
provider-compatible URLs. Requests that set `custom_endpoint` must also set
`custom_api_key`; the gateway uses that caller-supplied key for the custom URL
and never sends server-held provider keys to arbitrary endpoints. Meta Muse
Voice transcription rejects `custom_endpoint` because it uses Meta's fixed
realtime WebSocket protocol.

## Provider mapping

| Model prefix | Provider | Settings |
| --- | --- | --- |
| `ai21labs/*` | AI21 Labs | `AI21LABS_API_KEY` |
| `alibaba/*` | Alibaba DashScope (international) | `DASHSCOPE_API_KEY` |
| `alibaba/*-cn` | Alibaba DashScope (mainland China) | `DASHSCOPE_CN_API_KEY` |
| `amazon/*`, `bedrock/*` | Amazon Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`; optional `AWS_SESSION_TOKEN` |
| `ant/*` | Ant Group (Ling) | `ANT_API_KEY` |
| `anthropic/*` | Anthropic | `ANTHROPIC_API_KEY`; managed gateways also use contiguous `ANTHROPIC_API_KEY_2` through `ANTHROPIC_API_KEY_N` pools |
| `arcee-ai/*` | Arcee AI | `ARCEE_API_KEY` |
| `azure/*` | Azure OpenAI | `AZURE_API_KEY`, `AZURE_ENDPOINT`; optional `AZURE_API_VERSION` |
| `baseten/*` | Baseten | `BASETEN_API_KEY`; dedicated deployments also require `BASETEN_API_BASE_URL` (full OpenAI-compatible `/v1` base URL) |
| `cohere/*` | Cohere | `COHERE_API_KEY` |
| `deepseek/*` | DeepSeek | `DEEPSEEK_API_KEY` |
| `fireworks/*` | Fireworks | `FIREWORKS_API_KEY` |
| `google/*` | Google Gemini | `GOOGLE_API_KEY`; requests with `config.provider_config.use_vertex=true` use `GCP_REGION`, `GCP_PROJECT_ID`, `GCP_CREDS` |
| `grok/*`, `xai/*` | xAI | `XAI_API_KEY` |
| `inception/*` | Inception Mercury | `MERCURY_API_KEY` |
| `kimi/*` | Moonshot AI Kimi | `KIMI_API_KEY` |
| `minimax/*` | MiniMax | `MINIMAX_API_KEY` |
| `mistral/*`, `mistralai/*` | Mistral | `MISTRAL_API_KEY` |
| `openai/*` | OpenAI | `OPENAI_API_KEY` |
| `openrouter/*` | OpenRouter | `OPENROUTER_API_KEY` |
| `perplexity/*` | Perplexity | `PERPLEXITY_API_KEY` |
| `poolside/*` | Poolside | `POOLSIDE_API_KEY` |
| `thinkingmachines/*` | Thinking Machines (Tinker) | `TINKER_API_KEY` |
| `together/*` | Together AI | `TOGETHER_API_KEY` |
| `vercel/*` | Vercel AI Gateway | `VERCEL_API_KEY` |
| `xiaomi/*` | Xiaomi | `XIAOMI_API_KEY` |
| `zai/*` | ZhipuAI / Z.ai | `ZAI_API_KEY` |

## Speech-to-text provider mapping

| Model prefix | Provider | Settings |
| --- | --- | --- |
| `assemblyai/*` | AssemblyAI | `ASSEMBLYAI_API_KEY` |
| `aws_transcribe/*` | AWS Transcribe | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`; optional `AWS_SESSION_TOKEN`; falls back to the standard AWS credential chain |
| `azure_speech/*` | Azure AI Speech | `AZURE_SPEECH_KEY`; optional `AZURE_SPEECH_REGION` (defaults to `eastus`); `LLMConfig.custom_endpoint` overrides the region |
| `cartesia/*` | Cartesia | `CARTESIA_API_KEY` |
| `deepgram/*` | Deepgram | `DEEPGRAM_API_KEY` |
| `elevenlabs/*` | ElevenLabs | `ELEVENLABS_API_KEY` |
| `google_cloud/*` | Google Cloud Speech-to-Text | `GCP_CREDS`, `GCP_PROJECT_ID` |
| `groq/*` | Groq | `GROQ_API_KEY` |
| `inworld/*` | Inworld | `INWORLD_API_KEY` |
| `meta/muse_voice_transcribe` | Meta Muse Voice Transcribe | `META_API_KEY` |

Transcription models served by providers in the table above (`cohere/*`,
`google/*`, `mistral/*`, `openai/*`, `xai/*`) use that provider's key.

## Amazon Bedrock

Amazon Bedrock can also use the default boto3 credential chain when
`AWS_ACCESS_KEY_ID` is not set, and it does not support per-request
`custom_api_key`.
## Gateway settings

- `MODEL_GATEWAY_API_KEYS`: JSON object mapping stable names to server-side gateway client keys.
- `MODEL_GATEWAY_API_KEY`: single client key used to call a gateway server.
- `MODEL_GATEWAY_URL`: routes client calls through a gateway server.
- `MODEL_GATEWAY_HMAC_SECRET`: signs gateway history blobs server-side.
