# Model Library

Open-source model library for interacting with a variety of LLM providers. Originally developed for internal use at [vals.ai](https://vals.ai/) benchmarks. This tool is designed to be a general-purpose solution for projects that need a unified interface for multiple model providers.

Requires Python 3.11+.

```bash
pip install model-library
```

**Note:** This library is undergoing rapid development. Expect breaking changes.

## Start here

| Task | Start here |
| --- | --- |
| Use the installed Python library | [Usage](#usage) and [Environment setup](#environment-setup) |
| Browse models from a repo checkout | [Browse models](#browse-models) |
| Configure provider API keys | [API keys guide](docs/api-keys.md) |
| Run the gateway | [Gateway guide](docs/gateway.md) |
| Run examples from a repo checkout | [Examples guide](examples/README.md) |
| Run tests | [Tests guide](tests/README.md) |
| Contribute to model registry config | [Model config README](model_library/config/README.md) |

## Features

### Providers

- AI21 Labs
- Alibaba
- Amazon Bedrock
- Anthropic
- Azure OpenAI
- Cohere
- DeepSeek
- Fireworks
- Google Gemini
- Mistral
- Perplexity
- Together AI
- OpenAI
- X AI
- ZhipuAI (zai)

### Browse models

From a repo checkout, run this to browse the model registry interactively:

```bash
python -m scripts.browse_models
```

Installed-package users can inspect providers through the Python API:

```python
from model_library.registry_utils import get_model_names_by_provider, get_provider_names

print(get_provider_names())
print(get_model_names_by_provider("chosen-provider"))
```

### Supported input

- Images
- Files
- Tools with full history
- Batch
- Reasoning
- Custom parameters

## Usage

> **Warning:** This query makes a real provider call. Configure the provider key first, expect provider billing/rate limits, and do not send sensitive prompts unless intentional. Query logging can include request and response content; use `set_logging(enable=False)` or a redacting logger for sensitive workloads.

```python
import asyncio

from model_library import model


async def main():
    llm = model("anthropic/claude-opus-4-1-20250805-thinking")

    result = await llm.query(
        "What is QSBS? Explain your thinking in detail and make it concise."
    )

    print(result.output_text)
    print(result.metadata)  # cost, token, and performance telemetry


if __name__ == "__main__":
    asyncio.run(main())
```

The model registry holds model attributes such as reasoning, file support, tool support, and max tokens. You may also use models not included in the registry:

```python
from model_library import raw_model
from model_library.base import LLMConfig

llm = raw_model("grok/grok-code-fast", LLMConfig(max_tokens=10000))
```

You can extend the registry with custom configs from a local YAML file or URL using the same format as the bundled provider configs:

```python
from model_library import load_custom_model_configs, load_latest_vals_model_configs

load_custom_model_configs("/path/to/my_models.yaml")
load_custom_model_configs("https://raw.githubusercontent.com/org/repo/main/models.yaml")

# Pull latest bundled configs from GitHub without upgrading the package.
load_latest_vals_model_configs()
```

Root logger is named `llm`. To disable logging:

```python
from model_library import set_logging

set_logging(enable=False)
```

## Environment setup

The model library reads provider API keys from environment variables, including:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

You can also set values through `model_library_settings`:

```python
from model_library import model_library_settings

model_library_settings.set(MY_KEY="my-key")
```

See [docs/api-keys.md](docs/api-keys.md) for supported provider key names, [docs/config.md](docs/config.md) for YAML config structure, and [docs/result.md](docs/result.md) for result metadata, cost, tokens, and performance telemetry.

## Example snippets

The snippets below are excerpts. For runnable files and setup prerequisites, see [examples/README.md](examples/README.md).

### System prompt

```bash
uv run python examples/quickstart.py
```

```python
await llm.query(
    [
        SystemInput(
            text="You are a pirate. Answer in a pirate style under 10 words."
        ),
        TextInput(text="Hello, how are you?"),
    ],
)
```

### Image/file input

```bash
uv run python examples/inputs.py
```

```python
red_image_content = b"..."

await llm.query(
    [
        TextInput(text="What color is the image?"),
        FileWithBase64(
            type="image",
            name="red_image.png",
            mime="png",
            base64=base64.b64encode(red_image_content).decode("utf-8"),
        ),
    ]
)
```

### Tool calls

```bash
uv run python examples/tools.py <model> [--mode agent|direct|both]
```

```python
tools = [
    ToolDefinition(
        name="get_weather",
        body=ToolBody(
            name="get_weather",
            description="Get current temperature in a given location",
            properties={
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                },
            },
            required=["location"],
        ),
    )
]

output1 = await llm.query(
    [TextInput(text="What is the weather in SF right now?")],
    tools=tools,
)

output2 = await llm.query(
    [
        ToolResult(tool_call=output1.tool_calls[0], result="25C"),
        TextInput(text="Also include at least 8 emojis in your answer."),
    ],
    history=output1.history,
    tools=tools,
)
```

### Starter examples

Run these examples from a repo checkout. See [examples/README.md](examples/README.md) for validator coverage, model-release checks, agent loops, and one-off demos:

| Example | Command |
| --- | --- |
| Model validator | `uv run python examples/validate_model.py <model> [--json]` |
| Quickstart | `uv run python examples/quickstart.py` |
| Inputs | `uv run python examples/inputs.py` |
| Tools | `uv run python examples/tools.py <model> [--mode agent|direct|both]` |

Use the validator first for model-release checks. It exercises core text, declared image/file transports, bounded agent tool use, reasoning evidence, prompt caching, configured/live rate limits, and configured pricing. List example commands with `uv run examples` or `uv run python -m examples`. If you already activated `.venv`, bare `python examples/...` commands work too.

## Docs

- [Provider API Keys](docs/api-keys.md) — provider key names and gateway key rules
- [Model Configuration](docs/config.md) — YAML config structure, inheritance, deprecation, settings
- [Gateway](docs/gateway.md) — centralized FastAPI model proxy
- [Agent](docs/agent.md) — tool-augmented conversation loop
- [ATIF](docs/atif.md) — agent trajectory interchange format
- [Conductor](docs/conductor.md) — multi-agent conversation orchestration
- [Result Metadata](docs/result.md) — result shape, cost, tokens, and performance telemetry
- [Token Retry & Benchmark Queue](docs/token-retry.md) — rate-limit-aware scheduling via Redis

## Architecture

Designed to abstract different LLM providers:

- **LLM base class:** common interface for all models.
- **Model registry:** central registry that loads model configurations from YAML files.
- **Provider-specific implementations:** concrete classes for providers such as OpenAI, Google, and Anthropic.
- **Data models:** Pydantic models for input and output types such as `TextInput`, `FileWithBase64`, `ToolDefinition`, and `ToolResult`.
- **Retry logic:** retry strategies for provider errors and rate limiting.

## Contributing

### Setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management. A Makefile is provided to help with development.

```bash
make install
```

### Makefile commands

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies |
| `make test` | Run unit tests |
| `make test-integration` | Run integration tests; requires API keys and makes live provider calls |
| `make style` | Format and lint with fixes |
| `make style-check` | Check formatting and lint without fixes |
| `make typecheck` | Run basedpyright |
| `make config` | Generate `all_models.json` |
| `make run-models` | Run all configured model smoke tests |
| `make browse_models` | Browse models interactively |

The current Makefile help mentions `make test-all`, but that target has no recipe and does not run unit plus integration tests. Run `make test` and `make test-integration` separately.

### Testing

Unit tests do not require API keys:

```bash
make test
```

Integration tests require provider API keys and make live calls:

```bash
make test-integration
```

See [tests/README.md](tests/README.md) for model selection, raw pytest usage, and environment setup.
