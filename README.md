# Model Library

Open-source model library for interacting with a variety of LLM providers. Originally developed for internal use at [vals.ai](https://vals.ai/) benchmarks. This tool is designed to be a general-purpose solution for any project requiring a unified interface for multiple model providers.

`pip install model-library`

**Note**: This library is undergoing rapid development. Expect breaking changes.

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

Run `python -m scripts.browse_models` to browse the model registry or

```python
from model_library.registry_utils import get_model_names_by_provider, get_provider_names

print(get_provider_names())
print(get_model_names_by_provider("chosen-provider"))
```

### Supported Input

- Images
- Files
- Tools (with full history)
- Batch
- Reasoning
- Custom Parameters

## Usage

Here is a basic example of how to query a model:

```python
import asyncio
from model_library import model

async def main():
    # Load a model from the registry
    llm = model("anthropic/claude-opus-4-1-20250805-thinking")

    # Display the LLM instance
    llm.logger.info(llm)
    # or print(llm)

    # Query the model with a simple text input
    result = await llm.query(
        "What is QSBS? Explain your thinking in detail and make it concise."
    )

    # Logger automatically logs the result

    # Display only the output text
    llm.logger.info(result.output_text)


if __name__ == "__main__":
    asyncio.run(main())
```

The model registry holds model attributes, ex. reasoning, file support, tool support, max tokens. You may also use models not included in the registry.

```python
from model_library import raw_model
from model_library.base import LLMConfig

model = raw_model("grok/grok-code-fast", LLMConfig(max_tokens=10000))
```

Root logger is named "llm". To disable logging:

```python
from model_library import set_logging

set_logging(enable=False)
```

### Environment Setup

The model library will use:
- Environment varibles for API keys
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    - ...
    
- Variables set through model_library.settings
```python
from model_library import model_library_settings

model_library_settings.set(MY_KEY="my-key")
```


### System Prompt

```bash
python -m examples.basics
```

```python
await model.query(
    [TextInput(text="Hello, how are you?")],
    system_prompt="You are a pirate, answer in the speaking style of a pirate. Keeps responses under 10 words",
)
```

### Image/File Input

Supports base64, url, and file id (file upload)

```bash
python -m examples.images
```

```python
red_image_content = b"..."

await model.query(
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

### Tool Calls

```bash
python -m examples.tool_calls
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

output1 = await model.query(
    [TextInput(text="What is the weather in SF right now?")],
    tools=tools,
)

output2 = await model.query(
    [
        # assume one tool call was made
        ToolResult(tool_call=output1.tool_calls[0], result="25C"),
        TextInput(
            text="Also, includes some weird emojies in your answer (at least 8 of them)"
        ),
    ],
    history=output1.history,
    tools=tools,
```

### Full examples

You can run `make examples` (default models) or `make example <model>` to run all examples.

`python -m examples.basics`

`python -m examples.images`

`python -m examples.files`

`python -m examples.tool_calls`

`python -m examples.embeddings`

`python -m examples.advanced.batch`

`python -m examples.advanced.custom_retrier`

`python -m examples.advanced.stress`

`python -m examples.advanced.deep_research`


## Architecture

Designed to abstract different LLM providers:

- **LLM Base Class**: An abstract base class that defines a common interface for all models
- **Model Registry**: A central registry that loads model configurations from YAML files
- **Provider-Specific Implementations**: Concrete classes for each provider (e.g., OpenAI, Google, Anthropic) that inherit from the `LLM` base class
- **Data Models**: A set of `pydantic` models for representing various input and output types, such as `TextInput`, `FileWithBase64`, `ToolDefinition`, and `ToolResult`. This ensures code is model agnostic, and easy to maintain.
- **Retry Logic**: A set of retry strategies for handling errors and rate limiting

## Contributing

### Setup

We use [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management.
A Makefile is provided to help with development.

To install dependencies, run:

```bash
make install
```

### Makefile commands

```bash
make install          Install dependencies"
make test             Run unit tests"
make test-integration Run integration tests (requires API keys)"
make test-all         Run all tests (unit + integration)"
make style            Lint & Format"
make style-check      Check style"
make typecheck        Typecheck"
make config           Generate all_models.json"
make run-models       Run all models"
make examples         Run all examples"
make examples <model> Run all examples with specified model"
make browse-models    Browse all models"
```

### Testing

#### Unit Tests

Unit tests do not require API keys

```bash
make test-unit
```

#### Integration Tests

Make sure you have API keys configured

```bash
make test-integration
```
