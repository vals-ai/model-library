# Model Library Configuration

This directory contains YAML configuration files that define all available models in the model-proxy library.

## Configuration Structure

Each model configuration is organized into distinct sections:

### Core Sections

#### `properties`
Model-specific technical characteristics and capabilities:
- `context_window`: Maximum context window in tokens
- `max_tokens`: Maximum output tokens the model can generate
- `training_cutoff`: Training data cutoff date (string or null)
- `reasoning_model`: Whether the model is a reasoning/thinking model

```yaml
properties:
  context_window: 200_000
  max_tokens: 32_000
  training_cutoff: "2025-03"
  reasoning_model: false
```

#### `supports`
Feature support flags indicating model capabilities:
- `images`: Supports image inputs
- `videos`: Supports video inputs
- `files`: Supports file inputs
- `batch`: Supports batch requests
- `temperature`: Supports temperature parameter
- `tools`: Supports tool/function calling

```yaml
supports:
  images: true
  files: true
  tools: true
  batch: true
  temperature: true
  videos: false
```

#### `metadata`
Vals platform-specific metadata for model availability and status:
- `deprecated`: Model is deprecated and should not be used for new projects
- `available_for_everyone`: Model is available to all users
- `available_as_evaluator`: Model can be used as an evaluator
- `ignored_for_cost`: Exclude from cost calculations

```yaml
metadata:
  deprecated: false
  available_for_everyone: true
  available_as_evaluator: false
  ignored_for_cost: false
```

#### Other Sections

- `costs_per_million_token`: Pricing information (input, output, cache, batch, context)
- `default_parameters`: Default parameter values (temperature, top_p, reasoning_effort)
- `provider_properties`: Provider-specific configuration options
- `alternative_keys`: Alternative model identifiers/aliases

## Configuration Inheritance

Configurations support hierarchical inheritance through `base-config` blocks:

### 1. Provider-level base-config
```yaml
base-config:
  company: Anthropic
  open_source: false
  supports:
    images: true
    tools: true
  metadata:
    available_for_everyone: true
```

### 2. Model-block base-config
```yaml
claude-4-models:
  base-config:
    supports:
      temperature: true
    default_parameters:
      temperature: 1

  anthropic/claude-opus-4-1-20250805:
    # Inherits from both provider and block base-configs
    properties:
      context_window: 200_000
      max_tokens: 32_000
```

### 3. Individual model overrides
Models can override any inherited configuration:

```yaml
anthropic/claude-opus-4-1-20250805:
  properties:
    context_window: 200_000
    max_tokens: 32_000
  metadata:
    available_for_everyone: false  # Override base-config
```

## Alternative Keys

Models can define alternative identifiers that map to the same configuration:

```yaml
anthropic/claude-3-5-sonnet-20241022:
  label: Claude 3.5 Sonnet Latest
  properties:
    context_window: 200_000
    max_tokens: 8_192
  alternative_keys:
    - anthropic/claude-3-5-sonnet-latest
    - anthropic/claude-3.5-sonnet-latest
```

Alternative keys can also override configuration:

```yaml
alternative_keys:
  - anthropic/claude-opus-4-1-20250805-thinking:
      properties:
        reasoning_model: true
```

## Generating all_models.json

After making changes to any YAML configuration file, regenerate the compiled configuration:

```bash
make config
```

This generates `all_models.json` which is used by the model registry at runtime.

## Schema Validation

The configuration is validated using Pydantic models defined in `register_models.py`:
- `Properties` - Model properties
- `Supports` - Feature support flags
- `Metadata` - Platform metadata
- `DefaultParameters` - Default parameter values
- `CostProperties` - Pricing information
- `ProviderProperties` - Provider-specific config (dynamically generated)

## Migration Notes

### Previous Structure (Deprecated)
The old configuration used `class_properties` which mixed support flags and metadata:

```yaml
# OLD - Do not use
class_properties:
  supports_images: true
  supports_batch_requests: true
  deprecated: false
  available_for_everyone: true
properties:
  max_token_output: 32_000
```