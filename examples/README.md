# Model Library Examples

Examples are runnable demos for `model_library`. They make real provider calls unless you choose a local/custom endpoint.

## Before you run examples

> **Warning:** Use sandbox or least-privilege API keys, expect billing/rate limits, and do not send sensitive prompts unless intentional.

> **Note:** Examples load environment variables from the repo root `.env`. In internal checkouts, a missing `.env` may trigger the internal AWS Secrets Manager bootstrap and write a local `.env`. Public users should create `.env` themselves or export provider API keys such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.

## Core examples

Run these from a repo checkout. Start with the validator when checking a model release:

```bash
uv run python examples/validate_model.py <model> [--json]
uv run python examples/quickstart.py
uv run python examples/inputs.py
uv run python examples/tools.py <model> [--mode agent|direct|both]
```

If you already activated `.venv`, bare `python examples/...` commands work too.

The validator is the broad model smoke check. It exercises core text, declared image/file transports, bounded agent tool use, reasoning evidence for reasoning models, prompt-cache read/write metadata, configured and live rate limits, and configured pricing. Human output groups results by section; one-row sections render as `Caching pass` or `Pricing warn - no configured pricing`, with details indented below. `--json` emits the full validation report, including aggregated usage and configured pricing under `usage.price_per_million_tokens`.

Pass a model string after a command to override its default model:

```bash
uv run python examples/quickstart.py openai/gpt-5-nano-2025-08-07
```

## Model release checks

These are the examples most worth rerunning for model releases after the validator because they report rate-limit or retry behavior. They may bill, rate-limit, upload provider-side files, fetch external URLs, create local logs, require Redis, or intentionally stress provider limits.

```bash
uv run python examples/diagnostics/rate_limit.py <model>
uv run python examples/diagnostics/rate_limit.py <model> --probe
uv run python examples/diagnostics/token_retry.py <model>
```

`rate_limit.py --probe` is a quota-consuming TPM stress probe. Probe mode skips provider rate-limit preflight and estimates from successful tokens and 429s. It asks for confirmation before sending traffic; pass the hidden `--confirm-probe` flag for noninteractive runs. It sends bounded traffic under a 20M-token default budget that includes calibration, disables backoff/token retry wrappers, excludes unknown non-rate-limit errors from the TPM estimate, prints Rich live progress plus plan/round summaries, and reports a tightened TPM estimate range plus whether the result is clean, approximate, lower-bound, or failed. If no 429 is observed before the budget is exhausted, the result is reported as a lower bound such as `~20,000,000+`. Each probe round has a 60s wall-clock cap and stops earlier on budget, 429s, or errors. By default, a second round waits for refill and probes the midpoint of the first bounded range when budget remains.

## One-off demos

Less common demos live under `extras`. They are useful references for specific features, but usually are not model-release checks. List commands with:

```bash
uv run examples
```

You can also run the examples directory as a module:

```bash
uv run python -m examples
```

The command list puts core examples first, then recurring model-release checks, agent/tool loops, starter demos, one-off feature examples, and setup/compatibility asides such as custom endpoints and Google modes. The prompt-caching demo reports cached tokens and a `Caching` category result. The validator checks all implemented image/file transports and also reports caching, reasoning, rate-limit, and pricing diagnostics; other demos usually default to one representative live path and use flags such as `--mode both` for broader coverage.

## Support modules

- `examples/setup.py`, `examples/utils.py`, and `examples/data/` are support modules used by runnable demos.
