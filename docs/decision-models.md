# Typed-decision models

Models that take a `state` and typed `questions` and return a typed answer for each, instead of generating text: `choice` (one option plus a probability for every option), `noul` (a probability from 0 to 1) and `score` (a value on an ordered scale). They are served by one provider (`decision`) through OpenRouter's Decisions API (`https://openrouter.ai/api/alpha/decisions`) with `OPENROUTER_API_KEY`, or, for entries with `provider_properties.decision_api: openai`, OpenAI's (`https://api.openai.com/v1/decisions`) with `OPENAI_API_KEY`. Through the gateway the gateway's key pays; called directly, the caller's own key (or `custom_api_key`) does. A `DecisionConfig` override changes only the fields it sets, so `decision_api` stays as the registry entry has it.

| Key | Sends (pinned OpenRouter slug) | Input price / 1M tokens | Context |
| --- | --- | --- | --- |
| `decision/jev-1.13.0` | `typesafe/jev-1.13-20260917` | $0.042 | 64,000 |
| `decision/mercury-decide` | `inception/mercury-decide-20260930` | $0.02 | 65,536 |
| `decision/microsoft-decision-1` | `microsoft/microsoft-decision-1-20261009` | $0.042 | 32,768 |
| `decision/gpt-6-luna` | `gpt-6-luna` on OpenAI (no dated snapshot exists) | $0.10, $0.20 past 272K | 921,600 |

Output is free. OpenRouter prices and context windows are OpenRouter's; OpenAI's price is from its [Decisions guide](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability) and its context window is the largest input `/v1/decisions` accepted. Every OpenRouter key sends a pinned dated slug, never an alias. `metadata.served_model` is the model OpenRouter reports, when it reports one. OpenRouter has reported TypeSafe, Inception and Microsoft as the serving providers (kept as `upstream_provider`); the adapter sends no routing constraint. OpenRouter marks the Decisions API "alpha", so its path may change.

## Calling one

The query's single text input is the decision request as JSON; `model` is never part of it. The result's `output_text` is `{"answers": {...}}`.

The convenience helpers below require the full `model-library` distribution, even when routing through the gateway.
With `model-library-client`, pass `json.dumps({"state": state, "questions": questions})` directly to `model.query` and read `json.loads(result.output_text)["answers"]`.

```python
from model_library.providers.delegates.openrouter.decision import decision_request, parse_decision_output
from model_library.registry_utils import get_registry_model

model = get_registry_model("decision/jev-1.13.0")  # in-process, or through the gateway when MODEL_GATEWAY_URL is set
result = await model.query(decision_request(
    "The API returns 500 on every call.",
    {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Charges", "engineering": "Bugs"}}},
))
parse_decision_output(result.output_text)["team"]  # {"choice": "engineering", "probabilities": {...}, ...}
```

For OpenAI the request and answers are translated, so every key takes and returns the same shapes: `noul` is sent as `predicate`, choice `criteria` (`{value: description}`) as `choices`, score `criteria` (a list, lowest first) as `levels`, and the answers array comes back keyed by question name with `probabilities` keyed by value and a score's labels as `legend`, as OpenRouter returns them. A non-string `state` is sent as its JSON text. Images, which OpenAI's API accepts, are not exposed.

`metadata.in_tokens` / `out_tokens` come from the API's usage and cost comes from the registry price. OpenRouter's own `usage.cost` and `provider` (OpenAI reports neither) are kept in `metadata.extra` as `upstream_cost` and `upstream_provider` for reconciliation. A response with no usage raises instead of reporting $0. An OpenAI `refusal` answer is returned as `{"type": "refusal"}` and sets `finish_reason` to `content_filter` (raw `refusal`); the other answers are kept.

Not supported: system prompts, images, files, tools, `output_schema`, query options and `custom_endpoint`. A request that is not a valid decision request, or that cannot be sent as JSON (a lone surrogate, `NaN`), raises `DecisionError` before any request, as does `custom_endpoint` when the model is built (the Decisions URL is fixed). `output_schema` is rejected by the shared capability check. A response is checked only for an answers object, an answer for every question, and token usage; the answers themselves are not validated, so read them with `parse_decision_output`. `DecisionError` is never retried.

## Errors and retries

| Upstream answer | Raised | Retried |
| --- | --- | --- |
| 4xx other than 408, 409, 425, 429 (400 and 422 a request it rejects, 401 and 403 a key it rejects, 402 no credits, 404 an unknown model) | `DecisionRequestRejected` (`NoRetryException`) | No: one upstream call, fails at once |
| 429, 5xx, 408, 409, 425, timeouts, connection errors | the SDK error | Yes, by the library retrier (`ExponentialBackoffRetrier`) |
| 200 without answers or usage | `DecisionError` | No |

`DecisionRequestRejected` keeps the upstream message, `status_code` and error `code`, and reports the SDK error's class name (`UnprocessableEntityError`, `BadRequestError`, ...) as `exception_type`, also in the gateway's error envelope, so clients that classify by status or type see what they saw before. Without it a persistent 422 was retried for the library's whole backoff schedule (about 51 minutes), because `UnprocessableEntityError` is in `RETRIABLE_EXCEPTIONS`.

## How it works

`DecisionModel` (`model_library/providers/delegates/openrouter/decision.py`) reuses `OpenRouterModel`'s client setup (the key, `OPENROUTER_API_KEY` or a bring-your-own `custom_api_key`, and the retries), but not its chat query or endpoint; with `decision_api: openai` it points that client at `https://api.openai.com/v1` with `OPENAI_API_KEY`. It owns the Decisions POST and the translation of the response into a result, with cost from the registry price. The pinned slug is each registry entry's `provider_endpoint`.

## Deploying

`OPENROUTER_API_KEY` and `OPENAI_API_KEY` are already in the gateway's provider secrets. A deployed gateway serves these keys once both the provider code is deployed and the registry is published; the two pipelines are triggered separately.
