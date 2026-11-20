# Typed-decision models

Models that take a `state` and typed `questions` and return a typed answer for each, instead of generating text: `choice` (one option plus a probability for every option), `noul` (a probability from 0 to 1) and `score` (a value on an ordered scale). They are served by one provider (`decision`) through OpenRouter's Decisions API (`https://openrouter.ai/api/alpha/decisions`) with `OPENROUTER_API_KEY`, or, for entries with `provider_properties.decision_api: openai`, OpenAI's (`https://api.openai.com/v1/decisions`) with `OPENAI_API_KEY`, or, with `decision_api: perplexity`, Perplexity's (`https://api.perplexity.ai/v1/decisions`) with `PERPLEXITY_API_KEY`, or, with `decision_api: google`, Google's (`https://generativelanguage.googleapis.com/v1beta/decisions`) with `GOOGLE_API_KEY`. Through the gateway the gateway's key pays; called directly, the caller's own key (or `custom_api_key`) does. A `DecisionConfig` override changes only the fields it sets, so `decision_api` stays as the registry entry has it.

| Key | Sends (pinned OpenRouter slug) | Input price / 1M tokens | Context |
| --- | --- | --- | --- |
| `decision/joey-1` | `baseten/joey-1-20261029` | $0.01 | 16,384 |
| `decision/mammoth-1` | `baseten/mammoth-1-20261021` | $0.10 | 16,384 |
| `decision/elephant-1` | `baseten/elephant-1-20261021` | $0.035 | 16,384 |
| `decision/jev-1.13.0` | `typesafe/jev-1.13-20260917` | $0.042 | 64,000 |
| `decision/jev-1.14` | `typesafe/jev-1.14-20261014` | $0.042 | 262,144 |
| `decision/microsoft-decision-1` | `microsoft/microsoft-decision-1-20261009` | $0.042 | 32,768 |
| `decision/gpt-6-luna` | `gpt-6-luna` on OpenAI (no dated snapshot exists) | $0.10, $0.20 past 272K | 921,600 |
| `decision/gemini-decision-1` | `gemini-decision-1` on Google (no dated snapshot exists) | $0.10 | 262,144 |
| `decision/decider-v1.2` | `pplx-decider-v1.2` on Perplexity (no dated snapshot exists) | $0.02 | 262,144 |
| `decision/decider-flash-v1.2` | `pplx-decider-flash-v1.2` on Perplexity (no dated snapshot exists) | $0.01 | 262,144 |
| `decision/decider-v1.3` | `pplx-decider-v1.3` on Perplexity (no dated snapshot exists) | $0.02 | 262,144 |
| `decision/decider-flash-v1.3` | `pplx-decider-flash-v1.3` on Perplexity (no dated snapshot exists) | $0.01 | 262,144 |
| `decision/drex-v1.5` | `nace-ai/drex-v1.5-20261009` | $0.04 | 131,072 |
| `decision/drex-v2.0` | `nace-ai/drex-v2.0-20261009` | $0.06 | 131,072 |
| `decision/drex-max` | `nace-ai/drex-max-20261016` | $0.06 | 262,144 |
| `decision/d1` | `liquid/d1-20260930` | $0.02 | 65,536 |
| `decision/d1-ultra` | `liquid/d1-ultra-20261015` | $0.042 | 32,768 |
| `decision/d1.2` | `liquid/d1.2-20261116` | $0.02 | 65,536 |
| `decision/d1.2-ultra` | `liquid/d1.2-ultra-20261116` | $0.042 | 65,536 |
| `decision/kev-4b` | `jaredpalmer/kev-4b-20260924` | $0.042 | 8,192 |
| `decision/tev1-4b-experimental` | `togethercomputer/tev1-4b-experimental-20260923` | $0.042 | 32,768 |
| `decision/classify-1` | `scaledown/classify-1-20260928` | $0.04 | 1,000,000 |
| `decision/clef-omni` | `cloudflare/clef-omni-20261009` | $0.15 | 65,536 |
| `decision/clef` | `cloudflare/clef` on host `cloudflare` (no dated snapshot exists) | $0.24 | 65,536 |
| `decision/clef-flash` | `cloudflare/clef-flash` on host `cloudflare` (no dated snapshot exists) | $0.038 | 24,576 |
| `decision/solar-decide` | `upstage/solar-decide-20260928` on host `upstage/zdr` | $0.05 | 524,288 |
| `decision/solar-decide-flash` | `upstage/solar-decide-flash-20261008` on host `upstage/zdr` | $0.05 | 524,288 |

Output is free. Cached input on Drex Max, D1, D1 Ultra, Solar Decide and Solar Decide Flash costs the same as uncached input, as OpenRouter lists it. OpenRouter prices and context windows are OpenRouter's; OpenAI's price is from its [Decisions guide](https://developers.openai.com/api/docs/guides/decisions#pricing-and-availability) and its context window is the largest input `/v1/decisions` accepted. Perplexity's prices and its 256K input limit are from its [API reference](https://docs.perplexity.ai/api-reference/decisions-post); it takes OpenRouter's request and answer shapes. Google's price is from its [pricing page](https://ai.google.dev/gemini-api/docs/pricing#gemini-decision) and its context window from its [model page](https://ai.google.dev/gemini-api/docs/models/gemini-decision-1); it takes OpenRouter's questions and returns its answers, with `state` sent as `input` (JSON text if not a string), the key sent as `x-goog-api-key` and token counts read from `usage.total_input_tokens` / `total_output_tokens`. Every OpenRouter key sends a pinned dated slug, never an alias, except Clef and Clef Flash, which OpenRouter serves only under an alias. `metadata.served_model` is the model OpenRouter reports, when it reports one. OpenRouter has reported TypeSafe, Inception, Microsoft and BaseTen as the serving providers (kept as `upstream_provider`); an entry with `provider_properties.openrouter_provider_only` is sent as `provider: {"only": [...]}` so OpenRouter routes only to those hosts (Clef's hosts differ in price and context; Upstage also serves Solar Decide and Solar Decide Flash without ZDR). OpenRouter marks the Decisions API "alpha", so its path may change.

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

`DecisionModel` (`model_library/providers/delegates/openrouter/decision.py`) reuses `OpenRouterModel`'s client setup (the key, a bring-your-own `custom_api_key`, and the retries), but not its chat query. `DECISION_APIS` holds each API's base URL, key setting (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `PERPLEXITY_API_KEY`, `GOOGLE_API_KEY`) and how its request, auth header and usage differ from OpenRouter's; the client is pointed at that URL and posts to `/decisions`. Request validation (`parse_decision_request`) and the result (`decision_result`) are shared with `tinker_decision`. Cost comes from the registry price, and the pinned slug is each registry entry's `provider_endpoint`.

## Deploying

`OPENROUTER_API_KEY`, `OPENAI_API_KEY` and `GOOGLE_API_KEY` are already in the gateway's provider secrets. The Perplexity entries also need `PERPLEXITY_API_KEY` there. A deployed gateway serves these keys once both the provider code is deployed and the registry is published; the two pipelines are triggered separately.

## Tinker checkpoints

The `tinker_decision` provider (`model_library/providers/tinker_decision.py`, entries in `tinker_decision_models.yaml`) takes the same decision request and returns the same answers as `decision`, so a benchmark's decision adapter works with either. It answers with Tinker Cookbook's [decisions recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/decisions/), so a checkpoint trained with it is scored the way it was trained. Each question becomes a recipe `Decision`: the text is `state` (JSON if not a string), a blank line and the question's `instructions`; a `choice` question's choices are its `criteria` in order, a `noul` question's are `yes` / `no`. The recipe renders it with the base model's recommended chat renderer, prefills the answer, and takes the softmax of the choice tokens' log-probabilities from one `max_tokens=1` sample; there is no classifier head. A `choice` answer is `{"type": "choice", "choice": <most likely>, "probabilities": {...}}` and a `noul` answer `{"type": "noul", "noul": <P(yes)>}`. Questions of one request are sampled concurrently; `score` questions are rejected. `in_tokens` is the rendered prompt tokens and `out_tokens` one per question.

A checkpoint is one registry entry:

```yaml
  tinker_decision/<team>:
    provider_endpoint: tinker://<run>:train:0/sampler_weights/<name>
    provider_properties:
      base_model: Qwen/Qwen3-8B   # the checkpoint's base model: tokenizer, renderer and context
      # renderer: qwen3           # only to override the recommended renderer
```

The registry has one entry per Tinker base model: `tinker_decision/qwen3-8b` (32K context), `tinker_decision/qwen3.5-4b` and `tinker_decision/qwen3.5-9b` (64K), and `tinker_decision/qwen3.5-4b-256k` (Tinker's 256K variant of Qwen3.5-4B). Each entry's cache-read price is Tinker's cached prefill price from its [models.json](https://tinker-docs.thinkingmachines.ai/tinker/models.json): $0.039 per million tokens for Qwen3-8B, $0.066 for both Qwen3.5-4B variants and $0.06 for Qwen3.5-9B. Pretrained-only bases such as Qwen3.5-9B-Base are left out: with their `role_colon` renderer the recipe's `DecisionRenderer` cuts the prompt off inside its own system message, so every request gets the same answer. A base-model ID as `provider_endpoint` samples the untuned base, the parent to compare a checkpoint against. Any checkpoint on that base also runs under the base's entry, without an entry of its own, by setting `checkpoint` in the request's provider config:

```python
model = get_registry_model(
    "tinker_decision/qwen3-8b",
    override_config=LLMConfig(provider_config=TinkerDecisionConfig(checkpoint="tinker://<run>:train:0/sampler_weights/<name>")),
)
```

Through the gateway the same `config.provider_config` is sent with the query, and each checkpoint gets its own cached model. `TINKER_API_KEY`'s project must be able to load the checkpoint: it is in that project, the owner has run `tinker checkpoint publish` on it, or the key's user has the Project Sampler role on the owner's project. The `tinker` and `tinker-cookbook` packages are the `tinker` extra (they pull in torch and transformers); without it a query raises `DecisionError`. The gateway image does not install the extra yet and `TINKER_API_KEY` is not in its provider secrets.
