# ValSmith model catalog

ValSmith reads available models, credit rates, and plan requirements from the
[production Model Gateway](https://model-gateway.vals.ai/registry?include_valsmith=true)
with its Gateway credential. It lists public models that support tool calls; models
that don't, such as typed-decision models, can't run a ValSmith harness. Set
`metadata.available_for_everyone: false` to keep any other model out, as done for
host copies of models their lab serves directly, 2024 models, deep-research models,
cyber access-program variants, models from smaller labs (Qwen stays) and, for now, Meta models.
Alibaba, DeepSeek, Kimi, MiniMax and zAI models are listed only through Fireworks:
`VALSMITH_FIREWORKS_ONLY_PROVIDERS` leaves out their direct keys in ValSmith alone.

To change credits per task or required plans, edit `VALSMITH_OVERRIDES` in
`model_gateway/valsmith_catalog.py`, then deploy the Gateway. Models default to
1 credit per task and Individual access. Aliases inherit the base model's policy
unless overridden.
