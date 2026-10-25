# ValSmith model catalog

ValSmith reads available models, credit rates, and plan requirements from the
[production Model Gateway](https://model-gateway.vals.ai/registry?include_valsmith=true)
with its Gateway credential.

To change credits per task or required plans, edit `VALSMITH_OVERRIDES` in
`model_gateway/valsmith_catalog.py`, then deploy the Gateway. Models default to
1 credit per task and Individual access. Aliases inherit the base model's policy
unless overridden.
