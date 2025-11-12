import json
from datetime import date
from typing import Any, cast

from model_library.register_models import get_model_registry

PATH = "./model_library/config/all_models.json"


def remove_none(d: Any | dict[str, Any]) -> Any | dict[str, Any]:
    if isinstance(d, dict):
        d = cast(dict[str, Any], d)
        return {
            k: remove_none(v)
            for k, v in d.items()
            if (v is not None and (k != "costs_per_million_token" or v != {}))
            or k in {"max_token_output", "training_cutoff", "description"}
        }
    elif isinstance(d, list):
        d = cast(list[Any], d)
        return [remove_none(v) for v in d if v is not None]
    else:
        return d


model_registry = get_model_registry()
model_registry = dict(
    sorted(
        model_registry.items(),
        key=lambda kv: (kv[1].release_date or date.min, kv[0]),
        reverse=True,
    )
)

wrapper: dict[str, dict[str, Any]] = {}
for k, v in model_registry.items():
    model_data = v.model_dump(mode="json")

    wrapper[k] = remove_none(model_data)

with open(PATH, "w") as file:
    json.dump(wrapper, file, indent=4)
print(f"Successfully wrote all_models.json to {PATH}")
