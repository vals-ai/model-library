"""Move a model from active config to deprecated config.

Resolves the full model config from the registry (with all inheritance applied),
then writes a self-contained entry to the deprecated file.

Usage:
    uv run python scripts/deprecate_model.py <model_key>
    make deprecate model=openai/gpt-4o-2024-05-13
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import yaml

CONFIG_DIR = Path(__file__).parent.parent / "model_library" / "config"
DEPRECATED_DIR = CONFIG_DIR / "deprecated"
MODEL_KEY_PATTERN = re.compile(r"^  \S+/\S+.*:$")

# fields added by ModelConfig post-processing or inherited from provider base —
# not needed in the deprecated YAML entry
EXCLUDE_FIELDS = {"provider_name", "full_key", "slug"}


def find_model_lines(filepath: Path, model_key: str) -> tuple[int, int] | None:
    """Find the start and end line indices of a model entry in a YAML file."""
    lines = filepath.read_text().splitlines()
    target = f"  {model_key}:"
    start = None

    for i, line in enumerate(lines):
        if line.rstrip() == target or line.startswith(target + " "):
            start = i
            continue
        if start is not None and (
            MODEL_KEY_PATTERN.match(line)
            or (line and not line.startswith(" ") and line.rstrip().endswith(":"))
            or line.strip().startswith("# END-INTERNAL")
        ):
            end = i
            while end > start and not lines[end - 1].strip():
                end -= 1
            return start, end

    if start is not None:
        end = len(lines)
        while end > start and not lines[end - 1].strip():
            end -= 1
        return start, end

    return None


def find_model_file(model_key: str) -> tuple[Path, int, int]:
    """Search all active YAML files for the model key."""
    for filepath in sorted(CONFIG_DIR.glob("*.yaml")):
        result = find_model_lines(filepath, model_key)
        if result:
            return filepath, result[0], result[1]

    print(f"error: model '{model_key}' not found in any active config file")
    sys.exit(1)


def remove_none(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove None values."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            result[k] = remove_none(cast(dict[str, Any], v))
        elif isinstance(v, list):
            result[k] = [x for x in cast(list[Any], v) if x is not None]
        else:
            result[k] = v
    return result


def model_to_yaml_entry(model_key: str, data: dict[str, Any]) -> str:
    """Convert a model dict to a YAML entry with 2-space base indent."""
    raw = yaml.dump(
        {model_key: data},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    lines: list[str] = []
    for line in raw.splitlines():
        lines.append(f"  {line}" if line.strip() else line)
    return "\n".join(lines)


def deprecate(model_key: str) -> None:
    # load the resolved model from registry
    from model_library import model_library_settings

    model_library_settings.set(MODEL_LIBRARY_INCLUDE_DEPRECATED="false")

    from model_library.register_models import get_model_registry

    registry = get_model_registry()
    model_config = registry.get(model_key)
    if model_config is None:
        print(f"error: model '{model_key}' not found in registry")
        sys.exit(1)

    # find and remove from active YAML file (text-based to preserve formatting)
    filepath, start, end = find_model_file(model_key)
    lines = filepath.read_text().splitlines()
    print(f"found {model_key} in {filepath.name} (lines {start + 1}-{end})")

    # find or create the deprecated file
    deprecated_file = DEPRECATED_DIR / filepath.name
    if not deprecated_file.exists():
        base_lines: list[str] = []
        for line in lines:
            if line.startswith("base-config:") or (
                base_lines and (line.startswith("  ") or not line.strip())
            ):
                base_lines.append(line)
            elif base_lines and line.strip() and not line.startswith("  "):
                break
        base_text = "\n".join(base_lines)
        deprecated_file.write_text(
            f"{base_text}\n\ndeprecated:\n"
            f"  base-config:\n"
            f"    metadata:\n"
            f"      deprecated: true\n"
        )
        print(f"  created {deprecated_file}")

    # dump fully resolved model, clean up
    model_data = remove_none(model_config.model_dump(mode="json"))
    for field in EXCLUDE_FIELDS:
        model_data.pop(field, None)
    # remove empty provider_properties
    if not model_data.get("provider_properties"):
        model_data.pop("provider_properties", None)
    # remove metadata.deprecated — inherited from deprecated block base-config
    if "metadata" in model_data:
        model_data["metadata"].pop("deprecated", None)
        if not model_data["metadata"]:
            del model_data["metadata"]
    model_entry = model_data

    # generate YAML entry
    entry_yaml = model_to_yaml_entry(model_key, model_entry)

    # insert after deprecated base-config
    dep_lines = deprecated_file.read_text().splitlines()
    insert_at = None
    for i, line in enumerate(dep_lines):
        if line.strip() == "deprecated: true":
            insert_at = i + 1
            break
    if insert_at is None:
        print("error: could not find 'deprecated: true' in deprecated file")
        sys.exit(1)

    entry_lines = [""] + entry_yaml.splitlines()
    dep_lines = dep_lines[:insert_at] + entry_lines + dep_lines[insert_at:]
    deprecated_file.write_text("\n".join(dep_lines) + "\n")
    print(f"  inserted at top of deprecated/{filepath.name}")

    # remove model entry from active file
    remove_end = end
    while remove_end < len(lines) and not lines[remove_end].strip():
        remove_end += 1
    new_lines = lines[:start] + lines[remove_end:]
    filepath.write_text("\n".join(new_lines) + "\n")
    print(f"  removed from {filepath.name}")

    # regenerate config
    print("  running make config...")
    result = subprocess.run(
        ["make", "config"],
        cwd=CONFIG_DIR.parent.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  make config failed:\n{result.stderr}")
        sys.exit(1)
    print("  done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: uv run python scripts/deprecate_model.py <model_key>")
        sys.exit(1)
    deprecate(sys.argv[1])
