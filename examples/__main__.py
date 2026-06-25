from __future__ import annotations

import sys
from pathlib import Path

# Allow path execution (`uv run python examples`) from a source checkout.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _main() -> None:
    from examples._cli import main

    main()


if __name__ == "__main__":
    _main()
