from __future__ import annotations

import argparse
from typing import cast

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model-library chat UI.")
    _ = parser.add_argument("--host", default="127.0.0.1")
    _ = parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    host = cast(str, args.host)
    port = cast(int, args.port)

    uvicorn.run("model_library.web.chat:app", host=host, port=port)


if __name__ == "__main__":
    main()
