import asyncio
import base64
import sys
import uuid
from itertools import cycle
from typing import Callable

from model_library.base import LLM, FileWithBase64, TextInput
from model_library.registry_utils import get_registry_model

from ..data.images import red_image
from ..setup import console_log, setup

SPINNER = cycle(["|", "/", "-", "\\"])


def spinner_task(msg: str) -> Callable[[], None]:
    stop_event = asyncio.Event()
    spin_chars = cycle(["|", "/", "-", "\\"])

    async def spinner():
        while not stop_event.is_set():
            sys.stdout.write(f"\r{msg} {next(spin_chars)}")
            sys.stdout.flush()
            await asyncio.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(msg) + 4) + "\r")
        sys.stdout.flush()

    asyncio.create_task(spinner())
    return stop_event.set


async def batch(model: LLM):
    console_log("\n--- Batch ---\n")

    if not model.batch:
        raise Exception("Model does not have batch client")

    image_content = red_image()

    custom_id = uuid.uuid4().hex[:8]
    request = await model.batch.create_batch_query_request(
        custom_id,
        [
            TextInput(text="What color is the image?"),
            FileWithBase64(
                type="image",
                name="red_image.png",
                mime="png",
                base64=base64.b64encode(image_content).decode("utf-8"),
            ),
        ],
    )
    requests = [request]

    batch_id = await model.batch.batch_query(custom_id, requests)
    result_id = batch_id

    stop_spinner = spinner_task("Waiting for batch...")

    status_change = None
    while True:
        status = await model.batch.get_batch_status(batch_id)
        if status_change != status:
            status_change = status
            print(f"    [{status}]")

        if model.batch.is_batch_status_completed(status):
            if model.batch.is_batch_status_failed(status):
                model.logger.error(f"Batch failed: {status}")
                break
            if model.batch.is_batch_status_cancelled(status):
                model.logger.error(f"Batch cancelled: {status}")
                break

            results = await model.batch.get_batch_results(result_id)
            model.logger.info(f"Batch completed: {results}")
            break

        await asyncio.sleep(1)

    stop_spinner()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run basic examples with a specified model"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    if not model.supports_batch:
        raise Exception("Model does not support batch")

    await batch(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
