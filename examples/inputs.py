# ruff: noqa: E402
# Allow path execution (`uv run python examples/...`) from a source checkout.
from pathlib import Path as _Path
import sys as _sys

if __package__ in {None, ""}:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import base64
import logging
from io import BytesIO

from model_library.base import (
    LLM,
    FileWithBase64,
    FileWithBytes,
    FileWithId,
    FileWithUrl,
    TextInput,
)
from model_library.base.output import QueryResult
from model_library.registry_utils import get_registry_model

from examples.data.audio import tone_wav
from examples.data.files import secret_pdf
from examples.data.images import red_image
from examples.setup import console_log, setup

IMAGE_MIME = "png"
FILE_MIME = "application/pdf"
AUDIO_MIME = "audio/wav"


async def image_base64(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- Image ---\n")

    try:
        result = await model.query(
            [
                TextInput(text="What color is the image?"),
                FileWithBase64(
                    type="image",
                    name="red_image.png",
                    mime=IMAGE_MIME,
                    base64=base64.b64encode(red_image()).decode("utf-8"),
                ),
            ],
            logger=logger,
        )
        if not quiet:
            console_log(result.output_text_str)
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def image_id(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- Image Upload ID ---\n")

    try:
        image_content = red_image()
        uploaded_file: FileWithId = await model.upload_file(
            "image_id.png", IMAGE_MIME, BytesIO(image_content), type="image"
        )
        if not quiet:
            console_log(f"Uploaded File ID: {uploaded_file.file_id}")
        return await model.query(
            [TextInput(text="What color is the image?"), uploaded_file],
            logger=logger,
        )
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def image_url(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- Image URL ---\n")

    try:
        return await model.query(
            [
                TextInput(text="What is in this image?"),
                FileWithUrl(
                    type="image",
                    name="image.png",
                    mime=IMAGE_MIME,
                    url="https://pyxis.nymag.com/v1/imgs/424/858/e6c66c3a1992e711bca0137b754fea749f-cat-law.rsquare.w400.jpg",
                ),
            ],
            logger=logger,
        )
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def file_base64(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- File ---\n")

    try:
        result = await model.query(
            [
                TextInput(text="What is the secret?"),
                FileWithBase64(
                    type="file",
                    name="file_base64.pdf",
                    mime=FILE_MIME,
                    base64=base64.b64encode(secret_pdf()).decode("utf-8"),
                ),
            ],
            logger=logger,
        )
        if not quiet:
            console_log(result.output_text_str)
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def file_id(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- File Upload ID ---\n")

    try:
        uploaded_file: FileWithId = await model.upload_file(
            "file_id.pdf", FILE_MIME, BytesIO(secret_pdf())
        )
        if not quiet:
            console_log(f"Uploaded File ID: {uploaded_file.file_id}")
        return await model.query(
            [TextInput(text="What is the secret?"), uploaded_file], logger=logger
        )
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def file_url(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- File URL ---\n")

    try:
        return await model.query(
            [
                TextInput(text="What is the title of this document?"),
                FileWithUrl(
                    type="file",
                    name="file_url.pdf",
                    mime=FILE_MIME,
                    url="https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf",
                ),
            ],
            logger=logger,
        )
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def audio_bytes(
    model: LLM,
    *,
    quiet: bool = False,
    raise_errors: bool = False,
    logger: logging.Logger | None = None,
) -> QueryResult | None:
    if not quiet:
        console_log("\n--- Audio ---\n")

    try:
        result = await model.query(
            [
                TextInput(
                    text="Is this audio speech or a musical tone? Answer in one short sentence."
                ),
                FileWithBytes(
                    type="file",
                    name="tone.wav",
                    mime=AUDIO_MIME,
                    data=tone_wav(),
                ),
            ],
            logger=logger,
        )
        if not quiet:
            console_log(result.output_text_str)
        return result
    except Exception as e:
        if raise_errors:
            raise
        if not quiet:
            console_log(f"Error: {e}")
        return None


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run image, file, and audio demos")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07)",
    )
    parser.add_argument(
        "--all-transports",
        action="store_true",
        help="Also run upload-ID and URL variants",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.instance_logger.info(model)

    if model.supports_images:
        await image_base64(model)
        if args.all_transports:
            await image_id(model)
            await image_url(model)
    else:
        console_log(
            "Skipping image demo: model does not support images", color="yellow"
        )

    if model.supports_files:
        await file_base64(model)
        if args.all_transports:
            await file_id(model)
            await file_url(model)
    else:
        console_log("Skipping file demo: model does not support files", color="yellow")

    if model.supports_audio:
        await audio_bytes(model)
    else:
        console_log("Skipping audio demo: model does not support audio", color="yellow")


if __name__ == "__main__":
    setup()
    asyncio.run(main())
