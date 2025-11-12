import asyncio
import base64
from io import BytesIO

from model_library.base import (
    LLM,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    TextInput,
)
from model_library.registry_utils import get_registry_model

from .data.images import red_image
from .setup import console_log, setup

mime = "png"
red_image_content = red_image()


async def image_base64(model: LLM):
    console_log("\n--- Image Base64 ---\n")

    await model.query(
        [
            TextInput(text="What color is the image?"),
            FileWithBase64(
                type="image",
                name="red_image.png",
                mime=mime,
                base64=base64.b64encode(red_image_content).decode("utf-8"),
            ),
        ]
    )


async def image_id(model: LLM):
    console_log("\n--- File Id (Upload Bytes) ---\n")

    uploaded_file: FileWithId = await model.upload_file(
        "image_id.png", mime, BytesIO(red_image_content), type="image"
    )
    console_log(f"Uploaded File ID: {uploaded_file.file_id}")

    await model.query(
        [
            TextInput(text="What color is the image?"),
            uploaded_file,
        ],
    )


async def image_url(model: LLM):
    console_log("\n--- File URL ---\n")

    await model.query(
        [
            TextInput(text="What is in this image?"),
            FileWithUrl(
                type="image",
                name="image.png",
                mime=mime,
                url="https://pyxis.nymag.com/v1/imgs/424/858/e6c66c3a1992e711bca0137b754fea749f-cat-law.rsquare.w400.jpg",
            ),
        ],
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run image examples with a model")
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

    if not model.supports_images:
        raise Exception("Model does not support images")

    await image_base64(model)
    await image_id(model)
    await image_url(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
