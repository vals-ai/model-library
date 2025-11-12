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

from .data.files import secret_pdf
from .setup import console_log, setup

mime = "application/pdf"
secret_file_content = secret_pdf()


async def file_base64(model: LLM):
    console_log("\n--- File Base64 ---\n")

    await model.query(
        [
            TextInput(text="What is the secret?"),
            FileWithBase64(
                type="file",
                name="file_base64.pdf",
                mime=mime,
                base64=base64.b64encode(secret_file_content).decode("utf-8"),
            ),
        ],
    )


async def file_id(model: LLM):
    console_log("\n--- File Id (Upload Bytes) ---\n")

    uploaded_file: FileWithId = await model.upload_file(
        "file_id.pdf", mime, BytesIO(secret_file_content)
    )
    console_log(f"Uploaded File ID: {uploaded_file.file_id}")

    await model.query(
        [
            TextInput(text="What is the secret?"),
            uploaded_file,
        ],
    )


async def file_url(model: LLM):
    console_log("\n--- File URL ---\n")

    await model.query(
        [
            TextInput(text="What is in this document?"),
            FileWithUrl(
                type="file",
                name="file_url.pdf",
                mime=mime,
                url="https://ontheline.trincoll.edu/images/bookdown/sample-local-pdf.pdf",
            ),
        ],
    )


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run file examples with a model")
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

    if not model.supports_files:
        raise Exception("Model does not support files")

    await file_base64(model)
    await file_id(model)
    await file_url(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
