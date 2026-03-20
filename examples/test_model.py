import asyncio

from model_library.base import LLM, TextInput
from model_library.registry_utils import get_registry_model

from .setup import console_log, setup


async def test_query(model: LLM, num_tokens: int):
    console_log("\n--- Test Model ---\n")

    prompt = ("hello " * num_tokens).strip()

    result = await model.query(
        [TextInput(text=prompt)],
    )

    console_log(f"\nTokens requested: {num_tokens}")
    console_log(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
    console_log(
        f"Output: {result.output_text[:200] if result.output_text else 'No output'}"
    )
    if result.metadata:
        console_log(f"Input tokens: {result.metadata.in_tokens}")
        console_log(f"Output tokens: {result.metadata.out_tokens}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a model with a simple repeated-token query"
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="google/gemini-2.5-flash",
        type=str,
        help="Model endpoint (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10,
        help="Number of times to repeat the token in the prompt (default: 10)",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    await test_query(model, args.num_tokens)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
