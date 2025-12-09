import asyncio
import concurrent.futures
import logging
import sys
import time
from collections import defaultdict
from typing import Any, Coroutine

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.tree import Tree

from examples.setup import setup
from model_library.base import LLMConfig
from model_library.exceptions import exception_message, retry_llm_call
from model_library.register_models import get_model_registry
from model_library.registry_utils import get_registry_model

CONCURRENCY_PER_PROVIDER = 10
MAX_RETRIES = 5
DEFAULT_TIMEOUT = 120
DEEP_RESEARCH_TIMEOUT = 600  # 10 minutes for deep research models

# registry
model_registry = get_model_registry()
providers = {cfg.provider_name for cfg in model_registry.values()}

# concurrency
BLOCKING_PROVIDERS = {"bedrock"}
semaphores = {
    provider: asyncio.Semaphore(CONCURRENCY_PER_PROVIDER) for provider in providers
}
sync_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=10, thread_name_prefix="sync-models"
)

# analytics
exceptions: list[tuple[str, Exception]] = []
running_models: dict[str, dict[str, tuple[float, int]]] = defaultdict(
    dict
)  # provider -> {model: (start_time, retry_count)}
completed_models: dict[str, int] = defaultdict(int)  # provider -> success count
failed_model_names: dict[str, list[str]] = defaultdict(
    list
)  # provider -> [failed_model_names]


def create_dashboard(total: int, completed_count: int) -> Table:
    """Create the live dashboard display"""

    # container
    table = Table.grid()

    # header and progress bar
    progress_pct = completed_count / total if total > 0 else 0
    progress_bar_width = 40
    filled = int(progress_pct * progress_bar_width)
    bar = "▓" * filled + "░" * (progress_bar_width - filled)
    header = f"Processing Models: [{completed_count}/{total}] [[green]{bar}[/green]] {progress_pct:.1%}"
    table.add_row(f"[bold blue]{header}[/bold blue]\n")

    # active
    active_providers = [p for p in sorted(providers) if len(running_models[p]) > 0]
    if active_providers:
        tree = Tree("[bold]Active:[/bold]\n")
        for provider in active_providers:
            active_count = len(running_models[provider])
            completed = completed_models[provider]
            failed = len(failed_model_names[provider])

            provider_node = tree.add(
                f"[cyan]{provider}[/cyan]: {active_count} active, {completed} completed, {failed} failed"
            )

            # running time
            for model_name, (start_time, retry_count) in running_models[
                provider
            ].items():
                duration = time.time() - start_time
                retry_text = f" (retry {retry_count})" if retry_count > 0 else ""
                provider_node.add(
                    f"[yellow]{model_name:<50}[/yellow] [{duration:>6.1f}s]{retry_text}"
                )

        table.add_row(tree)

    # completed
    completed_providers = [
        (p, completed_models[p], len(failed_model_names[p]))
        for p in sorted(providers)
        if completed_models[p] > 0 or len(failed_model_names[p]) > 0
    ]
    if completed_providers:
        table.add_row("\n[bold]Completed:[/bold]")
        for provider, completed, failed in completed_providers:
            total = completed + failed
            result_line = f"[cyan]{provider}[/cyan]: {total} models, {failed} failed"
            table.add_row(result_line)

            if failed > 0 and failed_model_names[provider]:
                failed_names = ", ".join(failed_model_names[provider])
                table.add_row(f"  [red]Failed: {failed_names}[/red]")

    return table


async def process_model(model_str: str, provider_name: str):
    model = get_registry_model(
        model_str, override_config=LLMConfig(supports_batch=False)
    )

    # Use longer timeout for deep research models
    timeout = DEEP_RESEARCH_TIMEOUT if "deep-research" in model_str else DEFAULT_TIMEOUT

    start_time = time.time()

    # custom retry logico
    # capture retry attempts
    def custom_retrier(logger: logging.Logger):
        return retry_llm_call(
            logger,
            max_tries=MAX_RETRIES,
            max_time=timeout,
            backoff_callback=lambda tries, exception, elapsed, wait: (
                running_models[provider_name].update({model_str: (start_time, tries)})
            ),
        )

    model.custom_retrier = custom_retrier

    try:
        async with semaphores[provider_name]:
            running_models[provider_name][model_str] = (start_time, 0)

            # handle blocking providers
            def query():
                return (
                    asyncio.get_event_loop().run_in_executor(
                        sync_executor,
                        lambda: asyncio.run(
                            model.query("What is the capital of France?")
                        ),
                    )
                    if provider_name in BLOCKING_PROVIDERS
                    else model.query("What is the capital of France?")
                )

            output = await query()

            if not output.metadata.total_input_tokens:
                raise Exception("No in tokens")
            if not output.metadata.total_output_tokens:
                raise Exception("No out tokens")

            completed_models[provider_name] += 1

    except Exception as e:
        failed_model_names[provider_name].append(model_str)
        exceptions.append((model_str, e))
    finally:
        running_models[provider_name].pop(model_str, None)


MODEL_OVERRIDES = [
    "dumbmar",
    "bedrock",
]
IGNORED_MODELS = ["fireworks/deepseek-v3p1", "fireworks/deepseek-v3p2-thinking"]
ERROR_OVERRIDES = [
    "overloaded",
    "rate limit",
    "429",
    "503",
]


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run all models with a simple prompt")
    parser.add_argument(
        "--research",
        action="store_true",
        help="Run research models (default: False)",
    )
    args = parser.parse_args()

    tasks: list[Coroutine[Any, Any, None]] = []
    for key, config in model_registry.items():
        if config.metadata.deprecated:
            continue
        if "research" in key and not args.research:
            continue
        if getattr(config.provider_properties, "serverless", True) is False:
            continue
        if key in IGNORED_MODELS:
            continue

        tasks.append(process_model(key, config.provider_name))

    # start tasks
    running_tasks = [asyncio.create_task(task) for task in tasks]

    console = Console()

    # live dashboard
    total_models = len(tasks)
    completed_count = 0
    with Live(refresh_per_second=10, console=console) as live:
        while running_tasks:
            live.update(create_dashboard(total_models, completed_count))

            done, running_tasks = await asyncio.wait(
                running_tasks,
                timeout=0.1,  # Update at least every 100ms
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed_count += len(done)
            running_tasks = list(running_tasks)  # Convert back to list
        live.update(create_dashboard(total_models, completed_count))

    # show exceptions
    override_count = 0
    if exceptions:
        console.print(
            "\n[red bold]Exceptions encountered during processing:[/red bold]"
        )
        for model_str, exc in exceptions:
            error_override = next(
                (o for o in ERROR_OVERRIDES if o in str(exc).lower()), None
            )
            model_override = next((o for o in MODEL_OVERRIDES if o in model_str), None)

            override_text = ""
            if error_override or model_override:
                reason = error_override or model_override
                override_text = f" [green][OVERRIDDEN | {reason}][/green]"
                override_count += 1

            console.print(
                f"  [red]{model_str}[/red]{override_text}: {exception_message(exc)}"
            )

    if len(exceptions) != override_count:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    setup(disable_logging=True)
    asyncio.run(main())
