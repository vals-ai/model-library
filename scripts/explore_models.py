from typing import Any, List

from InquirerPy.prompts.list import ListPrompt
from rich.console import Console
from rich.pretty import Pretty

from model_library.registry_utils import (
    get_model_names_by_provider,
    get_provider_names,
    get_registry_config,
)

console = Console()


def display_model_details(model_name: str) -> None:
    console.clear()
    console.print(f"[bold cyan]Model:[/bold cyan] {model_name}\n")

    try:
        model_config: Any = get_registry_config(model_name)
    except Exception as e:
        console.print(f"[red]Error loading model config:[/red] {e}")
        input("\nPress Enter to go back.")
        return

    console.print(Pretty(model_config, expand_all=True))
    input("\nPress Enter to go back.")


def browse_models(provider: str) -> None:
    models: List[str] = get_model_names_by_provider(provider)
    if not models:
        console.print(f"[yellow]No models found for {provider}[/yellow]")
        input("\nPress Enter to return.")
        return

    while True:
        console.clear()
        console.print(
            f"[bold cyan]Provider:[/bold cyan] {provider} ({len(models)} models)\n"
        )

        model_prompt: ListPrompt = ListPrompt(
            message="Select a model:",
            choices=models + ["Back"],
            instruction="(Use ↑/↓ to navigate, Enter to select)",
            vi_mode=True,
            border=True,
        )
        model_choice: str = model_prompt.execute()

        if model_choice == "Back":
            break

        display_model_details(model_choice)


def browse_providers() -> None:
    providers: List[str] = get_provider_names()
    if not providers:
        console.print("[red]No providers found.[/red]")
        return

    while True:
        console.clear()
        provider_prompt = ListPrompt(
            message="Select a provider:",
            choices=sorted(providers) + ["Exit"],
            instruction="(Use ↑/↓ to navigate, Enter to select)",
            vi_mode=True,
            border=True,
        )
        provider_choice: str = provider_prompt.execute()

        if provider_choice == "Exit":
            break

        browse_models(provider_choice)


if __name__ == "__main__":
    browse_providers()
