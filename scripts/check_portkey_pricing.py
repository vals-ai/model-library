"""
Verify our model pricing against Portkey's pricing data.

Portkey maintains a comprehensive pricing database for LLMs at:
https://github.com/Portkey-AI/models

Portkey prices are in cents per token, while our prices are in dollars per million tokens.
Conversion: portkey_price * 10,000 = our_price ($/million tokens)

Usage:
    uv run python scripts/check_portkey_pricing.py

The script will:
1. Clone/use cached Portkey models repo from /tmp/portkey-models
2. Compare our model prices against Portkey's pricing data
3. Display a summary of matched, skipped, bypassed, and mismatched prices
4. Exit with code 1 if there are any mismatches that need review
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from model_library.register_models import get_model_registry

PORTKEY_REPO_URL = "https://github.com/Portkey-AI/models.git"
PORTKEY_CLONE_PATH = Path("/tmp/portkey-models")

PROVIDER_MAPPING = {
    "openai": "openai.json",
    "anthropic": "anthropic.json",
    "google": "google.json",
    "mistral": "mistral-ai.json",
    "deepseek": "deepseek.json",
    "xai": "x-ai.json",
    "fireworks": "fireworks-ai.json",
    "together": "together-ai.json",
    "cohere": "cohere.json",
    "amazon": "bedrock.json",
    "perplexity": "perplexity-ai.json",
    "azure": "azure-openai.json",
    "groq": "groq.json",
    "openrouter": "openrouter.json",
}

MANUALLY_VERIFIED_BYPASS: dict[str, str | None] = {
    "amazon/amazon.nova-pro-v1:0": "Portkey has wrong cache prices",
    "amazon/amazon.nova-lite-v1:0": "Portkey has wrong cache prices",
    "amazon/amazon.nova-micro-v1:0": "Portkey has wrong cache prices",
    #
    "deepseek/deepseek-chat": "Portkey has outdated prices",
    #
    "openai/gpt-4.1-nano-2025-04-14": "Portkey has wrong cache price",
    "azure/gpt-4.1-nano-2025-04-14": "Portkey has wrong cache price",
    "openai/o4-mini-deep-research-2025-06-26": "Portkey has o3-deep-research prices (5x too high)",
}


class PricingComparison(BaseModel):
    model_name: str
    price_type: str
    our_price: float
    portkey_price: float
    matches: bool


class PricingResults(BaseModel):
    """Aggregated results from pricing comparison.

    Attributes:
        matched: Comparisons where our price matches Portkey's price.
        mismatches: Comparisons where prices differ and need review.
        bypassed: Comparisons where prices differ but are in MANUALLY_VERIFIED_BYPASS
            (known differences where we've verified our price is correct).
        skipped_count: Number of models skipped entirely (not compared). Models are
            skipped if they are deprecated, their provider isn't in PROVIDER_MAPPING,
            or the model doesn't exist in Portkey's database.
    """

    matched: list[PricingComparison] = Field(default_factory=list)
    mismatches: list[PricingComparison] = Field(default_factory=list)
    bypassed: list[PricingComparison] = Field(default_factory=list)
    skipped_count: int = 0

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    @property
    def bypassed_count(self) -> int:
        return len(self.bypassed)

    def add_comparison(
        self,
        comparison: PricingComparison,
        bypass_models: dict[str, str | None],
    ) -> None:
        """Add a comparison to the appropriate list based on match status."""
        if comparison.matches:
            self.matched.append(comparison)
        elif comparison.model_name in bypass_models:
            self.bypassed.append(comparison)
        else:
            self.mismatches.append(comparison)

    def skip(self) -> None:
        """Increment skipped count."""
        self.skipped_count += 1


def clone_portkey_repo(console: Console) -> None:
    if PORTKEY_CLONE_PATH.exists():
        console.print("[dim]Updating cached Portkey repo...[/dim]")
        subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(PORTKEY_CLONE_PATH),
            check=True,
            capture_output=True,
        )
        console.print("[green]Portkey repo updated[/green]")
        return

    console.print("[yellow]Cloning Portkey models repo...[/yellow]")
    subprocess.run(
        ["git", "clone", "--depth", "1", PORTKEY_REPO_URL, str(PORTKEY_CLONE_PATH)],
        check=True,
        capture_output=True,
    )
    console.print("[green]Portkey repo cloned successfully[/green]")


def load_portkey_pricing(
    provider_file: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    pricing_path = PORTKEY_CLONE_PATH / "pricing" / provider_file
    if not pricing_path.exists():
        return {}

    with open(pricing_path) as f:
        return json.load(f)


def portkey_to_our_price(portkey_cents_per_token: float) -> float:
    """Convert Portkey price (cents/token) to our price ($/million tokens)."""
    return portkey_cents_per_token * 10_000


def compare_prices(our_price: float | None, portkey_price: float | None) -> bool:
    TOLERANCE = 0.001

    """Compare two prices with tolerance for floating point differences."""
    if our_price is None or portkey_price is None:
        return True

    if our_price == 0 and portkey_price == 0:
        return True

    if portkey_price == 0:
        return True

    return abs(our_price - portkey_price) <= TOLERANCE


def check_price_category(
    results: PricingResults,
    model_name: str,
    price_type: str,
    our_price: float | None,
    portkey_price_raw: float | None,
) -> None:
    """Check a single price category and add comparison to results."""
    if portkey_price_raw is None or our_price is None:
        return

    portkey_price = portkey_to_our_price(portkey_price_raw)
    results.add_comparison(
        PricingComparison(
            model_name=model_name,
            price_type=price_type,
            our_price=our_price,
            portkey_price=portkey_price,
            matches=compare_prices(our_price, portkey_price),
        ),
        MANUALLY_VERIFIED_BYPASS,
    )


def check_pricing(console: Console) -> PricingResults:
    """Check all model pricing against Portkey data."""
    model_registry = get_model_registry()
    results = PricingResults()

    # Load Portkey data
    portkey_data: dict[str, dict[str, Any]] = {}
    for provider, filename in PROVIDER_MAPPING.items():
        portkey_data[provider] = load_portkey_pricing(filename)

    for model_name, model_config in model_registry.items():
        # skip deprecated models
        if model_config.metadata.deprecated:
            results.skip()
            continue

        provider = model_config.provider_name
        if provider not in portkey_data:
            results.skip()
            continue

        portkey_models = portkey_data[provider]
        portkey_model_name = model_name.split("/", 1)[1]

        if portkey_model_name not in portkey_models:
            results.skip()
            continue

        portkey_pricing = portkey_models[portkey_model_name]["pricing_config"][
            "pay_as_you_go"
        ]
        our_costs = model_config.costs_per_million_token

        # Check input price
        check_price_category(
            results,
            model_name,
            "input",
            our_costs.input,
            portkey_pricing.get("request_token", {}).get("price"),
        )

        # Check output price
        check_price_category(
            results,
            model_name,
            "output",
            our_costs.output,
            portkey_pricing.get("response_token", {}).get("price"),
        )

        # Check cache prices
        if our_costs.cache is not None:
            our_cache_read, our_cache_write = our_costs.cache.get_costs(
                our_costs.input or 0
            )

            check_price_category(
                results,
                model_name,
                "cache_read",
                our_cache_read,
                portkey_pricing.get("cache_read_input_token", {}).get("price"),
            )

            check_price_category(
                results,
                model_name,
                "cache_write",
                our_cache_write,
                portkey_pricing.get("cache_write_input_token", {}).get("price"),
            )

    return results


def create_summary_table(results: PricingResults) -> Table:
    """Create a summary table."""
    table = Table(title="Pricing Comparison Summary", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Matched prices", f"[green]{results.matched_count}[/green]")
    table.add_row("Skipped models", f"[dim]{results.skipped_count}[/dim]")
    table.add_row(
        "Manually verified (bypassed)", f"[yellow]{results.bypassed_count}[/yellow]"
    )
    table.add_row(
        "Mismatches",
        f"[red]{results.mismatch_count}[/red]"
        if results.mismatch_count > 0
        else f"[green]{results.mismatch_count}[/green]",
    )

    return table


def create_mismatches_tree(
    mismatches: list[PricingComparison], title: str, style: str = "red"
) -> Tree:
    """Create a tree showing mismatches grouped by model."""
    tree = Tree(f"[bold {style}]{title}[/bold {style}]")

    # Group by model
    by_model: dict[str, list[PricingComparison]] = defaultdict(list)
    for m in mismatches:
        by_model[m.model_name].append(m)

    for model_name, comparisons in by_model.items():
        model_node = tree.add(f"[cyan]{model_name}[/cyan]")
        for m in comparisons:
            model_node.add(
                f"[yellow]{m.price_type:<12}[/yellow] "
                f"ours=[white]${m.our_price:<8.4f}[/white] "
                f"portkey=[white]${m.portkey_price:<8.4f}[/white] "
            )

    return tree


def create_bypassed_tree(bypassed: list[PricingComparison]) -> Tree:
    """Create a tree showing bypassed (manually verified) models grouped by model."""
    tree = Tree("[bold yellow]Manually Verified (Bypassed)[/bold yellow]")

    # Group by model
    by_model: dict[str, list[PricingComparison]] = defaultdict(list)
    for m in bypassed:
        by_model[m.model_name].append(m)

    for model_name, comparisons in by_model.items():
        reason = MANUALLY_VERIFIED_BYPASS.get(model_name)
        header = f"[cyan]{model_name}[/cyan]"
        if reason:
            header += f" [green]({reason})[/green]"
        model_node = tree.add(header)
        for m in comparisons:
            model_node.add(
                f"[yellow]{m.price_type:<12}[/yellow] "
                f"ours=[white]${m.our_price:<8.4f}[/white] "
                f"portkey=[white]${m.portkey_price:<8.4f}[/white]"
            )

    return tree


def display_results(console: Console, results: PricingResults) -> None:
    """Display pricing comparison results to the console."""
    console.print(create_summary_table(results))
    console.print()

    if results.bypassed:
        console.print(create_bypassed_tree(results.bypassed))
        console.print()

    if results.mismatches:
        console.print(
            create_mismatches_tree(
                results.mismatches, "Mismatches (NEED REVIEW)", "red"
            )
        )
        console.print()
        console.print(
            f"[red bold]Found {results.mismatch_count} pricing mismatches![/red bold]\n"
            "[dim]Review the mismatches above and either fix the pricing or add to MANUALLY_VERIFIED_BYPASS.[/dim]"
        )
    else:
        console.print("[green bold]All prices match![/green bold]")


def main():
    console = Console()

    console.print("\n[bold blue]Portkey Pricing Verification[/bold blue]\n")

    # Clone/use Portkey repo
    clone_portkey_repo(console)
    console.print()

    # Run the check
    console.print("[yellow]Checking model pricing...[/yellow]\n")
    results = check_pricing(console)

    # Display results
    display_results(console, results)

    # Exit with appropriate code
    sys.exit(1 if results.mismatches else 0)


if __name__ == "__main__":
    main()
