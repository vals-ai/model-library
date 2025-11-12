import asyncio
from typing import Any, cast

from model_library.base import LLM, ToolDefinition
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup


def create_web_search_tool(domains: list[str] | None = None) -> ToolDefinition:
    """Create a web search tool with optional domain filtering."""
    body: dict[str, Any] = {"type": "web_search"}
    if domains:
        body["filters"] = {"allowed_domains": domains}
    return ToolDefinition(name="web_search", body=body)


def get_openai_kwargs(reasoning_effort: str = "low") -> dict[str, Any]:
    """Get OpenAI-specific parameters for web search."""
    return {"background": True, "reasoning": {"effort": reasoning_effort}}


def print_search_details(tool_call: Any) -> None:
    """Print detailed information about a web search tool call."""
    console_log(f"- ID: {tool_call.id}")
    console_log(f"- Name: {tool_call.name}")

    if isinstance(tool_call.args, dict):
        args = cast(dict[str, Any], tool_call.args)
        action = args.get("action")
        if action:
            console_log(f"  Action: {action}")
            if action == "search":
                if query := args.get("query"):
                    console_log(f"  Search Query: {query}")
                if domains := args.get("domains"):
                    console_log(f"  Domains Searched: {domains}")
                if sources := args.get("sources"):
                    console_log("  Sources Consulted:")
                    for source in sources:
                        console_log(f"    - {source}")


def print_citations(response: Any) -> None:
    """Extract and print citations from response history."""
    if not response.history:
        return

    for item in response.history:
        if not (hasattr(item, "content") and isinstance(item.content, list)):
            continue

        content_list = cast(list[Any], item.content)
        for content_item in content_list:
            if not (hasattr(content_item, "annotations") and content_item.annotations):
                continue

            console_log("\nCitations:")
            annotations = cast(list[Any], content_item.annotations)
            for annotation in annotations:
                if hasattr(annotation, "url") and annotation.url:
                    title = getattr(annotation, "title", "Untitled")
                    url = annotation.url
                    location = getattr(annotation, "location", "Unknown")
                    console_log(f"- {title}: {url} (Location: {location})")


def print_web_search_results(response: Any) -> None:
    """Print comprehensive web search results."""
    console_log(f"Response: {response.output_text}")

    if response.tool_calls:
        console_log(f"\nWeb Search Calls: {len(response.tool_calls)}")
        for tool_call in response.tool_calls:
            print_search_details(tool_call)

    print_citations(response)


async def web_search_domain_filtered(model: LLM) -> None:
    """Web search with medical domain filtering."""
    console_log("\n--- Web Search (Medical Domains Filtered) ---\n")

    tools = [
        create_web_search_tool(
            [
                "pubmed.ncbi.nlm.nih.gov",
                "clinicaltrials.gov",
                "www.who.int",
                "www.cdc.gov",
                "www.fda.gov",
            ]
        )
    ]
    kwargs = get_openai_kwargs() if model.provider == "openai" else {}

    response = await model.query(
        "Please perform a web search on how semaglutide is used in the treatment of diabetes.",
        tools=tools,
        **kwargs,  # type: ignore
    )

    print_web_search_results(response)


async def simple_web_search(model: LLM) -> None:
    """Simple web search without domain filtering."""
    console_log("\n--- Simple Web Search ---\n")

    tools = [create_web_search_tool()]
    response = await model.query(
        "What is the weather right now in San Francisco? Answer in celsius.",
        tools=tools,
    )

    print_web_search_results(response)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run web search example with a model")
    parser.add_argument(
        "model",
        nargs="?",
        default="openai/gpt-5-nano-2025-08-07",
        type=str,
        help="Model endpoint (default: openai/gpt-5-nano-2025-08-07). Note: Web search requires models that support tools.",
    )
    args = parser.parse_args()

    model = get_registry_model(args.model)
    model.logger.info(model)

    if not model.supports_tools:
        raise Exception("Model does not support tools")

    await web_search_domain_filtered(model)
    await simple_web_search(model)


if __name__ == "__main__":
    setup()
    asyncio.run(main())
