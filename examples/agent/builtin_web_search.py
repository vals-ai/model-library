"""Agent example: OpenAI built-in web search across five turns.

Demonstrates using OpenAI's server-side web_search tool in a multi-turn
conversation where each turn builds on what was found before.

Built-in web search differs from function-call search:
- The model searches internally within a single API call (no round-trip)
- Search calls surface as ProviderToolEvents on QueryResult
- The full web_search_call history is preserved across turns via RawResponse

Sample output from a real run:

    === Turn 1: Who is the current CEO of Apple? ===
    INFO Turn 1/5 | 1 tool calls | in: 15551, out: 560
    INFO   ✓ submit (0.0s)
    INFO   ✓ web_search [web_search_call] (provider)
    Answer 1: The current CEO of Apple is Tim Cook...
    Searches: 1, Turns: 1

    === Turn 5: Summarize everything you found... ===
    INFO Turn 1/5 | 1 tool calls | in: 49986, out: 3783
    INFO   ✓ submit (0.0s)
    INFO   ✓ web_search [web_search_call] (provider)
    INFO   ✓ web_search [web_search_call] (provider)
    INFO   ✓ web_search [web_search_call] (provider)
    INFO   ✓ web_search [web_search_call] (provider)
    Answer 5: As of June 19 2026, Apple's CEO is Tim Cook...
    Searches: 4, Turns: 1

Note: web_search duration is not shown — the search happens
server-side inside the API call, not via a local execute().
"""

import asyncio

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    TimeLimit,
    TurnLimit,
    TurnSummary,
)
from model_library.agent.tool import ProviderTool
from model_library.agent.tools.submit import SubmitTool
from model_library.base.input import InputItem, SystemInput, TextInput
from model_library.registry_utils import get_registry_model

from ..setup import console_log, setup


async def main() -> None:
    setup()

    llm = get_registry_model("openai/gpt-5.5")
    tools = [ProviderTool(name="web_search", body={"type": "web_search"}), SubmitTool()]
    config = AgentConfig(
        turn_limit=TurnLimit(max_turns=5), time_limit=TimeLimit(max_seconds=120)
    )

    questions = [
        "Who is the current CEO of Apple?",
        "When did they become CEO?",
        "What is one major product they launched as CEO?",
        "What is Apple's current stock ticker and which exchange is it listed on?",
        "Summarize everything you found about Apple's CEO in 2-3 sentences and submit it.",
    ]

    history: list[InputItem] = [
        SystemInput(
            text=(
                "You are a research assistant. For every question, you MUST call the submit tool with your final answer. "
                "Use web search if needed, but always end by calling submit — never respond in text alone. "
                "Limit yourself to two web search per question."
            )
        ),
    ]

    for i, question in enumerate(questions, 1):
        history.append(TextInput(text=question))
        console_log(f"\n=== Turn {i}: {question} ===\n")
        agent = Agent(
            name="web-search", llm=llm, tools=tools, config=config, hooks=AgentHooks()
        )
        result = await agent.run(history, question_id=f"q{i}")
        console_log(f"Answer {i}: {result.final_answer}")
        searches = sum(
            len(t.provider_tool_events)
            for t in result.turns
            if isinstance(t, TurnSummary)
        )
        console_log(f"Searches: {searches}, Turns: {result.total_turns}")


if __name__ == "__main__":
    asyncio.run(main())
