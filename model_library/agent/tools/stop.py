import logging
from typing import Any

from model_library.agent.tool import Tool, ToolOutput


class StopTool(Tool):
    """Signals the agent to stop without submitting an answer."""

    name = "stop"
    description = (
        "Stop the agent. This tool takes no parameters, produces no outputs, and marks the task as complete. "
        "You MUST use this tool to stop the task - you will continue to be prompted to take actions until you use this tool to stop. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {}
    required: list[str] = []

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        return ToolOutput(output="", done=True)
