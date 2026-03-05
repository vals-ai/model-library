import logging
from typing import Any

from model_library.agent.tool import Tool, ToolOutput


class SubmitTool(Tool):
    """
    Signals the agent to stop and surfaces the submitted string as the final answer.
    """

    name = "submit"
    description = (
        "Submits the final answer to the user. This tool takes your answer as its sole parameter, produces no outputs, and marks the task as complete. "
        "You MUST use this tool to submit your final result. The user will not see your response if you do not use this tool to submit. "
        "You will not be able to continue working after this tool is called; the conversation will be ended."
    )
    parameters: dict[str, Any] = {
        "answer": {
            "type": "string",
            "description": "The final answer to submit.",
        }
    }
    required: list[str] = ["answer"]

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            answer = args["answer"]
            if answer is None:
                raise ValueError("answer not provided")
            return ToolOutput(output=answer, done=True)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Submission failed: {error_msg}")
            return ToolOutput(output=error_msg, error=error_msg, done=False)
