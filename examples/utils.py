"""Shared example tools used across examples."""

import json
import logging
from typing import Any

from model_library.agent.tool import Tool, ToolOutput


class GetWeather(Tool):
    """Returns fake weather data. Used in examples to avoid real API calls."""

    name = "get_weather"
    description = "Get current temperature in a given location."
    parameters = {
        "location": {
            "type": "string",
            "description": "City and country e.g. Bogotá, Colombia",
        }
    }

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        location = args.get("location", "unknown")
        return ToolOutput(
            output=json.dumps(
                {"location": location, "temp": "18C", "conditions": "foggy"}
            )
        )


class SaveNote(Tool):
    """Saves a note to shared state. Demonstrates state passing between tools."""

    name = "save_note"
    description = "Save a note to the session. Other tools can read it later."
    parameters = {
        "key": {"type": "string", "description": "Note identifier"},
        "content": {"type": "string", "description": "Note content"},
    }

    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        state[args["key"]] = args["content"]
        return ToolOutput(output=f"Saved note '{args['key']}'.")
