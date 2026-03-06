from collections.abc import Callable
from datetime import date
from pathlib import Path

from model_library.agent.tools.stop import StopTool
from model_library.agent.tools.submit import SubmitTool
from model_library.agent.tools.web_search import TavilyWebSearch
from model_library.agent.tools.bash import BashTool
from model_library.agent.tool import Tool

TOOL_REGISTRY: dict[str, Callable[[], Tool]] = {
    "stop": StopTool,
    "submit": SubmitTool,
    "web_search": lambda: TavilyWebSearch(max_end_date=date.today().isoformat()),
    "bash": lambda: BashTool(working_dir=str(Path.cwd())),
}
