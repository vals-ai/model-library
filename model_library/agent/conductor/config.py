from pydantic import Field

from model_library.agent.config import TimeLimit
from model_library.utils import ValsModel


class ConductorConfig(ValsModel):
    max_exchanges: int = Field(ge=1)
    time_limit: TimeLimit | None = None
