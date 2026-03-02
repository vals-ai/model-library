from model_library.base.input import InputItem, RawResponse
from model_library.utils import PrettyModel


def truncate_oldest(history: list[InputItem]) -> list[InputItem]:
    """Remove the oldest model response and associated inputs after it

    Always preserves the first message (initial prompt).
    Use with before_query hook for context window management:

        def before_query(history, last_error):
            if isinstance(last_error, MaxContextWindowExceededError):
                return truncate_oldest(history)
            if last_error:
                raise last_error
            return history
    """
    if len(history) <= 1:
        return history

    result = [history[0]]

    # skip RawResponse items (the first model response block)
    i = 1
    while i < len(history) and isinstance(history[i], RawResponse):
        i += 1

    # skip InputItems (ToolResults etc.) until next RawResponse or end
    while i < len(history) and not isinstance(history[i], RawResponse):
        i += 1

    # keep the rest
    result.extend(history[i:])
    return result


class AgentConfig(PrettyModel):
    """Configuration for agent execution

    - max_turns: maximum loop iterations (includes ErrorTurns), default 1000
    - max_time_seconds: wall-clock time limit, default 8 hours
    - serialize_histories: save per-turn histories to disk via FileHandler path
    """

    max_turns: int = 1000
    max_time_seconds: float = 28800
    serialize_histories: bool = True
