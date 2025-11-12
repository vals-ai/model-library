import re


def get_default_budget_tokens(max_tokens: int) -> int:
    return int(max(max_tokens * 0.75, 1024))


THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
THINK_TAG_REMOVE_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def get_reasoning_in_tag(text: str) -> tuple[str, str]:
    """
    Extract reasoning inside <think> tag
    """
    parsed_text: str = text
    parsed_reasoning: str = ""

    think_content_list = THINK_TAG_RE.findall(text)
    if len(think_content_list) == 1:
        parsed_text = THINK_TAG_REMOVE_RE.sub("", text)
        parsed_reasoning = think_content_list[0]
    else:
        parsed_reasoning = "Error: multiple or no reasoning tokens found"

    return parsed_text, parsed_reasoning
