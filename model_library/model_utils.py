import re


def get_default_budget_tokens(max_tokens: int) -> int:
    return int(max(max_tokens * 0.75, 1024))


THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
THINK_TAG_REMOVE_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)

# Google's OpenAI-compat endpoint uses <thought> tags instead of <think>
THOUGHT_TAG_RE = re.compile(r"<thought>(.*?)</thought>", flags=re.DOTALL)
THOUGHT_TAG_REMOVE_RE = re.compile(r"<thought>.*?</thought>\s*", flags=re.DOTALL)


def get_reasoning_in_tag(text: str) -> tuple[str, str]:
    """
    Extract reasoning inside <think> or <thought> tags.
    """
    parsed_text: str = text
    parsed_reasoning: str = ""

    for content_re, remove_re in [
        (THINK_TAG_RE, THINK_TAG_REMOVE_RE),
        (THOUGHT_TAG_RE, THOUGHT_TAG_REMOVE_RE),
    ]:
        content_list = content_re.findall(text)
        if len(content_list) == 1:
            parsed_text = remove_re.sub("", text)
            parsed_reasoning = content_list[0]
            return parsed_text, parsed_reasoning

    # multiple tags found — ambiguous, flag it; zero tags is normal (e.g. minimal effort)
    for content_re in [THINK_TAG_RE, THOUGHT_TAG_RE]:
        if len(content_re.findall(text)) > 1:
            parsed_reasoning = "Error: multiple reasoning tokens found"
            break

    return parsed_text, parsed_reasoning
