from model_library.model_utils import get_reasoning_in_tag


def test_think_tag_extracted():
    text, reasoning = get_reasoning_in_tag("<think>step by step</think>answer")
    assert text == "answer"
    assert reasoning == "step by step"


def test_thought_tag_extracted():
    text, reasoning = get_reasoning_in_tag("<thought>thinking here</thought>final answer")
    assert text == "final answer"
    assert reasoning == "thinking here"


