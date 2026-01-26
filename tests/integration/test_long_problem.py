"""
Integration tests for hard mathematical reasoning questions.
"""

from model_library.base import LLMConfig
from model_library.registry_utils import get_registry_model
from tests.conftest import parametrize_all_models, pytest
from tests.test_helpers import assert_basic_result


@parametrize_all_models
async def test_long_problem(model_key: str, require_model_flag: pytest.FixtureRequest):
    """
    Test that the system can handle a model thinking for a long time
    """
    problem = """Please reason step by step, and put your final answer within \\boxed{}. Let the sequence of rationals $x_1,x_2,\\dots$ be defined such that $x_1=\\frac{25}{11}$ and\\[x_{k+1}=\\frac{1}{3}\\left(x_k+\\frac{1}{x_k}-1\\right)\\]for all $k \\leq 1$. Then $x_{2025}$ can be expressed as $\\frac{m}{n}$ for relatively prime positive integers $m$ and $n$. Find the remainder when $m+n$ is divided by $1000$."""

    model = get_registry_model(model_key, LLMConfig(reasoning_effort="high"))
    if not model.reasoning:
        pytest.skip("Model does not support reasoning")

    result = await model.query(problem)

    assert_basic_result(result)
