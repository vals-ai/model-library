from typing import Literal, Sequence

from typing_extensions import override

from model_library import model_library_settings
from model_library.base import DelegateOnly, LLMConfig
from model_library.base.input import InputItem, ToolDefinition
from model_library.register_models import register_provider


@register_provider("minimax")
class MinimaxModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["minimax"] = "minimax",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        self.init_delegate(
            config=config,
            base_url="https://api.minimax.io/anthropic",
            api_key=model_library_settings.MINIMAX_API_KEY,
            delegate_provider="anthropic",
        )

    # minimax client shares anthropic's syntax
    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        assert self.delegate
        return await self.delegate.count_tokens(
            input, history=history, tools=tools, **kwargs
        )
