from typing import Any, Literal

from pydantic import SecretStr
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    DelegateOnly,
    LLMConfig,
)
from model_library.register_models import register_provider


@register_provider("ant")
class AntModel(DelegateOnly):
    def __init__(
        self,
        model_name: str,
        provider: Literal["ant"] = "ant",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        # OpenAI-compatible Chat Completions endpoint for Ant Group's Ling models.
        config = config or LLMConfig()
        config.custom_endpoint = (
            config.custom_endpoint or "https://api.ant-ling.com/v1/"
        )
        config.custom_api_key = config.custom_api_key or SecretStr(
            model_library_settings.ANT_API_KEY
        )

        self.init_delegate(
            config=config,
            delegate_provider="openai",
            use_completions=True,
        )

    @override
    def _get_extra_body(self) -> dict[str, Any]:
        """Toggle Ling's thinking mode.

        Thinking is enabled by default server-side, so we omit the parameter to
        keep it on and only send it to turn thinking off.

        Sending `{"type": "enable"}` -- the value the provider documents -- silently
        DISABLES thinking. Verified against the live API 2026-07-29: omitting the
        parameter and `{"type": "enabled"}` both return `reasoning_content`, while
        `"enable"`, `"disable"`, `"disabled"`, `""`, `"ENABLED"` and unknown values
        all return none, with no error for invalid input. Only the documented
        default-on path and the documented `"disable"` value are used here, so we
        never depend on the undocumented `"enabled"`.
        """
        if self.reasoning:
            return {}
        return {"thinking": {"type": "disable"}}
