import base64
import io
from typing import Any, Literal, Sequence, cast

from google.genai import Client
from google.genai import errors as genai_errors
from google.genai.types import (
    Content,
    File,
    FunctionDeclaration,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    ThinkingConfig,
    ThinkingLevel,
    Tool,
    ToolListUnion,
    UploadFileConfig,
    FinishReason,
)
from typing_extensions import override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    InputItem,
    LLMBatchMixin,
    LLMConfig,
    ProviderConfig,
    PydanticT,
    QueryResult,
    QueryResultCost,
    QueryResultMetadata,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.exceptions import (
    BadInputError,
    ImmediateRetryException,
    InvalidStructuredOutputError,
    ModelNoOutputError,
)
from model_library.providers.google.batch import GoogleBatchMixin
from model_library.register_models import register_provider
from model_library.utils import normalize_tool_result


class GoogleConfig(ProviderConfig):
    use_vertex: bool = False
    use_interactions: bool = False


@register_provider("google")
class GoogleModel(LLM):
    provider_config = GoogleConfig()

    DEFAULT_THINKING_BUDGET: int = -1

    SAFETY_CONFIG: list[SafetySetting] = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    @override
    def get_client(self) -> Client:
        if self.provider_config.use_vertex:
            # Preview Gemini releases from September 2025 are only served from the global
            # Vertex region. The public docs for these SKUs list `global` as the sole
            # availability region (see September 25, 2025 release notes), so we override
            # the default `us-central1` when we detect them.
            # https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash
            MODEL_REGION_OVERRIDES: dict[str, str] = {
                "gemini-2.5-flash-preview-09-2025": "global",
                "gemini-2.5-flash-lite-preview-09-2025": "global",
            }
            region = model_library_settings.GCP_REGION
            if self.model_name in MODEL_REGION_OVERRIDES:
                region = MODEL_REGION_OVERRIDES[self.model_name]

            return Client(
                vertexai=True,
                project=model_library_settings.GCP_PROJECT_ID,
                location=region,
                # Credentials object is not typed, so we have to ignore the error
                credentials=model_library_settings.GCP_CREDS,
            )

        return Client(api_key=model_library_settings.GOOGLE_API_KEY)

    def __init__(
        self,
        model_name: str,
        provider: Literal["google"] = "google",
        *,
        config: LLMConfig | None = None,
    ):
        super().__init__(model_name, provider, config=config)

        if self.provider_config.use_vertex:
            self.supports_batch = False

        if not getattr(model_library_settings, "GCP_CREDS", None):
            self.supports_batch = False
            self.logger.warning("GCP_CREDS not set, disabling batching")

        self.batch: LLMBatchMixin | None = (
            GoogleBatchMixin(self) if self.supports_batch else None
        )

        self.client = self.get_client()

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[Content]:
        parsed_input: list[Content] = []
        parts: list[Part] = []

        def flush_parts():
            nonlocal parts

            if parts:
                parsed_input.append(Content(parts=parts, role="user"))
                parts = []

        for item in input:
            match item:
                case TextInput():
                    if item.text.strip():
                        parts.append(Part.from_text(text=item.text))

                case FileWithBase64() | FileWithUrl() | FileWithId():
                    part = await self.parse_file(item)
                    parts.append(part)

                case ToolResult():
                    flush_parts()
                    result_str = normalize_tool_result(item.result)
                    parsed_input.append(
                        Content(
                            role="function",
                            parts=[
                                Part.from_function_response(
                                    name=item.tool_call.name,
                                    response={"result": result_str},
                                )
                            ],
                        )
                    )

                case GenerateContentResponse():
                    flush_parts()
                    candidates = item.candidates
                    if candidates and candidates[0]:
                        content0 = candidates[0].content
                        if content0 is not None:
                            parsed_input.append(content0)
                    else:
                        self.logger.debug(
                            "GenerateContentResponse missing candidates; skipping"
                        )

                case Content():
                    flush_parts()
                    parsed_input.append(item)

                case _:
                    raise BadInputError(f"Unsupported input type: {type(item)}")

        flush_parts()

        return parsed_input

    @override
    async def parse_file(self, file: FileInput) -> Part:
        match file:
            case FileWithBase64():
                # allows call from parse_image
                mime = f"image/{file.mime}" if file.type == "image" else file.mime
                return Part.from_bytes(
                    mime_type=mime,
                    data=base64.b64decode(file.base64),
                )
            case FileWithId():
                return Part.from_uri(file_uri=file.file_id, mime_type=file.mime)
            case FileWithUrl():
                raise BadInputError(
                    "Gemini does not support URL. Please fetch the image, convert it to bytes, and pass it as a FileWithBase64 object."
                )

    @override
    async def parse_image(self, image: FileInput) -> Part:
        return await self.parse_file(image)

    @override
    async def parse_tools(self, tools: list[ToolDefinition]) -> list[Tool]:
        parsed_tools: list[Tool] = []
        for tool in tools:
            body = tool.body
            if isinstance(body, Tool):
                parsed_tools.append(body)
                continue
            if isinstance(body, ToolBody):
                fn = FunctionDeclaration(
                    name=body.name,
                    description=body.description,
                    parametersJsonSchema={  # type: ignore
                        "type": "object",
                        "properties": body.properties or {},
                        "required": body.required or [],
                    },
                )
                parsed_tools.append(Tool(function_declarations=[fn]))
                continue

            raise ValueError(
                "Invalid ToolDefinition.body for Google provider; expected ToolBody or google.genai.types.Tool"
            )

        return parsed_tools

    @override
    async def upload_file(
        self,
        name: str,
        mime: str,
        bytes: io.BytesIO,
        type: Literal["image", "file"] = "file",
    ) -> FileWithId:
        if self.provider_config.use_vertex:
            raise Exception(
                "Vertex AI does not support file uploads. "
                "use FileWithBase64 to pass files as inline data"
                "or use genai for file uploads"
            )

        mime = f"image/{mime}" if type == "image" else mime  # TODO:
        response: File = self.client.files.upload(
            file=bytes, config=UploadFileConfig(mime_type=mime)
        )
        if not response.name:
            raise Exception(f"Failed to upload file {name} - no name returned")
        if not response.uri:
            raise Exception(f"Failed to upload file {name} - no uri returned")

        self.logger.info(f"File uploaded successfully: {response.name}")
        return FileWithId(
            type="file",
            file_id=response.uri,
            name=response.display_name or response.name,
            mime=mime,
        )

    async def create_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> dict[str, Any]:
        generation_config = GenerateContentConfig(
            max_output_tokens=self.max_tokens,
        )
        if self.supports_temperature:
            if self.temperature is not None:
                generation_config.temperature = self.temperature
            if self.top_p is not None:
                generation_config.top_p = self.top_p

        generation_config.safety_settings = self.SAFETY_CONFIG

        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
            generation_config.system_instruction = str(system_prompt)

        if self.reasoning:
            reasoning_config = ThinkingConfig(include_thoughts=True)
            if self.reasoning_effort:
                reasoning_config.thinking_level = ThinkingLevel(self.reasoning_effort)
            else:
                reasoning_config.thinking_budget = cast(
                    int, kwargs.pop("thinking_budget", self.DEFAULT_THINKING_BUDGET)
                )
            generation_config.thinking_config = reasoning_config

        if tools:
            generation_config.tools = cast(ToolListUnion, await self.parse_tools(tools))

        generation_config = generation_config.model_copy(update=kwargs)

        body: dict[str, Any] = {
            "model": self.model_name,
            "contents": await self.parse_input(input),
            "config": generation_config,
        }
        return body

    @override
    async def _query_impl(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        **kwargs: object,
    ) -> QueryResult:
        body: dict[str, Any] = await self.create_body(input, tools=tools, **kwargs)

        text: str = ""
        reasoning: str = ""
        tool_calls: list[ToolCall] = []

        metadata: GenerateContentResponseUsageMetadata | None = None

        stream = await self.client.aio.models.generate_content_stream(**body)
        contents: list[Content | None] = []
        finish_reason: FinishReason | None = None

        async for chunk in stream:
            candidates = chunk.candidates
            if not candidates:
                continue

            content = candidates[0].content

            if content and content.parts:
                for part in content.parts:
                    if part.function_call:
                        if not part.function_call.name:
                            raise Exception(f"Invalid function call: {part}")

                        call_args = part.function_call.args or {}
                        tool_calls.append(
                            # weirdly, id is not required
                            ToolCall(
                                id=part.function_call.id or "",
                                name=part.function_call.name,
                                args=call_args,
                            )
                        )
                    if not part.text:
                        continue
                    if part.thought:
                        reasoning += part.text
                    else:
                        text += part.text

            if chunk.usage_metadata:
                metadata = chunk.usage_metadata
            if content:
                contents.append(content)
            if candidates[0].finish_reason:
                finish_reason = candidates[0].finish_reason

        if finish_reason != FinishReason.STOP:
            self.logger.error(f"Unexpected finish reason: {finish_reason}")

        if not text and not reasoning and not tool_calls:
            raise ModelNoOutputError("Model returned empty response")

        result = QueryResult(
            output_text=text,
            reasoning=reasoning,
            history=[*input, *contents],
            tool_calls=tool_calls,
        )

        if metadata:
            # see _calculate_cost
            cache_read_tokens = metadata.cached_content_token_count or 0
            result.metadata = QueryResultMetadata(
                in_tokens=(metadata.prompt_token_count or 0) - cache_read_tokens,
                out_tokens=metadata.candidates_token_count or 0,
                reasoning_tokens=metadata.thoughts_token_count or 0,
                cache_read_tokens=cache_read_tokens,
            )
        return result

    @override
    async def _calculate_cost(
        self,
        metadata: QueryResultMetadata,
        batch: bool = False,
        bill_reasoning: bool = True,
    ) -> QueryResultCost | None:
        """
        Future Cost considerations
        Per 1000 prompts:
        - Google Search
        - Google Maps
        """
        # Implicit caching (automatically enabled on Gemini 2.5 models, no cost saving guarantee)
        # Explicit caching (can be manually enabled on most models, cost saving guarantee)

        # https://docs.cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview
        # For both implicit and explicit caching, there is no additional charge to write to cache other than the standard input token costs. For explicit caching, there are storage costs based on how long caches are stored. There are no storage costs for implicit caching

        # google has separate prices for (text/image/video) and audio

        # prompt_token_count includes cached_content_token_count
        # thoughts_token_count billed in addition to candidate_token_count

        # google cache tokens increse in price with long context

        return await super()._calculate_cost(metadata, batch, bill_reasoning=True)

    @override
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        **kwargs: object,
    ) -> PydanticT:
        # Create the request body with JSON schema
        body: dict[str, Any] = await self.create_body(input, tools=[], **kwargs)

        # Get the JSON schema from the Pydantic model
        json_schema = pydantic_model.model_json_schema()

        # Update the config to include response_json_schema
        config: GenerateContentConfig = body["config"]
        config = config.model_copy(
            update={
                "response_json_schema": json_schema,
                "response_mime_type": "application/json",
            }
        )

        body["config"] = config

        # Make the request with retry wrapper
        async def _query():
            try:
                return await self.client.aio.models.generate_content(**body)
            except (genai_errors.ServerError, genai_errors.UnknownApiResponseError):
                raise ImmediateRetryException("Failed to connect to Google API")

        response: GenerateContentResponse = await _query()

        if not response.text:
            raise ModelNoOutputError("Model returned empty response")

        # Parse the JSON response into the Pydantic model
        try:
            parsed = pydantic_model.model_validate_json(response.text)
        except Exception as e:
            raise InvalidStructuredOutputError() from e

        return parsed
