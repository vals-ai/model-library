import base64
import io
import json
import logging
import uuid
from typing import Any, Literal, Sequence, cast

from google.genai import Client
from google.genai import errors as genai_errors
from google.genai.types import (
    Content,
    CountTokensConfig,
    File,
    FinishReason,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    GoogleSearch,
    GroundingMetadata,
    HarmBlockThreshold,
    HarmCategory,
    HttpOptions,
    Part,
    SafetySetting,
    ThinkingConfig,
    ThinkingLevel,
    Tool,
    ToolConfig,
    ToolListUnion,
    UploadFileConfig,
)
from google.oauth2 import service_account
from pydantic import BaseModel, JsonValue, SecretStr
from typing_extensions import deprecated, override

from model_library import model_library_settings
from model_library.base import (
    LLM,
    FileBase,
    FileInput,
    FileWithBase64,
    FileWithId,
    FileWithUrl,
    FinishReasonInfo,
    InputItem,
    LLMBatchMixin,
    LLMConfig,
    ProviderConfig,
    PydanticT,
    QueryResult,
    QueryResultCost,
    QueryResultExtras,
    QueryResultMetadata,
    RawInput,
    RawResponse,
    SystemInput,
    TextInput,
    ToolBody,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from model_library.base.output.result import ProviderToolEvent
from model_library.base import (
    FinishReason as StandardFinishReason,
)
from model_library.base.output.builder import QueryResultBuilder
from model_library.base.input import normalize_query_input
from model_library.exceptions import (
    BadInputError,
    ImmediateRetryException,
    InvalidStructuredOutputError,
    ModelNoOutputError,
    RetryException,
    UnexpectedSystemInputError,
    handle_empty_response,
)
from model_library.providers.google.batch import GoogleBatchMixin
from model_library.providers.openai import OpenAIModel
from model_library.agent.tool import is_native_web_search
from model_library.register_models import register_provider
from model_library.utils import make_aiohttp_session


def map_google_finish_reason(
    finish_reason: FinishReason | None,
    has_tool_calls: bool = False,
) -> FinishReasonInfo:
    match finish_reason:
        case FinishReason.STOP:
            reason = (
                StandardFinishReason.TOOL_CALLS
                if has_tool_calls
                else StandardFinishReason.STOP
            )
        case FinishReason.MAX_TOKENS:
            reason = StandardFinishReason.MAX_TOKENS
        case (
            FinishReason.SAFETY
            | FinishReason.RECITATION
            | FinishReason.BLOCKLIST
            | FinishReason.PROHIBITED_CONTENT
            | FinishReason.SPII
            | FinishReason.LANGUAGE
            | FinishReason.IMAGE_SAFETY
            | FinishReason.IMAGE_PROHIBITED_CONTENT
            | FinishReason.IMAGE_RECITATION
        ):
            reason = StandardFinishReason.CONTENT_FILTER
        case FinishReason.MALFORMED_FUNCTION_CALL | FinishReason.UNEXPECTED_TOOL_CALL:
            reason = StandardFinishReason.MALFORMED_TOOL_CALL
        case _:
            reason = StandardFinishReason.UNKNOWN

    return FinishReasonInfo(
        reason=reason, raw=finish_reason.name if finish_reason else None
    )


def generate_tool_call_id(tool_name: str) -> str:
    return str(tool_name + "_" + str(uuid.uuid4()))


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

    def _get_default_api_key(self) -> str:
        if not self.provider_config.use_vertex:
            return model_library_settings.GOOGLE_API_KEY

        return json.dumps(
            {
                "GCP_REGION": model_library_settings.GCP_REGION,
                "GCP_PROJECT_ID": model_library_settings.GCP_PROJECT_ID,
                "GCP_CREDS": model_library_settings.GCP_CREDS,
            }
        )

    @override
    def get_client(
        self, api_key: str | None = None, base_url: str | None = None
    ) -> Client:
        if not self.has_client():
            assert api_key

            try:
                aiohttp_session = make_aiohttp_session()
            except RuntimeError:
                aiohttp_session = (
                    None  # no event loop — SDK will create its own session
                )

            http_options = (
                HttpOptions(base_url=base_url, aiohttp_client=aiohttp_session)
                if base_url
                else HttpOptions(aiohttp_client=aiohttp_session)
                if aiohttp_session
                else (HttpOptions(base_url=base_url) if base_url else None)
            )

            if self.provider_config.use_vertex:
                # Gemini preview releases are only server from the global Vertex region after September 2025.
                MODEL_REGION_OVERRIDES: dict[str, str] = {
                    "gemini-2.5-flash-preview-09-2025": "global",
                    "gemini-2.5-flash-lite-preview-09-2025": "global",
                    "gemini-3-flash-preview": "global",
                    "gemini-3-pro-preview": "global",
                }

                creds = json.loads(api_key)

                region = creds["GCP_REGION"]
                if self.model_name in MODEL_REGION_OVERRIDES:
                    region = MODEL_REGION_OVERRIDES[self.model_name]

                client = Client(
                    vertexai=True,
                    project=creds["GCP_PROJECT_ID"],
                    location=region,
                    credentials=service_account.Credentials.from_service_account_info(  # type: ignore
                        json.loads(creds["GCP_CREDS"]),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    ),
                    http_options=http_options,
                )
            else:
                client = Client(api_key=api_key, http_options=http_options)
            self.assign_client(client)
        return super().get_client()

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

        # https://ai.google.dev/gemini-api/docs/openai
        if self.native:
            self.delegate = None
        else:
            config = config or LLMConfig()
            config.custom_endpoint = (
                config.custom_endpoint
                or "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            config.custom_api_key = config.custom_api_key or SecretStr(
                model_library_settings.GOOGLE_API_KEY
            )

            # flip native back on for the delegate so it initializes its own
            # client registry; the outer model stays non-native.
            config.native = True
            self.delegate = OpenAIModel(
                model_name=self.model_name,
                provider=self.provider,
                config=config,
                use_completions=True,
            )
            config.native = False

        self.supports_batch = (
            self.supports_batch and self.native and not self.custom_endpoint
        )
        self.batch: LLMBatchMixin | None = (
            GoogleBatchMixin(self) if self.supports_batch else None
        )

    @override
    async def parse_input(
        self,
        input: Sequence[InputItem],
        **kwargs: Any,
    ) -> list[Content]:
        new_input: list[Content] = []

        content_user: list[Part] = []

        def flush_content_user():
            if content_user:
                new_input.append(Content(parts=content_user, role="user"))
                content_user.clear()

        for item in input:
            if isinstance(item, TextInput):
                content_user.append(Part.from_text(text=item.text))
                continue

            if isinstance(item, FileBase):
                parsed = await self.parse_file(item)
                content_user.append(parsed)
                continue

            # non content user item
            flush_content_user()

            match item:
                case ToolResult():
                    new_input.append(
                        Content(
                            role="user",
                            parts=[
                                Part(
                                    function_response=FunctionResponse(
                                        id=item.tool_call.id,
                                        name=item.tool_call.name,
                                        response={"result": item.result},
                                    )
                                )
                            ],
                        )
                    )

                case RawResponse():
                    new_input.extend(item.response)
                case RawInput():
                    new_input.append(item.input)
                case SystemInput():
                    raise UnexpectedSystemInputError()

        # in case content user item is the last item
        flush_content_user()

        return new_input

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

    @property
    @override
    def search_tool(self) -> Tool:
        return Tool(google_search=GoogleSearch())

    @override
    async def parse_tools(self, tools: list[ToolDefinition]) -> list[Tool]:
        parsed_tools: list[Tool] = []
        for tool in tools:
            body = tool.body
            if is_native_web_search(body):
                parsed_tools.append(self.search_tool)
                continue
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
        response: File = self.get_client().files.upload(
            file=bytes, config=UploadFileConfig(mime_type=mime)
        )
        if not response.name:
            raise Exception(f"Failed to upload file {name} - no name returned")
        if not response.uri:
            raise Exception(f"Failed to upload file {name} - no uri returned")

        self.instance_logger.info(f"File uploaded successfully: {response.name}")
        return FileWithId(
            type="file",
            file_id=response.uri,
            name=response.display_name or response.name,
            mime=mime,
        )

    @override
    async def build_body(
        self,
        input: Sequence[InputItem],
        *,
        tools: list[ToolDefinition],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
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
            if self.top_k is not None:
                generation_config.top_k = self.top_k

        generation_config.safety_settings = self.SAFETY_CONFIG

        if isinstance(input[0], SystemInput):
            generation_config.system_instruction = input[0].text
            input = input[1:]

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
            if any(is_native_web_search(t.body) for t in tools):
                generation_config.tool_config = ToolConfig(
                    include_server_side_tool_invocations=True
                )

        if output_schema is not None:
            if isinstance(output_schema, dict):
                # response_schema doesn't support additionalProperties in JSON schema
                # so we use response_json_schema instead of response_schema
                generation_config.response_json_schema = output_schema
            else:
                generation_config.response_schema = output_schema
            generation_config.response_mime_type = "application/json"

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
        query_logger: logging.Logger,
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> QueryResult:
        if self.delegate:
            return await self.delegate_query(
                input,
                tools=tools,
                query_logger=query_logger,
                output_schema=output_schema,
                **kwargs,
            )

        body: dict[str, Any] = await self.build_body(
            input, tools=tools, output_schema=output_schema, **kwargs
        )

        result_builder = QueryResultBuilder()
        tool_calls: list[ToolCall] = []

        metadata: GenerateContentResponseUsageMetadata | None = None
        response_id: str | None = None
        grounding_metadata: GroundingMetadata | None = None

        stream = await self.get_client().aio.models.generate_content_stream(**body)
        contents: list[Content | None] = []
        finish_reason: FinishReason | None = None

        chunks: list[GenerateContentResponse] = []

        async for chunk in stream:
            chunks.append(chunk)
            if chunk.response_id:
                response_id = chunk.response_id
            candidates = chunk.candidates
            if not candidates:
                continue

            content = candidates[0].content

            meaningful_content = False
            if content and content.parts:
                for part in content.parts:
                    if part.function_call:
                        meaningful_content = True
                        if not part.function_call.name:
                            raise Exception(f"Invalid function call: {part}")

                        call_args = part.function_call.args or {}
                        result_builder.start_tool_call_segment().record_tool_call_delta()
                        tool_calls.append(
                            # Weirdly, id is not required. If not provided, we generate one.
                            ToolCall(
                                id=part.function_call.id
                                or generate_tool_call_id(part.function_call.name),
                                name=part.function_call.name,
                                args=call_args,
                            )
                        )
                    if part.text is None:
                        continue
                    if part.thought:
                        meaningful_content = True
                        result_builder.append_reasoning_delta(part.text)
                    else:
                        meaningful_content = True
                        result_builder.append_content_delta(part.text)

            if chunk.usage_metadata:
                metadata = chunk.usage_metadata
            if content and meaningful_content:
                contents.append(content)
            if candidates[0].finish_reason:
                finish_reason = candidates[0].finish_reason
            if candidates[0].grounding_metadata:
                grounding_metadata = candidates[0].grounding_metadata

        if finish_reason != FinishReason.STOP:
            query_logger.error(
                f"Unexpected finish reason: {finish_reason}, chunks: {chunks}"
            )

        if finish_reason == FinishReason.MALFORMED_FUNCTION_CALL:
            # gemini handles malformed function calls server side
            # we don't want to return the content that was supposed to have a tool call, without that tool call
            # and since we don't get any info on the params, we throw an error

            query_logger.error("The function call was malformed")
            raise RetryException("The function call was malformed")

        provider_tool_events: list[ProviderToolEvent] = []
        if grounding_metadata and grounding_metadata.web_search_queries:
            sources: list[JsonValue] = [
                chunk.web.uri
                for chunk in (grounding_metadata.grounding_chunks or [])
                if chunk.web and chunk.web.uri
            ]
            queries = grounding_metadata.web_search_queries
            n = len(queries)
            for i, query in enumerate(queries):
                provider_tool_events.append(
                    ProviderToolEvent.web_search(
                        provider="google",
                        kind="google_search_call",
                        query=query,
                        sources=sources,
                        sequence=i - n,  # negative: sorts before function tool calls
                    )
                )

        mapped_finish_reason = map_google_finish_reason(
            finish_reason, has_tool_calls=bool(tool_calls)
        )
        if (
            not result_builder.has_output_text
            and not result_builder.has_reasoning
            and not tool_calls
            and not provider_tool_events
        ):
            query_logger.error(f"Empty response. Chunks: {chunks}")
            handle_empty_response(mapped_finish_reason, {"metadata": metadata})

        result_metadata = QueryResultMetadata()
        if metadata:
            # see _calculate_cost
            cache_read_tokens = metadata.cached_content_token_count or 0
            result_metadata = QueryResultMetadata(
                in_tokens=(metadata.prompt_token_count or 0) - cache_read_tokens,
                out_tokens=metadata.candidates_token_count or 0,
                reasoning_tokens=metadata.thoughts_token_count or 0,
                cache_read_tokens=cache_read_tokens,
            )
        return result_builder.build(
            finish_reason=mapped_finish_reason,
            history=[*input, RawResponse(response=contents)],
            tool_calls=tool_calls,
            provider_tool_events=provider_tool_events,
            metadata=result_metadata,
            extras=QueryResultExtras(
                provider_response_id=response_id,
            ),
        )

    @override
    async def count_tokens(
        self,
        input: Sequence[InputItem],
        *,
        history: Sequence[InputItem] = [],
        tools: list[ToolDefinition] = [],
        **kwargs: object,
    ) -> int:
        """
        Count the number of tokens using Google's native token counting API.
        https://ai.google.dev/gemini-api/docs/tokens

        Only Vertex AI supports system_instruction and tools in count_tokens.
        For Gemini API, fall back to the base implementation.
        TODO: implement token counting for non-Vertex models.
        """
        if not self.provider_config.use_vertex:
            return await super().count_tokens(
                input, history=history, tools=tools, **kwargs
            )

        input = normalize_query_input(input, history=history, kwargs=kwargs)
        if not input:
            return 0

        system_instruction: str | None = None
        if isinstance(input[0], SystemInput):
            system_instruction = input[0].text
            input = input[1:]
        contents = await self.parse_input(input, **kwargs)
        parsed_tools = await self.parse_tools(tools) if tools else None
        config = CountTokensConfig(
            system_instruction=system_instruction,
            tools=parsed_tools,
        )

        response = await self.get_client().aio.models.count_tokens(
            model=self.model_name,
            contents=cast(Any, contents),
            config=config,
        )

        if response.total_tokens is None:
            raise ValueError("count_tokens returned None")

        return response.total_tokens

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

    @deprecated("Use query(output_schema=...) instead")
    @override
    async def query_json(
        self,
        input: Sequence[InputItem],
        pydantic_model: type[PydanticT],
        output_schema: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: object,
    ) -> PydanticT:
        # Create the request body with JSON schema
        body: dict[str, Any] = await self.build_body(
            input, tools=[], output_schema=output_schema, **kwargs
        )

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
                return await self.get_client().aio.models.generate_content(**body)
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
