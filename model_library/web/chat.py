from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeGuard, cast

from pydantic import BaseModel, ConfigDict, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from model_library import model
from model_library.base import InputItem, QueryResultMetadata, RawResponse, TextInput
from model_library.register_models import ModelConfig, get_model_registry

STATIC_DIR = Path(__file__).parent / "static"


class ModelDetails(BaseModel):
    key: str
    label: str
    provider: str
    company: str
    context_window: int
    max_tokens: int
    reasoning: bool
    supports_images: bool
    supports_files: bool
    supports_tools: bool
    supports_temperature: bool
    internal_only: bool
    open_source: bool


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    output_text: str
    reasoning: str | None
    metadata: QueryResultMetadata


def _model_details(key: str, config: ModelConfig) -> ModelDetails:
    return ModelDetails(
        key=key,
        label=config.label,
        provider=config.provider_name,
        company=config.company,
        context_window=config.properties.context_window,
        max_tokens=config.properties.max_tokens,
        reasoning=config.properties.reasoning_model,
        supports_images=bool(config.supports.images),
        supports_files=bool(config.supports.files),
        supports_tools=bool(config.supports.tools),
        supports_temperature=bool(config.supports.temperature),
        internal_only=config.metadata.internal_only,
        open_source=config.open_source,
    )


def list_models() -> list[ModelDetails]:
    registry = get_model_registry()
    return [
        _model_details(key, config)
        for key, config in sorted(registry.items(), key=lambda item: item[0])
        if not config.metadata.deprecated
    ]


def _is_history_item(item: object) -> TypeGuard[InputItem]:
    return isinstance(item, (TextInput, RawResponse))


def _build_query_parts(messages: Sequence[ChatMessage]) -> tuple[list[InputItem], str]:
    if not messages:
        raise ValueError("At least one message is required")
    if messages[-1].role != "user":
        raise ValueError("The last message must be from the user")

    history: list[InputItem] = []
    if messages[:-1]:
        transcript = "\n\n".join(
            f"{message.role.title()}: {message.content}" for message in messages[:-1]
        )
        history.append(TextInput(text=f"Conversation so far:\n{transcript}"))

    return history, messages[-1].content


async def run_chat_query(request: ChatRequest) -> ChatResponse:
    history, prompt = _build_query_parts(request.messages)
    llm = model(request.model)
    result = await llm.query(prompt, history=history)
    if result.history:
        supported_history = [item for item in result.history if _is_history_item(item)]
        if supported_history:
            history[:] = supported_history

    return ChatResponse(
        output_text=result.output_text_str,
        reasoning=result.reasoning,
        metadata=result.metadata,
    )


async def homepage(_request: Request) -> Response:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


async def models_endpoint(_request: Request) -> JSONResponse:
    return JSONResponse([details.model_dump(mode="json") for details in list_models()])


async def chat_endpoint(request: Request) -> JSONResponse:
    try:
        payload = ChatRequest.model_validate(cast(object, await request.json()))
        response = await run_chat_query(payload)
    except ValidationError as exc:
        return JSONResponse({"error": exc.errors()}, status_code=422)
    except (KeyError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(response.model_dump(mode="json"))


async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    while True:
        try:
            payload = ChatRequest.model_validate(
                cast(object, await websocket.receive_json())
            )
            response = await run_chat_query(payload)
            await websocket.send_json(
                {
                    "type": "message",
                    **response.model_dump(mode="json"),
                }
            )
        except WebSocketDisconnect:
            return
        except ValidationError as exc:
            await websocket.send_json({"type": "error", "error": exc.errors()})
        except Exception as exc:
            if websocket.application_state == WebSocketState.DISCONNECTED:
                return
            await websocket.send_json({"type": "error", "error": str(exc)})


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/", homepage),
            Route("/api/models", models_endpoint),
            Route("/api/chat", chat_endpoint, methods=["POST"]),
            WebSocketRoute("/ws/chat", websocket_endpoint),
            Mount("/", app=StaticFiles(directory=STATIC_DIR, html=False)),
        ]
    )


app: ASGIApp = create_app()
