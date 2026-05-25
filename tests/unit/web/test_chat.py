from __future__ import annotations

from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from model_library.base import QueryResult, QueryResultMetadata, TextInput
from model_library.web import chat


def test_models_endpoint_lists_registry_models() -> None:
    client = TestClient(chat.create_app())

    response = client.get("/api/models")

    assert response.status_code == 200
    models = response.json()
    assert any(model["key"] == "openai/gpt-4o" for model in models)


async def test_run_chat_query_sends_history(monkeypatch) -> None:
    mock_llm = AsyncMock()
    mock_llm.query.return_value = QueryResult(
        output_text="Hello from the model",
        metadata=QueryResultMetadata(in_tokens=10, out_tokens=5),
    )
    monkeypatch.setattr(chat, "model", lambda _model_key: mock_llm)

    response = await chat.run_chat_query(
        chat.ChatRequest(
            model="openai/gpt-4o",
            messages=[
                chat.ChatMessage(role="user", content="Hi"),
                chat.ChatMessage(role="assistant", content="Hello"),
                chat.ChatMessage(role="user", content="How are you?"),
            ],
        )
    )

    mock_llm.query.assert_awaited_once_with(
        "How are you?",
        history=[TextInput(text="Conversation so far:\nUser: Hi\n\nAssistant: Hello")],
    )
    assert response.output_text == "Hello from the model"
    assert response.metadata.in_tokens == 10


def test_chat_endpoint_validates_last_message(monkeypatch) -> None:
    mock_llm = AsyncMock()
    monkeypatch.setattr(chat, "model", lambda _model_key: mock_llm)
    client = TestClient(chat.create_app())

    response = client.post(
        "/api/chat",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "assistant", "content": "Hello"}],
        },
    )

    assert response.status_code == 400
    assert "last message" in response.json()["error"]
    mock_llm.query.assert_not_awaited()
