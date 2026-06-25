import pytest

from load import LoadConfig, build_body, operation_url


@pytest.mark.unit
def test_operation_url_uses_gateway_base_url():
    assert (
        operation_url("https://gateway.example.com", "query")
        == "https://gateway.example.com/query"
    )


@pytest.mark.unit
def test_operation_url_deduplicates_base_url_slash():
    assert (
        operation_url("https://gateway.example.com/", "upload")
        == "https://gateway.example.com/files/upload"
    )


@pytest.mark.unit
def test_query_body_does_not_include_provider_extra_params():
    config = LoadConfig(
        url="https://gateway.example.com",
        api_key="key",
        model="openai/gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        operations=("query",),
        duration=1,
        total=None,
        workers=1,
        rps_per_worker=1,
        concurrency=1,
        timeout=30,
        prompt="hello",
        max_tokens=16,
        temperature=0,
        include_token_retry=False,
    )

    body = build_body(config, "query", 123)

    assert "extra_params" not in body
