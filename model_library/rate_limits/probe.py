from model_library.rate_limits import RateLimit, rate_limit_from_headers
from model_library.utils import default_httpx_client


async def probe_chat_completions_rate_limit(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
) -> RateLimit | None:
    """Read rate-limit headers from an OpenAI-compatible /chat/completions reply.

    Used by providers whose SDK does not expose response headers (xAI speaks
    gRPC, Mistral's client returns parsed bodies only), so the limits are read
    from one minimal direct HTTP call instead.
    """
    async with default_httpx_client() as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_name,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Do not think. Say 'ok'"}],
            },
        )
    response.raise_for_status()
    return rate_limit_from_headers(response.headers)
