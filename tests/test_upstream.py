import json

import httpx
import pytest

from remote_web_search.config import Settings
from remote_web_search.upstream import UpstreamClient


@pytest.mark.asyncio
async def test_upstream_uses_native_search_without_tools():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"found","results":[]}'
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    settings = Settings(
        upstream_base_url="https://upstream.example",
        upstream_api_key="secret",
        upstream_model="native-search",
        upstream_max_retries=0,
    )
    client = UpstreamClient(settings)
    async with httpx.AsyncClient(transport=transport) as http_client:
        response = await client.search(
            "what happened today?",
            max_results=3,
            http_client=http_client,
        )

    assert response.answer == "found"
    assert captured["json"]["model"] == "native-search"
    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]
    assert captured["json"]["stream"] is False
    assert captured["headers"]["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_upstream_supports_content_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": '{"answer":"block"}'},
                            ]
                        }
                    }
                ]
            },
        )

    settings = Settings(upstream_base_url="https://upstream.example", upstream_max_retries=0)
    client = UpstreamClient(settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        response = await client.search("q", max_results=1, http_client=http_client)

    assert response.answer == "block"
