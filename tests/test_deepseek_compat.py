import json

import httpx
import pytest

from remote_web_search.config import Settings
from remote_web_search.deepseek_compat import (
    CompatibilityError,
    DeepSeekCompatibility,
    FetchRequest,
    SearchRequest,
)
from remote_web_search.upstream import UpstreamClient


@pytest.mark.asyncio
async def test_deepseek_search_endpoint_shape_uses_native_upstream():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "answer",
                                    "results": [
                                        {
                                            "title": "Example",
                                            "url": "https://example.com/a",
                                            "snippet": "Snippet",
                                            "published_at": "2026-10-24",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    settings = Settings(upstream_base_url="https://upstream.example", upstream_max_retries=0)
    compatibility = DeepSeekCompatibility(UpstreamClient(settings))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await compatibility.search(
            {"query": "q", "count": 1, "need_content": False},
            http_client=client,
        )

    assert response == {
        "results": [
            {
                "rerank_score": 0.0,
                "title": "Example",
                "url": "https://example.com/a",
                "snippet": "Snippet",
                "summary": "Snippet",
                "content": "",
                "site_name": "example.com",
                "site_icon": None,
                "date": "2026-10-24",
                "authority": -1,
            }
        ]
    }
    assert captured["path"] == "/v1/chat/completions"
    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]


@pytest.mark.asyncio
async def test_deepseek_fetch_uses_deployment_network_and_returns_markdown():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://example.com/article"
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title>Title</title></head><body><h1>Heading</h1><p>Body</p></body></html>",
        )

    compatibility = DeepSeekCompatibility(
        UpstreamClient(Settings(upstream_base_url="https://upstream.example"))
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await compatibility.fetch(
            {"url": "https://example.com/article"},
            http_client=client,
        )

    assert response["title"] == "Title"
    assert "# Heading" in response["content"]
    assert "Body" in response["content"]


def test_deepseek_payload_validation():
    with pytest.raises(CompatibilityError, match="query"):
        SearchRequest.from_payload({"query": " "})
    with pytest.raises(CompatibilityError, match="username"):
        FetchRequest.from_payload({"url": "https://user:pass@example.com"})
