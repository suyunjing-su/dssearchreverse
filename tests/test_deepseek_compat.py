import json

import httpx
import pytest
from starlette.testclient import TestClient

from remote_web_search import server
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
                                            "summary": "Native summary",
                                            "content": "Native content",
                                            "site_name": "Example Site",
                                            "site_icon": "https://example.com/icon.png",
                                            "date": "2026-10-24T08:30:00+08:00",
                                            "rerank_score": 0.87,
                                            "authority": 4,
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
                "rerank_score": 0.87,
                "title": "Example",
                "url": "https://example.com/a",
                "snippet": "Snippet",
                "summary": "Native summary",
                "content": "",
                "site_name": "Example Site",
                "site_icon": "https://example.com/icon.png",
                "date": "2026-10-24T00:30:00+00:00",
                "authority": 4,
            }
        ]
    }
    assert captured["path"] == "/v1/chat/completions"
    prompt = captured["json"]["messages"][1]["content"]
    for field in ("rerank_score", "content", "site_icon", "authority", "summary", "site_name", "date"):
        assert f'"{field}"' in prompt
    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]


@pytest.mark.asyncio
async def test_deepseek_search_preserves_native_content_when_requested():
    def handler(request: httpx.Request) -> httpx.Response:
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
                                            "summary": "Long summary",
                                            "content": "Full native page content",
                                            "site_name": "Example Site",
                                            "site_icon": "https://example.com/icon.png",
                                            "date": "2026-10-24T08:30:00+08:00",
                                            "rerank_score": 2.4,
                                            "authority": 3,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    class FailingFetcher:
        async def fetch(self, *args, **kwargs):
            raise AssertionError("native content should not trigger a fallback fetch")

    compatibility = DeepSeekCompatibility(
        UpstreamClient(Settings(upstream_base_url="https://upstream.example", upstream_max_retries=0)),
        fetcher=FailingFetcher(),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await compatibility.search(
            {"query": "q", "count": 1, "need_content": True},
            http_client=client,
        )

    result = response["results"][0]
    assert result["rerank_score"] == 2.4
    assert result["content"] == "Full native page content"
    assert result["site_icon"] == "https://example.com/icon.png"
    assert result["authority"] == 3
    assert result["summary"] == "Long summary"


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


def test_deepseek_http_routes_are_mounted(monkeypatch):
    class FakeCompatibility:
        async def search(self, payload):
            return {"results": [{"url": payload["query"]}]}

        async def fetch(self, payload):
            return {"title": "local", "content": payload["url"]}

    monkeypatch.setattr(server, "deepseek", FakeCompatibility())
    headers = {}
    if server.settings.mcp_auth_token:
        headers["Authorization"] = f"Bearer {server.settings.mcp_auth_token}"
    with TestClient(server.mcp.streamable_http_app()) as client:
        search = client.post("/v1/tools/search", headers=headers, json={"query": "q"})
        fetch = client.post(
            "/v1/tools/fetch",
            headers=headers,
            json={"url": "https://example.com"},
        )

    assert search.status_code == 200
    assert search.json() == {"results": [{"url": "q"}]}
    assert fetch.status_code == 200
    assert fetch.json() == {"title": "local", "content": "https://example.com"}
