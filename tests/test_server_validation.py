from dataclasses import replace

import pytest
from starlette.requests import Request

from remote_web_search import server


def test_validate_query_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        server._validate_query("  ")


def test_validate_domains_rejects_non_domain_values():
    with pytest.raises(ValueError, match="invalid domain"):
        server._validate_domains(["https://example.com/path"])


def test_validate_options():
    assert server._validate_max_results(1) == 1
    assert server._validate_recency(30) == 30
    assert server._validate_domains(["Example.COM"]) == ["example.com"]


@pytest.mark.asyncio
async def test_static_token_verifier():
    verifier = server.StaticTokenVerifier("correct-token")
    assert await verifier.verify_token("correct-token") is not None
    assert await verifier.verify_token("wrong-token") is None


def test_bearer_auth_rejects_missing_and_wrong_token(monkeypatch):
    monkeypatch.setattr(
        server,
        "settings",
        replace(server.settings, mcp_auth_token="correct-token"),
    )
    def request(headers):
        return Request({"type": "http", "headers": headers, "method": "POST", "path": "/mcp"})

    assert server._require_bearer(request([])).status_code == 401
    assert server._require_bearer(request([(b"authorization", b"Bearer wrong")])).status_code == 401
    assert server._require_bearer(request([(b"authorization", b"Bearer correct-token")])) is None
