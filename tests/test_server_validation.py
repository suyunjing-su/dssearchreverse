import pytest

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
