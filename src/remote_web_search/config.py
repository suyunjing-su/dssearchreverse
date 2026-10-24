from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


DEFAULT_SYSTEM_PROMPT = (
    "You are a web research assistant with native web search enabled. "
    "Use that native search capability for every request that needs current or external information."
)


class ConfigurationError(ValueError):
    """Raised when an environment setting cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Settings:
    upstream_base_url: str = "https://api.example.com"
    upstream_chat_path: str = "/v1/chat/completions"
    upstream_api_key: str | None = None
    upstream_model: str = "search-enabled-model"
    upstream_timeout_seconds: float = 60.0
    upstream_max_retries: int = 2
    max_query_chars: int = 4000
    max_results: int = 10
    max_domains: int = 10
    upstream_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    upstream_extra_headers: dict[str, str] = field(default_factory=dict)
    server_host: str = "127.0.0.1"
    server_port: int = 8000

    @property
    def chat_url(self) -> str:
        base = self.upstream_base_url.rstrip("/")
        path = self.upstream_chat_path
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        values = {
            "upstream_base_url": _env("UPSTREAM_BASE_URL", defaults.upstream_base_url),
            "upstream_chat_path": _env("UPSTREAM_CHAT_PATH", defaults.upstream_chat_path),
            "upstream_api_key": os.getenv("UPSTREAM_API_KEY") or None,
            "upstream_model": _env("UPSTREAM_MODEL", defaults.upstream_model),
            "upstream_timeout_seconds": _float_env(
                "UPSTREAM_TIMEOUT_SECONDS", defaults.upstream_timeout_seconds, minimum=1
            ),
            "upstream_max_retries": _int_env(
                "UPSTREAM_MAX_RETRIES", defaults.upstream_max_retries, minimum=0, maximum=10
            ),
            "max_query_chars": _int_env("MAX_QUERY_CHARS", defaults.max_query_chars, minimum=1),
            "max_results": _int_env("MAX_RESULTS", defaults.max_results, minimum=1, maximum=50),
            "max_domains": _int_env("MAX_DOMAINS", defaults.max_domains, minimum=0, maximum=50),
            "upstream_system_prompt": _env(
                "UPSTREAM_SYSTEM_PROMPT", defaults.upstream_system_prompt
            ),
            "upstream_extra_headers": _json_headers(),
            "server_host": _env("SERVER_HOST", defaults.server_host),
            "server_port": _int_env("SERVER_PORT", defaults.server_port, minimum=1, maximum=65535),
        }
        _validate_url(values["upstream_base_url"], "UPSTREAM_BASE_URL")
        if not values["upstream_model"].strip():
            raise ConfigurationError("UPSTREAM_MODEL must not be empty")
        return cls(**values)


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _int_env(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigurationError(f"{name} must be >= {minimum}{upper}")
    return value


def _float_env(name: str, default: float, *, minimum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


def _json_headers() -> dict[str, str]:
    raw = os.getenv("UPSTREAM_EXTRA_HEADERS_JSON", "{}").strip() or "{}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("UPSTREAM_EXTRA_HEADERS_JSON must be valid JSON") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ConfigurationError("UPSTREAM_EXTRA_HEADERS_JSON must be an object of string values")
    return value


def _validate_url(value: str, name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute http(s) URL")
