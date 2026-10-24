from __future__ import annotations

import html
import ipaddress
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx

from .parsing import SearchResponse


class CompatibilityError(ValueError):
    """Raised when a DeepSeek-compatible request is invalid."""


class FetchError(RuntimeError):
    """Raised when the local machine cannot fetch a page."""


# Backwards-compatible name used by the HTTP adapter.
DeepSeekCompatibilityError = CompatibilityError

_MAX_FETCH_BYTES = 5 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    count: int = 10
    ori_question: str | None = None
    localization: str = ""
    need_content: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "SearchRequest":
        if not isinstance(payload, dict):
            raise CompatibilityError("request body must be a JSON object")

        query = _required_text(payload, "query", maximum=128)
        count = payload.get("count", 10)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 50:
            raise CompatibilityError("count must be an integer between 1 and 50")

        ori_question = payload.get("ori_question")
        if ori_question is not None:
            ori_question = _text(ori_question, "ori_question", maximum=128)

        localization = payload.get("localization", "")
        if not isinstance(localization, str):
            raise CompatibilityError("localization must be a string")
        localization = localization.strip()
        if localization:
            try:
                ipaddress.ip_address(localization)
            except ValueError as exc:
                raise CompatibilityError("localization must be an IP address") from exc

        need_content = payload.get("need_content", False)
        if not isinstance(need_content, bool):
            raise CompatibilityError("need_content must be a boolean")
        return cls(query, count, ori_question, localization, need_content)


def search_to_deepseek(response: SearchResponse) -> dict[str, list[dict[str, Any]]]:
    """Adapt the existing search result model to DeepSeek's result contract."""
    return {
        "results": [
            {
                "rerank_score": 0.0,
                "title": item.title,
                "url": item.url,
                "snippet": item.snippet,
                "summary": item.snippet,
                "content": "",
                "site_name": _site_name(item.url),
                "site_icon": None,
                "date": _rfc3339_date(item.published_at),
                "authority": -1,
            }
            for item in response.results
        ]
    }


@dataclass(frozen=True, slots=True)
class FetchRequest:
    url: str

    @classmethod
    def from_payload(cls, payload: Any) -> "FetchRequest":
        if not isinstance(payload, dict):
            raise CompatibilityError("request body must be a JSON object")
        url = payload.get("url")
        if not isinstance(url, str):
            raise CompatibilityError("url must be a string")
        url = url.strip()
        if len(url) > 8192:
            raise CompatibilityError("url must be at most 8192 characters")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise CompatibilityError("url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise CompatibilityError("url must not contain a username or password")
        return cls(url)


class DirectFetcher:
    """Fetch a page using the deployment machine's network egress."""

    def __init__(self, timeout_seconds: float = 60.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def fetch(
        self,
        url: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> dict[str, str]:
        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True)
        try:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "remote-web-search-fetch/1.0"},
                    follow_redirects=True,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise FetchError(f"fetch failed: {exc}") from exc

            if len(response.content) > _MAX_FETCH_BYTES:
                raise FetchError("fetched response exceeds 5 MiB")
            content_type = response.headers.get("content-type", "").lower()
            if "html" in content_type or _looks_like_html(response.text):
                title, content = _html_to_markdown(response.text)
            else:
                title, content = "", response.text.strip()
            return {"title": title, "content": content}
        finally:
            if owns_client:
                await client.aclose()


class DeepSeekCompatibility:
    """Serve DeepSeek Search/Fetch response shapes from local capabilities."""

    def __init__(self, upstream: Any, *, fetcher: DirectFetcher | None = None) -> None:
        self.upstream = upstream
        self.fetcher = fetcher or DirectFetcher()

    async def search(
        self,
        payload: Any,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        request = SearchRequest.from_payload(payload)
        response = await self.upstream.search(
            request.query,
            max_results=request.count,
            http_client=http_client,
        )
        result = search_to_deepseek(response)
        if request.need_content:
            for item in result["results"]:
                try:
                    fetched = await self.fetcher.fetch(item["url"], http_client=http_client)
                except FetchError:
                    continue
                item["content"] = fetched["content"]
        return result

    async def fetch(
        self,
        payload: Any,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> dict[str, str]:
        request = FetchRequest.from_payload(payload)
        return await self.fetcher.fetch(request.url, http_client=http_client)


def _parse_search_payload(payload: Any) -> SearchRequest:
    """Compatibility wrapper for callers that validate search payloads directly."""
    return SearchRequest.from_payload(payload)


def _parse_fetch_payload(payload: Any) -> str:
    """Compatibility wrapper for callers that validate fetch payloads directly."""
    return FetchRequest.from_payload(payload).url


def _required_text(payload: dict[str, Any], key: str, *, maximum: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise CompatibilityError(f"{key} must be a string")
    return _text(value, key, maximum=maximum)


def _text(value: str, key: str, *, maximum: int) -> str:
    value = value.strip()
    if not value:
        raise CompatibilityError(f"{key} must not be empty")
    if len(value) > maximum:
        raise CompatibilityError(f"{key} must be at most {maximum} characters")
    return value


def _site_name(url: str) -> str | None:
    hostname = urlparse(url).hostname
    return hostname or None


def _rfc3339_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        if len(value) == 10:
            parsed_date = date.fromisoformat(value)
            return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


class _MarkdownParser(HTMLParser):
    _BLOCK_TAGS = {"address", "article", "blockquote", "div", "dl", "dt", "dd", "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section", "table", "tr", "ul"}
    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._heading: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            self._link_href = dict(attrs).get("href")
            self._link_text = []
        elif tag in {f"h{i}" for i in range(1, 7)}:
            self._heading = tag
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a":
            text = _normalize_space("".join(self._link_text))
            href = self._link_href
            if text and href and urlparse(href).scheme in {"http", "https"}:
                self.parts.append(f"[{text}]({href})")
            else:
                self.parts.append(text)
            self._link_href = None
            self._link_text = []
        elif tag == self._heading:
            self.parts.append("\n")
            self._heading = None
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
            return
        if self._link_href is not None:
            self._link_text.append(data)
        elif self._heading and data.strip():
            level = int(self._heading[1])
            self.parts.append(f"{'#' * level} {_normalize_space(data)}")
        else:
            self.parts.append(data)


def _looks_like_html(source: str) -> bool:
    return bool(re.search(r"<\s*(?:!doctype\s+html|html\b|head\b|body\b)", source[:4096], re.IGNORECASE))


def _html_to_markdown(source: str) -> tuple[str, str]:
    parser = _MarkdownParser()
    parser.feed(source)
    title = _normalize_space("".join(parser.title_parts))
    content = _normalize_markdown("".join(parser.parts))
    return title, content


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _normalize_markdown(value: str) -> str:
    value = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    compact: list[str] = []
    blank = False
    for line in lines:
        if not line:
            if not blank:
                compact.append("")
            blank = True
        else:
            compact.append(line)
            blank = False
    return "\n".join(compact).strip()
