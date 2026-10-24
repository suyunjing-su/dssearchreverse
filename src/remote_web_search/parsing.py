from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    answer: str
    results: tuple[SearchResult, ...]


_MARKDOWN_LINK = re.compile(r"\[([^\]]{1,300})\]\((https?://[^)\s]+)\)")
_RAW_URL = re.compile(r"https?://[^\s<>\"')\]]+")


def parse_upstream_content(content: str, *, max_results: int = 10) -> SearchResponse:
    """Parse the preferred JSON contract, with a safe text/citation fallback."""
    text = content.strip()
    if not text:
        raise ValueError("upstream returned an empty response")

    payload = _try_json(text)
    if isinstance(payload, dict):
        answer = _first_text(payload, "answer", "summary", "response", "text")
        raw_results = payload.get("results")
        if raw_results is None:
            raw_results = payload.get("sources")
        results = _parse_results(raw_results, max_results=max_results)
        if answer or results:
            return SearchResponse(answer=answer or text, results=results)

    return SearchResponse(
        answer=text,
        results=_extract_link_results(text, max_results=max_results),
    )


def response_to_dict(response: SearchResponse) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "results": [asdict(result) for result in response.results],
    }


def _try_json(text: str) -> Any:
    candidates = [text]
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            candidates.append("\n".join(lines[1:-1]).strip())
    first = text.find("{")
    last = text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_results(raw: Any, *, max_results: int) -> tuple[SearchResult, ...]:
    if not isinstance(raw, list):
        return ()
    parsed: list[SearchResult] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link")
        if not isinstance(url, str) or not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        title = item.get("title") or item.get("name") or url
        snippet = item.get("snippet") or item.get("description") or item.get("content") or ""
        date = item.get("published_at") or item.get("publishedAt") or item.get("date")
        parsed.append(
            SearchResult(
                title=str(title).strip()[:300] or url,
                url=url,
                snippet=str(snippet).strip()[:2000],
                published_at=str(date).strip() if date else None,
            )
        )
        if len(parsed) >= max_results:
            break
    return tuple(parsed)


def _extract_link_results(text: str, *, max_results: int) -> tuple[SearchResult, ...]:
    found: list[SearchResult] = []
    seen: set[str] = set()
    for title, url in _MARKDOWN_LINK.findall(text):
        url = _clean_url(url)
        if _is_http_url(url) and url not in seen:
            seen.add(url)
            found.append(SearchResult(title=title.strip(), url=url))
        if len(found) >= max_results:
            return tuple(found)
    for raw_url in _RAW_URL.findall(text):
        url = _clean_url(raw_url)
        if _is_http_url(url) and url not in seen:
            seen.add(url)
            found.append(SearchResult(title=url, url=url))
        if len(found) >= max_results:
            break
    return tuple(found)


def _clean_url(url: str) -> str:
    return url.rstrip(".,;:!?\"'")


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
