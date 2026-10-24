from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import Settings
from .deepseek_compat import (
    CompatibilityError,
    DeepSeekCompatibility,
    FetchError,
)
from .parsing import response_to_dict
from .upstream import UpstreamClient, UpstreamError

LOGGER = logging.getLogger(__name__)
_DOMAIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::[0-9]{1,5})?$")

settings = Settings.from_env()
upstream = UpstreamClient(settings)
deepseek = DeepSeekCompatibility(upstream)
mcp = FastMCP("remote-web-search")


@mcp.custom_route("/v1/tools/search", methods=["POST"])
async def deepseek_search_endpoint(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        return JSONResponse(await deepseek.search(payload))
    except CompatibilityError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": {"message": f"invalid JSON: {exc}"}}, status_code=400)
    except (FetchError, UpstreamError) as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=502)


@mcp.custom_route("/v1/tools/fetch", methods=["POST"])
async def deepseek_fetch_endpoint(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        return JSONResponse(await deepseek.fetch(payload))
    except CompatibilityError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    except ValueError as exc:
        return JSONResponse({"error": {"message": f"invalid JSON: {exc}"}}, status_code=400)
    except (FetchError, UpstreamError) as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=502)


@mcp.tool()
async def web_search(
    query: Annotated[str, "The web search query."],
    max_results: Annotated[int, "Maximum number of sources to return (1-10)."] = 5,
    recency_days: Annotated[int | None, "Optional freshness window in days."] = None,
    domains: Annotated[list[str] | None, "Optional domains to prioritize."] = None,
) -> str:
    """Search the web through an upstream model with native search enabled.

    The upstream request deliberately contains no MCP or provider tool definitions;
    the configured model is expected to perform search as part of its normal reply.
    """
    clean_query = _validate_query(query)
    clean_max_results = _validate_max_results(max_results)
    clean_recency = _validate_recency(recency_days)
    clean_domains = _validate_domains(domains)

    result = await upstream.search(
        clean_query,
        max_results=clean_max_results,
        recency_days=clean_recency,
        domains=clean_domains,
    )
    return json.dumps(
        {
            "query": clean_query,
            "provider": "upstream-native-search",
            **response_to_dict(result),
        },
        ensure_ascii=False,
    )


def _validate_query(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    clean = query.strip()
    if not clean:
        raise ValueError("query must not be empty")
    if len(clean) > settings.max_query_chars:
        raise ValueError(f"query must be at most {settings.max_query_chars} characters")
    return clean


def _validate_max_results(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_results must be an integer")
    if not 1 <= value <= settings.max_results:
        raise ValueError(f"max_results must be between 1 and {settings.max_results}")
    return value


def _validate_recency(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3650:
        raise ValueError("recency_days must be an integer between 1 and 3650")
    return value


def _validate_domains(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if not isinstance(values, list) or len(values) > settings.max_domains:
        raise ValueError(f"domains must contain at most {settings.max_domains} items")
    cleaned: list[str] = []
    for value in values:
        domain = value.strip().lower()
        if not domain or not _DOMAIN.fullmatch(domain) or ".." in domain:
            raise ValueError(f"invalid domain: {value!r}")
        cleaned.append(domain)
    return cleaned


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default=settings.server_host)
    parser.add_argument("--port", type=int, default=settings.server_port)
    args = parser.parse_args(argv)

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    if args.transport == "streamable-http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        LOGGER.info("starting Streamable HTTP MCP server on %s:%s", args.host, args.port)
    else:
        LOGGER.info("starting stdio MCP server")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
