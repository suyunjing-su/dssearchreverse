import json

from remote_web_search.parsing import parse_upstream_content


def test_parse_structured_json_and_normalize_sources():
    response = parse_upstream_content(
        json.dumps(
            {
                "answer": "最新结论",
                "results": [
                    {"title": "一", "url": "https://example.com/a", "snippet": "摘要"},
                    {"title": "重复", "url": "https://example.com/a"},
                    {"title": "无效", "url": "javascript:alert(1)"},
                ],
            }
        ),
        max_results=5,
    )

    assert response.answer == "最新结论"
    assert len(response.results) == 1
    assert response.results[0].url == "https://example.com/a"


def test_parse_markdown_fallback_extracts_citations():
    response = parse_upstream_content(
        "结论见 [官方文档](https://example.com/docs). 也可看 https://example.org/news.",
        max_results=5,
    )

    assert response.answer.startswith("结论见")
    assert [item.url for item in response.results] == [
        "https://example.com/docs",
        "https://example.org/news",
    ]


def test_parse_fenced_json():
    response = parse_upstream_content(
        '```json\n{"summary":"ok","sources":[{"link":"https://example.com"}]}\n```'
    )

    assert response.answer == "ok"
    assert response.results[0].url == "https://example.com"


def test_parse_sources_when_answer_is_missing():
    response = parse_upstream_content(
        '{"results":[{"title":"source","url":"https://example.com"}]}'
    )

    assert response.results[0].title == "source"
    assert response.results[0].url == "https://example.com"
