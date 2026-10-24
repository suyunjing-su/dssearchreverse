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


def test_parse_preserves_deepseek_result_metadata():
    response = parse_upstream_content(
        json.dumps(
            {
                "answer": "ok",
                "results": [
                    {
                        "rerank_score": 0.91,
                        "title": "Native title",
                        "url": "https://example.com/article",
                        "snippet": "Snippet",
                        "summary": "Summary",
                        "content": "Full content",
                        "site_name": "Example",
                        "site_icon": "https://example.com/icon.svg",
                        "date": "2026-10-24T00:00:00Z",
                        "authority": 7,
                    }
                ],
            }
        )
    )

    result = response.results[0]
    assert result.rerank_score == 0.91
    assert result.summary == "Summary"
    assert result.content == "Full content"
    assert result.site_name == "Example"
    assert result.site_icon == "https://example.com/icon.svg"
    assert result.date == "2026-10-24T00:00:00Z"
    assert result.authority == 7
