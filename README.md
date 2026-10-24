# Remote Web Search MCP

一个把“上游原生具备 web search 的大模型”包装成 MCP `web_search` 工具的服务器，同时提供 DeepSeek Search/Fetch API 兼容 HTTP 接口。

这里的关键点是：服务器向上游发起普通的 Chat Completions 请求，**不携带 `tools`、`tool_choice` 或任何搜索工具定义**。搜索由上游模型自身的原生能力完成，服务器只负责转发查询、约束输出、解析答案和提取来源。

## 支持的上游接口

默认对接 OpenAI-compatible 的：

```text
POST {UPSTREAM_BASE_URL}{UPSTREAM_CHAT_PATH}
```

请求体是普通的：

```json
{
  "model": "search-enabled-model",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Search the web ..."}
  ],
  "stream": false
}
```

上游需要在模型侧默认执行原生搜索。服务器不会替上游补充搜索工具，也不会把搜索结果当成 MCP tool call 发送。

## DeepSeek Search/Fetch 兼容接口

服务器还提供与 DeepSeek Search/Fetch API 相同路径和主要响应结构的本地接口：

```text
POST /v1/tools/search
POST /v1/tools/fetch
```

这两个接口不是把 DeepSeek 当作上游：Search 仍使用配置的原生搜索模型，Fetch 则由部署服务器直接发起 HTTP 请求访问目标网页。

Search 请求示例：

```json
{
  "query": "人工智能最新进展",
  "count": 10,
  "need_content": false
}
```

Search returns the DeepSeek-compatible `results` array with all documented fields. When the configured native search model supplies `rerank_score`, `content`, `site_icon`, `authority`, `summary`, `site_name`, or `date`, those native values are preserved; missing values use explicit fallbacks (`rerank_score` is `0.0`, `authority` is `-1`, `site_icon` is `null`, and `summary` reuses the available snippet). Dates are normalized to UTC RFC 3339; unparseable dates become `null`. Set `need_content: true` to ask the upstream model for full content and locally fetch a result URL only when the upstream result has no content. Fetch failures leave that result's `content` unchanged. This keeps native-model metadata intact while retaining a safe compatibility fallback.

Fetch 请求示例：

```json
{"url": "https://example.com/article"}
```

Fetch 返回：

```json
{"title": "页面标题", "content": "Markdown 正文"}
```

Fetch 仅允许 HTTP(S) URL，不允许 URL 中包含用户名或密码，并限制目标响应体为 5 MiB。网页 HTML 会提取标题并转换为基础 Markdown；非 HTML 内容会原样作为正文返回。生产环境应在出站防火墙、DNS 策略和响应大小限制之外，再配置网关认证与限流。

## 快速开始

需要 Python 3.10+，推荐使用 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra test
cp .env.example .env
# 编辑 .env，至少设置 UPSTREAM_BASE_URL、UPSTREAM_API_KEY、UPSTREAM_MODEL
uv run remote-web-search --transport stdio
```

Windows PowerShell 可以使用：

```powershell
Copy-Item .env.example .env
uv sync --extra test
uv run remote-web-search --transport stdio
```

### 使用 requirements.txt 部署

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
python -m remote_web_search.server --transport streamable-http --host 0.0.0.0 --port 8000
```

`requirements.txt` 固定了当前验证通过的运行时依赖版本；开发测试依赖仍由 `pyproject.toml` 的 `test` extra 管理。

### MCP 客户端配置（stdio）

```json
{
  "mcpServers": {
    "remote-web-search": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "D:/path/to/remote-web-search-mcp",
        "remote-web-search",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

## 远程 HTTP 模式

启动 Streamable HTTP：

```bash
uv run remote-web-search --transport streamable-http --host 0.0.0.0 --port 8000
```

MCP endpoint 默认是：

```text
http://127.0.0.1:8000/mcp
```

远程部署时请在反向代理层添加 TLS、认证、限流和来源控制；本项目本身不提供用户认证。不要把带有 `UPSTREAM_API_KEY` 的进程直接暴露到公网。

## `web_search` 工具

输入参数：

- `query`：必填，最多 `MAX_QUERY_CHARS` 个字符。
- `max_results`：1 到 `MAX_RESULTS`，默认 5。
- `recency_days`：可选的新鲜度窗口，1 到 3650 天。
- `domains`：可选的优先域名列表，例如 `["gov.cn", "who.int"]`。

返回 JSON 字符串：

```json
{
  "query": "今天的 AI 新闻",
  "provider": "upstream-native-search",
  "answer": "...",
  "results": [
    {
      "title": "...",
      "url": "https://example.com/article",
      "snippet": "...",
      "published_at": "2026-10-24"
    }
  ]
}
```

服务器优先解析上游返回的 JSON。如果上游仍返回普通文本，服务器会原样保留答案，并从 Markdown 链接和裸 URL 中尽力提取来源。

## 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UPSTREAM_BASE_URL` | `https://api.example.com` | 上游 API 根地址 |
| `UPSTREAM_CHAT_PATH` | `/v1/chat/completions` | Chat Completions 路径 |
| `UPSTREAM_API_KEY` | 空 | 以 Bearer token 发送 |
| `UPSTREAM_MODEL` | `search-enabled-model` | 上游原生搜索模型名称 |
| `UPSTREAM_TIMEOUT_SECONDS` | `60` | 单次请求超时 |
| `UPSTREAM_MAX_RETRIES` | `2` | 网络错误和 5xx 重试次数 |
| `MAX_QUERY_CHARS` | `4000` | 查询长度上限 |
| `MAX_RESULTS` | `10` | 单次最大来源数量 |
| `MAX_DOMAINS` | `10` | 域名过滤数量上限 |
| `UPSTREAM_EXTRA_HEADERS_JSON` | `{}` | 额外请求头 JSON 对象 |
| `UPSTREAM_SYSTEM_PROMPT` | 内置提示词 | 上游 system prompt |
| `SERVER_HOST` | `127.0.0.1` | HTTP 监听地址 |
| `SERVER_PORT` | `8000` | HTTP 监听端口 |

## 测试

```bash
uv run pytest
```

测试使用 `httpx.MockTransport`，不会访问真实上游；其中明确断言请求体不含 `tools` 和 `tool_choice`。

## 安全边界

- 查询作为不可信数据放在 user message 中，并在 system/user 提示中明确不能改变模型角色。
- 只接受 `http`/`https` 来源 URL；不会执行上游返回内容中的代码。
- 上游地址完全由部署者配置，未实现任意 URL 代理；生产环境仍应锁定出站网络和 DNS 策略。
- 错误信息会截断上游响应正文，避免把大段响应写入 MCP 错误。
- HTTP 模式没有内置认证，公网部署必须由网关负责认证、TLS、限流和审计。
