"""Perplexity MCP server — web-grounded search for LLM tool use.

Tools registered in escalating order (cheapest first):
  web_search    — quick facts via Sonar (2-5s)
  web_ask       — thorough answer via Sonar Pro (5-15s)
  deep_research — comprehensive multi-step analysis (2-10 min)

Requires PERPLEXITY_API_KEY environment variable.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

# ── Constants ────────────────────────────────────────────────────────

SONAR_URL = "https://api.perplexity.ai/chat/completions"
AGENT_URL = "https://api.perplexity.ai/chat/completions"

VALID_FOCUS = {"", "academic", "finance"}

mcp = FastMCP("perplexity")


# ── Helpers ──────────────────────────────────────────────────────────


def _get_api_key() -> str:
    return os.environ.get("PERPLEXITY_API_KEY", "")


def _sonar_call(
    query: str,
    model: str,
    focus: str = "",
    recency: str = "",
    timeout: int = 60,
) -> dict[str, Any]:
    """Call the Sonar API and return the raw response dict."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not set")

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "return_citations": True,
        "web_search_options": {"search_context_size": "high"},
    }

    if focus in ("academic", "finance"):
        mode = "sec" if focus == "finance" else focus
        payload["search_mode"] = mode

    if recency in ("day", "week", "month", "year"):
        payload["search_recency_filter"] = recency

    resp = httpx.post(
        SONAR_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _agent_call(
    query: str,
    preset: str = "deep-research",
    timeout: int = 600,
) -> dict[str, Any]:
    """Call the Agent API with a preset."""
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY not set")

    payload: dict[str, Any] = {
        "model": preset,
        "messages": [{"role": "user", "content": query}],
        "return_citations": True,
    }

    resp = httpx.post(
        AGENT_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _format_response(data: dict[str, Any]) -> str:
    """Format API response as compact text with sources."""
    choices = data.get("choices", [])
    if not choices:
        return "No response from Perplexity."

    content = choices[0].get("message", {}).get("content", "")
    citations = data.get("citations", [])

    parts = [content.strip()]

    if citations:
        parts.append("\n\nSources:")
        for i, url in enumerate(citations, 1):
            parts.append(f"[{i}] {url}")

    return "\n".join(parts)


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool()
def web_search(query: str, focus: str = "", recency: str = "") -> str:
    """Quick web search — fast factual answers with sources (2-5 seconds).

    Use for: simple facts, definitions, current events, quick lookups.

    Args:
        query: Natural language question or search query.
        focus: Search focus — "" (general web), "academic" (scholarly sources),
               or "finance" (SEC filings).
        recency: Time filter — "day", "week", "month", "year", or "" (all time).
    """
    try:
        data = _sonar_call(query, model="sonar", focus=focus, recency=recency, timeout=30)
        return _format_response(data)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def web_ask(query: str, focus: str = "", recency: str = "") -> str:
    """Thorough web research — detailed answer with many citations (5-15 seconds).

    Use for: complex questions needing multiple sources, comparisons,
    literature discovery, nuanced answers.

    Args:
        query: Natural language question or research query.
        focus: Search focus — "" (general web), "academic" (scholarly sources),
               or "finance" (SEC filings).
        recency: Time filter — "day", "week", "month", "year", or "" (all time).
    """
    try:
        data = _sonar_call(query, model="sonar-pro", focus=focus, recency=recency, timeout=60)
        return _format_response(data)
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def deep_research(query: str) -> str:
    """Comprehensive multi-step research — takes 2-10 MINUTES.

    Performs extensive multi-step web research with up to 10 reasoning
    iterations. Use only for comprehensive analysis that justifies the
    wait time. Ask the user for confirmation before calling this tool.

    Args:
        query: Detailed research question or analysis request.
    """
    try:
        data = _agent_call(query, preset="deep-research", timeout=600)
        return _format_response(data)
    except Exception as exc:
        return f"Error: {exc}"


# ── Entry point ──────────────────────────────────────────────────────


def main():
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
