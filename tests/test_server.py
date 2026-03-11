"""Tests for perplexity-mcp server — unit tests with mocked HTTP."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from perplexity_sonar_mcp.server import (
    _format_response,
    _sonar_call,
    web_ask,
    web_search,
)


# ── _format_response ─────────────────────────────────────────────────


class TestFormatResponse:
    def test_basic(self):
        data = {
            "choices": [{"message": {"content": "The answer is 42."}}],
            "citations": ["https://example.com/1", "https://example.com/2"],
        }
        result = _format_response(data)
        assert "The answer is 42." in result
        assert "[1] https://example.com/1" in result
        assert "[2] https://example.com/2" in result

    def test_no_citations(self):
        data = {"choices": [{"message": {"content": "Just text."}}]}
        result = _format_response(data)
        assert result == "Just text."
        assert "Sources:" not in result

    def test_empty_choices(self):
        result = _format_response({"choices": []})
        assert "No response" in result

    def test_empty_dict(self):
        result = _format_response({})
        assert "No response" in result


# ── _sonar_call ──────────────────────────────────────────────────────


class TestSonarCall:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
                _sonar_call("test", model="sonar")

    @patch("perplexity_sonar_mcp.server.httpx.post")
    def test_basic_call(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "answer"}}],
            "citations": [],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "test-key"}):
            result = _sonar_call("what is water", model="sonar")

        assert result["choices"][0]["message"]["content"] == "answer"

        # Verify the payload
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["model"] == "sonar"
        assert payload["messages"][0]["content"] == "what is water"
        assert payload["return_citations"] is True

    @patch("perplexity_sonar_mcp.server.httpx.post")
    def test_academic_focus(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "a"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "key"}):
            _sonar_call("enzyme kinetics", model="sonar-pro", focus="academic")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["search_mode"] == "academic"

    @patch("perplexity_sonar_mcp.server.httpx.post")
    def test_finance_focus(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "a"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "key"}):
            _sonar_call("AAPL 10-K", model="sonar", focus="finance")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["search_mode"] == "sec"

    @patch("perplexity_sonar_mcp.server.httpx.post")
    def test_recency_filter(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "a"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "key"}):
            _sonar_call("latest AI news", model="sonar", recency="week")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert payload["search_recency_filter"] == "week"

    @patch("perplexity_sonar_mcp.server.httpx.post")
    def test_no_focus_no_recency(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": "a"}}]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "key"}):
            _sonar_call("test", model="sonar")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1]["json"]
        assert "search_mode" not in payload
        assert "search_recency_filter" not in payload


# ── Tool wrappers ────────────────────────────────────────────────────


class TestWebSearch:
    @patch("perplexity_sonar_mcp.server._sonar_call")
    def test_returns_formatted(self, mock_call):
        mock_call.return_value = {
            "choices": [{"message": {"content": "MoS2 band gap is 1.8 eV."}}],
            "citations": ["https://example.com"],
        }
        result = web_search("MoS2 band gap")
        assert "1.8 eV" in result
        assert "[1]" in result
        mock_call.assert_called_once()
        # Verify model is sonar
        assert mock_call.call_args.kwargs.get("model") == "sonar" or mock_call.call_args[1].get("model") == "sonar"

    @patch("perplexity_sonar_mcp.server._sonar_call")
    def test_error_handling(self, mock_call):
        mock_call.side_effect = RuntimeError("connection refused")
        result = web_search("test")
        assert "Error" in result
        assert "connection refused" in result


class TestWebAsk:
    @patch("perplexity_sonar_mcp.server._sonar_call")
    def test_uses_sonar_pro(self, mock_call):
        mock_call.return_value = {
            "choices": [{"message": {"content": "Detailed answer."}}],
            "citations": [],
        }
        web_ask("compare Pd vs Pt catalysts")
        assert mock_call.call_args.kwargs.get("model") == "sonar-pro" or mock_call.call_args[1].get("model") == "sonar-pro"

    @patch("perplexity_sonar_mcp.server._sonar_call")
    def test_passes_focus(self, mock_call):
        mock_call.return_value = {
            "choices": [{"message": {"content": "a"}}],
            "citations": [],
        }
        web_ask("enzyme kinetics", focus="academic")
        assert mock_call.call_args.kwargs.get("focus") == "academic" or mock_call.call_args[1].get("focus") == "academic"
