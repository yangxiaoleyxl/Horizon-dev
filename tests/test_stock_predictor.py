"""Tests for AI stock prediction Markdown generation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.ai.stock_predictor import AIStockPredictor, extract_ai_stock_items
from src.models import ContentItem, SourceType


class FakeAIClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    async def complete(self, system: str, user: str, max_tokens: int | None = None, **kwargs) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, **kwargs})
        return self.response


def _item(
    title: str,
    symbols: list[str],
    score: float = 7.0,
    category: str = "ai-stocks",
    watchlist: str = "ai-infrastructure",
) -> ContentItem:
    return ContentItem(
        id=f"openbb:news:{title}",
        source_type=SourceType.OPENBB,
        title=title,
        url="https://example.com/news",
        content="AI server and data-center demand update.",
        author="wire",
        published_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        metadata={"symbols": symbols, "watchlist": watchlist, "category": category},
        ai_score=score,
        ai_reason="Relevant AI infrastructure stock news",
        ai_summary="AI infrastructure demand is changing expectations.",
        ai_tags=["ai-infrastructure", "stocks"],
    )


def test_extract_ai_stock_items_keeps_openbb_ai_stock_news_only():
    ai_item = _item("NVIDIA rises on AI server demand", ["NVDA"])
    mega_item = _item("Generic megacap news", ["AAPL"], category="equities", watchlist="megacaps")
    non_openbb = ContentItem(
        id="rss:1",
        source_type=SourceType.RSS,
        title="AI software release",
        url="https://example.com/rss",
        published_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        metadata={},
    )

    assert extract_ai_stock_items([mega_item, non_openbb, ai_item]) == [ai_item]


def test_generate_prediction_markdown_contains_rise_and_descend_probabilities():
    response = """
    {
      "market_overview": "AI infrastructure demand is positive, but valuation risk remains high.",
      "predictions": [
        {
          "symbol": "NVDA",
          "company": "NVIDIA",
          "rise_probability": 64,
          "descend_probability": 36,
          "confidence": "medium",
          "key_news": ["AI server demand remains strong"],
          "rationale": "Demand tailwinds outweigh valuation risk over the short horizon."
        },
        {
          "symbol": "SMCI",
          "company": "Super Micro Computer",
          "rise_probability": 48,
          "descend_probability": 52,
          "confidence": "low",
          "key_news": ["AI factory rollout is positive but execution risk remains"],
          "rationale": "The news is constructive, but volatility and execution risk dominate."
        }
      ],
      "disclaimer": "Not financial advice."
    }
    """
    predictor = AIStockPredictor(FakeAIClient(response))

    md = asyncio.run(
        predictor.generate_prediction(
            [_item("NVIDIA AI server demand update", ["NVDA", "SMCI"])],
            date="2026-06-02",
        )
    )

    assert md.startswith("# AI-Related Stock Prediction - 2026-06-02")
    assert "| NVDA | NVIDIA | 64% | 36% | medium |" in md
    assert "| SMCI | Super Micro Computer | 48% | 52% | low |" in md
    assert "Not financial advice" in md


def test_generate_prediction_empty_markdown_when_no_ai_stock_news():
    predictor = AIStockPredictor(FakeAIClient("{}"))

    md = asyncio.run(predictor.generate_prediction([], date="2026-06-02"))

    assert "No AI-related stock news was available" in md
