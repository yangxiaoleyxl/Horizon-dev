"""Tests for stock prediction item selection in the orchestrator."""

import asyncio
from datetime import datetime, timezone

from src.models import (
    AIConfig,
    AIProvider,
    Config,
    ContentItem,
    FilteringConfig,
    SourceType,
    SourcesConfig,
)
from src.orchestrator import HorizonOrchestrator


class _Storage:
    def __init__(self):
        self.stock_reports = []
        self.daily_summaries = []

    def save_stock_prediction(self, date, summary, language="en"):
        self.stock_reports.append((date, summary, language))
        return "stock.md"

    def save_daily_summary(self, date, summary, language="en"):
        self.daily_summaries.append((date, summary, language))
        return "summary.md"


class _FakePredictor:
    calls = []

    def __init__(self, _client):
        pass

    async def generate_prediction(self, items, date):
        self.calls.append(list(items))
        return "# stock report"


def test_stock_prediction_uses_analyzed_stock_items_before_importance_filter(monkeypatch, tmp_path):
    """Low-scored OpenBB stock news should still feed the stock predictor."""
    monkeypatch.chdir(tmp_path)
    stock_item = ContentItem(
        id="openbb:news:nvda",
        source_type=SourceType.OPENBB,
        title="NVIDIA AI infrastructure update",
        url="https://example.com/nvda",
        content="NVIDIA expands AI infrastructure.",
        published_at=datetime.now(timezone.utc),
        metadata={
            "symbols": ["NVDA"],
            "watchlist": "ai-infrastructure",
            "category": "ai-stocks",
        },
        ai_score=3.0,
        ai_reason="Stock-specific but below general summary threshold.",
    )
    important_non_stock = ContentItem(
        id="rss:news:general-ai",
        source_type=SourceType.RSS,
        title="General AI platform news",
        url="https://example.com/general-ai",
        content="Important general AI news.",
        published_at=datetime.now(timezone.utc),
        metadata={},
        ai_score=9.0,
    )

    config = Config(
        ai=AIConfig(
            provider=AIProvider.OLLAMA,
            model="test-model",
            api_key_env="IGNORED",
            languages=["en"],
        ),
        sources=SourcesConfig(),
        filtering=FilteringConfig(ai_score_threshold=7.0),
    )
    orchestrator = HorizonOrchestrator(config, _Storage())

    async def fake_fetch_all_sources(since):
        return [stock_item, important_non_stock]

    monkeypatch.setattr(orchestrator, "fetch_all_sources", fake_fetch_all_sources)

    async def fake_analyze(items):
        return list(items)

    async def identity_dedupe(items):
        return list(items)

    async def noop_expand(items):
        return None

    async def noop_enrich(items):
        return None

    monkeypatch.setattr(orchestrator, "_analyze_content", fake_analyze)
    monkeypatch.setattr(orchestrator, "merge_topic_duplicates", identity_dedupe)
    monkeypatch.setattr(orchestrator, "_expand_twitter_discussion", noop_expand)
    monkeypatch.setattr(orchestrator, "_enrich_important_items", noop_enrich)
    monkeypatch.setattr("src.orchestrator.create_ai_client", lambda ai_config: object())
    monkeypatch.setattr("src.orchestrator.AIStockPredictor", _FakePredictor)
    _FakePredictor.calls = []

    asyncio.run(orchestrator.run())

    assert _FakePredictor.calls == [[stock_item]]
