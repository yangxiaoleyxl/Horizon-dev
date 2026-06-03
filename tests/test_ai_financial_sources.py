"""Tests for AI-related financial source presets and config examples."""

import json
from pathlib import Path

from src.models import AIConfig, AIProvider
from src.setup.wizard import build_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))


def test_example_config_enables_ai_infrastructure_openbb_watchlist():
    config = _load_json("data/config.example.json")

    openbb = config["sources"]["openbb"]
    assert openbb["enabled"] is True

    watchlist = next(
        wl for wl in openbb["watchlists"] if wl["name"] == "ai-infrastructure"
    )
    assert watchlist["enabled"] is True
    assert watchlist["provider"] == "yfinance"
    assert watchlist["category"] == "ai-stocks"
    assert watchlist["symbols"] == [
        "NVDA",
        "AMD",
        "AVGO",
        "TSM",
        "ASML",
        "ARM",
        "SMCI",
        "DELL",
        "ANET",
        "MU",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "ORCL",
        "PLTR",
    ]


def test_local_presets_include_ai_finance_domain_with_openbb_watchlist():
    presets = _load_json("data/presets.json")

    domain = next(d for d in presets["domains"] if d["id"] == "ai-finance")
    assert "AI Finance / AI Stocks" == domain["name"]
    assert "ai stocks" in domain["keywords"]
    assert "人工智能股票" in domain["keywords"]

    source = next(s for s in domain["sources"] if s["type"] == "openbb_watchlist")
    assert source["config"]["name"] == "ai-infrastructure"
    assert source["config"]["provider"] == "yfinance"
    assert source["config"]["category"] == "ai-stocks"
    assert "NVDA" in source["config"]["symbols"]
    assert "PLTR" in source["config"]["symbols"]


def test_setup_wizard_builds_openbb_watchlist_from_selected_source():
    ai_config = AIConfig(
        provider=AIProvider.OPENAI,
        model="gpt-4",
        api_key_env="OPENAI_API_KEY",
    )

    config = build_config(
        ai_config,
        [
            {
                "type": "openbb_watchlist",
                "config": {
                    "name": "ai-infrastructure",
                    "provider": "yfinance",
                    "fetch_limit": 25,
                    "category": "ai-stocks",
                    "symbols": ["NVDA", "MSFT"],
                },
            }
        ],
    )

    assert config.sources.openbb is not None
    assert config.sources.openbb.enabled is True
    assert len(config.sources.openbb.watchlists) == 1
    watchlist = config.sources.openbb.watchlists[0]
    assert watchlist.name == "ai-infrastructure"
    assert watchlist.provider == "yfinance"
    assert watchlist.fetch_limit == 25
    assert watchlist.category == "ai-stocks"
    assert watchlist.symbols == ["NVDA", "MSFT"]
