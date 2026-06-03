"""Generate only the AI-related stock prediction Markdown report.

Usage:
    uv run python scripts/generate_ai_stock_md.py
    uv run python scripts/generate_ai_stock_md.py --days 30 --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

from src.ai.analyzer import ContentAnalyzer
from src.ai.client import create_ai_client
from src.ai.stock_predictor import AIStockPredictor
from src.scrapers.openbb import OpenBBScraper
from src.storage.manager import StorageManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate only Horizon's AI-related stock prediction Markdown."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Lookback window for OpenBB news fetch. Default: 14.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum fetched items to send through AI scoring. Default: 20.",
    )
    parser.add_argument(
        "--watchlist",
        default="ai-infrastructure",
        help="OpenBB watchlist name from config.json. Default: ai-infrastructure.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv(".env")

    storage = StorageManager("data")
    cfg = storage.load_config()

    openbb = cfg.sources.openbb
    if openbb is None:
        raise SystemExit("OpenBB config is missing from data/config.json")

    openbb.watchlists = [
        wl for wl in openbb.watchlists
        if wl.name == args.watchlist
    ]
    if not openbb.watchlists:
        raise SystemExit(f"No OpenBB watchlist named {args.watchlist!r} found in data/config.json")

    openbb.enabled = True
    cfg.ai.analysis_concurrency = 1
    cfg.ai.throttle_sec = 0

    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    async with httpx.AsyncClient(timeout=30.0) as client:
        scraper = OpenBBScraper(openbb, client)
        items = await scraper.fetch(since)

    if not items:
        raise SystemExit(
            f"No OpenBB items fetched for watchlist {args.watchlist!r}. "
            "Try increasing --days or check OpenBB/yfinance availability."
        )

    analyzer = ContentAnalyzer(create_ai_client(cfg.ai))
    analyzed = await analyzer.analyze_batch(items[: max(args.limit, 1)])

    selected = [
        item for item in analyzed
        if (item.ai_score or 0) >= cfg.filtering.ai_score_threshold
    ]
    if not selected:
        selected = sorted(analyzed, key=lambda item: item.ai_score or 0, reverse=True)[:5]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    predictor = AIStockPredictor(create_ai_client(cfg.ai))
    markdown = await predictor.generate_prediction(selected, today)

    path = storage.save_stock_prediction(today, markdown, language="en")
    print(f"saved: {path}")
    print(
        f"watchlist={args.watchlist} fetched={len(items)} "
        f"analyzed={len(analyzed)} selected={len(selected)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
