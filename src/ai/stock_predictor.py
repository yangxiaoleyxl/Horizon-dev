"""AI-related stock prediction Markdown generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from .client import AIClient
from .utils import parse_json_response
from ..models import ContentItem, SourceType


STOCK_PREDICTION_SYSTEM = """You are an AI-sector equity news analyst.

Use only the supplied news items. Estimate short-term directional probabilities for each mentioned AI-related stock over the next 1-5 trading days.

Return valid JSON only. This is not financial advice. Probabilities are subjective news-impact estimates, not guarantees.
"""

STOCK_PREDICTION_USER = """Analyze these AI-related stock news items and estimate rise/descend probabilities for each symbol.

Rules:
- Output one row per important symbol explicitly listed in the news metadata.
- rise_probability and descend_probability must be integers from 0 to 100 and should sum to 100.
- Use confidence: low, medium, or high.
- Base rationale only on the supplied news text, scores, and reasons.
- Prefer conservative probabilities when evidence is weak or only price-chatter.

Date: {date}

News items:
{news_items}

Respond with valid JSON only:
{{
  "market_overview": "<brief overview>",
  "predictions": [
    {{
      "symbol": "NVDA",
      "company": "NVIDIA or Unknown",
      "rise_probability": 55,
      "descend_probability": 45,
      "confidence": "low|medium|high",
      "key_news": ["<short evidence bullet>", "..."],
      "rationale": "<short rationale>"
    }}
  ],
  "disclaimer": "Not financial advice."
}}"""


def extract_ai_stock_items(items: Iterable[ContentItem]) -> List[ContentItem]:
    """Return OpenBB AI-stock watchlist items that have symbol metadata."""
    out: List[ContentItem] = []
    for item in items:
        if item.source_type != SourceType.OPENBB:
            continue
        symbols = item.metadata.get("symbols") or []
        if not symbols:
            continue
        category = str(item.metadata.get("category") or "").lower()
        watchlist = str(item.metadata.get("watchlist") or "").lower()
        if category == "ai-stocks" or watchlist == "ai-infrastructure":
            out.append(item)
    return out


class AIStockPredictor:
    """Generate a stock prediction Markdown report from AI-related stock news."""

    def __init__(self, ai_client: AIClient):
        self.client = ai_client

    async def generate_prediction(
        self,
        items: List[ContentItem],
        date: str,
    ) -> str:
        """Generate an AI-related stock prediction Markdown document."""
        stock_items = extract_ai_stock_items(items)
        if not stock_items:
            return self._empty_report(date)

        prompt = STOCK_PREDICTION_USER.format(
            date=date,
            news_items=self._format_news_items(stock_items),
        )
        response = await self.client.complete(
            system=STOCK_PREDICTION_SYSTEM,
            user=prompt,
            max_tokens=4096,
        )
        data = parse_json_response(response) or {}
        return self._render_markdown(date, stock_items, data)

    @staticmethod
    def _format_news_items(items: List[ContentItem]) -> str:
        lines = []
        for idx, item in enumerate(items, start=1):
            symbols = ", ".join(item.metadata.get("symbols") or [])
            lines.append(
                "\n".join(
                    [
                        f"{idx}. Title: {item.title}",
                        f"   Symbols: {symbols}",
                        f"   URL: {item.url}",
                        f"   Published: {item.published_at.isoformat()}",
                        f"   AI score: {item.ai_score if item.ai_score is not None else 'unknown'}",
                        f"   AI reason: {item.ai_reason or ''}",
                        f"   Summary: {item.ai_summary or item.content or ''}",
                    ]
                )
            )
        return "\n\n".join(lines)

    @staticmethod
    def _empty_report(date: str) -> str:
        return (
            f"# AI-Related Stock Prediction - {date}\n\n"
            "> No AI-related stock news was available for prediction.\n\n"
            "No OpenBB `ai-infrastructure` / `ai-stocks` items were selected. "
            "Try increasing `--hours`, lowering `ai_score_threshold`, or checking the OpenBB source.\n\n"
            "_Not financial advice._\n"
        )

    def _render_markdown(
        self,
        date: str,
        source_items: List[ContentItem],
        data: dict,
    ) -> str:
        predictions = data.get("predictions") or []
        overview = data.get("market_overview") or "No market overview returned."
        disclaimer = data.get("disclaimer") or "Not financial advice."

        lines = [
            f"# AI-Related Stock Prediction - {date}",
            "",
            "> Short-term directional probabilities inferred from the selected AI-related news items.",
            "",
            "**Market overview**: " + str(overview),
            "",
            "| Symbol | Company | Rise probability | Descend probability | Confidence |",
            "|---|---|---:|---:|---|",
        ]

        for pred in predictions:
            symbol = self._cell(pred.get("symbol", "?"))
            company = self._cell(pred.get("company", "Unknown"))
            rise = self._prob(pred.get("rise_probability"))
            descend = self._prob(pred.get("descend_probability"))
            confidence = self._cell(pred.get("confidence", "low"))
            lines.append(f"| {symbol} | {company} | {rise} | {descend} | {confidence} |")

        if not predictions:
            lines.append("| N/A | No prediction returned | N/A | N/A | N/A |")

        lines += ["", "## Rationale", ""]
        for pred in predictions:
            symbol = self._cell(pred.get("symbol", "?"))
            rationale = str(pred.get("rationale") or "No rationale returned.").strip()
            lines.append(f"### {symbol}")
            lines.append("")
            key_news = pred.get("key_news") or []
            if key_news:
                lines.append("Key news:")
                for news in key_news:
                    lines.append(f"- {news}")
                lines.append("")
            lines.append(rationale)
            lines.append("")

        lines += ["## Source news", ""]
        for idx, item in enumerate(source_items, start=1):
            symbols = ", ".join(item.metadata.get("symbols") or [])
            score = item.ai_score if item.ai_score is not None else "?"
            lines.append(f"{idx}. [{item.title}]({item.url}) — {symbols} — score {score}/10")

        lines += ["", f"_Disclaimer: {disclaimer}_", ""]
        return "\n".join(lines)

    @staticmethod
    def _cell(value) -> str:
        return str(value).replace("|", "\\|").strip()

    @staticmethod
    def _prob(value) -> str:
        try:
            return f"{int(round(float(value)))}%"
        except (TypeError, ValueError):
            return "N/A"
