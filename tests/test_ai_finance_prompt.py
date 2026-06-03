"""Tests for finance-aware AI stock analysis prompting."""

from src.ai.prompts import CONTENT_ANALYSIS_SYSTEM


def test_content_analysis_prompt_treats_ai_stock_news_as_relevant():
    prompt = CONTENT_ANALYSIS_SYSTEM.lower()

    assert "openbb" in prompt
    assert "ai infrastructure" in prompt
    assert "semiconductor" in prompt
    assert "hyperscaler" in prompt
    assert "stock" in prompt
