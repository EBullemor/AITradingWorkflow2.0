"""
LLM Module

Provides LLM integration for news summarization and analysis.

Components:
- LLMClient: Claude API wrapper with retry logic
- NewsSummarizer: News article summarization
- Asset-specific summarizers (FX, Commodities, Crypto)

Usage:
    from src.llm import create_news_summarizer, Article
    
    # Create summarizer
    summarizer = create_news_summarizer()
    
    # Create articles
    articles = [
        Article(id="1", title="Fed Signals...", text="...", source="Reuters"),
        Article(id="2", title="ECB Meeting...", text="...", source="Bloomberg"),
    ]
    
    # Summarize
    summary = summarizer.summarize(articles)
    print(summary.format_for_display())
"""

from .client import (
    LLMResponse,
    LLMClient,
    MockLLMClient,
    create_llm_client,
)

from .news_summarizer import (
    Article,
    NewsSummary,
    NewsSummarizer,
    FXNewsSummarizer,
    CommodityNewsSummarizer,
    CryptoNewsSummarizer,
    create_news_summarizer,
)


__all__ = [
    # Client
    "LLMResponse",
    "LLMClient",
    "MockLLMClient",
    "create_llm_client",
    
    # Summarizer
    "Article",
    "NewsSummary",
    "NewsSummarizer",
    "FXNewsSummarizer",
    "CommodityNewsSummarizer",
    "CryptoNewsSummarizer",
    "create_news_summarizer",
]
