"""
Outputs Module

Handles formatting and delivering trading recommendations to various channels:
- Notion databases
- Slack notifications
- Email digests
- Markdown reports

Components:
- NotionRecommendationClient: Push recommendations to Notion
- NotionTradeLogger: Log executed trades
- RecommendationFormatter: Format signals for various outputs
- TradeCard: Rich trade card format

Usage:
    from src.outputs import (
        create_notion_client,
        RecommendationFormatter,
        format_recommendations_report,
    )
    
    # Create Notion client
    notion = create_notion_client()
    
    # Push recommendations
    page_ids = notion.create_recommendations_batch([rec.to_dict() for rec in recs])
    
    # Format for display
    formatter = RecommendationFormatter()
    markdown = formatter.format_markdown(recommendation)
"""

from .notion_client import (
    NotionConfig,
    NotionRecommendationClient,
    NotionTradeLogger,
    RecommendationStatus,
    MockNotionClient,
    create_notion_client,
)

from .formatter import (
    TradeCard,
    RecommendationFormatter,
    format_recommendations_report,
)


__all__ = [
    # Notion
    "NotionConfig",
    "NotionRecommendationClient",
    "NotionTradeLogger",
    "RecommendationStatus",
    "MockNotionClient",
    "create_notion_client",
    
    # Formatter
    "TradeCard",
    "RecommendationFormatter",
    "format_recommendations_report",
]
