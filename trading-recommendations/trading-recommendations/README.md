# AI Trading Recommendations Platform

AI-driven trading recommendations platform generating 2-10 actionable trade ideas per day for FX, Bitcoin, and Commodities, combining quantitative signals with LLM-powered narrative synthesis.

## Overview

This system:
- Ingests market data from Bloomberg Terminal
- Calculates features (momentum, carry, volatility, regime)
- Runs 5 strategy pods to generate signals
- Aggregates and deduplicates signals
- Enriches with LLM-generated rationale (grounded in news sources)
- Applies risk management (position sizing, correlation limits)
- Outputs recommendations to Notion and Slack

## Quick Start

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd trading-recommendations
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run daily pipeline
python pipelines/daily_run.py
```

## Project Structure

```
trading-recommendations/
├── config/                     # Configuration files
│   ├── instruments.yaml        # Tradeable instruments
│   ├── risk_limits.yaml        # Portfolio risk parameters
│   ├── strategy_params.yaml    # Strategy-specific settings
│   ├── feature_registry.yaml   # All features with definitions
│   ├── model_registry.yaml     # Strategy → features mapping
│   ├── data_dependencies.yaml  # Feature → data source mapping
│   ├── backtest_config.yaml    # Backtest settings
│   ├── grounding_config.yaml   # LLM grounding thresholds
│   ├── aggregator_config.yaml  # Signal combination rules
│   ├── monitoring_config.yaml  # Health check thresholds
│   └── data_quality_rules.yaml # Validation thresholds
│
├── data/                       # Data pipeline stages
│   ├── raw/                    # Untouched Bloomberg exports
│   ├── validated/              # After quality checks
│   ├── processed/              # Feature-ready data
│   └── quarantine/             # Rejected data for review
│
├── src/                        # Source code
│   ├── data/
│   │   ├── ingest/             # Data loading scripts
│   │   └── validate/           # Quality checks
│   ├── features/               # Feature engineering
│   ├── strategies/             # Strategy pods
│   ├── aggregator/             # Signal combination
│   ├── llm/
│   │   └── grounding/          # Claim verification
│   ├── risk/                   # Position sizing
│   ├── recommendations/        # Formatting
│   ├── outputs/                # Notion, Slack
│   ├── monitoring/             # Health checks
│   └── registry/               # Config validators
│
├── pipelines/                  # Orchestration
│   ├── daily_run.py            # Main pipeline
│   ├── backtest.py             # Strategy backtesting
│   └── health_check.py         # Post-pipeline monitoring
│
├── prompts/                    # LLM prompt templates
├── tests/                      # Test suites
└── reports/                    # Generated reports
```

## Configuration

All configuration is in YAML files in `config/`. Key files:

- **instruments.yaml**: Define which instruments to trade
- **risk_limits.yaml**: Position sizing, drawdown limits, kill switches
- **strategy_params.yaml**: Strategy-specific thresholds and parameters
- **feature_registry.yaml**: Document all features for traceability

## Data Flow

```
Bloomberg CSV → raw/ → Validation → validated/ → Features → processed/
                           ↓
                      quarantine/ (if invalid)
```

## Strategy Pods

1. **FX Carry + Momentum** - Regime-gated carry and momentum for FX majors
2. **BTC Trend + Volatility** - Trend following with vol breakout detection
3. **Commodities Term Structure** - Backwardation/contango signals
4. **Cross-Asset Risk Sentiment** - Portfolio-level risk-on/off
5. **Mean Reversion** - Vol spike snapback trades

## Environment Variables

Required in `.env`:
- `ANTHROPIC_API_KEY` - Claude API for LLM features
- `NOTION_API_KEY` - Notion integration token
- `NOTION_RECOMMENDATIONS_DB` - Recommendations database ID
- `NOTION_SIGNALS_DB` - Signals database ID
- `NOTION_TRADES_DB` - Trades database ID
- `SLACK_WEBHOOK_URL` - Slack notifications

## Development

```bash
# Run tests
pytest tests/

# Run linting
ruff check src/

# Type checking
mypy src/
```

## License

Proprietary - All rights reserved
