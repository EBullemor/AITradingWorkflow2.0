"""
Backtest Module

Provides backtesting framework with:
- Walk-forward validation
- Realistic transaction cost modeling
- Regime-aware evaluation
- Performance metrics and attribution

Usage:
    from src.backtest import BacktestEngine, BacktestConfig, TransactionCostCalculator
    
    config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2025, 12, 31),
        use_walk_forward=True
    )
    
    engine = BacktestEngine(config)
    
    cost_calc = TransactionCostCalculator()
    
    result = engine.run(
        signals=signals_df,
        prices=prices_df,
        cost_calculator=cost_calc.calculate_cost
    )
    
    print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
    print(f"Max DD: {result.metrics.max_drawdown:.1%}")
"""

from .engine import (
    Trade,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    WalkForwardPeriod,
    BacktestEngine,
    FillAssumption,
)

from .costs import (
    CostConfig,
    TransactionCostCalculator,
    calculate_cost,
    DEFAULT_COSTS,
)

from .metrics import (
    PerformanceMetrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cagr,
    calculate_var,
    bootstrap_sharpe_ci,
    calculate_trade_statistics,
    calculate_all_metrics,
)


__all__ = [
    # Engine
    "Trade",
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "WalkForwardPeriod",
    "BacktestEngine",
    "FillAssumption",
    
    # Costs
    "CostConfig",
    "TransactionCostCalculator",
    "calculate_cost",
    "DEFAULT_COSTS",
    
    # Metrics
    "PerformanceMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_cagr",
    "calculate_var",
    "bootstrap_sharpe_ci",
    "calculate_trade_statistics",
    "calculate_all_metrics",
]
