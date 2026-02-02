"""
Risk Management Module

Provides position sizing and portfolio risk management.

Components:
- PositionSizer: Calculate position sizes based on risk rules
- PortfolioRiskManager: Enforce portfolio-level constraints
- Risk checking utilities

Usage:
    from src.risk import PositionSizer, PortfolioRiskManager
    
    # Position sizing
    sizer = PositionSizer(default_risk_pct=0.01)  # 1% risk
    result = sizer.calculate_position_size(
        portfolio_value=100000,
        entry_price=1.1050,
        stop_loss=1.1000,
        direction="LONG",
        instrument="EURUSD"
    )
    print(f"Size: {result.position_size}, Risk: ${result.risk_amount}")
    
    # Portfolio risk
    risk_mgr = PortfolioRiskManager()
    can_trade, reason = risk_mgr.can_add_position("EURUSD", "LONG", 10000)
"""

from .position_sizer import (
    SizingMethod,
    PositionSizeResult,
    PositionSizer,
    calculate_position_size,
    calculate_stop_from_atr,
)

from .portfolio_risk import (
    RiskLevel,
    Position,
    PortfolioRiskReport,
    PortfolioRiskManager,
    check_trade_risk,
)


__all__ = [
    # Position sizing
    "SizingMethod",
    "PositionSizeResult",
    "PositionSizer",
    "calculate_position_size",
    "calculate_stop_from_atr",
    
    # Portfolio risk
    "RiskLevel",
    "Position",
    "PortfolioRiskReport",
    "PortfolioRiskManager",
    "check_trade_risk",
]
