"""
Data Validation Module

Provides comprehensive data validation for the trading pipeline:
- Schema validation
- Quality checks (completeness, freshness, outliers)
- Forward-looking bias prevention
- Quarantine management for invalid data

Usage:
    from src.data.validate import validate_dataframe, ValidationResult
    
    result = validate_dataframe(df, ticker="EURUSD")
    if result.passed:
        # Use the data
    else:
        # Handle validation failure
"""

from .schema import (
    InstrumentType,
    DataFrequency,
    FieldSchema,
    InstrumentSchema,
    get_schema,
    get_required_fields,
    get_all_tickers,
    get_tickers_by_type,
    INSTRUMENT_SCHEMAS,
)

from .quality_checks import (
    ValidationSeverity,
    ValidationStatus,
    ValidationIssue,
    ValidationResult,
    validate_dataframe,
    validate_all_instruments,
    load_validation_config,
    check_required_fields,
    check_missing_values,
    check_row_count,
    check_duplicate_timestamps,
    check_data_freshness,
    check_gaps_in_data,
    check_price_outliers,
    check_negative_prices,
    check_value_bounds,
)

from .bias_checks import (
    BiasCheckConfig,
    check_no_future_data,
    check_point_in_time_consistency,
    check_settlement_vs_realtime,
    check_announcement_timing,
    check_data_revisions,
    run_all_bias_checks,
    create_point_in_time_snapshot,
)

from .quarantine import (
    QuarantineManager,
    quarantine_bad_data,
)


__all__ = [
    # Schema
    "InstrumentType",
    "DataFrequency", 
    "FieldSchema",
    "InstrumentSchema",
    "get_schema",
    "get_required_fields",
    "get_all_tickers",
    "get_tickers_by_type",
    "INSTRUMENT_SCHEMAS",
    
    # Quality Checks
    "ValidationSeverity",
    "ValidationStatus",
    "ValidationIssue",
    "ValidationResult",
    "validate_dataframe",
    "validate_all_instruments",
    "load_validation_config",
    "check_required_fields",
    "check_missing_values",
    "check_row_count",
    "check_duplicate_timestamps",
    "check_data_freshness",
    "check_gaps_in_data",
    "check_price_outliers",
    "check_negative_prices",
    "check_value_bounds",
    
    # Bias Checks
    "BiasCheckConfig",
    "check_no_future_data",
    "check_point_in_time_consistency",
    "check_settlement_vs_realtime",
    "check_announcement_timing",
    "check_data_revisions",
    "run_all_bias_checks",
    "create_point_in_time_snapshot",
    
    # Quarantine
    "QuarantineManager",
    "quarantine_bad_data",
]
