"""
Data Ingestion Module

Provides data loading from various sources:
- Bloomberg Terminal CSV exports
- On-chain data (Glassnode) for BTC
- Future: Alternative data sources

Usage:
    from src.data.ingest import ingest_and_validate, load_processed_data
    
    # Ingest new data
    data, results = ingest_and_validate(datetime.now())
    
    # Load existing processed data
    df = load_processed_data("EURUSD", start_date, end_date)
"""

from .bloomberg import (
    load_bloomberg_data,
    load_bloomberg_file,
    ingest_and_validate,
    save_validated_data,
    load_processed_data,
    get_available_dates,
    get_data_dirs,
    ensure_data_dirs,
    split_by_ticker,
    BLOOMBERG_FILE_PATTERNS,
    TICKER_MAPPINGS,
)

from .onchain import (
    fetch_glassnode_metric,
    fetch_all_onchain_metrics,
    ingest_onchain_data,
    save_onchain_data,
    create_mock_onchain_data,
    load_onchain_csv,
    GLASSNODE_METRICS,
)


__all__ = [
    # Bloomberg
    "load_bloomberg_data",
    "load_bloomberg_file",
    "ingest_and_validate",
    "save_validated_data",
    "load_processed_data",
    "get_available_dates",
    "get_data_dirs",
    "ensure_data_dirs",
    "split_by_ticker",
    "BLOOMBERG_FILE_PATTERNS",
    "TICKER_MAPPINGS",
    
    # On-chain
    "fetch_glassnode_metric",
    "fetch_all_onchain_metrics",
    "ingest_onchain_data",
    "save_onchain_data",
    "create_mock_onchain_data",
    "load_onchain_csv",
    "GLASSNODE_METRICS",
]
