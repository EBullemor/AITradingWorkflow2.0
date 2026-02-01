"""
Unit Tests for Data Validation Module

Tests validation checks with intentionally good and bad data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

# Import validation functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.validate import (
    # Schema
    InstrumentType,
    get_schema,
    get_required_fields,
    get_all_tickers,
    
    # Quality checks
    ValidationSeverity,
    ValidationStatus,
    ValidationResult,
    validate_dataframe,
    check_required_fields,
    check_missing_values,
    check_duplicate_timestamps,
    check_data_freshness,
    check_price_outliers,
    check_negative_prices,
    
    # Bias checks
    check_no_future_data,
    run_all_bias_checks,
    create_point_in_time_snapshot,
    
    # Quarantine
    QuarantineManager,
    quarantine_bad_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def good_fx_data():
    """Create valid FX data for testing."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    np.random.seed(42)
    
    # Generate realistic EURUSD prices around 1.10
    base_price = 1.10
    returns = np.random.normal(0, 0.005, 100)  # 0.5% daily vol
    prices = base_price * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": prices,
        "PX_HIGH": prices * (1 + np.abs(np.random.normal(0, 0.003, 100))),
        "PX_LOW": prices * (1 - np.abs(np.random.normal(0, 0.003, 100))),
    })
    
    return df


@pytest.fixture
def bad_fx_data_missing():
    """FX data with missing values."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    prices = np.random.uniform(1.05, 1.15, 100)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": prices,
        "PX_HIGH": prices * 1.01,
        "PX_LOW": prices * 0.99,
    })
    
    # Add missing values
    df.loc[10:15, "PX_LAST"] = np.nan
    
    return df


@pytest.fixture
def bad_fx_data_outliers():
    """FX data with outliers."""
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    prices = np.ones(100) * 1.10  # Stable prices
    
    # Add extreme outlier
    prices[50] = 2.0  # 80% jump - clearly an outlier
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": prices,
        "PX_HIGH": prices * 1.01,
        "PX_LOW": prices * 0.99,
    })
    
    return df


@pytest.fixture
def bad_fx_data_future():
    """FX data with future timestamps."""
    now = datetime.now()
    dates = pd.date_range(start=now - timedelta(days=10), periods=20, freq="D")
    prices = np.random.uniform(1.05, 1.15, 20)
    
    df = pd.DataFrame({
        "timestamp": dates,
        "PX_LAST": prices,
        "PX_HIGH": prices * 1.01,
        "PX_LOW": prices * 0.99,
    })
    
    return df


@pytest.fixture
def temp_quarantine_dir():
    """Create temporary quarantine directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# =============================================================================
# Schema Tests
# =============================================================================

class TestSchema:
    """Tests for schema module."""
    
    def test_get_schema_eurusd(self):
        """Test getting EURUSD schema."""
        schema = get_schema("EURUSD")
        assert schema is not None
        assert schema.ticker == "EURUSD"
        assert schema.instrument_type == InstrumentType.FX_MAJOR
    
    def test_get_schema_case_insensitive(self):
        """Test schema lookup is case-insensitive."""
        schema1 = get_schema("EURUSD")
        schema2 = get_schema("eurusd")
        assert schema1 == schema2
    
    def test_get_schema_unknown_ticker(self):
        """Test getting schema for unknown ticker."""
        schema = get_schema("UNKNOWN123")
        assert schema is None
    
    def test_get_required_fields(self):
        """Test getting required fields for an instrument."""
        required = get_required_fields("EURUSD")
        assert "PX_LAST" in required
        assert "timestamp" in required
    
    def test_get_all_tickers(self):
        """Test getting all registered tickers."""
        tickers = get_all_tickers()
        assert len(tickers) > 0
        assert "EURUSD" in tickers or "eurusd" in [t.lower() for t in tickers]


# =============================================================================
# Quality Check Tests
# =============================================================================

class TestQualityChecks:
    """Tests for quality check functions."""
    
    def test_validate_good_data_passes(self, good_fx_data):
        """Test that good data passes validation."""
        result = validate_dataframe(good_fx_data, "EURUSD")
        
        assert result.passed
        assert result.status in [ValidationStatus.PASSED, ValidationStatus.PASSED_WITH_WARNINGS]
    
    def test_validate_missing_values_detected(self, bad_fx_data_missing):
        """Test that missing values are detected."""
        result = validate_dataframe(bad_fx_data_missing, "EURUSD")
        
        # Should have issues about missing values
        missing_issues = [i for i in result.issues if i.check_name == "missing_values"]
        assert len(missing_issues) > 0
    
    def test_check_required_fields_missing(self):
        """Test detection of missing required fields."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10),
            # Missing PX_LAST, PX_HIGH, PX_LOW
        })
        
        issues = check_required_fields(df, "EURUSD")
        assert len(issues) > 0
        assert issues[0].severity == ValidationSeverity.CRITICAL
    
    def test_check_required_fields_present(self, good_fx_data):
        """Test that complete data has no missing field issues."""
        issues = check_required_fields(good_fx_data, "EURUSD")
        assert len(issues) == 0
    
    def test_check_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        df = pd.DataFrame({
            "timestamp": ["2025-01-01", "2025-01-01", "2025-01-02"],  # Duplicate
            "PX_LAST": [1.10, 1.11, 1.12],
        })
        
        issues = check_duplicate_timestamps(df, "timestamp")
        assert len(issues) > 0
    
    def test_check_price_outliers(self, bad_fx_data_outliers):
        """Test detection of price outliers."""
        issues = check_price_outliers(bad_fx_data_outliers, "EURUSD")
        
        # Should detect the 80% price jump
        assert len(issues) > 0
        outlier_issues = [i for i in issues if "outlier" in i.check_name.lower() or "jump" in i.check_name.lower()]
        assert len(outlier_issues) > 0
    
    def test_check_negative_prices(self):
        """Test detection of negative prices."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=10),
            "PX_LAST": [1.10, 1.11, -0.5, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18],
            "PX_HIGH": [1.11] * 10,
            "PX_LOW": [1.09] * 10,
        })
        
        issues = check_negative_prices(df, "EURUSD")
        assert len(issues) > 0
    
    def test_check_data_freshness_stale(self):
        """Test detection of stale data."""
        old_date = datetime.now() - timedelta(days=5)
        df = pd.DataFrame({
            "timestamp": pd.date_range(end=old_date, periods=10),
            "PX_LAST": [1.10] * 10,
        })
        
        issues = check_data_freshness(df, "EURUSD", as_of_date=datetime.now())
        assert len(issues) > 0


# =============================================================================
# Bias Check Tests
# =============================================================================

class TestBiasChecks:
    """Tests for forward-looking bias prevention."""
    
    def test_check_no_future_data_clean(self, good_fx_data):
        """Test that historical data passes future data check."""
        as_of_date = datetime.now()
        issues = check_no_future_data(good_fx_data, as_of_date, "timestamp")
        assert len(issues) == 0
    
    def test_check_no_future_data_fails(self, bad_fx_data_future):
        """Test that future data is detected."""
        # Use a date 5 days ago as "now"
        as_of_date = datetime.now() - timedelta(days=5)
        issues = check_no_future_data(bad_fx_data_future, as_of_date, "timestamp")
        
        # Should detect future data
        assert len(issues) > 0
        assert issues[0].check_name == "forward_looking_bias"
        assert issues[0].severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
    
    def test_create_point_in_time_snapshot(self, bad_fx_data_future):
        """Test creating point-in-time snapshot."""
        as_of_date = datetime.now() - timedelta(days=5)
        
        original_len = len(bad_fx_data_future)
        snapshot = create_point_in_time_snapshot(bad_fx_data_future, as_of_date, "timestamp")
        
        # Snapshot should have fewer rows (no future data)
        assert len(snapshot) < original_len
        
        # All timestamps should be <= as_of_date
        max_ts = pd.to_datetime(snapshot["timestamp"]).max()
        assert max_ts <= as_of_date
    
    def test_run_all_bias_checks(self, good_fx_data):
        """Test running all bias checks."""
        as_of_date = datetime.now()
        issues = run_all_bias_checks(good_fx_data, "EURUSD", as_of_date, "timestamp")
        
        # Good data should have no critical bias issues
        critical = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        assert len(critical) == 0


# =============================================================================
# Quarantine Tests
# =============================================================================

class TestQuarantine:
    """Tests for quarantine functionality."""
    
    def test_quarantine_manager_init(self, temp_quarantine_dir):
        """Test quarantine manager initialization."""
        manager = QuarantineManager(temp_quarantine_dir)
        assert manager.quarantine_dir.exists()
    
    def test_quarantine_dataframe(self, bad_fx_data_missing, temp_quarantine_dir):
        """Test quarantining a dataframe."""
        manager = QuarantineManager(temp_quarantine_dir)
        
        # Create a validation result
        result = validate_dataframe(bad_fx_data_missing, "EURUSD")
        
        # Quarantine the data
        quarantine_path = manager.quarantine_dataframe(
            bad_fx_data_missing,
            "EURUSD",
            result
        )
        
        assert quarantine_path.exists()
        assert (quarantine_path / "data.csv").exists()
        assert (quarantine_path / "metadata.json").exists()
        assert (quarantine_path / "validation_report.json").exists()
    
    def test_list_quarantined(self, bad_fx_data_missing, temp_quarantine_dir):
        """Test listing quarantined data."""
        manager = QuarantineManager(temp_quarantine_dir)
        result = validate_dataframe(bad_fx_data_missing, "EURUSD")
        
        # Quarantine some data
        manager.quarantine_dataframe(bad_fx_data_missing, "EURUSD", result)
        
        # List quarantined
        quarantined = manager.list_quarantined()
        assert len(quarantined) == 1
        assert quarantined[0]["ticker"] == "EURUSD"
    
    def test_quarantine_rows(self, bad_fx_data_missing, temp_quarantine_dir):
        """Test quarantining specific rows."""
        manager = QuarantineManager(temp_quarantine_dir)
        
        original_len = len(bad_fx_data_missing)
        
        # Quarantine rows 10-15 (the ones with missing values)
        clean_df = manager.quarantine_rows(
            bad_fx_data_missing,
            row_indices=list(range(10, 16)),
            ticker="EURUSD",
            reason="Missing values"
        )
        
        assert len(clean_df) == original_len - 6


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full validation pipeline."""
    
    def test_full_validation_pipeline_good_data(self, good_fx_data):
        """Test full validation pipeline with good data."""
        result = validate_dataframe(
            good_fx_data,
            ticker="EURUSD",
            as_of_date=datetime.now()
        )
        
        assert result.passed
        assert result.ticker == "EURUSD"
        assert result.stats["row_count"] == len(good_fx_data)
    
    def test_full_validation_pipeline_bad_data(self, bad_fx_data_missing):
        """Test full validation pipeline with bad data."""
        result = validate_dataframe(
            bad_fx_data_missing,
            ticker="EURUSD",
            as_of_date=datetime.now()
        )
        
        # Should detect issues
        assert len(result.issues) > 0
        
        # Result should be serializable
        json_str = result.to_json()
        assert "EURUSD" in json_str
    
    def test_validation_result_serialization(self, good_fx_data):
        """Test that validation results can be serialized."""
        result = validate_dataframe(good_fx_data, "EURUSD")
        
        # Test to_dict
        result_dict = result.to_dict()
        assert "ticker" in result_dict
        assert "status" in result_dict
        assert "issues" in result_dict
        assert "stats" in result_dict
        
        # Test to_json
        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
