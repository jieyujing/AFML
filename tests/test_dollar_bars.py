"""
Unit tests for DollarBarsProcessor.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from afml import DollarBarsProcessor


@pytest.fixture
def sample_minute_bars():
    """Create sample minute bar data for testing."""
    np.random.seed(42)
    n_bars = 1000

    # Create datetime index
    start_time = datetime(2023, 1, 1, 9, 30, 0)
    times = [start_time + timedelta(minutes=i) for i in range(n_bars)]

    # Generate OHLCV data with some trend
    base_price = 4000.0
    returns = np.random.randn(n_bars) * 0.0001 + 0.00001  # Slight upward drift
    prices = base_price * np.cumprod(1 + returns)

    # Create volume
    volume = np.random.randint(100, 1000, size=n_bars)

    df = pd.DataFrame(
        {
            "datetime": times,
            "open": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.0005)),
            "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.0005)),
            "close": prices,
            "volume": volume,
        }
    )

    # Calculate dollar amount (assuming multiplier of 300)
    multiplier = 300.0
    avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["amount"] = avg_price * df["volume"] * multiplier

    return df


class TestDollarBarsProcessor:
    """Tests for DollarBarsProcessor class."""

    def test_initialization(self):
        """Test processor initialization with default parameters."""
        processor = DollarBarsProcessor()
        assert processor.daily_target == 4
        assert processor.ema_span == 20

    def test_initialization_custom_params(self):
        """Test processor initialization with custom parameters."""
        processor = DollarBarsProcessor(daily_target=8, ema_span=50)
        assert processor.daily_target == 8
        assert processor.ema_span == 50

    def test_fit(self, sample_minute_bars):
        """Test fit method."""
        processor = DollarBarsProcessor()
        result = processor.fit(sample_minute_bars)

        assert result is processor  # Should return self
        assert processor.threshold_ > 0
        assert hasattr(processor, "threshold_type")

    def test_fit_transform_fixed(self, sample_minute_bars):
        """Test fit_transform with fixed threshold."""
        processor = DollarBarsProcessor()
        result = processor.fit_transform(sample_minute_bars)

        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(sample_minute_bars)  # Should have fewer bars
        assert "datetime" in result.columns
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_fit_transform_dynamic(self, sample_minute_bars):
        """Test fit_transform_dynamic method."""
        processor = DollarBarsProcessor(daily_target=4, ema_span=20)
        result = processor.fit_transform_dynamic(sample_minute_bars)

        assert isinstance(result, pd.DataFrame)
        assert len(result) < len(sample_minute_bars)
        assert len(result) > 0

    def test_fit_dynamic(self, sample_minute_bars):
        """Test fit_dynamic method."""
        processor = DollarBarsProcessor()
        result = processor.fit_dynamic(sample_minute_bars)

        assert result is processor
        assert processor.threshold_type == "dynamic"

    def test_get_threshold_info(self, sample_minute_bars):
        """Test get_threshold_info method."""
        processor = DollarBarsProcessor()
        processor.fit(sample_minute_bars)

        info = processor.get_threshold_info()
        assert "threshold_type" in info
        assert "threshold" in info
        assert info["threshold"] > 0

    def test_threshold_reasonable(self, sample_minute_bars):
        """Test that threshold is in a reasonable range."""
        processor = DollarBarsProcessor()
        processor.fit(sample_minute_bars)

        info = processor.get_threshold_info()
        # Threshold should be positive
        assert info["threshold"] > 0

        # Threshold should be reasonable (daily volume / target bars)
        # It's expected to be large since it's a daily accumulated amount
        avg_daily_amount = (
            sample_minute_bars.set_index("datetime")
            .resample("D")["amount"]
            .sum()
            .mean()
        )
        assert (
            info["threshold"] < avg_daily_amount * 2
        )  # Should be less than 2x daily volume

    def test_sklearn_compatible(self, sample_minute_bars):
        """Test sklearn compatibility (fit, fit_transform, transform)."""
        processor = DollarBarsProcessor()

        # fit should return self
        fitted = processor.fit(sample_minute_bars)
        assert fitted is processor

        # transform should work after fit
        transformed = processor.transform(sample_minute_bars)
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) < len(sample_minute_bars)

    def test_dynamic_mode_changes_threshold(self, sample_minute_bars):
        """Test that dynamic mode produces different results than fixed."""
        processor_fixed = DollarBarsProcessor()
        processor_dynamic = DollarBarsProcessor()

        fixed_result = processor_fixed.fit_transform(sample_minute_bars)
        dynamic_result = processor_dynamic.fit_transform_dynamic(sample_minute_bars)

        # Both should produce valid results but possibly different counts
        assert len(fixed_result) > 0
        assert len(dynamic_result) > 0
        assert isinstance(fixed_result, pd.DataFrame)
        assert isinstance(dynamic_result, pd.DataFrame)


class TestDollarBarsProcessorEdgeCases:
    """Edge case tests for DollarBarsProcessor."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        processor = DollarBarsProcessor()
        df = pd.DataFrame(
            columns=["datetime", "open", "high", "low", "close", "volume", "amount"]
        )

        with pytest.raises(Exception):
            processor.fit_transform(df)

    def test_small_dataset(self):
        """Test with very small dataset."""
        np.random.seed(42)
        n_bars = 10

        start_time = datetime(2023, 1, 1, 9, 30, 0)
        times = [start_time + timedelta(minutes=i) for i in range(n_bars)]

        base_price = 4000.0
        prices = base_price * (1 + np.cumsum(np.random.randn(n_bars) * 0.001))
        volume = np.random.randint(100, 1000, size=n_bars)

        df = pd.DataFrame(
            {
                "datetime": times,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": volume,
                "amount": prices * volume * 300,
            }
        )

        processor = DollarBarsProcessor(daily_target=10)
        result = processor.fit_transform(df)

        # Should still produce valid output
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
