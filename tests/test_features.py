"""
Unit tests for FeatureEngineer.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from afml import FeatureEngineer


@pytest.fixture
def sample_dollar_bars():
    """Create sample dollar bar data for testing."""
    np.random.seed(42)
    n_bars = 200

    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i * 4) for i in range(n_bars)]

    base_price = 4000.0
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price * np.cumprod(1 + returns)

    df = pl.DataFrame(
        {
            "datetime": times,
            "open": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, size=n_bars),
        }
    )

    return df


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        engineer = FeatureEngineer()
        assert engineer.windows == [5, 10, 20, 30, 50]
        assert engineer.ffd_check_stationarity is True
        assert engineer.ffd_d == 0.4

    def test_initialization_custom(self):
        """Test custom initialization."""
        engineer = FeatureEngineer(
            windows=[10, 20, 30], ffd_check_stationarity=False, ffd_d=0.5
        )
        assert engineer.windows == [10, 20, 30]
        assert engineer.ffd_check_stationarity is False
        assert engineer.ffd_d == 0.5

    def test_fit(self, sample_dollar_bars):
        """Test fit method."""
        engineer = FeatureEngineer()
        result = engineer.fit(sample_dollar_bars)

        assert result is engineer
        assert engineer.selected_features_ is not None or engineer._metadata is not None

    def test_transform(self, sample_dollar_bars):
        """Test transform method."""
        engineer = FeatureEngineer(windows=[5, 10, 20])
        engineer.fit(sample_dollar_bars)
        features = engineer.transform(sample_dollar_bars)

        assert isinstance(features, pl.DataFrame)
        assert len(features) == len(sample_dollar_bars)
        assert len(features) > 0
        assert len(features.columns) > 0

    def test_fit_transform(self, sample_dollar_bars):
        """Test fit_transform method."""
        engineer = FeatureEngineer(windows=[5, 10, 20])
        features = engineer.fit_transform(sample_dollar_bars)

        assert isinstance(features, pl.DataFrame)
        assert len(features) > 0
        assert len(features.columns) > 0

    def test_feature_names(self, sample_dollar_bars):
        """Test get_feature_names method."""
        engineer = FeatureEngineer()
        engineer.fit(sample_dollar_bars)
        names = engineer.get_feature_names()

        assert isinstance(names, list)
        assert len(names) > 0

    def test_sklearn_compatible(self, sample_dollar_bars):
        """Test sklearn compatibility."""
        engineer = FeatureEngineer(windows=[5, 10])

        fitted = engineer.fit(sample_dollar_bars)
        assert fitted is engineer

        features = engineer.transform(sample_dollar_bars)
        assert isinstance(features, pl.DataFrame)

    def test_d_values_stored(self, sample_dollar_bars):
        """Test that d_values_ is stored after transform."""
        engineer = FeatureEngineer(windows=[5, 10], ffd_check_stationarity=True)
        engineer.fit_transform(sample_dollar_bars)

        assert hasattr(engineer, "d_values_")
        assert engineer.d_values_ is not None


class TestFeatureEngineerEdgeCases:
    """Edge case tests for FeatureEngineer."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        engineer = FeatureEngineer()
        df = pl.DataFrame(
            schema={
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

        result = engineer.fit_transform(df)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_small_dataset(self):
        """Test with small dataset."""
        np.random.seed(42)
        n_bars = 10

        start_time = datetime(2023, 1, 1)
        times = [start_time + timedelta(hours=i) for i in range(n_bars)]

        prices = 4000.0 + np.cumsum(np.random.randn(n_bars))

        df = pl.DataFrame(
            {
                "datetime": times,
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.random.randint(1000, 10000, size=n_bars),
            }
        )

        engineer = FeatureEngineer(windows=[3, 5])
        features = engineer.fit_transform(df)

        assert isinstance(features, pl.DataFrame)

    def test_missing_columns(self):
        """Test with missing required columns."""
        engineer = FeatureEngineer()
        df = pl.DataFrame(
            {
                "open": [4000.0],
                "close": [4001.0],
            }
        )

        with pytest.raises(Exception):
            engineer.fit_transform(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
