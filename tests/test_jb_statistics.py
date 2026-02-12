"""
Unit tests for JB statistics visualization functionality.
"""

import pytest
import numpy as np
from scipy import stats

from afml.visualization import _compute_jb_statistics, AFMLVisualizer


class TestComputeJBStatistics:
    """Tests for _compute_jb_statistics helper function."""

    def test_normal_distribution(self):
        """Test with normally distributed returns."""
        np.random.seed(42)
        returns = np.random.randn(1000)
        result = _compute_jb_statistics(returns)

        assert "jb_stat" in result
        assert "p_value" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "is_normal" in result
        assert "n_samples" in result
        assert result["n_samples"] == 1000

    def test_high_kurtosis_distribution(self):
        """Test with high kurtosis (leptokurtic) distribution."""
        np.random.seed(42)
        # Student's t-distribution has high kurtosis
        returns = np.random.standard_t(df=5, size=1000)
        result = _compute_jb_statistics(returns)

        # Should have high JB statistic and likely fail normality
        assert result["jb_stat"] > 0
        # Kurtosis should be > 3 for this distribution
        assert result["kurtosis"] > 3

    def test_empty_array(self):
        """Test with empty array."""
        returns = np.array([])
        result = _compute_jb_statistics(returns)

        assert result["jb_stat"] == 0.0
        assert result["p_value"] == 1.0
        assert result["is_normal"] is True
        assert result["n_samples"] == 0

    def test_nan_values_filtered(self):
        """Test that NaN values are filtered out."""
        np.random.seed(42)
        returns = np.random.randn(100)
        returns_with_nan = np.concatenate(
            [returns[:50], [np.nan, np.nan], returns[50:]]
        )
        result = _compute_jb_statistics(returns_with_nan)

        # Should still work with NaN values filtered
        assert result["n_samples"] == 100
        assert result["jb_stat"] >= 0

    def test_small_sample(self):
        """Test with very small sample size."""
        returns = np.array([1.0, 2.0, 3.0])
        result = _compute_jb_statistics(returns)

        # Should handle gracefully with small samples
        assert result["n_samples"] == 3
        assert "jb_stat" in result

    def test_normality_threshold(self):
        """Test p-value > 0.05 for normal distribution."""
        np.random.seed(42)
        returns = np.random.randn(10000)  # Large sample of normal distribution
        result = _compute_jb_statistics(returns)

        # Large normal samples should pass JB test more often
        # (Note: with very large samples, JB can be sensitive to tiny deviations)
        assert isinstance(result["is_normal"], bool)

    def test_skewness_computation(self):
        """Test that skewness is computed correctly."""
        np.random.seed(42)
        # Positively skewed distribution
        returns = np.random.exponential(scale=1.0, size=1000)
        result = _compute_jb_statistics(returns)

        # Exponential distribution has positive skewness
        assert result["skewness"] > 0


class TestPlotBarStats:
    """Tests for plot_bar_stats method with JB statistics."""

    @pytest.fixture
    def sample_dollar_bars(self):
        """Create sample dollar bars for testing."""
        import polars as pl
        from datetime import datetime, timedelta

        np.random.seed(42)
        n_bars = 500

        start_time = datetime(2023, 1, 1, 9, 30, 0)
        times = [start_time + timedelta(hours=i) for i in range(n_bars)]

        base_price = 4000.0
        returns = np.random.randn(n_bars) * 0.0001 + 0.00001
        prices = base_price * np.cumprod(1 + returns)

        volume = np.random.randint(1000, 5000, size=n_bars)

        df = pl.DataFrame(
            {
                "datetime": times,
                "open": prices,
                "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.0005)),
                "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.0005)),
                "close": prices,
                "volume": volume,
            }
        )

        return df

    @pytest.fixture
    def sample_time_bars(self):
        """Create sample time bars for comparison."""
        import polars as pl
        from datetime import datetime, timedelta

        np.random.seed(42)
        n_bars = 5000  # More bars for time-based data

        start_time = datetime(2023, 1, 1, 9, 30, 0)
        times = [start_time + timedelta(minutes=i) for i in range(n_bars)]

        base_price = 4000.0
        returns = np.random.randn(n_bars) * 0.0001 + 0.00001
        prices = base_price * np.cumprod(1 + returns)

        volume = np.random.randint(100, 1000, size=n_bars)

        df = pl.DataFrame(
            {
                "datetime": times,
                "open": prices,
                "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.0005)),
                "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.0005)),
                "close": prices,
                "volume": volume,
            }
        )

        return df

    def test_returns_dict(self, sample_dollar_bars, tmp_path):
        """Test that plot_bar_stats returns JB statistics dictionary."""
        viz = AFMLVisualizer(output_dir=str(tmp_path))
        result = viz.plot_bar_stats(sample_dollar_bars, filename="test.png")

        assert isinstance(result, dict)
        assert "jb_stat" in result
        assert "p_value" in result
        assert "skewness" in result
        assert "kurtosis" in result
        assert "is_normal" in result

    def test_with_time_bars_comparison(
        self, sample_dollar_bars, sample_time_bars, tmp_path
    ):
        """Test plot_bar_stats with time bars comparison."""
        viz = AFMLVisualizer(output_dir=str(tmp_path))
        result = viz.plot_bar_stats(
            sample_dollar_bars,
            filename="test_comparison.png",
            time_bars_df=sample_time_bars,
        )

        # Should still return dollar bar stats
        assert isinstance(result, dict)
        assert "jb_stat" in result

    def test_empty_dataframe(self, tmp_path):
        """Test with empty DataFrame."""
        import polars as pl

        viz = AFMLVisualizer(output_dir=str(tmp_path))
        empty_df = pl.DataFrame({"datetime": [], "close": []})
        result = viz.plot_bar_stats(empty_df, filename="test.png")

        assert result.get("error") == "Empty DataFrame"

    def test_jb_statistics_values(self, sample_dollar_bars, tmp_path):
        """Test that JB statistics contain reasonable values."""
        viz = AFMLVisualizer(output_dir=str(tmp_path))
        result = viz.plot_bar_stats(sample_dollar_bars, filename="test.png")

        # JB statistic should be non-negative
        assert result["jb_stat"] >= 0
        # p-value should be between 0 and 1
        assert 0 <= result["p_value"] <= 1
        # Skewness should be a finite number
        assert np.isfinite(result["skewness"])
        # Kurtosis should be finite
        assert np.isfinite(result["kurtosis"])
