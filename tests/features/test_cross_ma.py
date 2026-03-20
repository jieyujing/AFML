"""
Tests for cross moving average features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.cross_ma import (
    cross_ma_ratio,
    cross_ma_signal,
    cross_ma_both,
    cross_ma_distance,
    CrossMARatioTransform,
    CrossMASignalTransform,
    CrossMAsTransform,
    cross_ma
)


class TestCrossMARatio:
    """Tests for cross MA ratio core function."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        prices = np.random.randn(n) + 100

        result = cross_ma_ratio(prices, fast_window=10, slow_window=20)

        assert result.shape == (n,)

    def test_first_slow_window_minus_one_are_nan(self):
        """First slow_window-1 values should be NaN."""
        n = 100
        fast_window = 10
        slow_window = 20
        prices = np.random.randn(n) + 100

        result = cross_ma_ratio(prices, fast_window, slow_window)

        assert np.all(np.isnan(result[:slow_window - 1]))

    def test_uptrend_positive_ratio(self):
        """Uptrending prices should have positive ratio."""
        n = 100
        prices = 100 + np.arange(n) * 0.5  # Linear uptrend

        result = cross_ma_ratio(prices, fast_window=5, slow_window=20)

        # For uptrend, fast_ma > slow_ma, so ratio > 0
        valid = result[19:]  # After warmup
        assert np.all(valid > 0), "Uptrend should have positive cross MA ratio"

    def test_downtrend_negative_ratio(self):
        """Downtrending prices should have negative ratio."""
        n = 100
        prices = 200 - np.arange(n) * 0.5  # Linear downtrend

        result = cross_ma_ratio(prices, fast_window=5, slow_window=20)

        # For downtrend, fast_ma < slow_ma, so ratio < 0
        valid = result[19:]
        assert np.all(valid < 0), "Downtrend should have negative cross MA ratio"

    def test_constant_prices_zero_ratio(self):
        """Constant prices should have zero ratio."""
        n = 100
        prices = np.ones(n) * 100

        result = cross_ma_ratio(prices, fast_window=5, slow_window=20)

        valid = result[19:]
        assert_array_almost_equal(valid, np.zeros_like(valid), decimal=10)

    def test_fast_window_greater_than_slow_raises(self):
        """fast_window >= slow_window should raise ValueError."""
        prices = np.random.randn(100) + 100

        with pytest.raises(ValueError, match="must be less than"):
            cross_ma_ratio(prices, fast_window=20, slow_window=10)

    def test_insufficient_data(self):
        """Data shorter than slow_window should return all NaN."""
        prices = np.random.randn(10) + 100

        result = cross_ma_ratio(prices, fast_window=3, slow_window=20)

        assert np.all(np.isnan(result))

    def test_mathematical_correctness(self):
        """Verify ratio formula: fast_ma / slow_ma - 1."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        result = cross_ma_ratio(prices, fast_window=2, slow_window=3)

        # At index 2 (slow_window - 1), we can verify manually:
        # fast_ma = (102 + 101) / 2 = 101.5
        # slow_ma = (102 + 101 + 100) / 3 = 101.0
        # ratio = 101.5 / 101.0 - 1 ≈ 0.00495

        expected_ratio = 101.5 / 101.0 - 1
        assert abs(result[2] - expected_ratio) < 1e-6


class TestCrossMASignal:
    """Tests for cross MA signal core function."""

    def test_uptrend_positive_signal(self):
        """Uptrending prices should have +1 signal."""
        n = 100
        prices = 100 + np.arange(n) * 0.5

        result = cross_ma_signal(prices, fast_window=5, slow_window=20)

        valid = result[19:]
        assert np.all(valid == 1.0), "Uptrend should have +1 signal"

    def test_downtrend_negative_signal(self):
        """Downtrending prices should have -1 signal."""
        n = 100
        prices = 200 - np.arange(n) * 0.5

        result = cross_ma_signal(prices, fast_window=5, slow_window=20)

        valid = result[19:]
        assert np.all(valid == -1.0), "Downtrend should have -1 signal"

    def test_constant_prices_zero_signal(self):
        """Constant prices should have 0 signal."""
        n = 100
        prices = np.ones(n) * 100

        result = cross_ma_signal(prices, fast_window=5, slow_window=20)

        valid = result[19:]
        assert np.all(valid == 0.0), "Constant prices should have 0 signal"

    def test_signal_values_in_valid_set(self):
        """Signal values should only be -1, 0, or 1."""
        np.random.seed(42)
        n = 200
        prices = 100 + np.random.randn(n).cumsum()

        result = cross_ma_signal(prices, fast_window=10, slow_window=50)

        valid = result[~np.isnan(result)]
        unique_values = set(valid)
        assert unique_values.issubset({-1.0, 0.0, 1.0})


class TestCrossMABoth:
    """Tests for cross_ma_both function."""

    def test_returns_two_arrays(self):
        """Should return tuple of two arrays."""
        prices = np.random.randn(100) + 100

        result = cross_ma_both(prices, fast_window=10, slow_window=20)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fast_ma_available_earlier(self):
        """Fast MA should have values before slow_window."""
        n = 50
        fast_window = 10
        slow_window = 20
        prices = np.random.randn(n) + 100

        fast_ma, slow_ma = cross_ma_both(prices, fast_window, slow_window)

        # Fast MA should have values from index fast_window - 1
        assert not np.isnan(fast_ma[fast_window - 1])
        # Slow MA should still be NaN before slow_window - 1
        assert np.isnan(slow_ma[slow_window - 2])

    def test_ma_values_correct(self):
        """Verify MA calculations are correct."""
        prices = np.array([100, 101, 102, 103, 104, 105])

        fast_ma, slow_ma = cross_ma_both(prices, fast_window=2, slow_window=3)

        # Fast MA at index 1: (101 + 100) / 2 = 100.5
        assert abs(fast_ma[1] - 100.5) < 1e-6

        # Slow MA at index 2: (102 + 101 + 100) / 3 = 101
        assert abs(slow_ma[2] - 101.0) < 1e-6


class TestCrossMADistance:
    """Tests for cross MA distance function."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        prices = np.random.randn(100) + 100

        result = cross_ma_distance(prices, fast_window=10, slow_window=20)

        assert result.shape == (100,)

    def test_normalized_distance(self):
        """Normalized distance should be percentage of price."""
        prices = np.arange(100, 200) * 1.0  # Uptrend

        result = cross_ma_distance(prices, fast_window=5, slow_window=20, normalize=True)

        # For normalized, result should be small (< 1 typically)
        valid = result[19:]
        assert np.all(np.abs(valid) < 1.0)


class TestCrossMARatioTransform:
    """Tests for CrossMARatioTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        prices = 100 + np.random.randn(n).cumsum()
        return pd.DataFrame({'close': prices}, index=dates)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = CrossMARatioTransform('close', fast_window=10, slow_window=50)

        assert transform.output_name == 'close_cross_ma_10_50'

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = CrossMARatioTransform('close', fast_window=10, slow_window=50)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_pandas_backend(self, sample_data):
        """Test pandas backend produces same results."""
        transform = CrossMARatioTransform('close', fast_window=10, slow_window=50)

        result_nb = transform(sample_data, backend='nb')
        result_pd = transform(sample_data, backend='pd')

        valid_mask = ~(result_nb.isna() | result_pd.isna())
        assert_array_almost_equal(
            result_nb[valid_mask].values,
            result_pd[valid_mask].values,
            decimal=10
        )

    def test_index_preserved(self, sample_data):
        """Output index should match input index."""
        transform = CrossMARatioTransform('close', fast_window=10, slow_window=50)

        result = transform(sample_data, backend='nb')

        assert result.index.equals(sample_data.index)

    def test_invalid_column_raises(self, sample_data):
        """Invalid column should raise ValueError."""
        transform = CrossMARatioTransform('nonexistent', fast_window=10, slow_window=50)

        with pytest.raises(ValueError, match="not found"):
            transform(sample_data, backend='nb')

    def test_fast_greater_than_slow_raises(self):
        """fast_window >= slow_window should raise in constructor."""
        with pytest.raises(ValueError, match="must be less than"):
            CrossMARatioTransform('close', fast_window=50, slow_window=10)


class TestCrossMASignalTransform:
    """Tests for CrossMASignalTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        prices = 100 + np.random.randn(n).cumsum()
        return pd.DataFrame({'close': prices}, index=dates)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = CrossMASignalTransform('close', fast_window=10, slow_window=50)

        assert 'signal' in transform.output_name

    def test_signal_values_valid(self, sample_data):
        """Signal values should be in {-1, 0, 1}."""
        transform = CrossMASignalTransform('close', fast_window=10, slow_window=50)

        result = transform(sample_data, backend='nb')

        valid = result.dropna()
        unique = set(valid.unique())
        assert unique.issubset({-1.0, 0.0, 1.0})


class TestCrossMAsTransform:
    """Tests for CrossMAsTransform (SIMO) class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        prices = 100 + np.random.randn(n).cumsum()
        return pd.DataFrame({'close': prices}, index=dates)

    def test_returns_tuple(self, sample_data):
        """Transform should return tuple of Series."""
        transform = CrossMAsTransform('close', fast_window=10, slow_window=50)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_names(self, sample_data):
        """Test output name generation."""
        transform = CrossMAsTransform('close', fast_window=10, slow_window=50)

        assert transform.output_name == ['close_ma_10', 'close_ma_50']


class TestCrossMAConvenienceFunction:
    """Tests for the convenience function."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price Series."""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        return pd.Series(100 + np.random.randn(n).cumsum(), index=dates, name='price')

    def test_ratio_output(self, sample_prices):
        """Test ratio output type."""
        result = cross_ma(sample_prices, fast_window=10, slow_window=50, output='ratio')

        assert isinstance(result, pd.Series)
        assert 'cross_ma' in result.name

    def test_signal_output(self, sample_prices):
        """Test signal output type."""
        result = cross_ma(sample_prices, fast_window=10, slow_window=50, output='signal')

        assert isinstance(result, pd.Series)
        assert 'signal' in result.name

    def test_both_output(self, sample_prices):
        """Test both output returns DataFrame."""
        result = cross_ma(sample_prices, fast_window=10, slow_window=50, output='both')

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2

    def test_invalid_output_raises(self, sample_prices):
        """Invalid output type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown output"):
            cross_ma(sample_prices, fast_window=10, slow_window=50, output='invalid')


class TestPerformance:
    """Performance comparison tests."""

    @pytest.mark.slow
    @pytest.mark.skip(reason="Performance test, run manually for benchmarking")
    def test_performance_vs_pandas_composition(self):
        """Compare performance with pandas feature composition."""
        import time

        n = 100_000
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(n).cumsum())

        # Numba version
        start = time.time()
        result_nb = cross_ma_ratio(prices.values, fast_window=10, slow_window=50)
        nb_time = time.time() - start

        # Pandas composition version
        start = time.time()
        fast_ma = prices.rolling(10).mean()
        slow_ma = prices.rolling(50).mean()
        result_pd = fast_ma / slow_ma - 1
        pd_time = time.time() - start

        # Results should be similar (use lower precision due to floating point differences)
        valid_mask = ~(np.isnan(result_nb) | result_pd.isna())
        assert_array_almost_equal(
            result_nb[valid_mask],
            result_pd[valid_mask].values,
            decimal=6
        )

        # Numba should be faster (print but don't assert for CI stability)
        print(f"\nNumba: {nb_time:.4f}s, Pandas: {pd_time:.4f}s, Speedup: {pd_time/nb_time:.1f}x")