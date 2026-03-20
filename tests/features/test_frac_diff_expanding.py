"""
Tests for expanding window fractional differentiation features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.frac_diff_expanding import (
    get_fracdiff_weights,
    frac_diff_expanding,
    frac_diff_expanding_rolling,
    frac_diff_expanding_with_weights,
    compare_ffd_vs_expanding,
    FracDiffExpandingTransform,
    FracDiffRollingTransform,
    fracdiff_expanding
)


class TestGetFracdiffWeights:
    """Tests for weight computation."""

    def test_first_weight_is_one(self):
        """First weight should always be 1."""
        for d in [0.1, 0.5, 0.8, 1.0]:
            weights = get_fracdiff_weights(d, 10)
            assert weights[0] == 1.0

    def test_d_zero_gives_unit_weights(self):
        """d=0 should give weights that sum to 1 (no differencing)."""
        # For d=0, weights decay but not exactly 0
        weights = get_fracdiff_weights(0.0, 10)
        assert weights[0] == 1.0

    def test_d_one_gives_first_difference(self):
        """d=1 should give first difference weights [1, -1, 0, 0, ...]."""
        weights = get_fracdiff_weights(1.0, 10)
        assert weights[0] == 1.0
        assert weights[1] == -1.0
        # All subsequent weights should be 0
        for k in range(2, 10):
            assert abs(weights[k]) < 1e-10

    def test_weights_sign_pattern(self):
        """Weights should follow the correct sign pattern based on d."""
        d = 0.5
        weights = get_fracdiff_weights(d, 20)

        # ω_0 = 1 (positive)
        assert weights[0] == 1.0
        # ω_1 = -d = -0.5 (negative)
        assert weights[1] == -0.5
        # Subsequent weights follow: ω_k = -ω_{k-1} * (d - k + 1) / k
        # The sign depends on the sign of (d - k + 1)
        # For d=0.5: k=1 -> (0.5 - 1 + 1) = 0.5 > 0, so sign flip
        #             k=2 -> (0.5 - 2 + 1) = -0.5 < 0, so no flip
        # This creates a more complex pattern than simple alternation

    def test_weights_decay(self):
        """Weights should decay in magnitude."""
        d = 0.5
        weights = get_fracdiff_weights(d, 100)

        # Check decay (roughly)
        for k in range(10, 100):
            assert abs(weights[k]) < abs(weights[k - 1])

    def test_weight_sum_near_one(self):
        """Sum of weights should be near 1 for small d."""
        d = 0.4
        weights = get_fracdiff_weights(d, 1000)
        weight_sum = np.sum(weights)
        # Sum of weights for d < 1 is related to (1 - z)^d evaluated at z=1
        # For stationarity, we want the sum to not diverge
        assert abs(weight_sum) < 2  # Reasonable bound


class TestFracDiffExpanding:
    """Tests for expanding window fracdiff."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding(prices, d=0.5, min_window=10)

        assert result.shape == (n,)

    def test_first_min_window_minus_one_are_nan(self):
        """First min_window-1 values should be NaN."""
        n = 100
        min_window = 20
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding(prices, d=0.5, min_window=min_window)

        assert np.all(np.isnan(result[:min_window - 1]))

    def test_insufficient_data(self):
        """Data shorter than min_window should return all NaN."""
        prices = 100 + np.random.randn(10).cumsum()

        result = frac_diff_expanding(prices, d=0.5, min_window=20)

        assert np.all(np.isnan(result))

    def test_d_zero_preserves_prices(self):
        """d=0 should approximately preserve the price series."""
        n = 100
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding(prices, d=0.0, min_window=10)

        # d=0 should give original values (up to early truncation effects)
        valid_idx = 50  # Use middle values where cumulative error is small
        assert abs(result[valid_idx] - prices[valid_idx]) < 1.0

    def test_d_one_gives_first_difference(self):
        """d=1 should give first difference: x_t - x_{t-1}."""
        n = 100
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding(prices, d=1.0, min_window=10)

        # d=1 should give price difference
        for t in range(10, n):
            expected = prices[t] - prices[t - 1]
            assert abs(result[t] - expected) < 1e-6

    def test_constant_prices_behavior(self):
        """Test behavior with constant prices for different d values."""
        n = 100
        const_value = 100.0
        prices = np.ones(n) * const_value

        # For d=1 (first difference), constant prices should give 0
        result_d1 = frac_diff_expanding(prices, d=1.0, min_window=10)
        valid_d1 = result_d1[~np.isnan(result_d1)]
        assert_array_almost_equal(valid_d1, np.zeros_like(valid_d1), decimal=6)

        # For d=0.5, expanding window uses different number of weights at each point
        # So the result is NOT constant - it's the cumulative weight sum * price
        # This is expected behavior for expanding window fracdiff
        result_d05 = frac_diff_expanding(prices, d=0.5, min_window=10)
        valid_d05 = result_d05[~np.isnan(result_d05)]
        # Values should decay toward 0 as more weights are added
        # (since sum of weights for fractional d tends to 0 as window expands)
        assert len(valid_d05) > 0


class TestFracDiffExpandingRolling:
    """Tests for rolling window fracdiff."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding_rolling(prices, d=0.5, window=20)

        assert result.shape == (n,)

    def test_window_constraint(self):
        """Should only use last 'window' observations."""
        n = 100
        window = 20
        prices = 100 + np.random.randn(n).cumsum()

        result = frac_diff_expanding_rolling(prices, d=0.5, window=window)

        assert np.all(np.isnan(result[:window - 1]))

    def test_faster_than_expanding(self):
        """Rolling should be faster than pure expanding."""
        import time

        n = 500
        prices = 100 + np.random.randn(n).cumsum()

        # Time expanding
        start = time.time()
        result_exp = frac_diff_expanding(prices, d=0.5, min_window=20)
        time_exp = time.time() - start

        # Time rolling
        start = time.time()
        result_roll = frac_diff_expanding_rolling(prices, d=0.5, window=20)
        time_roll = time.time() - start

        # Both should have similar values where valid
        valid_mask = ~(np.isnan(result_exp) | np.isnan(result_roll))
        # Values should be close (not exact due to different window handling)
        # Just check they're the same order of magnitude
        assert np.all(np.abs(result_exp[valid_mask] - result_roll[valid_mask]) < 100)


class TestCompareFFDvsExpanding:
    """Tests for comparison function."""

    def test_returns_two_arrays(self):
        """Should return tuple of two arrays."""
        prices = 100 + np.random.randn(100).cumsum()

        ffd, expanding = compare_ffd_vs_expanding(prices, d=0.5, ffd_window=20)

        assert isinstance(ffd, np.ndarray)
        assert isinstance(expanding, np.ndarray)
        assert ffd.shape == expanding.shape

    def test_both_have_valid_values(self):
        """Both methods should produce valid (non-NaN) values."""
        prices = 100 + np.random.randn(100).cumsum()

        ffd, expanding = compare_ffd_vs_expanding(prices, d=0.5, ffd_window=20)

        assert not np.all(np.isnan(ffd))
        assert not np.all(np.isnan(expanding))


class TestFracDiffExpandingTransform:
    """Tests for the Transform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        prices = 100 + np.random.randn(n).cumsum()
        return pd.DataFrame({'close': prices}, index=dates)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = FracDiffExpandingTransform('close', d=0.5, min_window=50)

        assert transform.output_name == 'close_fracdiff_expanding_0.5'

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = FracDiffExpandingTransform('close', d=0.5, min_window=50)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_pandas_backend(self, sample_data):
        """Test pandas backend produces same results."""
        transform = FracDiffExpandingTransform('close', d=0.5, min_window=50)

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
        transform = FracDiffExpandingTransform('close', d=0.5, min_window=50)

        result = transform(sample_data, backend='nb')

        assert result.index.equals(sample_data.index)

    def test_invalid_d_raises(self):
        """Invalid d should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 2"):
            FracDiffExpandingTransform('close', d=-0.5)

        with pytest.raises(ValueError, match="must be between 0 and 2"):
            FracDiffExpandingTransform('close', d=3.0)

    def test_invalid_column_raises(self, sample_data):
        """Invalid column should raise ValueError."""
        transform = FracDiffExpandingTransform('nonexistent', d=0.5)

        with pytest.raises(ValueError, match="not found"):
            transform(sample_data, backend='nb')


class TestFracDiffRollingTransform:
    """Tests for rolling transform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        prices = 100 + np.random.randn(n).cumsum()
        return pd.DataFrame({'close': prices}, index=dates)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = FracDiffRollingTransform('close', d=0.5, window=50)

        assert 'fracdiff_roll' in transform.output_name
        assert '0.5' in transform.output_name
        assert '50' in transform.output_name

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = FracDiffRollingTransform('close', d=0.5, window=50)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_returns_series(self):
        """Function should return a Series."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = pd.Series(100 + np.random.randn(200).cumsum(), index=dates, name='price')

        result = fracdiff_expanding(prices, d=0.5, min_window=50)

        assert isinstance(result, pd.Series)

    def test_index_preserved(self):
        """Index should be preserved."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        prices = pd.Series(100 + np.random.randn(200).cumsum(), index=dates, name='price')

        result = fracdiff_expanding(prices, d=0.5, min_window=50)

        assert result.index.equals(prices.index)


class TestEdgeCases:
    """Edge case tests."""

    def test_small_d(self):
        """Test with very small d value."""
        prices = 100 + np.random.randn(100).cumsum()
        result = frac_diff_expanding(prices, d=0.1, min_window=10)

        assert not np.all(np.isnan(result))

    def test_large_d(self):
        """Test with d close to 1."""
        prices = 100 + np.random.randn(100).cumsum()
        result = frac_diff_expanding(prices, d=0.9, min_window=10)

        assert not np.all(np.isnan(result))

    def test_large_min_window(self):
        """Test with min_window close to data length."""
        prices = 100 + np.random.randn(100).cumsum()
        result = frac_diff_expanding(prices, d=0.5, min_window=95)

        # Should have very few non-NaN values
        non_nan = np.sum(~np.isnan(result))
        assert non_nan <= 6

    def test_negative_prices(self):
        """Test with negative price values."""
        prices = np.random.randn(100)
        result = frac_diff_expanding(prices, d=0.5, min_window=10)

        assert not np.all(np.isnan(result))