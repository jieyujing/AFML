"""
Tests for serial correlation (autocorrelation) features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.serial_corr import (
    rolling_serial_correlation,
    serial_correlation_at_lag,
    ljung_box_statistic,
    SerialCorrelationTransform,
    LjungBoxTransform,
    autocorr
)


class TestRollingSerialCorrelation:
    """Tests for the core Numba function."""

    def test_output_shape(self):
        """Output should have shape (n, len(lags))."""
        n = 100
        window = 20
        lags = np.array([1, 5, 10], dtype=np.int64)

        data = np.random.randn(n)
        result = rolling_serial_correlation(data, window, lags)

        assert result.shape == (n, len(lags))

    def test_first_window_minus_one_are_nan(self):
        """First window-1 rows should be NaN."""
        n = 100
        window = 20
        lags = np.array([1, 5], dtype=np.int64)

        data = np.random.randn(n)
        result = rolling_serial_correlation(data, window, lags)

        # All values before window-1 should be NaN
        assert np.all(np.isnan(result[:window - 1, :]))

    def test_random_data_near_zero(self):
        """Random data should have autocorrelation near zero."""
        np.random.seed(42)
        n = 1000
        window = 100
        lags = np.array([1, 5, 10], dtype=np.int64)

        data = np.random.randn(n)
        result = rolling_serial_correlation(data, window, lags)

        # Mean autocorrelation for random data should be close to 0
        valid_results = result[window - 1:, :]
        mean_autocorr = np.nanmean(valid_results, axis=0)

        for ma in mean_autocorr:
            assert abs(ma) < 0.1, f"Mean autocorr {ma} too far from 0"

    def test_ar1_positive_autocorrelation(self):
        """AR(1) process with positive coefficient should have positive autocorr at lag 1."""
        np.random.seed(42)
        n = 500
        phi = 0.8  # AR(1) coefficient
        window = 100
        lags = np.array([1], dtype=np.int64)

        # Generate AR(1) process: x_t = phi * x_{t-1} + epsilon
        data = np.zeros(n)
        for t in range(1, n):
            data[t] = phi * data[t - 1] + np.random.randn() * 0.1

        result = rolling_serial_correlation(data, window, lags)

        # Lag 1 autocorrelation should be positive for AR(1) with phi > 0
        mean_autocorr = np.nanmean(result[window - 1:, 0])
        assert mean_autocorr > 0.5, f"AR(1) autocorr {mean_autocorr} should be > 0.5"

    def test_lag_greater_than_window_returns_nan(self):
        """Lag greater than window should return NaN."""
        n = 100
        window = 10
        lags = np.array([20], dtype=np.int64)  # lag > window

        data = np.random.randn(n)
        result = rolling_serial_correlation(data, window, lags)

        # All values should be NaN
        assert np.all(np.isnan(result[:, 0]))

    def test_constant_data_returns_zero(self):
        """Constant data should have zero autocorrelation."""
        n = 100
        window = 20
        lags = np.array([1, 5], dtype=np.int64)

        data = np.ones(n) * 5.0  # Constant
        result = rolling_serial_correlation(data, window, lags)

        # Valid autocorrelations should be 0
        valid = result[window - 1:, :]
        assert_array_almost_equal(valid, np.zeros_like(valid))


class TestSerialCorrelationAtLag:
    """Tests for single-lag autocorrelation."""

    def test_lag_1_basic(self):
        """Test lag-1 autocorrelation calculation."""
        # Create simple test data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # This is a perfectly trending series
        # Autocorr should be high

        rho = serial_correlation_at_lag(data, lag=1)
        assert not np.isnan(rho)
        assert rho > 0  # Trending series has positive autocorr

    def test_random_data(self):
        """Random data should have autocorr near 0."""
        np.random.seed(42)
        data = np.random.randn(1000)

        rho = serial_correlation_at_lag(data, lag=1)
        assert abs(rho) < 0.1

    def test_insufficient_data(self):
        """Too little data for lag should return NaN."""
        data = np.array([1.0, 2.0])
        rho = serial_correlation_at_lag(data, lag=5)
        assert np.isnan(rho)


class TestLjungBoxStatistic:
    """Tests for Ljung-Box Q statistic."""

    def test_random_data_small_q(self):
        """Random data should have small Q statistic."""
        np.random.seed(42)
        data = np.random.randn(500)
        lags = np.array([1, 5, 10], dtype=np.int64)

        q = ljung_box_statistic(data, lags)

        # For n=500, df=3, critical value at 0.05 is ~7.8
        # Random data should typically have Q < critical value
        # But we use a loose threshold due to randomness
        assert q < 20, f"Q={q} is unexpectedly large for random data"

    def test_ar1_large_q(self):
        """AR(1) process should have large Q statistic."""
        np.random.seed(42)
        n = 500
        phi = 0.8

        data = np.zeros(n)
        for t in range(1, n):
            data[t] = phi * data[t - 1] + np.random.randn() * 0.1

        lags = np.array([1, 5, 10], dtype=np.int64)
        q = ljung_box_statistic(data, lags)

        # Strong AR(1) should have very large Q
        assert q > 100, f"Q={q} should be larger for AR(1) process"


class TestSerialCorrelationTransform:
    """Tests for the SIMO Transform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        df = pd.DataFrame({'returns': returns}, index=dates)
        return df

    def test_output_names(self, sample_data):
        """Test that output names are correctly generated."""
        transform = SerialCorrelationTransform(
            input_col='returns',
            window=50,
            lags=[1, 5, 10]
        )

        assert transform.output_name == ['returns_autocorr_1', 'returns_autocorr_5', 'returns_autocorr_10']

    def test_transform_returns_tuple(self, sample_data):
        """Transform should return tuple of Series."""
        transform = SerialCorrelationTransform(
            input_col='returns',
            window=50,
            lags=[1, 5, 10]
        )

        result = transform(sample_data, backend='nb')

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(s, pd.Series) for s in result)

    def test_pandas_backend(self, sample_data):
        """Test pandas backend produces same results as numba."""
        transform = SerialCorrelationTransform(
            input_col='returns',
            window=50,
            lags=[1, 5, 10]
        )

        result_nb = transform(sample_data, backend='nb')
        result_pd = transform(sample_data, backend='pd')

        for s_nb, s_pd in zip(result_nb, result_pd):
            # Compare non-NaN values
            valid_mask = ~(s_nb.isna() | s_pd.isna())
            assert_array_almost_equal(
                s_nb[valid_mask].values,
                s_pd[valid_mask].values,
                decimal=10
            )

    def test_index_preserved(self, sample_data):
        """Output index should match input index."""
        transform = SerialCorrelationTransform(
            input_col='returns',
            window=50,
            lags=[1, 5]
        )

        result = transform(sample_data, backend='nb')

        for s in result:
            assert s.index.equals(sample_data.index)

    def test_invalid_column_raises(self, sample_data):
        """Invalid input column should raise ValueError."""
        transform = SerialCorrelationTransform(
            input_col='nonexistent',
            window=50,
            lags=[1]
        )

        with pytest.raises(ValueError, match="not found"):
            transform(sample_data, backend='nb')

    def test_window_too_large(self):
        """Window larger than data should raise ValueError."""
        df = pd.DataFrame({'returns': np.random.randn(10)})
        transform = SerialCorrelationTransform(
            input_col='returns',
            window=100,
            lags=[1]
        )

        with pytest.raises(ValueError, match="must be >="):
            transform(df, backend='nb')


class TestLjungBoxTransform:
    """Tests for the Ljung-Box Transform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        returns = np.random.randn(n) * 0.02
        df = pd.DataFrame({'returns': returns}, index=dates)
        return df

    def test_output_is_series(self, sample_data):
        """Transform should return a Series."""
        transform = LjungBoxTransform(
            input_col='returns',
            window=50,
            lags=[1, 5, 10]
        )

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = LjungBoxTransform(
            input_col='returns',
            window=50,
            lags=[1, 5, 10]
        )

        assert 'ljung_box' in transform.output_name


class TestAutocorrConvenienceFunction:
    """Tests for the convenience function."""

    def test_returns_dataframe(self):
        """Function should return DataFrame."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.02, index=dates, name='returns')

        result = autocorr(returns, window=50, lags=[1, 5, 10])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (200, 3)

    def test_column_names(self):
        """Test column naming."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        returns = pd.Series(np.random.randn(200) * 0.02, index=dates, name='returns')

        result = autocorr(returns, window=50, lags=[1, 5, 10])

        expected_cols = ['returns_autocorr_1', 'returns_autocorr_5', 'returns_autocorr_10']
        assert result.columns.tolist() == expected_cols


class TestEdgeCases:
    """Edge case tests."""

    def test_single_lag(self):
        """Test with single lag value."""
        np.random.seed(42)
        n = 100
        data = np.random.randn(n)
        lags = np.array([1], dtype=np.int64)

        result = rolling_serial_correlation(data, window=20, lags=lags)

        assert result.shape == (n, 1)

    def test_many_lags(self):
        """Test with many lag values."""
        np.random.seed(42)
        n = 200
        data = np.random.randn(n)
        lags = np.array(list(range(1, 21)), dtype=np.int64)  # 1 to 20

        result = rolling_serial_correlation(data, window=50, lags=lags)

        assert result.shape == (n, 20)

    def test_data_with_nans(self):
        """Test behavior with NaN in input."""
        np.random.seed(42)
        n = 100
        data = np.random.randn(n)
        data[30:35] = np.nan  # Inject NaNs

        lags = np.array([1], dtype=np.int64)
        result = rolling_serial_correlation(data, window=20, lags=lags)

        # NaN in window will propagate
        # Check that we don't crash and get some results
        assert not np.all(np.isnan(result))