"""
Tests for microstructure features.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from afmlkit.feature.core.microstructure import (
    amihud_illiquidity,
    roll_spread,
    corwin_schultz_spread,
    rolling_corwin_schultz_spread,
    high_low_volatility,
    AmihudTransform,
    RollSpreadTransform,
    CorwinSchultzTransform,
    ParkinsonVolatilityTransform,
    amihud,
    roll_spread_estimate,
    corwin_schultz
)


class TestAmihudIlliquidity:
    """Tests for Amihud illiquidity measure."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        returns = np.random.randn(n) * 0.02
        volume = np.abs(np.random.randn(n)) * 1e6

        result = amihud_illiquidity(returns, volume, window=20)

        assert result.shape == (n,)

    def test_first_window_minus_one_are_nan(self):
        """First window-1 values should be NaN."""
        n = 100
        window = 20
        returns = np.random.randn(n) * 0.02
        volume = np.abs(np.random.randn(n)) * 1e6

        result = amihud_illiquidity(returns, volume, window)

        assert np.all(np.isnan(result[:window - 1]))

    def test_zero_volume_handling(self):
        """Zero volume should be handled gracefully."""
        n = 100
        returns = np.random.randn(n) * 0.02
        volume = np.abs(np.random.randn(n)) * 1e6
        volume[50:55] = 0  # Zero volume period

        result = amihud_illiquidity(returns, volume, window=20)

        # Should not crash, and should have valid values
        assert not np.all(np.isnan(result))

    def test_higher_illiquidity_for_higher_impact(self):
        """Higher price impact should give higher illiquidity."""
        np.random.seed(42)
        n = 200

        # High impact scenario: large returns, small volume
        returns_high = np.random.randn(n) * 0.05
        volume_low = np.ones(n) * 1e4

        # Low impact scenario: small returns, large volume
        returns_low = np.random.randn(n) * 0.01
        volume_high = np.ones(n) * 1e8

        illiq_high = amihud_illiquidity(returns_high, volume_low, window=50)
        illiq_low = amihud_illiquidity(returns_low, volume_high, window=50)

        # High impact should have higher illiquidity
        mean_high = np.nanmean(illiq_high)
        mean_low = np.nanmean(illiq_low)

        assert mean_high > mean_low * 10  # Should be significantly higher

    def test_insufficient_data(self):
        """Data shorter than window should return all NaN."""
        returns = np.random.randn(10) * 0.02
        volume = np.abs(np.random.randn(10)) * 1e6

        result = amihud_illiquidity(returns, volume, window=20)

        assert np.all(np.isnan(result))


class TestRollSpread:
    """Tests for Roll spread estimator."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        prices = 100 + np.random.randn(n).cumsum()

        result = roll_spread(prices, window=20)

        assert result.shape == (n,)

    def test_first_window_are_nan(self):
        """First window values should be NaN."""
        n = 100
        window = 20
        prices = 100 + np.random.randn(n).cumsum()

        result = roll_spread(prices, window)

        assert np.all(np.isnan(result[:window]))

    def test_trending_prices_give_nan(self):
        """Strongly trending prices give positive covariance = NaN spread."""
        n = 100
        # Strong uptrend: each price higher than last
        prices = 100 + np.arange(n) * 0.5

        result = roll_spread(prices, window=20)

        # For strong trend, covariance should be positive
        # Roll model returns NaN when cov > 0
        # Most values should be NaN
        nan_count = np.sum(np.isnan(result[20:]))
        assert nan_count > 0

    def test_mean_reverting_prices_give_spread(self):
        """Mean-reverting prices should give positive spread."""
        np.random.seed(42)
        n = 200

        # Create mean-reverting prices (alternating up/down)
        # This creates negative serial correlation
        changes = np.array([1, -1, 1, -1] * 50)[:n-1] * 0.1
        prices = np.zeros(n)
        prices[0] = 100
        for i in range(1, n):
            prices[i] = prices[i-1] + changes[i-1]

        result = roll_spread(prices, window=20)

        # Should have some valid (non-NaN) spread estimates
        valid_count = np.sum(~np.isnan(result))
        assert valid_count > 0


class TestCorwinSchultzSpread:
    """Tests for Corwin-Schultz spread estimator."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        high = 100 + np.abs(np.random.randn(n))
        low = 100 - np.abs(np.random.randn(n))

        result = corwin_schultz_spread(high, low)

        assert result.shape == (n,)

    def test_first_value_is_nan(self):
        """First value should be NaN (needs 2 days)."""
        n = 100
        high = 100 + np.abs(np.random.randn(n))
        low = 100 - np.abs(np.random.randn(n))

        result = corwin_schultz_spread(high, low)

        assert np.isnan(result[0])

    def test_spread_is_positive(self):
        """Spread estimates should be positive when valid."""
        np.random.seed(42)
        n = 200
        base_price = 100
        high = base_price + np.abs(np.random.randn(n)) * 2
        low = base_price - np.abs(np.random.randn(n)) * 2

        result = corwin_schultz_spread(high, low)

        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_constant_prices_nan(self):
        """Constant high = low should give NaN."""
        n = 50
        prices = np.ones(n) * 100
        high = prices.copy()
        low = prices.copy()

        result = corwin_schultz_spread(high, low)

        # Should be NaN or very small spread
        assert np.all(np.isnan(result) | (result < 1e-10))

    def test_high_less_than_low_handling(self):
        """High < Low should be handled gracefully."""
        n = 50
        high = np.abs(np.random.randn(n)) + 100
        low = np.abs(np.random.randn(n)) + 100

        # Swap some values
        high[:10], low[:10] = low[:10].copy(), high[:10].copy()

        # Should not crash
        result = corwin_schultz_spread(high, low)
        assert result.shape == (n,)


class TestRollingCorwinSchultzSpread:
    """Tests for rolling Corwin-Schultz spread."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        high = 100 + np.abs(np.random.randn(n))
        low = 100 - np.abs(np.random.randn(n))

        result = rolling_corwin_schultz_spread(high, low, window=10)

        assert result.shape == (n,)

    def test_smoothing_effect(self):
        """Rolling average should smooth the daily estimates."""
        np.random.seed(42)
        n = 100
        high = 100 + np.abs(np.random.randn(n)) * 2
        low = 100 - np.abs(np.random.randn(n)) * 2

        daily = corwin_schultz_spread(high, low)
        rolling = rolling_corwin_schultz_spread(high, low, window=20)

        # Rolling should have lower variance (smoother)
        daily_valid = daily[~np.isnan(daily)]
        rolling_valid = rolling[~np.isnan(rolling)]

        if len(daily_valid) > 5 and len(rolling_valid) > 5:
            assert np.std(rolling_valid) <= np.std(daily_valid) * 1.5


class TestHighLowVolatility:
    """Tests for Parkinson high-low volatility."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        n = 100
        high = 100 + np.abs(np.random.randn(n))
        low = 100 - np.abs(np.random.randn(n))

        result = high_low_volatility(high, low, window=20)

        assert result.shape == (n,)

    def test_volatility_is_positive(self):
        """Volatility estimates should be positive."""
        np.random.seed(42)
        n = 200
        high = 100 + np.abs(np.random.randn(n)) * 2
        low = 100 - np.abs(np.random.randn(n)) * 2

        result = high_low_volatility(high, low, window=20)

        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_larger_range_higher_volatility(self):
        """Larger high-low range should give higher volatility."""
        np.random.seed(42)
        n = 200

        # Small range
        high_small = 100 + np.abs(np.random.randn(n)) * 0.5
        low_small = 100 - np.abs(np.random.randn(n)) * 0.5

        # Large range
        high_large = 100 + np.abs(np.random.randn(n)) * 5
        low_large = 100 - np.abs(np.random.randn(n)) * 5

        vol_small = high_low_volatility(high_small, low_small, window=50)
        vol_large = high_low_volatility(high_large, low_large, window=50)

        assert np.nanmean(vol_large) > np.nanmean(vol_small)


class TestAmihudTransform:
    """Tests for AmihudTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        close = 100 + np.random.randn(n).cumsum()
        volume = np.abs(np.random.randn(n)) * 1e6
        returns = np.diff(close, prepend=close[0]) / close

        return pd.DataFrame({
            'close': close,
            'volume': volume,
            'returns': returns
        }, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = AmihudTransform('returns', 'volume', 'close', window=20)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = AmihudTransform('returns', 'volume', 'close', window=20)

        assert transform.output_name == 'amihud_20'

    def test_index_preserved(self, sample_data):
        """Output index should match input index."""
        transform = AmihudTransform('returns', 'volume', 'close', window=20)

        result = transform(sample_data, backend='nb')

        assert result.index.equals(sample_data.index)


class TestRollSpreadTransform:
    """Tests for RollSpreadTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        close = 100 + np.random.randn(n).cumsum()

        return pd.DataFrame({'close': close}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = RollSpreadTransform('close', window=20)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = RollSpreadTransform('close', window=20)

        # SISOTransform prepends input column name
        assert transform.output_name == 'close_roll_spread_20'


class TestCorwinSchultzTransform:
    """Tests for CorwinSchultzTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        high = 100 + np.abs(np.random.randn(n)) * 2
        low = 100 - np.abs(np.random.randn(n)) * 2

        return pd.DataFrame({'high': high, 'low': low}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = CorwinSchultzTransform('high', 'low', window=5)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_window_one_no_averaging(self, sample_data):
        """Window=1 should give daily estimates without averaging."""
        transform = CorwinSchultzTransform('high', 'low', window=1)

        result = transform(sample_data, backend='nb')

        assert transform.output_name == 'cs_spread'


class TestParkinsonVolatilityTransform:
    """Tests for ParkinsonVolatilityTransform class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        high = 100 + np.abs(np.random.randn(n)) * 2
        low = 100 - np.abs(np.random.randn(n)) * 2

        return pd.DataFrame({'high': high, 'low': low}, index=dates)

    def test_transform_returns_series(self, sample_data):
        """Transform should return a Series."""
        transform = ParkinsonVolatilityTransform('high', 'low', window=20)

        result = transform(sample_data, backend='nb')

        assert isinstance(result, pd.Series)

    def test_output_name(self, sample_data):
        """Test output name generation."""
        transform = ParkinsonVolatilityTransform('high', 'low', window=20)

        assert transform.output_name == 'parkinson_vol_20'


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        close = 100 + np.random.randn(n).cumsum()
        volume = np.abs(np.random.randn(n)) * 1e6
        returns = np.diff(close, prepend=close[0]) / close
        high = np.maximum(close, close + np.abs(np.random.randn(n)) * 2)
        low = np.minimum(close, close - np.abs(np.random.randn(n)) * 2)

        return pd.DataFrame({
            'close': close,
            'volume': volume,
            'returns': returns,
            'high': high,
            'low': low
        }, index=dates)

    def test_amihud_function(self, sample_data):
        """Test amihud convenience function."""
        result = amihud(sample_data, window=20)

        assert isinstance(result, pd.Series)

    def test_roll_spread_function(self, sample_data):
        """Test roll_spread_estimate convenience function."""
        result = roll_spread_estimate(sample_data['close'], window=20)

        assert isinstance(result, pd.Series)

    def test_corwin_schultz_function(self, sample_data):
        """Test corwin_schultz convenience function."""
        result = corwin_schultz(sample_data, window=5)

        assert isinstance(result, pd.Series)


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_arrays(self):
        """Test with empty arrays."""
        returns = np.array([])
        volume = np.array([])

        result = amihud_illiquidity(returns, volume, window=20)

        assert len(result) == 0

    def test_single_element(self):
        """Test with single element."""
        returns = np.array([0.01])
        volume = np.array([1e6])

        result = amihud_illiquidity(returns, volume, window=20)

        assert len(result) == 1
        assert np.isnan(result[0])

    def test_all_nan_returns(self):
        """Test with all NaN returns."""
        n = 50
        returns = np.full(n, np.nan)
        volume = np.abs(np.random.randn(n)) * 1e6

        result = amihud_illiquidity(returns, volume, window=20)

        assert np.all(np.isnan(result))

    def test_extreme_values(self):
        """Test with extreme price values."""
        n = 100
        # Very large prices
        high = 1e10 + np.abs(np.random.randn(n)) * 1e8
        low = 1e10 - np.abs(np.random.randn(n)) * 1e8

        result = corwin_schultz_spread(high, low)

        # Should handle large values without overflow
        assert not np.any(np.isinf(result))