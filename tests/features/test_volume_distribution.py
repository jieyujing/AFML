"""Tests for volume distribution transforms."""
import numpy as np
import pandas as pd
import pytest

from afmlkit.feature.core.volume_distribution import (
    VolEntropyTransform,
    VolSkewTransform,
    VolKurtTransform,
    VolPeakTransform,
    VolDiffStdTransform,
    AmbiguityTransform,
    AmbiguityVolTransform,
    AmbiguityCountTransform,
    PShapeTransform,
    BShapeTransform,
    PShapeDiffTransform,
    UnanimousBuyingTransform,
    VolStdTransform,
    VolumeVarTransform,
    UnanimousTradingTransform,
    TailVolumeRatioTransform,
    VolumeRatioTransform,
    VolumeShareTransform,
    AmountQuantileTransform,
    TradingIntensityTransform,
    TailVolumeRatioVTransform,
    TVDTransform,
)


def make_ohlcv(n=200, code='BTC', freq='1min') -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame."""
    dates = pd.date_range('2024-01-01', periods=n, freq=freq)
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'code': code,
        'amount': rng.randint(1000, 100000, n).astype(float),
        'count': rng.randint(1, 100, n),
        'open': 100 + rng.randn(n).cumsum(),
        'close': 100 + rng.randn(n).cumsum(),
        'high': 100 + rng.randn(n).cumsum() + 1,
        'low': 100 + rng.randn(n).cumsum() - 1,
        'volume': rng.randint(10, 1000, n),
    }, index=dates)


class TestVolEntropyTransform:
    """Tests for VolEntropyTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolEntropyTransform(n_bins=5, frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolEntropyTransform(n_bins=5, frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_has_name(self):
        """Test that result has the expected output name."""
        df = make_ohlcv(200)
        t = VolEntropyTransform(n_bins=5, frequency='h')
        result = t(df, backend='pd')
        assert result.name == t.output_name

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolEntropyTransform(n_bins=5, frequency='h')
        result = t(df, backend='pd')
        assert not np.all(np.isnan(result))

    def test_params_get_set(self):
        """Test get_params and set_params methods."""
        t = VolEntropyTransform(n_bins=5, frequency='h')
        params = t.get_params()
        assert isinstance(params, dict)
        assert 'frequency' in params
        assert 'n_bins' in params

        t.set_params(frequency='15min')
        assert t.frequency == '15min'

    def test_n_bins_affects_output_name(self):
        """Test that n_bins affects the output name."""
        t5 = VolEntropyTransform(n_bins=5)
        t10 = VolEntropyTransform(n_bins=10)
        assert t5.output_name != t10.output_name


class TestVolSkewTransform:
    """Tests for VolSkewTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolSkewTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolSkewTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_has_name(self):
        """Test that result has the expected output name."""
        df = make_ohlcv(200)
        t = VolSkewTransform(frequency='h')
        result = t(df, backend='pd')
        assert result.name == t.output_name

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolSkewTransform(frequency='h')
        result = t(df, backend='pd')
        assert not np.all(np.isnan(result))

    def test_params_get_set(self):
        """Test get_params and set_params methods."""
        t = VolSkewTransform(frequency='h')
        params = t.get_params()
        assert isinstance(params, dict)
        assert 'frequency' in params

        t.set_params(frequency='15min')
        assert t.frequency == '15min'


class TestVolKurtTransform:
    """Tests for VolKurtTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolKurtTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolKurtTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_has_name(self):
        """Test that result has the expected output name."""
        df = make_ohlcv(200)
        t = VolKurtTransform(frequency='h')
        result = t(df, backend='pd')
        assert result.name == t.output_name

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolKurtTransform(frequency='h')
        result = t(df, backend='pd')
        assert not np.all(np.isnan(result))

    def test_params_get_set(self):
        """Test get_params and set_params methods."""
        t = VolKurtTransform(frequency='h')
        params = t.get_params()
        assert isinstance(params, dict)
        assert 'frequency' in params

        t.set_params(frequency='15min')
        assert t.frequency == '15min'


class TestVolPeakTransform:
    """Tests for VolPeakTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolPeakTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolPeakTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_has_name(self):
        """Test that result has the expected output name."""
        df = make_ohlcv(200)
        t = VolPeakTransform(frequency='h')
        result = t(df, backend='pd')
        assert result.name == t.output_name

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolPeakTransform(frequency='h')
        result = t(df, backend='pd')
        assert not np.all(np.isnan(result))

    def test_params_get_set(self):
        """Test get_params and set_params methods."""
        t = VolPeakTransform(frequency='h')
        params = t.get_params()
        assert isinstance(params, dict)
        assert 'frequency' in params

        t.set_params(frequency='15min')
        assert t.frequency == '15min'

    def test_output_in_zero_one(self):
        """Test that peak ratio values are in [0, 1] after dropna."""
        df = make_ohlcv(200)
        t = VolPeakTransform(frequency='h')
        result = t(df, backend='pd')
        valid = result.dropna()
        assert np.all(valid >= 0)
        assert np.all(valid <= 1)


class TestVolDiffStdTransform:
    """Tests for VolDiffStdTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolDiffStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolDiffStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_has_name(self):
        """Test that result has the expected output name."""
        df = make_ohlcv(200)
        t = VolDiffStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert result.name == t.output_name

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolDiffStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert not np.all(np.isnan(result))

    def test_params_get_set(self):
        """Test get_params and set_params methods."""
        t = VolDiffStdTransform(frequency='h')
        params = t.get_params()
        assert isinstance(params, dict)
        assert 'frequency' in params

        t.set_params(frequency='15min')
        assert t.frequency == '15min'

    def test_requires_count_column(self):
        """Test that calling on DataFrame without 'count' raises ValueError."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        rng = np.random.RandomState(42)
        df_no_count = pd.DataFrame({
            'code': 'BTC',
            'amount': rng.randint(1000, 100000, 200).astype(float),
            'open': 100 + rng.randn(200).cumsum(),
            'close': 100 + rng.randn(200).cumsum(),
            'high': 100 + rng.randn(200).cumsum() + 1,
            'low': 100 + rng.randn(200).cumsum() - 1,
            'volume': rng.randint(10, 1000, 200),
        }, index=dates)
        t = VolDiffStdTransform(frequency='h')
        with pytest.raises(ValueError):
            t(df_no_count, backend='pd')


class TestAmbiguityTransform:
    """Tests for AmbiguityTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = AmbiguityTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = AmbiguityTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = AmbiguityTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestAmbiguityVolTransform:
    """Tests for AmbiguityVolTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = AmbiguityVolTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = AmbiguityVolTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = AmbiguityVolTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestAmbiguityCountTransform:
    """Tests for AmbiguityCountTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = AmbiguityCountTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = AmbiguityCountTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = AmbiguityCountTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestPShapeTransform:
    """Tests for PShapeTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = PShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = PShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = PShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestBShapeTransform:
    """Tests for BShapeTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = BShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = BShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = BShapeTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestPShapeDiffTransform:
    """Tests for PShapeDiffTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = PShapeDiffTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = PShapeDiffTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = PShapeDiffTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestUnanimousBuyingTransform:
    """Tests for UnanimousBuyingTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = UnanimousBuyingTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = UnanimousBuyingTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = UnanimousBuyingTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestVolStdTransform:
    """Tests for VolStdTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolStdTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestVolumeVarTransform:
    """Tests for VolumeVarTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolumeVarTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolumeVarTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolumeVarTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestUnanimousTradingTransform:
    """Tests for UnanimousTradingTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = UnanimousTradingTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = UnanimousTradingTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = UnanimousTradingTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestTailVolumeRatioTransform:
    """Tests for TailVolumeRatioTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = TailVolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = TailVolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = TailVolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestVolumeRatioTransform:
    """Tests for VolumeRatioTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolumeRatioTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestVolumeShareTransform:
    """Tests for VolumeShareTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = VolumeShareTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = VolumeShareTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = VolumeShareTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestAmountQuantileTransform:
    """Tests for AmountQuantileTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = AmountQuantileTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = AmountQuantileTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = AmountQuantileTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()

    def test_requires_count_column(self):
        """Test that calling on DataFrame without 'count' raises ValueError."""
        dates = pd.date_range('2024-01-01', periods=200, freq='1min')
        rng = np.random.RandomState(42)
        df_no_count = pd.DataFrame({
            'code': 'BTC',
            'amount': rng.randint(1000, 100000, 200).astype(float),
            'open': 100 + rng.randn(200).cumsum(),
            'close': 100 + rng.randn(200).cumsum(),
            'high': 100 + rng.randn(200).cumsum() + 1,
            'low': 100 + rng.randn(200).cumsum() - 1,
            'volume': rng.randint(10, 1000, 200),
        }, index=dates)
        t = AmountQuantileTransform(frequency='h')
        with pytest.raises(ValueError):
            t(df_no_count, backend='pd')


class TestTradingIntensityTransform:
    """Tests for TradingIntensityTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = TradingIntensityTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = TradingIntensityTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = TradingIntensityTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestTailVolumeRatioVTransform:
    """Tests for TailVolumeRatioVTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = TailVolumeRatioVTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = TailVolumeRatioVTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = TailVolumeRatioVTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()


class TestTVDTransform:
    """Tests for TVDTransform."""

    def test_output_length(self):
        """Test output length matches input length."""
        df = make_ohlcv(200)
        t = TVDTransform(frequency='h')
        result = t(df, backend='pd')
        assert len(result) == len(df)

    def test_returns_series(self):
        """Test that transform returns a pandas Series."""
        df = make_ohlcv(200)
        t = TVDTransform(frequency='h')
        result = t(df, backend='pd')
        assert isinstance(result, pd.Series)

    def test_no_all_nan(self):
        """Test that not all values are NaN (some windows have enough data)."""
        df = make_ohlcv(200)
        t = TVDTransform(frequency='h')
        result = t(df, backend='pd')
        assert not result.isna().all()