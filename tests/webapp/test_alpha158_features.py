"""Tests for Alpha158 FFD features module."""
import numpy as np
import pandas as pd
import pytest


def test_compute_ffd_base_basic():
    """Test basic FFD base series computation."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Check return types
    assert isinstance(ffd_series, pd.Series)
    assert isinstance(optimal_d, float)

    # Check optimal_d is in valid range
    assert 0.0 <= optimal_d <= 1.0

    # Check series length (should be <= input due to FFD lag)
    assert len(ffd_series) <= len(close)
    assert len(ffd_series) > 0

    # Check no NaN in result
    assert ffd_series.isna().sum() == 0


def test_compute_ffd_base_already_stationary():
    """Test FFD returns d=0 for already stationary series."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    stationary = pd.Series(np.random.randn(200))

    ffd_series, optimal_d = compute_ffd_base(stationary)

    # Should return d=0 or very small
    assert optimal_d < 0.1


def test_compute_ffd_base_custom_params():
    """Test FFD with custom threshold and d_step."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(
        close,
        thres=1e-5,
        d_step=0.1
    )

    assert isinstance(ffd_series, pd.Series)
    assert 0.0 <= optimal_d <= 1.0


def test_compute_ffd_base_preserves_index():
    """Test FFD preserves datetime index."""
    from webapp.utils.alpha158_features import compute_ffd_base

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='min')
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Index should be preserved (though first few rows may be dropped)
    assert isinstance(ffd_series.index, pd.DatetimeIndex)


# ============== compute_ffd_volatility tests ==============

def test_compute_ffd_volatility_basic():
    """Test FFD volatility feature computation."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100), name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5, 10, 20])

    # Check all expected columns exist
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' in result.columns
    assert 'ffd_vol_ewm_5' in result.columns
    assert 'ffd_vol_ewm_10' in result.columns
    assert 'ffd_vol_ewm_20' in result.columns

    # Check data types
    assert all(result.dtypes == np.float64)


def test_compute_ffd_volatility_custom_spans():
    """Test FFD volatility with custom spans."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100))

    result = compute_ffd_volatility(ffd_series, spans=[3, 7])

    assert 'ffd_vol_std_3' in result.columns
    assert 'ffd_vol_std_7' in result.columns
    assert 'ffd_vol_ewm_3' in result.columns
    assert 'ffd_vol_ewm_7' in result.columns


def test_compute_ffd_volatility_preserves_index():
    """Test FFD volatility preserves datetime index."""
    from webapp.utils.alpha158_features import compute_ffd_volatility

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='min')
    ffd_series = pd.Series(np.random.randn(100), index=dates, name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5])

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 100


# ============== compute_ffd_ma tests ==============

def test_compute_ffd_ma_basic():
    """Test FFD moving average feature computation."""
    from webapp.utils.alpha158_features import compute_ffd_ma

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100), name="ffd_log_price")

    result = compute_ffd_ma(ffd_series, windows=[5, 10, 20])

    # Check all expected columns exist
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_ma_10' in result.columns
    assert 'ffd_ma_20' in result.columns
    assert 'ffd_ema_5' in result.columns
    assert 'ffd_ema_10' in result.columns
    assert 'ffd_ema_20' in result.columns


def test_compute_ffd_ma_custom_windows():
    """Test FFD MA with custom windows."""
    from webapp.utils.alpha158_features import compute_ffd_ma

    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100))

    result = compute_ffd_ma(ffd_series, windows=[3, 7])

    assert 'ffd_ma_3' in result.columns
    assert 'ffd_ma_7' in result.columns
    assert 'ffd_ema_3' in result.columns
    assert 'ffd_ema_7' in result.columns


def test_compute_ffd_ma_preserves_index():
    """Test FFD MA preserves datetime index."""
    from webapp.utils.alpha158_features import compute_ffd_ma

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='min')
    ffd_series = pd.Series(np.random.randn(100), index=dates, name="ffd_log_price")

    result = compute_ffd_ma(ffd_series, windows=[5])

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 100


# ============== compute_ffd_rank tests ==============

def test_compute_ffd_rank_basic():
    """Test FFD rank feature computation."""
    from webapp.utils.alpha158_features import compute_ffd_rank

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='min')

    df = pd.DataFrame({
        'ffd_ma_5': pd.Series(np.random.randn(100), index=dates),
        'ffd_vol_std_10': pd.Series(np.random.randn(100), index=dates)
    })

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5', 'ffd_vol_std_10'], rank_window=20)

    # Check rank columns exist
    assert 'ffd_rank_ffd_ma_5_20' in result.columns
    assert 'ffd_rank_ffd_vol_std_10_20' in result.columns

    # Check values are in [0, 1] range
    assert result['ffd_rank_ffd_ma_5_20'].between(0, 1).all()
    assert result['ffd_rank_ffd_vol_std_10_20'].between(0, 1).all()


def test_compute_ffd_rank_single_feature():
    """Test FFD rank with single feature."""
    from webapp.utils.alpha158_features import compute_ffd_rank

    np.random.seed(42)
    df = pd.DataFrame({'ffd_ma_5': np.random.randn(50)})

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5'], rank_window=10)

    assert 'ffd_rank_ffd_ma_5_10' in result.columns
    assert len(result) == 50


def test_compute_ffd_rank_preserves_original():
    """Test FFD rank doesn't modify original columns."""
    from webapp.utils.alpha158_features import compute_ffd_rank

    np.random.seed(42)
    df = pd.DataFrame({
        'ffd_ma_5': np.random.randn(50),
        'other_col': np.random.randn(50)
    })

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5'], rank_window=10)

    # Original columns preserved
    assert 'ffd_ma_5' in result.columns
    assert 'other_col' in result.columns
    assert 'ffd_rank_ffd_ma_5_10' in result.columns


# ============== compute_volume_features tests ==============

def test_compute_volume_features_basic():
    """Test volume feature computation."""
    from webapp.utils.alpha158_features import compute_volume_features

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='min')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100),
        'volume': np.random.exponential(1000, 100)
    }, index=dates)

    result = compute_volume_features(df)

    # Check expected columns
    assert 'ffd_vwap' in result.columns
    assert 'ffd_amount' in result.columns
    assert 'ffd_amplification' in result.columns


def test_compute_volume_amount():
    """Test amount = close * volume."""
    from webapp.utils.alpha158_features import compute_volume_features

    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0],
        'volume': [1000, 1100, 1200]
    })

    result = compute_volume_features(df)

    expected_amount = df['close'] * df['volume']
    # Skip first row due to rolling window
    assert result['ffd_amount'].iloc[1:].equals(expected_amount.iloc[1:])


# ============== compute_alpha158_features tests ==============

def test_compute_alpha158_features_complete():
    """Test complete Alpha158 feature computation."""
    from webapp.utils.alpha158_features import compute_alpha158_features

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='min')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    result, metadata = compute_alpha158_features(df)

    # Check base FFD feature
    assert 'ffd_log_price' in result.columns

    # Check volatility features
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' in result.columns

    # Check MA features
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_ma_10' in result.columns
    assert 'ffd_ma_20' in result.columns

    # Check volume features
    assert 'ffd_vwap' in result.columns
    assert 'ffd_amount' in result.columns

    # Check metadata
    assert 'optimal_d' in metadata
    assert 0.0 <= metadata['optimal_d'] <= 1.0
    assert 'feature_columns' in metadata


def test_compute_alpha158_features_with_config():
    """Test Alpha158 with custom config."""
    from webapp.utils.alpha158_features import compute_alpha158_features

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='min')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    config = {
        'volatility': {'spans': [5, 10]},
        'ma': {'windows': [5, 10]},
        'rank': {'enabled': True, 'window': 20}
    }

    result, metadata = compute_alpha158_features(df, config=config)

    # Check custom spans
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' not in result.columns  # Not in config

    # Check rank features enabled
    assert 'ffd_rank_ffd_ma_5_20' in result.columns


def test_compute_alpha158_features_missing_columns():
    """Test Alpha158 gracefully handles missing columns."""
    from webapp.utils.alpha158_features import compute_alpha158_features

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='min')
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    # Should not raise error, just skip features requiring missing columns
    result, metadata = compute_alpha158_features(df)

    # Core features should still be computed
    assert 'ffd_log_price' in result.columns
    assert 'ffd_ma_5' in result.columns
