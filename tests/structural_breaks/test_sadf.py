import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.sadf import (
    _sadf_core,
    sadf_test,
    SADFTest,
)


def test_sadf_core_basic():
    """Test SADF core computation."""
    np.random.seed(42)
    t = np.arange(100, dtype=np.float64)
    prices = 100 * np.exp(0.01 * t + 0.001 * t**1.5)

    min_window = 20
    max_window = 50
    result = _sadf_core(prices, min_window, max_window, max_lag=0, trend=True)

    assert len(result) == len(prices)
    assert np.all(np.isnan(result[:min_window]))
    assert np.all(np.isfinite(result[min_window:]))
    assert np.mean(result[min_window:]) > 0


def test_sadf_test_series():
    """Test SADF with pandas Series input."""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(200) * 0.02)),
        index=dates
    )

    result = sadf_test(prices, min_window=20, max_window=100, max_lag=2)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)
    assert result.index.equals(prices.index)


def test_sadf_test_array():
    """Test SADF with numpy array input."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.02))

    result = sadf_test(prices, min_window=20, max_window=80, use_log=True)

    assert len(result) == len(prices)
    assert isinstance(result, np.ndarray)


def test_sadf_transform():
    """Test SADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    }, index=dates)

    transform = SADFTest(min_window=20, max_window=60, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_sadf'
    assert len(result) == len(df)


def test_sadf_no_bubble_series():
    """Test SADF on random walk (no bubble)."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

    result = sadf_test(prices, min_window=20, max_window=100)

    valid = result[~np.isnan(result)]
    assert np.percentile(valid, 90) < 2.0
