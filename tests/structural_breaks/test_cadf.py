import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.cadf import (
    _conditional_expectation_core,
    cadf_test,
    CADFTest,
)


def test_conditional_expectation_core():
    """Test conditional expectation calculation."""
    adf = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    quantile = np.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5], dtype=np.float64)
    window = 5

    result = _conditional_expectation_core(adf, quantile, window)

    assert len(result) == len(adf)
    assert np.isfinite(result[9])


def test_cadf_from_series():
    """Test CADF from price series."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))

    result = cadf_test(
        prices,
        min_window=20,
        max_window=100,
        quantile_window=20,
        quantile=0.95
    )

    assert len(result) == len(prices)
    assert np.all(np.isfinite(result[~np.isnan(result)]))


def test_cadf_transform():
    """Test CADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(150) * 0.02))
    }, index=dates)

    transform = CADFTest(
        min_window=20,
        max_window=80,
        quantile_window=20,
        quantile=0.95
    )
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'cadf'
    assert len(result) == len(df)


def test_cadf_with_injected_values():
    """Test CADF with pre-computed SADF/QADF injection."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    sadf = np.random.randn(100).cumsum() * 0.5 + 1
    qadf = pd.Series(np.convolve(sadf, np.ones(20)/20, mode='same'), index=dates)
    sadf = pd.Series(sadf, index=dates)

    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = CADFTest(
        sadf_values=sadf.values,
        qadf_values=qadf.values,
        quantile_window=20
    )
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
