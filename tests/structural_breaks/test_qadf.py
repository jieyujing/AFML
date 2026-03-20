import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.qadf import (
    _rolling_quantile_core,
    qadf_test,
    QADFTest,
)


def test_rolling_quantile_core():
    """Test rolling quantile computation."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    window = 5
    quantile = 0.95

    result = _rolling_quantile_core(data, window, quantile)

    assert len(result) == len(data)
    assert np.all(np.isnan(result[:window-1]))
    assert np.all(np.isfinite(result[window-1:]))
    assert result[4] > 4.5


def test_qadf_from_sadf():
    """Test QADF computed from SADF values."""
    np.random.seed(42)
    sadf = np.random.randn(200) * 2 + np.sin(np.arange(200) * 0.1)

    result = qadf_test(sadf, window=20, quantile=0.95)

    assert len(result) == len(sadf)
    assert np.all(np.isnan(result[:19]))
    assert np.all(np.isfinite(result[20:]))


def test_qadf_transform():
    """Test QADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    np.random.seed(42)
    sadf = pd.Series(
        np.random.randn(150).cumsum() * 0.5 + 1,
        index=dates,
        name='sadf'
    )
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(150))})
    df['sadf'] = sadf

    transform = QADFTest(window=30, quantile=0.95)
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'qadf'
    assert len(result) == len(df)


def test_qadf_high_quantile():
    """Test that higher quantile gives higher values."""
    np.random.seed(42)
    sadf = np.random.randn(100) * 2

    q95 = qadf_test(sadf, window=20, quantile=0.95)
    q50 = qadf_test(sadf, window=20, quantile=0.50)

    valid_idx = ~np.isnan(q95) & ~np.isnan(q50)
    assert np.all(q95[valid_idx] >= q50[valid_idx])
