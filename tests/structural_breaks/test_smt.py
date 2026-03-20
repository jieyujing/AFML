import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.smt import (
    _sub_martingale_core,
    _super_martingale_core,
    sub_martingale_test,
    super_martingale_test,
    martingale_test,
    SubMartingaleTest,
    SuperMartingaleTest,
    MartingaleTest,
)


def test_sub_martingale_core_uptrend():
    """Test sub-martingale detects uptrend."""
    prices = np.exp(np.arange(100) * 0.02) * 100

    result = _sub_martingale_core(prices, decay=0.95, window_size=20)

    assert len(result) == len(prices)
    assert np.all(np.isfinite(result[20:]))
    assert np.mean(result[20:]) > 0


def test_sub_martingale_core_downtrend():
    """Test sub-martingale on downtrend."""
    prices = np.exp(-np.arange(100) * 0.02) * 100

    result = _sub_martingale_core(prices, decay=0.95, window_size=20)

    assert np.mean(result[20:]) < 1.0


def test_super_martingale_core():
    """Test super-martingale is negative of sub-martingale."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))

    sub = _sub_martingale_core(prices, decay=0.95, window_size=20)
    sup = _super_martingale_core(prices, decay=0.95, window_size=20)

    assert len(sub) == len(sup)
    # Compare only finite values (first element is NaN)
    assert np.allclose(sub[1:], -sup[1:], atol=1e-10)


def test_sub_martingale_test():
    """Test public sub-martingale function."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    result = sub_martingale_test(prices, decay=0.95, window=30)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)


def test_super_martingale_test():
    """Test public super-martingale function."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    result = super_martingale_test(prices, decay=0.95, window=30)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)


def test_martingale_test():
    """Test combined martingale test."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    sub, sup = martingale_test(prices, decay=0.95, window=30)

    assert len(sub) == len(prices)
    assert len(sup) == len(prices)


def test_sub_martingale_transform():
    """Test SubMartingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = SubMartingaleTest(decay=0.95, window=20, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_sub_martingale'


def test_super_martingale_transform():
    """Test SuperMartingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = SuperMartingaleTest(decay=0.95, window=20, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_super_martingale'


def test_martingale_transform():
    """Test combined Martingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = MartingaleTest(decay=0.95, window=20, input_col='close')
    sub, sup = transform(df, backend='nb')

    assert isinstance(sub, pd.Series)
    assert isinstance(sup, pd.Series)
    assert sub.name == 'close_sub_martingale'
    assert sup.name == 'close_super_martingale'


def test_martingale_decay_effect():
    """Test that higher decay gives more persistent statistics."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200))

    sub_90 = sub_martingale_test(prices, decay=0.90, window=50)
    sub_99 = sub_martingale_test(prices, decay=0.99, window=50)

    var_90 = np.nanvar(sub_90)
    var_99 = np.nanvar(sub_99)
    assert var_99 < var_90
