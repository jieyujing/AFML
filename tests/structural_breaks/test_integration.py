"""
Integration tests for structural break features.

Tests the complete pipeline and ensures all modules integrate correctly.
"""
import pandas as pd
import numpy as np


def test_import_all_features():
    """Test all structural break features can be imported."""
    from afmlkit.feature.core.structural_break import (
        # ADF base
        adf_test,
        adf_test_rolling,
        # SADF
        sadf_test,
        SADFTest,
        # QADF
        qadf_test,
        QADFTest,
        # CADF
        cadf_test,
        CADFTest,
        # SMT
        sub_martingale_test,
        super_martingale_test,
        martingale_test,
        SubMartingaleTest,
        SuperMartingaleTest,
        MartingaleTest,
        # CUSUM
        cusum_test_developing,
        cusum_test_last,
        cusum_test_rolling,
    )

    assert callable(adf_test)
    assert callable(adf_test_rolling)
    assert callable(sadf_test)
    assert callable(qadf_test)
    assert callable(cadf_test)
    assert callable(sub_martingale_test)
    assert callable(super_martingale_test)
    assert callable(martingale_test)
    assert callable(cusum_test_developing)
    assert callable(cusum_test_last)
    assert callable(cusum_test_rolling)

    # Check classes exist
    assert isinstance(SADFTest, type)
    assert isinstance(QADFTest, type)
    assert isinstance(CADFTest, type)
    assert isinstance(SubMartingaleTest, type)
    assert isinstance(SuperMartingaleTest, type)
    assert isinstance(MartingaleTest, type)


def test_end_to_end_pipeline():
    """Test end-to-end feature pipeline."""
    from afmlkit.feature.core.structural_break import (
        SADFTest, QADFTest, SubMartingaleTest, SuperMartingaleTest
    )

    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))
    }, index=dates)

    # Apply transforms
    sadf = SADFTest(min_window=20, max_window=100)(df, backend='nb')
    df['sadf'] = sadf

    qadf = QADFTest(window=20)(df, backend='nb')

    sub = SubMartingaleTest(window=30)(df, backend='nb')
    sup = SuperMartingaleTest(window=30)(df, backend='nb')

    assert len(qadf) == len(df)
    assert len(sub) == len(df)
    assert len(sup) == len(df)


def test_cusum_functions():
    """Test CUSUM functions can be called."""
    from afmlkit.feature.core.structural_break import (
        cusum_test_last,
        cusum_test_rolling,
    )

    np.random.seed(42)
    y = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))

    # Test cusum_test_last
    result = cusum_test_last(y)
    assert len(result) == 4  # Returns tuple of 4 values

    # Test cusum_test_rolling - uses window_size, not window
    up_values, down_values, up_critical, down_critical = cusum_test_rolling(y, window_size=20, warmup_period=10)
    assert len(up_values) == len(y)
    assert len(down_values) == len(y)


def test_adf_functions():
    """Test ADF functions can be called."""
    from afmlkit.feature.core.structural_break import (
        adf_test,
        adf_test_rolling,
    )

    np.random.seed(42)
    y = np.cumsum(np.random.randn(50) * 0.01)

    # Test adf_test - returns tuple (statistic, pvalue, usedlag)
    result = adf_test(y)
    assert isinstance(result, tuple)
    assert len(result) == 3

    # Test adf_test_rolling
    result = adf_test_rolling(y, window=20)
    assert len(result) == len(y)


def test_cadf_functions():
    """Test CADF functions can be called."""
    from afmlkit.feature.core.structural_break import (
        cadf_test,
        CADFTest,
    )

    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    }, index=dates)

    # Test CADFTest class - uses min_window/max_window, not window
    cadf = CADFTest(min_window=20, max_window=80, quantile_window=20)(df, backend='nb')
    assert len(cadf) == len(df)


def test_martingale_functions():
    """Test martingale test functions can be called."""
    from afmlkit.feature.core.structural_break import (
        sub_martingale_test,
        super_martingale_test,
        martingale_test,
    )

    np.random.seed(42)
    y = np.cumsum(np.random.randn(50) * 0.01)

    # Test sub_martingale_test - returns array, not float
    result = sub_martingale_test(y, window=20)
    assert len(result) == len(y)

    # Test super_martingale_test - returns array
    result = super_martingale_test(y, window=20)
    assert len(result) == len(y)

    # Test martingale_test - returns tuple of two arrays
    sub_result, sup_result = martingale_test(y, window=20)
    assert len(sub_result) == len(y)
    assert len(sup_result) == len(y)
