"""
Pytest configuration and fixtures for AFML tests.
"""

import numpy as np
import polars as pl
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_dollar_bars() -> pl.DataFrame:
    """Create sample dollar bars for testing."""
    np.random.seed(42)
    n_bars = 200

    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i) for i in range(n_bars)]

    close = 4000.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
    volume = np.random.randint(1000, 10000, n_bars).astype(float)

    return pl.DataFrame(
        {
            "datetime": times,
            "close": close,
            "open": close - np.random.randn(n_bars) * 0.1,
            "high": close + np.abs(np.random.randn(n_bars) * 0.2),
            "low": close - np.abs(np.random.randn(n_bars) * 0.2),
            "volume": volume,
        }
    )


@pytest.fixture
def df() -> pl.DataFrame:
    """Create sample OHLCV data for integration tests."""
    np.random.seed(42)
    n_rows = 1000

    start_time = datetime(2020, 1, 1)
    dates = [start_time + timedelta(hours=i) for i in range(n_rows)]

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    open_prices = close + np.random.randn(n_rows) * 0.1
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n_rows) * 0.1)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n_rows) * 0.1)
    volume = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
