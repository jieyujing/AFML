"""
Unit tests for TripleBarrierLabeler.
"""

import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta

from afml import TripleBarrierLabeler


@pytest.fixture
def sample_dollar_bars():
    """Create sample dollar bar data for testing."""
    np.random.seed(42)
    n_bars = 500

    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i * 4) for i in range(n_bars)]

    base_price = 4000.0
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price * np.cumprod(1 + returns)

    df = pl.DataFrame(
        {
            "datetime": times,
            "open": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, size=n_bars),
        }
    )

    return df


@pytest.fixture
def sample_events(sample_dollar_bars):
    """Create sample CUSUM events."""
    np.random.seed(42)
    n = len(sample_dollar_bars)

    event_indices = list(range(10, n, 20))
    events = sample_dollar_bars["datetime"][event_indices]

    return events


class TestTripleBarrierLabeler:
    """Tests for TripleBarrierLabeler class."""

    def test_initialization_default(self):
        """Test default initialization."""
        labeler = TripleBarrierLabeler()
        assert labeler.pt_sl == [1.0, 1.0]
        assert labeler.vertical_barrier_bars == 12
        assert labeler.min_ret == 0.001
        assert labeler.volatility_span == 100

    def test_initialization_custom(self):
        """Test custom initialization."""
        labeler = TripleBarrierLabeler(
            pt_sl=[2.0, 2.0],
            vertical_barrier_bars=20,
            min_ret=0.001,
            volatility_span=50,
        )
        assert labeler.pt_sl == [2.0, 2.0]
        assert labeler.vertical_barrier_bars == 20
        assert labeler.min_ret == 0.001
        assert labeler.volatility_span == 50

    def test_fit(self, sample_dollar_bars):
        """Test fit method."""
        labeler = TripleBarrierLabeler()
        result = labeler.fit(sample_dollar_bars["close"])

        assert result is labeler
        assert hasattr(labeler, "volatility_")
        assert labeler.volatility_ is not None
        assert len(labeler.volatility_) == len(sample_dollar_bars)

    def test_get_cusum_events(self, sample_dollar_bars):
        """Test CUSUM event generation."""
        labeler = TripleBarrierLabeler()
        labeler.fit(sample_dollar_bars["close"])

        threshold = labeler.volatility_.mean() * 2
        events = labeler.get_cusum_events(
            sample_dollar_bars["close"], threshold=threshold
        )

        assert isinstance(events, pl.DataFrame)
        assert len(events) > 0

    def test_label_basic(self, sample_dollar_bars):
        """Test basic labeling functionality."""
        labeler = TripleBarrierLabeler(
            pt_sl=[1.0, 1.0], vertical_barrier_bars=10, min_ret=0.0
        )

        labeler.fit(sample_dollar_bars["close"])

        events = labeler.get_cusum_events(
            sample_dollar_bars["close"],
            threshold=0.0002,
        )

        if len(events) > 0:
            labels = labeler.label(close=sample_dollar_bars["close"], events=events)

            assert isinstance(labels, pl.DataFrame)
            assert "t1" in labels.columns
            assert "label" in labels.columns
            assert "tr" in labels.columns
            assert len(labels) <= len(events)

    def test_volatility_calculation(self, sample_dollar_bars):
        """Test that volatility is calculated correctly."""
        labeler = TripleBarrierLabeler(volatility_span=50)
        labeler.fit(sample_dollar_bars["close"])

        assert labeler.volatility_ is not None
        assert len(labeler.volatility_) == len(sample_dollar_bars)

    def test_sklearn_compatible(self, sample_dollar_bars):
        """Test sklearn compatibility."""
        labeler = TripleBarrierLabeler()

        fitted = labeler.fit(sample_dollar_bars["close"])
        assert fitted is labeler


class TestTripleBarrierLabelerEdgeCases:
    """Edge case tests for TripleBarrierLabeler."""

    def test_empty_series(self):
        """Test with empty price series."""
        labeler = TripleBarrierLabeler()
        empty_series = pl.Series("close", [])

        try:
            labeler.fit(empty_series)
        except Exception:
            pass

    def test_single_bar(self):
        """Test with single bar."""
        labeler = TripleBarrierLabeler()
        single_bar = pl.Series("close", [4000.0])

        labeler.fit(single_bar)

        assert hasattr(labeler, "volatility_")

    def test_constant_price(self):
        """Test with constant price series."""
        labeler = TripleBarrierLabeler()
        constant = pl.Series("close", [4000.0] * 10)

        labeler.fit(constant)

        assert hasattr(labeler, "volatility_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
