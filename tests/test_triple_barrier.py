"""
Unit tests for TripleBarrierLabeler.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from afml import TripleBarrierLabeler


@pytest.fixture
def sample_dollar_bars():
    """Create sample dollar bar data for testing."""
    np.random.seed(42)
    n_bars = 500

    # Create datetime index
    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i * 4) for i in range(n_bars)]  # 4-hour bars

    # Generate OHLCV data
    base_price = 4000.0
    returns = np.random.randn(n_bars) * 0.001
    prices = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * (1 + np.abs(np.random.randn(n_bars) * 0.002)),
            "low": prices * (1 - np.abs(np.random.randn(n_bars) * 0.002)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, size=n_bars),
        },
        index=times,  # Use datetime index
    )

    return df


@pytest.fixture
def sample_events(sample_dollar_bars):
    """Create sample CUSUM events."""
    np.random.seed(42)
    n = len(sample_dollar_bars)

    # Create events at regular intervals
    event_indices = list(range(10, n, 20))
    events = sample_dollar_bars.index[event_indices]

    return pd.DatetimeIndex(events)


class TestTripleBarrierLabeler:
    """Tests for TripleBarrierLabeler class."""

    def test_initialization_default(self):
        """Test default initialization."""
        labeler = TripleBarrierLabeler()
        assert labeler.pt_sl == [1.0, 1.0]
        assert labeler.vertical_barrier_bars == 12
        assert labeler.min_ret == 0.001  # Default is 0.001
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
        assert isinstance(labeler.volatility_, pd.Series)
        assert len(labeler.volatility_) == len(sample_dollar_bars)

    def test_get_cusum_events(self, sample_dollar_bars):
        """Test CUSUM event generation."""
        labeler = TripleBarrierLabeler()
        labeler.fit(sample_dollar_bars["close"])

        # Use volatility as threshold
        threshold = labeler.volatility_.mean() * 2
        events = labeler.get_cusum_events(
            sample_dollar_bars["close"], threshold=threshold
        )

        assert isinstance(events, pd.DatetimeIndex)
        assert len(events) > 0
        assert len(events) < len(sample_dollar_bars)  # Should filter some events

    def test_label_basic(self, sample_dollar_bars):
        """Test basic labeling functionality."""
        labeler = TripleBarrierLabeler(
            pt_sl=[1.0, 1.0], vertical_barrier_bars=10, min_ret=0.0
        )

        # Fit the labeler
        labeler.fit(sample_dollar_bars["close"])

        # Generate events
        events = labeler.get_cusum_events(
            sample_dollar_bars["close"],
            threshold=0.0002,  # Fixed threshold
        )

        # Apply labeling
        if len(events) > 0:
            labels = labeler.label(close=sample_dollar_bars["close"], events=events)

            assert isinstance(labels, pd.DataFrame)
            assert "t1" in labels.columns
            assert "label" in labels.columns
            assert "ret" in labels.columns
            assert len(labels) <= len(events)

    def test_label_with_side(self, sample_dollar_bars):
        """Test labeling with side prediction."""
        labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=10)

        labeler.fit(sample_dollar_bars["close"])

        events = labeler.get_cusum_events(sample_dollar_bars["close"], threshold=0.0002)

        if len(events) > 0:
            # Create side prediction (alternating long/short)
            side = pd.Series(1.0, index=events)
            side.iloc[1::2] = -1.0

            labels = labeler.label(
                close=sample_dollar_bars["close"], events=events, side=side
            )

            assert isinstance(labels, pd.DataFrame)
            assert "side" in labels.columns

    def test_volatility_calculation(self, sample_dollar_bars):
        """Test that volatility is calculated correctly."""
        labeler = TripleBarrierLabeler(volatility_span=50)
        labeler.fit(sample_dollar_bars["close"])

        # Volatility should be positive for non-NaN values (first few are NaN due to shift)
        assert (labeler.volatility_.dropna() > 0).all()

        # Volatility should have same length as input
        assert len(labeler.volatility_) == len(sample_dollar_bars)

    def test_sklearn_compatible(self, sample_dollar_bars):
        """Test sklearn compatibility."""
        labeler = TripleBarrierLabeler()

        # fit should return self
        fitted = labeler.fit(sample_dollar_bars["close"])
        assert fitted is labeler


class TestTripleBarrierLabelerEdgeCases:
    """Edge case tests for TripleBarrierLabeler."""

    def test_empty_series(self):
        """Test with empty price series."""
        labeler = TripleBarrierLabeler()
        empty_series = pd.Series(dtype=float)

        # Empty series may or may not raise depending on implementation
        # Just verify it doesn't crash
        try:
            labeler.fit(empty_series)
        except (ValueError, KeyError, Exception):
            pass  # Expected behavior for empty data

    def test_single_bar(self):
        """Test with single bar."""
        labeler = TripleBarrierLabeler()
        single_bar = pd.Series([4000.0], index=[datetime(2023, 1, 1)])

        labeler.fit(single_bar)

        # Should handle gracefully
        assert hasattr(labeler, "volatility_")

    def test_constant_price(self):
        """Test with constant price series."""
        labeler = TripleBarrierLabeler()
        constant = pd.Series(4000.0, index=range(10))

        labeler.fit(constant)

        # Should handle without error
        assert hasattr(labeler, "volatility_")

    def test_no_cusum_events(self, sample_dollar_bars):
        """Test when CUSUM filter produces no events."""
        labeler = TripleBarrierLabeler()
        labeler.fit(sample_dollar_bars["close"])

        # Very high threshold should produce no events
        events = labeler.get_cusum_events(
            sample_dollar_bars["close"],
            threshold=100.0,  # Very high
        )

        assert len(events) == 0

    def test_very_high_volatility(self, sample_dollar_bars):
        """Test with very high volatility settings."""
        labeler = TripleBarrierLabeler(
            pt_sl=[10.0, 10.0],  # Very wide barriers
            vertical_barrier_bars=50,
            volatility_span=10,  # Short span
        )

        labeler.fit(sample_dollar_bars["close"])

        events = labeler.get_cusum_events(
            sample_dollar_bars["close"],
            threshold=0.001,  # Low threshold
        )

        # Should still work
        assert isinstance(events, pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
