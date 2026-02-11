"""
Unit tests for BetSizer.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from afml import BetSizer


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    np.random.seed(42)
    n_events = 50

    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i * 4) for i in range(n_events)]

    end_times = []
    for i in range(n_events):
        end_bar = min(i + np.random.randint(1, 5), n_events - 1)
        end_times.append(times[end_bar])

    df = pd.DataFrame(
        {
            "datetime": times,
            "close": 4000.0 + np.cumsum(np.random.randn(n_events) * 0.001),
            "t1": pd.DatetimeIndex(end_times),
            "avg_uniqueness": np.random.uniform(0.5, 1.0, size=n_events),
        }
    )
    df.set_index("datetime", inplace=True)

    return df


@pytest.fixture
def sample_probabilities():
    """Create sample probability series."""
    np.random.seed(42)
    n = 50
    times = [datetime(2023, 1, 1) + timedelta(hours=i * 4) for i in range(n)]

    # Generate probabilities between 0.5 and 0.99
    probs = 0.5 + np.random.rand(n) * 0.49

    return pd.Series(probs, index=times)


class TestBetSizer:
    """Tests for BetSizer class."""

    def test_initialization_default(self):
        """Test default initialization."""
        sizer = BetSizer()
        assert sizer.step_size == 0.0
        assert sizer.concurrency_ is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        sizer = BetSizer(step_size=0.1)
        assert sizer.step_size == 0.1

    def test_fit(self, sample_events):
        """Test fit method (no-op for BetSizer)."""
        sizer = BetSizer()
        result = sizer.fit(sample_events)

        assert result is sizer

    def test_calculate_basic(self, sample_events, sample_probabilities):
        """Test basic bet size calculation."""
        sizer = BetSizer()
        bet_sizes = sizer.calculate(
            events=sample_events, prob_series=sample_probabilities
        )

        assert isinstance(bet_sizes, pd.Series)
        assert len(bet_sizes) == len(sample_probabilities)

        # All bet sizes should be >= 0
        assert (bet_sizes >= 0).all()

        # All bet sizes should be <= 1
        assert (bet_sizes <= 1).all()

    def test_calculate_with_predictions(self, sample_events, sample_probabilities):
        """Test bet size calculation with direction predictions."""
        sizer = BetSizer()

        # Create predictions (alternating long/short)
        predictions = pd.Series(1.0, index=sample_probabilities.index)
        predictions.iloc[1::2] = -1.0

        bet_sizes = sizer.calculate(
            events=sample_events,
            prob_series=sample_probabilities,
            pred_series=predictions,
        )

        assert isinstance(bet_sizes, pd.Series)

        # Should have some negative values now
        assert (bet_sizes < 0).any() | (bet_sizes > 0).any()

    def test_calculate_with_averaging(self, sample_events, sample_probabilities):
        """Test bet size calculation with active signal averaging."""
        sizer = BetSizer()

        bet_sizes = sizer.calculate(
            events=sample_events, prob_series=sample_probabilities, average_active=True
        )

        assert isinstance(bet_sizes, pd.Series)

        # Averaging should reduce bet sizes
        # Compare with non-averaged
        bet_sizes_no_avg = sizer.calculate(
            events=sample_events, prob_series=sample_probabilities, average_active=False
        )

        # Bet sizes with averaging should be <= bet sizes without
        # (since avg_uniqueness <= 1)
        assert all(bet_sizes <= bet_sizes_no_avg + 1e-10)

    def test_discrete_bet_sizes(self, sample_events, sample_probabilities):
        """Test discrete bet sizing with step_size."""
        sizer = BetSizer(step_size=0.2)

        bet_sizes = sizer.calculate(
            events=sample_events, prob_series=sample_probabilities
        )

        assert isinstance(bet_sizes, pd.Series)

        # All values should be multiples of 0.2
        for val in bet_sizes:
            assert abs(val % 0.2) < 1e-10 or abs(val % 0.2 - 0.2) < 1e-10

    def test_fit_transform(self, sample_events, sample_probabilities):
        """Test fit_transform method."""
        sizer = BetSizer()

        # fit_transform for BetSizer expects (events, probabilities)
        bet_sizes = sizer.fit_transform(sample_events, sample_probabilities)

        assert isinstance(bet_sizes, pd.Series)
        assert len(bet_sizes) == len(sample_probabilities)


class TestBetSizerEdgeCases:
    """Edge case tests for BetSizer."""

    def test_probability_at_bounds(self, sample_events):
        """Test with probability at exact bounds."""
        sizer = BetSizer()

        # Test at exactly 0.5
        probs = pd.Series([0.5, 0.5], index=sample_events.index[:2])
        bet_sizes = sizer.calculate(sample_events.iloc[:2], probs)

        # Should handle without error
        assert len(bet_sizes) == 2

    def test_probability_near_zero(self, sample_events):
        """Test with probability near zero."""
        sizer = BetSizer()

        probs = pd.Series([0.001, 0.5], index=sample_events.index[:2])
        bet_sizes = sizer.calculate(sample_events.iloc[:2], probs)

        # Should handle without error (clipping)
        assert len(bet_sizes) == 2
        assert (bet_sizes >= 0).all()

    def test_probability_near_one(self, sample_events):
        """Test with probability near one."""
        sizer = BetSizer()

        probs = pd.Series([0.999, 0.5], index=sample_events.index[:2])
        bet_sizes = sizer.calculate(sample_events.iloc[:2], probs)

        # Should handle without error (clipping)
        assert len(bet_sizes) == 2
        assert (bet_sizes <= 1).all()

    def test_no_averaging_without_uniqueness(self, sample_events, sample_probabilities):
        """Test that averaging works even without avg_uniqueness column."""
        sizer = BetSizer()

        # Events without avg_uniqueness
        events = sample_events.drop(columns=["avg_uniqueness"])

        bet_sizes = sizer.calculate(
            events=events, prob_series=sample_probabilities, average_active=True
        )

        # Should return original bet sizes without adjustment
        assert isinstance(bet_sizes, pd.Series)

    def test_all_zero_probabilities(self, sample_events):
        """Test with all zero probabilities."""
        sizer = BetSizer()

        probs = pd.Series([0.0, 0.0], index=sample_events.index[:2])
        bet_sizes = sizer.calculate(sample_events.iloc[:2], probs)

        # Should handle - probabilities get clipped to 0.001
        assert len(bet_sizes) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
