"""
Unit tests for SampleWeightCalculator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from afml import SampleWeightCalculator


@pytest.fixture
def sample_events():
    """Create sample events with t1 column for testing."""
    np.random.seed(42)
    n_events = 50

    start_time = datetime(2023, 1, 1)
    times = [start_time + timedelta(hours=i * 4) for i in range(n_events)]

    # Create events with varying end times
    event_times = []
    t1_times = []

    for i, t in enumerate(times):
        event_times.append(t)
        # End time is between 1-3 bars after start
        end_bar = min(i + np.random.randint(1, 4), n_events - 1)
        t1_times.append(times[end_bar])

    df = pd.DataFrame(
        {
            "datetime": event_times,
            "close": 4000.0 + np.cumsum(np.random.randn(n_events) * 0.001),
            "t1": pd.DatetimeIndex(t1_times),
            "label": np.random.choice([-1, 0, 1], size=n_events),
            "ret": np.random.randn(n_events) * 0.001,
        }
    )
    df.set_index("datetime", inplace=True)

    return df


class TestSampleWeightCalculator:
    """Tests for SampleWeightCalculator class."""

    def test_initialization_default(self):
        """Test default initialization."""
        calculator = SampleWeightCalculator()
        assert calculator.decay == 0.9
        assert calculator.concurrency_ is None
        assert calculator.uniqueness_ is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        calculator = SampleWeightCalculator(decay=0.8)
        assert calculator.decay == 0.8

    def test_fit(self, sample_events):
        """Test fit method."""
        calculator = SampleWeightCalculator()
        result = calculator.fit(sample_events)

        assert result is calculator
        assert calculator.concurrency_ is not None
        assert calculator.uniqueness_ is not None
        assert isinstance(calculator.concurrency_, pd.Series)
        assert isinstance(calculator.uniqueness_, pd.Series)

    def test_transform(self, sample_events):
        """Test transform method."""
        calculator = SampleWeightCalculator(decay=0.9)
        calculator.fit(sample_events)
        result = calculator.transform(sample_events)

        assert isinstance(result, pd.DataFrame)
        assert "sample_weight" in result.columns
        assert "avg_uniqueness" in result.columns
        assert len(result) == len(sample_events)

    def test_fit_transform(self, sample_events):
        """Test fit_transform method."""
        calculator = SampleWeightCalculator()
        result = calculator.fit_transform(sample_events)

        assert isinstance(result, pd.DataFrame)
        assert "sample_weight" in result.columns
        assert "avg_uniqueness" in result.columns

    def test_uniqueness_values(self, sample_events):
        """Test that uniqueness values are reasonable."""
        calculator = SampleWeightCalculator()
        calculator.fit(sample_events)

        uniqueness = calculator.uniqueness_

        # All uniqueness values should be between 0 and 1
        assert (uniqueness > 0).all()
        assert (uniqueness <= 1).all()

    def test_concurrency_values(self, sample_events):
        """Test that concurrency values are reasonable."""
        calculator = SampleWeightCalculator()
        calculator.fit(sample_events)

        concurrency = calculator.concurrency_

        # All concurrency values should be >= 1
        assert (concurrency >= 1).all()

    def test_sklearn_compatible(self, sample_events):
        """Test sklearn compatibility."""
        calculator = SampleWeightCalculator()

        # fit should return self
        fitted = calculator.fit(sample_events)
        assert fitted is calculator

        # transform should work after fit
        result = calculator.transform(sample_events)
        assert isinstance(result, pd.DataFrame)


class TestSampleWeightCalculatorEdgeCases:
    """Edge case tests for SampleWeightCalculator."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        calculator = SampleWeightCalculator()
        df = pd.DataFrame(columns=["t1"])

        with pytest.raises(Exception):
            calculator.fit(df)

    def test_single_event(self):
        """Test with single event."""
        start_time = datetime(2023, 1, 1)
        df = pd.DataFrame(
            {
                "datetime": [start_time],
                "t1": [start_time + timedelta(hours=4)],
                "close": [4000.0],
            }
        )
        df.set_index("datetime", inplace=True)

        calculator = SampleWeightCalculator()
        calculator.fit(df)

        # Should handle without error
        assert calculator.concurrency_ is not None
        assert calculator.uniqueness_ is not None

    def test_no_overlapping_events(self):
        """Test with non-overlapping events."""
        start_time = datetime(2023, 1, 1)
        times = [
            start_time + timedelta(hours=i * 10) for i in range(5)
        ]  # Non-overlapping

        df = pd.DataFrame(
            {
                "datetime": times,
                "t1": [t + timedelta(hours=4) for t in times],
                "close": 4000.0 + np.arange(5),
            }
        )
        df.set_index("datetime", inplace=True)

        calculator = SampleWeightCalculator()
        calculator.fit(df)

        # All uniqueness should be 1.0 (no overlap)
        assert (calculator.uniqueness_ == 1.0).all()

    def test_all_overlapping_events(self):
        """Test with all events overlapping."""
        start_time = datetime(2023, 1, 1)
        base_time = start_time

        df = pd.DataFrame(
            {
                "datetime": [base_time] * 5,
                "t1": [base_time + timedelta(hours=20)] * 5,
                "close": 4000.0,
            }
        )
        df.set_index("datetime", inplace=True)

        calculator = SampleWeightCalculator()
        calculator.fit(df)

        # Uniqueness should be 1/5 = 0.2
        assert (calculator.uniqueness_ == 0.2).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
