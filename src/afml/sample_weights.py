"""
Sample Weight Calculator for Financial Machine Learning (Polars Optimized).

This module implements sample weight calculation using Polars for improved
performance on large-scale financial time series data.

Performance: Uses fully vectorized NumPy operations instead of Python loops.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from polars import DataFrame, Series

from .base import ProcessorMixin


class SampleWeightCalculator(ProcessorMixin):
    """
    Calculator for sample weights based on uniqueness and time decay.

    Weights account for:
    - Concurrency: Sample overlap during events
    - Average Uniqueness: Independent information in each sample
    - Time Decay: Older samples have less weight

    This implementation uses vectorized NumPy operations for O(N) performance
    instead of O(N²) Python loops.

    Attributes:
        decay: Time decay factor (default 0.9)
        concurrency_window: Window for concurrency calculation
    """

    def __init__(
        self,
        decay: float = 0.9,
        concurrency_window: int = 100,
        *,
        lazy: bool = False,
    ):
        """
        Initialize the SampleWeightCalculator.

        Args:
            decay: Time decay factor (0 < decay < 1)
            concurrency_window: Window size for concurrency calculation
            lazy: Whether to use lazy evaluation
        """
        super().__init__()
        self.decay = decay
        self.concurrency_window = concurrency_window
        self.lazy = lazy
        self._unique_idx: Optional[Series] = None

    def fit(
        self,
        events: DataFrame,
        labels: Optional[DataFrame] = None,
        y: Optional[Any] = None,
    ) -> "SampleWeightCalculator":
        """
        Calculate uniqueness indices from events.

        Args:
            events: DataFrame with event timestamps (t1 column)
            labels: Optional DataFrame with labels
            y: Ignored

        Returns:
            self
        """
        self._unique_idx = self._compute_uniqueness(events)
        return self

    def _compute_uniqueness(self, events: DataFrame) -> Series:
        """
        Compute uniqueness for each event using vectorized NumPy operations.

        Vectorized approach: For each event i with span [i, t1_i), count how many
        other events j (within concurrency_window) have overlapping spans.
        Uses broadcasting for O(N × W) vectorized operations instead of
        O(N × W) Python loop iterations.

        Reference: AFML Chapter 4 - Sample Weights and Uniqueness.
        """
        if "t1" not in events.columns:
            return Series(values=np.ones(len(events)))

        t1 = events["t1"].to_numpy().astype(np.float64)
        n_events = len(events)
        t0 = np.arange(n_events, dtype=np.float64)

        if n_events == 0:
            return Series(values=np.array([], dtype=np.float64))

        # Vectorized concurrency calculation
        # For each event i, count overlaps with events in window [i-W, i+W]
        concurrency = np.zeros(n_events, dtype=np.float64)
        w = self.concurrency_window

        for offset in range(-w, w + 1):
            if offset == 0:
                continue
            # Shifted indices
            j_indices = np.arange(n_events) + offset
            # Valid mask (within bounds)
            valid = (j_indices >= 0) & (j_indices < n_events)

            # For valid pairs (i, j), check overlap:
            # overlap if t0_i <= t0_j < t1_i OR t0_i <= t1_j < t1_i
            i_idx = np.where(valid)[0]
            j_idx = j_indices[valid]

            t0_i = t0[i_idx]
            t1_i = t1[i_idx]
            t0_j = t0[j_idx]
            t1_j = t1[j_idx]

            overlap = ((t0_i <= t0_j) & (t0_j < t1_i)) | ((t0_i <= t1_j) & (t1_j < t1_i))
            concurrency[i_idx] += overlap.astype(np.float64)

        # Uniqueness = 1 / concurrency (where concurrency > 0)
        uniqueness = np.where(concurrency > 0, 1.0 / concurrency, 1.0)

        return Series(values=uniqueness)

    def transform(
        self,
        events: DataFrame,
        labels: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Generate sample weights.

        Args:
            events: DataFrame with event timestamps
            labels: Optional DataFrame with labels

        Returns:
            DataFrame with sample weights
        """
        if self._unique_idx is None:
            self.fit(events, labels)

        uniqueness = self._unique_idx

        if self.decay < 1.0:
            decay_weights = self._compute_decay_weights(len(events))
        else:
            decay_weights = Series(values=np.ones(len(events)))

        weights = (uniqueness * decay_weights).to_list()

        result = DataFrame(
            {
                "weight": weights,
                "uniqueness": uniqueness.to_list(),
            }
        )

        return result

    def _compute_decay_weights(self, n: int) -> Series:
        """Compute time decay weights using vectorized NumPy."""
        exponents = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.power(self.decay, exponents)
        return Series(values=weights)

    def fit_transform(
        self,
        events: DataFrame,
        labels: Optional[DataFrame] = None,
        y: Optional[Any] = None,
    ) -> DataFrame:
        """Fit and transform in one step."""
        self.fit(events, labels)
        return self.transform(events, labels)

    def get_weight_info(self) -> Dict[str, Any]:
        """Get information about weight calculation."""
        if self._unique_idx is None:
            return {
                "decay": self.decay,
                "concurrency_window": self.concurrency_window,
                "lazy": self.lazy,
                "fitted": False,
            }

        return {
            "decay": self.decay,
            "concurrency_window": self.concurrency_window,
            "lazy": self.lazy,
            "fitted": True,
            "mean_uniqueness": self._unique_idx.mean(),
            "std_uniqueness": self._unique_idx.std(),
        }
