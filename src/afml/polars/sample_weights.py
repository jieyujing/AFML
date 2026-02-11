"""
Polars Sample Weight Calculator for Financial Machine Learning.

This module implements sample weight calculation using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from polars import DataFrame, Series

from afml.base import ProcessorMixin


class PolarsSampleWeightCalculator(ProcessorMixin):
    """
    Calculator for sample weights based on uniqueness and time decay.

    Weights account for:
    - Concurrency: Sample overlap during events
    - Average Uniqueness: Independent information in each sample
    - Time Decay: Older samples have less weight

    This implementation uses Polars for improved performance on large datasets,
    with support for lazy evaluation.

    Attributes:
        decay: Time decay factor (default 0.9)
        concurrency_window: Window for concurrency calculation

    Example:
        >>> calculator = PolarsSampleWeightCalculator(decay=0.9)
        >>> weights = calculator.fit_transform(events, labels)
    """

    def __init__(
        self,
        decay: float = 0.9,
        concurrency_window: int = 100,
        *,
        lazy: bool = False,
    ):
        """
        Initialize the PolarsSampleWeightCalculator.

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
    ) -> "PolarsSampleWeightCalculator":
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
        Compute uniqueness for each event.

        Uniqueness measures how much independent information
        each sample contributes.

        Args:
            events: DataFrame with t1 (vertical barrier times)

        Returns:
            Series with uniqueness values
        """
        if "t1" not in events.columns:
            return Series(values=np.ones(len(events)))

        t1 = events["t1"]

        n_events = len(events)
        uniqueness = np.ones(n_events)

        for i in range(n_events):
            t0_i = i
            t1_i = t1[i] if i < len(t1) else t0_i

            concurrent = 0
            for j in range(n_events):
                if i != j:
                    t0_j = j
                    t1_j = t1[j] if j < len(t1) else t0_j

                    if t0_i <= t0_j < t1_i or t0_i <= t1_j < t1_i:
                        concurrent += 1

            if concurrent > 0:
                uniqueness[i] = 1.0 / concurrent

        return Series(values=uniqueness)

    def _compute_concurrency(
        self,
        events: DataFrame,
    ) -> Series:
        """
        Calculate concurrency for each timestamp.

        Args:
            events: DataFrame with t1 column

        Returns:
            Series with concurrency values
        """
        if "t1" not in events.columns:
            return Series(values=np.ones(len(events)))

        t1 = events["t1"]

        n = len(events)
        concurrency = np.zeros(n)

        for i in range(n):
            t0_i = i
            t1_i = t1[i] if i < len(t1) else t0_i

            for j in range(n):
                if i != j:
                    t0_j = j
                    t1_j = t1[j] if j < len(t1) else t0_j

                    if t0_i <= t0_j < t1_i or t0_i <= t1_j < t1_i:
                        concurrency[i] += 1

        return Series(values=concurrency + 1)

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

        if labels is not None and "label" in labels.columns:
            result = result.with_columns(
                pl.col("weight").alias("weight_positive"),
            )
            result = result.with_columns(
                pl.when(pl.col("label") == 1)
                .then(pl.col("weight"))
                .otherwise(0.0)
                .alias("weight_negative")
            )

        return result

    def _compute_decay_weights(self, n: int) -> Series:
        """
        Compute time decay weights.

        Args:
            n: Number of samples

        Returns:
            Series with decay weights
        """
        weights = np.zeros(n)
        for i in range(n):
            weights[i] = self.decay ** (n - 1 - i)

        return Series(values=weights)

    def fit_transform(
        self,
        events: DataFrame,
        labels: Optional[DataFrame] = None,
        y: Optional[Any] = None,
    ) -> DataFrame:
        """
        Fit and transform in one step.

        Args:
            events: DataFrame with event timestamps
            labels: Optional DataFrame with labels
            y: Ignored

        Returns:
            DataFrame with sample weights
        """
        self.fit(events, labels)
        return self.transform(events, labels)

    def get_weight_info(self) -> Dict[str, Any]:
        """
        Get information about weight calculation.

        Returns:
            Dict with configuration and statistics
        """
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
