"""
Sample Weight Calculator for Financial Machine Learning.

This module implements sample weighting strategies from AFML Chapter 4,
calculating weights based on concurrency and uniqueness.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base import ProcessorMixin


class SampleWeightCalculator(ProcessorMixin):
    """
    Calculates sample weights for financial ML.

    Addresses the non-IID nature of financial data where overlapping
    labels introduce redundancy.

    Attributes:
        decay: Time decay factor for weights
        concurrency_: Computed concurrency after fitting
        uniqueness_: Computed average uniqueness after fitting

    Example:
        >>> calculator = SampleWeightCalculator(decay=0.9)
        >>> calculator.fit(events_df)
        >>> weights = calculator.transform(df)
    """

    def __init__(self, decay: float = 0.9):
        """
        Initialize the SampleWeightCalculator.

        Args:
            decay: Time decay factor (default: 0.9)
        """
        super().__init__()
        self.decay = decay
        self.concurrency_: Optional[pd.Series] = None
        self.uniqueness_: Optional[pd.Series] = None

    def fit(
        self, events: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "SampleWeightCalculator":
        """
        Compute concurrency and uniqueness metrics.

        Args:
            events: DataFrame with 't1' (barrier time) column
            y: Ignored

        Returns:
            self
        """
        self.concurrency_ = self._calculate_concurrency(events)
        self.uniqueness_ = self._calculate_uniqueness(events)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weight columns to DataFrame.

        Args:
            df: DataFrame to add weights to

        Returns:
            DataFrame with 'sample_weight' and 'avg_uniqueness' columns
        """
        if self.uniqueness_ is None:
            raise ValueError("Calculator has not been fitted. Call fit() first.")

        df = df.copy()
        df["sample_weight"] = self._compute_weights(df)
        df["avg_uniqueness"] = self.uniqueness_
        return df

    def _calculate_concurrency(self, events: pd.DataFrame) -> pd.Series:
        """Calculate concurrency (number of overlapping events at each time)."""
        label_endtime = events["t1"].fillna(events.index[-1])
        label_endtime = label_endtime[label_endtime >= events.index[0]]

        count = pd.Series(0, index=events.index)
        for t_in, t_out in events["t1"].items():
            count.loc[t_in:t_out] += 1

        return count

    def _calculate_uniqueness(self, events: pd.DataFrame) -> pd.Series:
        """Calculate average uniqueness for each event."""
        concurrency = self.concurrency_
        if concurrency is None:
            concurrency = self._calculate_concurrency(events)

        uniqueness = pd.Series(index=events.index, dtype="float64")
        for t_in, t_out in events["t1"].items():
            uniqueness.loc[t_in] = (1.0 / concurrency.loc[t_in:t_out]).mean()

        return uniqueness

    def _compute_weights(self, df: pd.DataFrame) -> pd.Series:
        """Compute final sample weights."""
        if "ret" in df.columns and "avg_uniqueness" in df.columns:
            ret = np.log(df["close"]).diff() if "close" in df.columns else df["ret"]
            weights = df["avg_uniqueness"] * np.abs(ret) * self.decay
        else:
            weights = (
                df["avg_uniqueness"]
                if "avg_uniqueness" in df.columns
                else pd.Series(1.0, index=df.index)
            )

        return weights.fillna(0)

    def fit_transform(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame
            y: Ignored

        Returns:
            DataFrame with weight columns
        """
        self.fit(df)
        return self.transform(df)
