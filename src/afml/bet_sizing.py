"""
Bet Sizer for Financial Machine Learning.

This module implements probability-based bet sizing from AFML Chapter 10,
converting model probabilities into position sizes.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Optional

from .base import ProcessorMixin


class BetSizer(ProcessorMixin):
    """
    Computes bet sizes from prediction probabilities.

    Uses CDF-based mapping to convert probabilities to position sizes,
    with optional concurrency adjustment for overlapping positions.

    Attributes:
        step_size: Discrete step size (0.0 for continuous)
        concurrency_: Concurrency factor for overlapping events

    Example:
        >>> sizer = BetSizer(step_size=0.0)
        >>> bet_sizes = sizer.calculate(events, probabilities, average_active=True)
    """

    def __init__(self, step_size: float = 0.0):
        """
        Initialize the BetSizer.

        Args:
            step_size: Discrete step size (0.0 for continuous sizing)
        """
        super().__init__()
        self.step_size = step_size
        self.concurrency_: Optional[pd.Series] = None

    def calculate(
        self,
        events: pd.DataFrame,
        prob_series: pd.Series,
        pred_series: Optional[pd.Series] = None,
        average_active: bool = False,
    ) -> pd.Series:
        """
        Calculate bet sizes from probabilities.

        Args:
            events: DataFrame with 't1' (barrier time) column
            prob_series: Series of probabilities from meta-model
            pred_series: Primary model predictions for direction
            average_active: Whether to average concurrent signals

        Returns:
            Series of bet sizes (0 to 1)
        """
        p = prob_series.copy()
        p = p.clip(0.001, 0.999)

        z = (p - 0.5) / np.sqrt(p * (1 - p))
        m = 2 * norm.cdf(z) - 1

        bet_sizes = pd.Series(m, index=p.index).clip(lower=0.0)

        if pred_series is not None:
            bet_sizes = bet_sizes * pred_series.loc[bet_sizes.index]

        if self.step_size > 0:
            bet_sizes = (bet_sizes / self.step_size).round() * self.step_size

        if average_active:
            return self._avg_active_signals(events, bet_sizes)

        return bet_sizes

    def _avg_active_signals(self, events: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Average active signals across concurrent events."""
        if "avg_uniqueness" in events.columns:
            return signal * events["avg_uniqueness"]
        return signal

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BetSizer":
        """
        Fit is a no-op for BetSizer.

        Args:
            X: Ignored
            y: Ignored

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform is not applicable for BetSizer.

        Use calculate() method instead.
        """
        raise NotImplementedError("Use calculate() method for bet sizing.")

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Fit and calculate in one step.

        Args:
            X: Events DataFrame
            y: Probabilities Series

        Returns:
            Bet sizes Series
        """
        self.fit(X)
        return self.calculate(X, y)
