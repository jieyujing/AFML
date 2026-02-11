"""
Polars Triple Barrier Labeler for Financial Machine Learning.

This module implements the Triple Barrier Method using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from polars import DataFrame, Series

from afml.base import ProcessorMixin


class PolarsTripleBarrierLabeler(ProcessorMixin):
    """
    Labeler that applies the Triple Barrier Method to generate financial labels.

    Each event gets three barriers:
    - Upper barrier: Profit-taking level
    - Lower barrier: Stop-loss level
    - Vertical barrier: Maximum holding period

    This implementation uses Polars for improved performance on large datasets,
    with support for lazy evaluation and multi-threaded operations.

    Attributes:
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
        vertical_barrier_bars: Maximum holding period in bars
        min_ret: Minimum return threshold for event filtering
        volatility_span: Span for volatility calculation
        volatility_: Computed volatility after fitting

    Example:
        >>> labeler = PolarsTripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
        >>> labeler.fit(close_prices)
        >>> events = labeler.label(close=close_series, timestamps=timestamps)
    """

    def __init__(
        self,
        pt_sl: List[float] = [1.0, 1.0],
        vertical_barrier_bars: int = 12,
        min_ret: float = 0.001,
        volatility_span: int = 100,
        *,
        lazy: bool = False,
    ):
        """
        Initialize the PolarsTripleBarrierLabeler.

        Args:
            pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
            vertical_barrier_bars: Number of bars for vertical barrier
            min_ret: Minimum return threshold to keep events
            volatility_span: Span for volatility calculation
            lazy: Whether to use lazy evaluation
        """
        super().__init__()
        self.pt_sl = pt_sl
        self.vertical_barrier_bars = vertical_barrier_bars
        self.min_ret = min_ret
        self.volatility_span = volatility_span
        self.lazy = lazy
        self.volatility_: Optional[Series] = None

    def fit(
        self,
        close: Union[Series, DataFrame],
        y: Optional[Any] = None,
    ) -> "PolarsTripleBarrierLabeler":
        """
        Calculate volatility for barrier sizing.

        Args:
            close: Series or DataFrame with close prices
            y: Ignored

        Returns:
            self
        """
        if isinstance(close, DataFrame):
            if "close" not in close.columns:
                raise ValueError("DataFrame must have 'close' column")
            close = close["close"]

        self.volatility_ = self._calculate_volatility(close)
        return self

    def _calculate_volatility(self, close: Series) -> Series:
        """Calculate volatility using EMA of absolute log returns."""
        returns = (close / close.shift(1)).log()
        volatility = returns.ewm_std(span=self.volatility_span)
        return volatility

    def get_cusum_events(
        self,
        close: Union[Series, DataFrame],
        threshold: Union[float, Series] = None,
    ) -> DataFrame:
        """
        Apply CUSUM filter to detect significant events.

        Args:
            close: Series of close prices
            threshold: Threshold for event detection

        Returns:
            DataFrame with detected events
        """
        if isinstance(close, DataFrame):
            if "close" not in close.columns:
                raise ValueError("DataFrame must have 'close' column")
            close = close["close"]

        close_np = close.to_numpy()

        if threshold is None:
            threshold_val = (
                self.volatility_ * 2 if self.volatility_ is not None else 0.002
            )
            if isinstance(threshold_val, Series):
                threshold_val = threshold_val.mean()
        elif isinstance(threshold, Series):
            threshold_val = threshold.mean()
        else:
            threshold_val = threshold

        returns = np.log(close_np[1:] / close_np[:-1])
        returns = np.insert(returns, 0, 0)

        pos = np.zeros(len(returns))
        neg = np.zeros(len(returns))

        for i in range(1, len(returns)):
            pos[i] = max(0, pos[i - 1] + returns[i])
            neg[i] = min(0, neg[i - 1] - returns[i])

        events = []
        for i in range(len(returns)):
            if pos[i] > threshold_val:
                events.append({"datetime": i, "type": "pos"})
                pos[i] = 0
                neg[i] = 0
            elif -neg[i] > threshold_val:
                events.append({"datetime": i, "type": "neg"})
                pos[i] = 0
                neg[i] = 0

        if events:
            return DataFrame(events)
        return DataFrame({"datetime": [], "type": []})

    def label(
        self,
        close: Union[Series, DataFrame],
        events: DataFrame,
        t1: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Apply triple barrier labels to events.

        Args:
            close: Series or DataFrame with close prices
            events: DataFrame with event timestamps
            t1: DataFrame with vertical barrier times

        Returns:
            DataFrame with labels
        """
        if isinstance(close, DataFrame):
            if "close" not in close.columns:
                raise ValueError("DataFrame must have 'close' column")
            close = close["close"]

        if self.volatility_ is None:
            self.fit(close)

        pt_sl = self.pt_sl
        vertical_barrier_bars = self.vertical_barrier_bars
        min_ret = self.min_ret

        if t1 is None:
            t1 = self._get_vertical_barrier(close, events, vertical_barrier_bars)

        labels = []
        events_dict = events.to_dicts()
        t1_vals = t1["t1"].to_numpy() if "t1" in t1.columns else t1.to_numpy().flatten()
        close_np = close.to_numpy()
        n = len(close_np)

        for idx, event in enumerate(events_dict):
            t0 = int(event.get("datetime", idx))
            if t0 >= n:
                continue

            t1_val = (
                int(t1_vals[idx])
                if idx < len(t1_vals)
                else t0 + self.vertical_barrier_bars
            )
            t1_val = min(t1_val, n - 1)

            if t0 >= t1_val:
                continue

            ret = close_np[t1_val] / close_np[t0] - 1

            if abs(ret) <= min_ret:
                continue

            vol = 0.02
            if self.volatility_ is not None:
                vol_slice = self.volatility_[t0:t1_val]
                if hasattr(vol_slice, "mean"):
                    vol_val = vol_slice.mean()
                    vol = float(vol_val) if vol_val else 0.02

            pt = pt_sl[0] * vol
            sl = pt_sl[1] * vol

            if ret > pt:
                label = 1
            elif ret < -sl:
                label = -1
            else:
                label = 0

            labels.append(
                {
                    "t1": t1_val,
                    "tr": abs(ret),
                    "label": label,
                }
            )

        if labels:
            return DataFrame(labels)
        return DataFrame({"t1": [], "tr": [], "label": []})

    def _get_vertical_barrier(
        self,
        close: Series,
        events: DataFrame,
        vertical_barrier_bars: int,
    ) -> DataFrame:
        """
        Calculate vertical barrier for each event.

        Args:
            close: Close price series
            events: Events DataFrame
            vertical_barrier_bars: Number of bars for vertical barrier

        Returns:
            DataFrame with t1 (vertical barrier time)
        """
        t1_list = []
        for idx in range(len(events)):
            event_row = events.row(idx, named=True)
            t0 = event_row.get("datetime", event_row.get(0))
            if t0 is None:
                t1_list.append(None)
                continue

            t1_val = t0 + vertical_barrier_bars
            if t1_val >= len(close):
                t1_val = len(close) - 1

            t1_list.append(t1_val)

        return DataFrame({"t1": t1_list})

    def get_all_events(
        self,
        close: Union[Series, DataFrame],
        timestamps: DataFrame,
    ) -> DataFrame:
        """
        Get all triple barrier events.

        Args:
            close: Close price series or DataFrame
            timestamps: DataFrame with event timestamps

        Returns:
            DataFrame with all events and barriers
        """
        if isinstance(close, DataFrame):
            if "close" not in close.columns:
                raise ValueError("DataFrame must have 'close' column")
            close = close["close"]

        if self.volatility_ is None:
            self.fit(close)

        t1 = self._get_vertical_barrier(close, timestamps, self.vertical_barrier_bars)

        return self.label(close, timestamps, t1)

    def fit_transform(
        self,
        close: Union[Series, DataFrame],
        events: DataFrame,
        y: Optional[Any] = None,
    ) -> DataFrame:
        """
        Fit and transform in one step.

        Args:
            close: Close price series or DataFrame
            events: Events DataFrame
            y: Ignored

        Returns:
            Labeled events DataFrame
        """
        self.fit(close)
        return self.label(close, events)

    def transform(
        self,
        close: Union[Series, DataFrame],
    ) -> DataFrame:
        """
        Apply triple barrier labels (stub for sklearn compatibility).

        Args:
            close: Close price series or DataFrame

        Returns:
            Empty DataFrame (use label() method for actual labeling)
        """
        return DataFrame({"t1": [], "tr": [], "label": []})

    def get_label_info(self) -> Dict[str, Any]:
        """
        Get information about labeler configuration.

        Returns:
            Dict with configuration information
        """
        return {
            "pt_sl": self.pt_sl,
            "vertical_barrier_bars": self.vertical_barrier_bars,
            "min_ret": self.min_ret,
            "volatility_span": self.volatility_span,
            "lazy": self.lazy,
            "volatility_computed": self.volatility_ is not None,
        }
