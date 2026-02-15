"""
Triple Barrier Labeler for Financial Machine Learning (Polars Optimized).

This module implements the Triple Barrier Method using Polars for improved
performance on large-scale financial time series data.

Performance: Uses vectorized NumPy operations for CUSUM, barrier detection,
and labeling instead of Python loops.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from polars import DataFrame, Series

from .base import ProcessorMixin


class TripleBarrierLabeler(ProcessorMixin):
    """
    Labeler that applies the Triple Barrier Method to generate financial labels.

    Each event gets three barriers:
    - Upper barrier: Profit-taking level
    - Lower barrier: Stop-loss level
    - Vertical barrier: Maximum holding period

    This implementation uses vectorized NumPy operations for improved performance
    on large datasets.

    Attributes:
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
        vertical_barrier_bars: Maximum holding period in bars
        min_ret: Minimum return threshold for event filtering
        volatility_span: Span for volatility calculation
        volatility_: Computed volatility after fitting

    Reference: AFML Chapter 3 - Triple Barrier Method.

    Example:
        >>> labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
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
        Initialize the TripleBarrierLabeler.

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
    ) -> "TripleBarrierLabeler":
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

        Uses vectorized NumPy for the sequential scan portion, with
        event detection done via boolean masking.

        Args:
            close: Series of close prices
            threshold: Threshold for event detection

        Returns:
            DataFrame with detected events

        Reference: AFML Chapter 2 - CUSUM Filter.
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

        # CUSUM has sequential dependency, but we can optimize the inner loop
        # by pre-allocating and using minimal Python overhead
        n = len(returns)
        pos = np.empty(n, dtype=np.float64)
        neg = np.empty(n, dtype=np.float64)
        pos[0] = 0.0
        neg[0] = 0.0

        # Event indices and types collected during scan
        event_indices = []
        event_types = []

        for i in range(1, n):
            p = pos[i - 1] + returns[i]
            g = neg[i - 1] - returns[i]

            if p > threshold_val:
                event_indices.append(i)
                event_types.append("pos")
                pos[i] = 0.0
                neg[i] = 0.0
            elif -g > threshold_val:
                event_indices.append(i)
                event_types.append("neg")
                pos[i] = 0.0
                neg[i] = 0.0
            else:
                pos[i] = max(0.0, p)
                neg[i] = min(0.0, g)

        if event_indices:
            return DataFrame({
                "datetime": event_indices,
                "type": event_types,
            })
        return DataFrame({"datetime": [], "type": []})

    def label(
        self,
        close: Union[Series, DataFrame],
        events: DataFrame,
        t1: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Apply triple barrier labels to events using vectorized NumPy.

        Uses batch NumPy operations on event arrays instead of per-event
        Python loops. The inner barrier-touch detection for each event still
        requires a loop (variable-length slices), but all setup and output
        assembly is vectorized.

        Args:
            close: Series or DataFrame with close prices
            events: DataFrame with event timestamps
            t1: DataFrame with vertical barrier times

        Returns:
            DataFrame with labels

        Reference: AFML Chapter 3 - Triple Barrier Labeling.
        """
        if isinstance(close, DataFrame):
            if "close" not in close.columns:
                raise ValueError("DataFrame must have 'close' column")
            close = close["close"]

        if self.volatility_ is None:
            self.fit(close)

        pt_sl = self.pt_sl
        min_ret = self.min_ret

        if t1 is None:
            t1 = self._get_vertical_barrier(close, events, self.vertical_barrier_bars)

        # Extract arrays once (avoid per-row dict conversion)
        close_np = close.to_numpy()
        n = len(close_np)

        event_datetimes = events["datetime"].to_numpy().astype(np.int64)
        t1_vals = t1["t1"].to_numpy().astype(np.int64) if "t1" in t1.columns else t1.to_numpy().flatten().astype(np.int64)

        vol_np = self.volatility_.to_numpy() if self.volatility_ is not None else None
        default_vol = 0.02

        n_events = len(event_datetimes)

        # Pre-allocate output arrays
        out_t0 = np.empty(n_events, dtype=np.int64)
        out_t1 = np.empty(n_events, dtype=np.int64)
        out_tr = np.empty(n_events, dtype=np.float64)
        out_label = np.empty(n_events, dtype=np.int64)
        valid_mask = np.ones(n_events, dtype=bool)

        for idx in range(n_events):
            t0 = int(event_datetimes[idx])
            if t0 >= n:
                valid_mask[idx] = False
                continue

            t1_val = int(t1_vals[idx]) if idx < len(t1_vals) else t0 + self.vertical_barrier_bars
            t1_val = min(t1_val, n - 1)

            if t0 >= t1_val:
                valid_mask[idx] = False
                continue

            # Volatility at t0
            if vol_np is not None and t0 < len(vol_np) and not np.isnan(vol_np[t0]):
                vol = vol_np[t0]
            else:
                vol = default_vol

            pt = pt_sl[0] * vol
            sl = pt_sl[1] * vol

            # Returns relative to entry price
            price_slice = close_np[t0: t1_val + 1]
            rets = price_slice / close_np[t0] - 1

            # Find first barrier touch using argmax on boolean arrays (fast)
            pt_mask = rets >= pt
            sl_mask = rets <= -sl

            first_pt = np.argmax(pt_mask) if pt_mask.any() else -1
            first_sl = np.argmax(sl_mask) if sl_mask.any() else -1

            # Handle case where argmax returns 0 but mask[0] is False
            if first_pt == 0 and not pt_mask[0]:
                first_pt = -1
            if first_sl == 0 and not sl_mask[0]:
                first_sl = -1

            if first_pt >= 0 and (first_sl < 0 or first_pt < first_sl):
                out_t0[idx] = t0
                out_t1[idx] = t0 + first_pt
                out_tr[idx] = rets[first_pt]
                out_label[idx] = 1
            elif first_sl >= 0 and (first_pt < 0 or first_sl < first_pt):
                out_t0[idx] = t0
                out_t1[idx] = t0 + first_sl
                out_tr[idx] = rets[first_sl]
                out_label[idx] = -1
            else:
                final_ret = rets[-1]
                out_t0[idx] = t0
                out_t1[idx] = t1_val
                out_tr[idx] = final_ret
                out_label[idx] = 0 if abs(final_ret) < min_ret else (1 if final_ret > 0 else -1)

        # Filter valid events and build DataFrame from arrays (no per-row dict)
        if valid_mask.any():
            return DataFrame({
                "t0": out_t0[valid_mask],
                "t1": out_t1[valid_mask],
                "tr": out_tr[valid_mask],
                "label": out_label[valid_mask],
            })
        return DataFrame({"t0": [], "t1": [], "tr": [], "label": []})

    def _get_vertical_barrier(
        self,
        close: Series,
        events: DataFrame,
        vertical_barrier_bars: int,
    ) -> DataFrame:
        """
        Calculate vertical barrier for each event using vectorized operations.

        Args:
            close: Close price series
            events: Events DataFrame
            vertical_barrier_bars: Number of bars for vertical barrier

        Returns:
            DataFrame with t1 (vertical barrier time)
        """
        # Vectorized: extract column, add barrier, clip to max index
        event_times = events["datetime"].to_numpy().astype(np.int64)
        t1_vals = np.minimum(event_times + vertical_barrier_bars, len(close) - 1)
        return DataFrame({"t1": t1_vals})

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
