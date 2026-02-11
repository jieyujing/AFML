"""
Triple Barrier Labeler for Financial Machine Learning.

This module implements the Triple Barrier Method from AFML Chapter 3,
generating labels based on profit-taking, stop-loss, and time barriers.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List

from .base import ProcessorMixin


class TripleBarrierLabeler(ProcessorMixin):
    """
    Labeler that applies the Triple Barrier Method to generate financial labels.

    Each event gets three barriers:
    - Upper barrier: Profit-taking level
    - Lower barrier: Stop-loss level
    - Vertical barrier: Maximum holding period

    Attributes:
        pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
        vertical_barrier_bars: Maximum holding period in bars
        min_ret: Minimum return threshold for event filtering
        volatility_: Computed volatility after fitting

    Example:
        >>> labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
        >>> labeler.fit(close_prices)
        >>> events = labeler.label(events=timestamps, t1=barrier_times)
    """

    def __init__(
        self,
        pt_sl: List[float] = [1.0, 1.0],
        vertical_barrier_bars: int = 12,
        min_ret: float = 0.001,
        volatility_span: int = 100,
    ):
        """
        Initialize the TripleBarrierLabeler.

        Args:
            pt_sl: [profit_taking_multiplier, stop_loss_multiplier]
            vertical_barrier_bars: Number of bars for vertical barrier
            min_ret: Minimum return threshold to keep events
            volatility_span: Span for volatility calculation
        """
        super().__init__()
        self.pt_sl = pt_sl
        self.vertical_barrier_bars = vertical_barrier_bars
        self.min_ret = min_ret
        self.volatility_span = volatility_span
        self.volatility_: Optional[pd.Series] = None

    def fit(
        self, close: pd.Series, y: Optional[pd.Series] = None
    ) -> "TripleBarrierLabeler":
        """
        Calculate volatility for barrier sizing.

        Args:
            close: Series of close prices
            y: Ignored

        Returns:
            self
        """
        self.volatility_ = self._calculate_volatility(close)
        return self

    def _calculate_volatility(self, close: pd.Series) -> pd.Series:
        """Calculate volatility using EMA of absolute log returns."""
        returns = np.log(close / close.shift(1))
        volatility = returns.ewm(span=self.volatility_span).std()
        return volatility

    def get_cusum_events(
        self,
        close: pd.Series,
        threshold: Union[float, pd.Series] = None,
    ) -> pd.DatetimeIndex:
        """
        Apply CUSUM filter to detect significant events.

        Args:
            close: Series of close prices
            threshold: Threshold for event detection

        Returns:
            DatetimeIndex of detected events
        """
        if threshold is None:
            threshold = self.volatility_ * 2 if self.volatility_ is not None else 0.002

        t_events = []
        s_pos = 0
        s_neg = 0

        diff = np.log(close / close.shift(1))
        times = diff.index

        if isinstance(threshold, pd.Series):
            thresh = threshold.reindex(times).ffill()
        else:
            thresh = pd.Series(threshold, index=times)

        for i in range(1, len(times)):
            t = times[i]
            ret = diff.iloc[i]
            h = thresh.iloc[i]

            if pd.isna(ret) or pd.isna(h):
                continue

            s_pos = max(0, s_pos + ret)
            s_neg = min(0, s_neg + ret)

            if s_neg < -h:
                s_neg = 0
                t_events.append(t)
            elif s_pos > h:
                s_pos = 0
                t_events.append(t)

        return pd.DatetimeIndex(t_events)

    def label(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex,
        t1: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Apply triple barrier and generate labels.

        Args:
            close: Series of close prices with datetime index
            events: DatetimeIndex of events to label
            t1: Series with vertical barrier timestamps
            side: Series with position side (1 for long, -1 for short)

        Returns:
            DataFrame with columns: t1, trgt, side, ret, label
        """
        if self.volatility_ is None:
            raise ValueError("Labeler has not been fitted. Call fit() first.")

        # Prepare events
        events_ = events

        # Get target (volatility-based barrier width)
        trgt = self.volatility_.reindex(events_, method="ffill")

        # Set vertical barrier
        if t1 is None:
            t1 = self._create_vertical_barrier(events_, close)
        else:
            t1 = t1.reindex(events_)

        # Set side
        if side is None:
            side_ = pd.Series(1.0, index=trgt.index)
            pt_sl_ = self.pt_sl[:2]
        else:
            side_ = side.reindex(trgt.index, method="ffill")
            pt_sl_ = self.pt_sl[:2]

        # Apply barriers
        out = pd.DataFrame(index=events_)
        out["t1"] = t1
        out["trgt"] = trgt
        out["side"] = side_

        for loc in events_:
            df0 = close[loc:]

            if pd.isna(out.loc[loc, "t1"]):
                df0 = df0[: close.index[-1]]
            else:
                df0 = df0[: out.loc[loc, "t1"]]

            if len(df0) <= 1:
                out.loc[loc, "ret"] = 0
                out.loc[loc, "label"] = 0
                continue

            ret = (df0 / close[loc] - 1) * out.loc[loc, "side"]
            upper = pt_sl_[0] * out.loc[loc, "trgt"]
            lower = -pt_sl_[1] * out.loc[loc, "trgt"]

            touch_upper = ret[ret >= upper]
            touch_lower = ret[ret <= lower]

            if len(touch_upper) > 0 and len(touch_lower) > 0:
                if touch_upper.index[0] < touch_lower.index[0]:
                    out.loc[loc, "t1"] = touch_upper.index[0]
                    out.loc[loc, "ret"] = ret.loc[touch_upper.index[0]]
                    out.loc[loc, "label"] = 1
                else:
                    out.loc[loc, "t1"] = touch_lower.index[0]
                    out.loc[loc, "ret"] = ret.loc[touch_lower.index[0]]
                    out.loc[loc, "label"] = -1
            elif len(touch_upper) > 0:
                out.loc[loc, "t1"] = touch_upper.index[0]
                out.loc[loc, "ret"] = ret.loc[touch_upper.index[0]]
                out.loc[loc, "label"] = 1
            elif len(touch_lower) > 0:
                out.loc[loc, "t1"] = touch_lower.index[0]
                out.loc[loc, "ret"] = ret.loc[touch_lower.index[0]]
                out.loc[loc, "label"] = -1
            else:
                out.loc[loc, "ret"] = ret.iloc[-1]
                out.loc[loc, "label"] = np.sign(ret.iloc[-1])

        # Filter by minimum return
        if self.min_ret > 0:
            out = out[out["ret"].abs() >= self.min_ret]

        return out

    def _create_vertical_barrier(
        self, events: pd.DatetimeIndex, close: pd.Series
    ) -> pd.Series:
        """Create vertical barrier timestamps based on bar count."""
        indices = close.index.get_indexer(events)
        t1_indices = indices + self.vertical_barrier_bars
        t1_indices = np.clip(t1_indices, 0, len(close) - 1)
        t1 = pd.Series(close.index[t1_indices], index=events)
        return t1

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform is not applicable for labeler.

        Use label() method instead.
        """
        raise NotImplementedError(
            "Use label() method to generate labels after fitting."
        )

    def fit_label(
        self,
        close: pd.Series,
        events: pd.DatetimeIndex,
        t1: Optional[pd.Series] = None,
        side: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fit and label in one step.

        Args:
            close: Close price series
            events: Event timestamps
            t1: Vertical barrier timestamps
            side: Position sides

        Returns:
            Labeled events DataFrame
        """
        self.fit(close)
        return self.label(close, events, t1, side)

    def get_bins(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Convert events to discrete bins.

        Args:
            events: DataFrame from label() method

        Returns:
            DataFrame with bin labels
        """
        events_ = events.dropna(subset=["t1"])
        bins = events_[["ret", "label"]].copy()
        bins["bin"] = events_["label"]
        return bins
