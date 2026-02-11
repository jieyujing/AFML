"""
Dollar Bars Processor for Financial Machine Learning.

This module implements Dollar Bars generation - transforming raw tick data
into bars based on dollar volume thresholds.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .base import ProcessorMixin


class DollarBarsProcessor(ProcessorMixin):
    """
    Processor for generating dollar bars from tick data.

    Dollar bars aggregate ticks based on a dollar amount threshold, providing
    more statistically reliable bars than time-based sampling.

    Attributes:
        daily_target: Target number of bars per trading day
        ema_span: Span for EMA calculation (used in dynamic mode)
        threshold_: Computed dollar threshold after fitting

    Example:
        >>> processor = DollarBarsProcessor(daily_target=4)
        >>> dollar_bars = processor.fit_transform(raw_df)
    """

    def __init__(
        self,
        daily_target: int = 4,
        ema_span: int = 20,
    ):
        """
        Initialize the DollarBarsProcessor.

        Args:
            daily_target: Target number of bars per trading day (default: 4)
            ema_span: Span for EMA calculation in dynamic mode (default: 20)
        """
        super().__init__()
        self.daily_target = daily_target
        self.ema_span = ema_span
        self.threshold_: Optional[float] = None
        self.threshold_type: str = "fixed"

    def fit(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DollarBarsProcessor":
        """
        Calculate threshold parameters from the input data.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        if "amount" not in df.columns:
            # Calculate amount if not present
            multiplier = 300.0
            avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            df = df.copy()
            df["amount"] = avg_price * df["volume"] * multiplier

        # Calculate average daily volume
        df_daily = df.set_index("datetime").resample("D")["amount"].sum()
        df_daily = df_daily[df_daily > 0]  # Filter out zero-volume days
        avg_daily_volume = df_daily.mean()

        # Store fixed threshold
        self.threshold_ = avg_daily_volume / self.daily_target
        self.threshold_type = "fixed"

        return self

    def fit_dynamic(self, df: pd.DataFrame) -> "DollarBarsProcessor":
        """
        Fit using dynamic threshold based on EMA of daily volume.

        Args:
            df: DataFrame with 'datetime' and price/volume columns

        Returns:
            self
        """
        if "amount" not in df.columns:
            multiplier = 300.0
            avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            df = df.copy()
            df["amount"] = avg_price * df["volume"] * multiplier

        # Calculate daily stats
        daily_stats = (
            df.set_index("datetime")
            .resample("D")["amount"]
            .sum()
            .to_frame(name="daily_amt")
        )

        # Calculate EMA threshold
        daily_stats["ema_amt"] = (
            daily_stats["daily_amt"]
            .replace(0, np.nan)
            .ewm(span=self.ema_span, adjust=False)
            .mean()
        )
        daily_stats["threshold"] = daily_stats["ema_amt"].shift(1) / self.daily_target

        # Store daily thresholds for transform
        global_mean = daily_stats["daily_amt"][daily_stats["daily_amt"] > 0].mean()
        start_threshold = global_mean / self.daily_target
        daily_stats["threshold"] = daily_stats["threshold"].fillna(start_threshold)

        # Create mapping
        daily_stats["date"] = daily_stats.index.date
        self._daily_thresholds = daily_stats.set_index("date")["threshold"].to_dict()

        self.threshold_type = "dynamic"
        self.threshold_ = start_threshold

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate dollar bars from raw tick data.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            DataFrame with dollar bars
        """
        if self.threshold_ is None:
            raise ValueError("Processor has not been fitted. Call fit() first.")

        # Ensure amount column exists
        if "amount" not in df.columns:
            multiplier = 300.0
            avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
            df = df.copy()
            df["amount"] = avg_price * df["volume"] * multiplier

        if self.threshold_type == "fixed":
            return self._transform_fixed(df)
        else:
            return self._transform_dynamic(df)

    def _transform_fixed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fixed dollar bars."""
        threshold = self.threshold_

        # Cumulative sum and group assignment
        cum_amount = df["amount"].cumsum()
        group_ids = (cum_amount.shift(1).fillna(0) // threshold).astype(int)

        # Aggregate bars
        dollar_bars = df.groupby(group_ids).apply(
            self._aggregate_bar, include_groups=False
        )
        dollar_bars = dollar_bars.reset_index(drop=True)

        return dollar_bars

    def _transform_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate dynamic dollar bars with daily EMA threshold."""
        # Map thresholds
        df = df.copy()
        df["date"] = df["datetime"].dt.date
        df["dynamic_threshold"] = df["date"].map(self._daily_thresholds)
        df["dynamic_threshold"] = df["dynamic_threshold"].fillna(self.threshold_)

        # Normalize amount
        df["norm_amount"] = df["amount"] / df["dynamic_threshold"]
        df["cum_norm_amount"] = df["norm_amount"].cumsum()

        group_ids = (df["cum_norm_amount"].shift(1).fillna(0)).astype(int)

        dollar_bars = df.groupby(group_ids).apply(
            self._aggregate_bar, include_groups=False
        )
        dollar_bars = dollar_bars.reset_index(drop=True)

        return dollar_bars

    @staticmethod
    def _aggregate_bar(group: pd.DataFrame) -> pd.Series:
        """Aggregate a group of ticks into a single bar."""
        return pd.Series(
            {
                "datetime": group["datetime"].iloc[-1],
                "open": group["open"].iloc[0],
                "high": group["high"].max(),
                "low": group["low"].min(),
                "close": group["close"].iloc[-1],
                "volume": group["volume"].sum(),
                "amount": group["amount"].sum(),
            }
        )

    def fit_transform(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and transform in one step (fixed threshold).

        Args:
            df: Input DataFrame
            y: Ignored

        Returns:
            Dollar bars DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def fit_transform_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit with dynamic threshold and transform.

        Args:
            df: Input DataFrame

        Returns:
            Dollar bars DataFrame
        """
        self.fit_dynamic(df)
        return self.transform(df)

    def get_threshold_info(self) -> dict:
        """Get information about the computed threshold."""
        if self.threshold_ is None:
            raise ValueError("Processor has not been fitted.")
        return {
            "threshold": self.threshold_,
            "threshold_type": self.threshold_type,
            "daily_target": self.daily_target,
            "ema_span": self.ema_span,
        }
