"""
Polars Dollar Bars Processor for Financial Machine Learning.

This module implements Dollar Bars generation using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame

from afml.base import ProcessorMixin


class PolarsDollarBarsProcessor(ProcessorMixin):
    """
    Processor for generating dollar bars from tick data using Polars.

    Dollar bars aggregate ticks based on a dollar amount threshold, providing
    more statistically reliable bars than time-based sampling.

    This implementation uses Polars for improved performance on large datasets,
    with support for lazy evaluation and multi-threaded operations.

    Attributes:
        daily_target: Target number of bars per trading day
        ema_span: Span for EMA calculation (used in dynamic mode)
        threshold_: Computed dollar threshold after fitting
        threshold_type: Either 'fixed' or 'dynamic'

    Example:
        >>> processor = PolarsDollarBarsProcessor(daily_target=4)
        >>> dollar_bars = processor.fit_transform(raw_df)

        >>> # For large datasets, use lazy mode
        >>> processor = PolarsDollarBarsProcessor(daily_target=4, lazy=True)
        >>> dollar_bars = processor.fit_transform(lazy_df).collect()
    """

    def __init__(
        self,
        daily_target: int = 4,
        ema_span: int = 20,
        *,
        lazy: bool = False,
    ):
        """
        Initialize the PolarsDollarBarsProcessor.

        Args:
            daily_target: Target number of bars per trading day (default: 4)
            ema_span: Span for EMA calculation in dynamic mode (default: 20)
            lazy: Whether to use lazy evaluation (default: False)
        """
        super().__init__()
        self.daily_target = daily_target
        self.ema_span = ema_span
        self.lazy = lazy
        self.threshold_: Optional[float] = None
        self.threshold_type: str = "fixed"
        self._daily_thresholds: Optional[Dict[str, float]] = None

    def fit(
        self,
        df: Union[DataFrame, LazyFrame],
        y: Optional[Any] = None,
    ) -> "PolarsDollarBarsProcessor":
        """
        Calculate threshold parameters from the input data.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        if isinstance(df, LazyFrame):
            df = df.collect()

        # Calculate amount if not present
        if "amount" not in df.columns:
            multiplier = 300.0
            df = df.with_columns(
                (
                    (
                        (
                            pl.col("open")
                            + pl.col("high")
                            + pl.col("low")
                            + pl.col("close")
                        )
                        / 4.0
                    )
                    * pl.col("volume")
                    * multiplier
                ).alias("amount")
            )

        # Calculate average daily volume
        df_daily = (
            df.sort("datetime")
            .group_by_dynamic("datetime", every="1d")
            .agg(pl.col("amount").sum().alias("daily_amount"))
            .filter(pl.col("daily_amount") > 0)
        )

        avg_daily_volume = df_daily["daily_amount"].mean()

        # Store fixed threshold
        self.threshold_ = avg_daily_volume / self.daily_target
        self.threshold_type = "fixed"

        return self

    def fit_dynamic(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> "PolarsDollarBarsProcessor":
        """
        Fit using dynamic threshold based on EMA of daily volume.

        Args:
            df: DataFrame with 'datetime' and price/volume columns

        Returns:
            self
        """
        if isinstance(df, LazyFrame):
            df = df.collect()

        # Calculate amount if not present
        if "amount" not in df.columns:
            multiplier = 300.0
            df = df.with_columns(
                (
                    (
                        (
                            pl.col("open")
                            + pl.col("high")
                            + pl.col("low")
                            + pl.col("close")
                        )
                        / 4.0
                    )
                    * pl.col("volume")
                    * multiplier
                ).alias("amount")
            )

        # Calculate daily stats
        df_daily = (
            df.sort("datetime")
            .group_by_dynamic("datetime", every="1d")
            .agg(pl.col("amount").sum().alias("daily_amount"))
            .with_columns(
                pl.col("daily_amount")
                .replace(0, None)
                .ewm_mean(span=self.ema_span, adjust=False)
                .alias("ema_amt")
            )
        )

        # Calculate threshold with shift
        df_daily = df_daily.with_columns(
            (pl.col("ema_amt").shift(1) / self.daily_target).alias("threshold")
        )

        # Calculate global mean for initial threshold
        global_mean = df_daily.filter(pl.col("daily_amount") > 0)["daily_amount"].mean()
        start_threshold = global_mean / self.daily_target

        # Fill NaN thresholds with start_threshold
        df_daily = df_daily.with_columns(pl.col("threshold").fill_nan(start_threshold))

        # Create mapping from date to threshold
        df_daily = df_daily.with_columns(pl.col("datetime").dt.date().alias("date"))

        self._daily_thresholds = dict(zip(df_daily["date"], df_daily["threshold"]))

        self.threshold_type = "dynamic"
        self.threshold_ = start_threshold

        return self

    def transform(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Generate dollar bars from raw tick data.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            DataFrame or LazyFrame with dollar bars
        """
        if self.threshold_ is None:
            raise ValueError("Processor has not been fitted. Call fit() first.")

        if isinstance(df, LazyFrame):
            result = self._transform_fixed(df.lazy())
            return result.lazy() if self.lazy else result.collect()
        else:
            if self.threshold_type == "fixed":
                return self._transform_fixed(df)
            else:
                return self._transform_dynamic(df)

    def _transform_fixed(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Generate fixed dollar bars."""
        threshold = self.threshold_

        if isinstance(df, LazyFrame):
            df = df.collect()

        # Calculate amount if not present
        if "amount" not in df.columns:
            multiplier = 300.0
            df = df.with_columns(
                (
                    (
                        (
                            pl.col("open")
                            + pl.col("high")
                            + pl.col("low")
                            + pl.col("close")
                        )
                        / 4.0
                    )
                    * pl.col("volume")
                    * multiplier
                ).alias("amount")
            )

        # Cumulative sum and bar assignment
        df = df.with_columns(pl.col("amount").cum_sum().alias("cum_amount"))

        # Assign bar IDs
        df = df.with_columns(
            ((pl.col("cum_amount") / threshold).floor().cast(pl.Int64)).alias("bar_id")
        )

        # Aggregate bars
        dollar_bars = (
            df.group_by("bar_id")
            .agg(
                pl.col("datetime").last().alias("datetime"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
            )
            .drop("bar_id")
            .sort("datetime")
        )

        return dollar_bars

    def _transform_dynamic(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Generate dynamic dollar bars with daily EMA threshold."""
        if isinstance(df, LazyFrame):
            df = df.collect()

        # Add date column for threshold mapping
        df = df.with_columns(pl.col("datetime").dt.date().alias("date"))

        # Map dynamic thresholds
        if self._daily_thresholds is None:
            raise ValueError("Dynamic thresholds not fitted. Call fit_dynamic() first.")

        # Create expression for threshold lookup
        date_to_threshold = pl.lit(self._daily_thresholds)
        df = df.with_columns(
            pl.struct(
                [
                    pl.col("date")
                    .map_dict(date_to_threshold, default=self.threshold_)
                    .alias("dynamic_threshold")
                ]
            )
        )

        # Fill any remaining NaN with default threshold
        df = df.with_columns(pl.col("dynamic_threshold").fill_nan(self.threshold_))

        # Normalize amount and cumulative sum
        df = df.with_columns(
            (pl.col("amount") / pl.col("dynamic_threshold")).alias("norm_amount")
        ).with_columns(pl.col("norm_amount").cum_sum().alias("cum_norm_amount"))

        # Assign bar IDs
        df = df.with_columns(
            pl.col("cum_norm_amount").floor().cast(pl.Int64).alias("bar_id")
        )

        # Aggregate bars
        dollar_bars = (
            df.group_by("bar_id")
            .agg(
                pl.col("datetime").last().alias("datetime"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
            )
            .drop("bar_id")
            .sort("datetime")
        )

        return dollar_bars

    def fit_transform(
        self,
        df: Union[DataFrame, LazyFrame],
        y: Optional[Any] = None,
    ) -> Union[DataFrame, LazyFrame]:
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

    def fit_transform_dynamic(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Fit with dynamic threshold and transform.

        Args:
            df: Input DataFrame

        Returns:
            Dollar bars DataFrame
        """
        self.fit_dynamic(df)
        return self.transform(df)

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get information about the computed threshold.

        Returns:
            Dict with threshold information

        Raises:
            ValueError: If processor has not been fitted
        """
        if self.threshold_ is None:
            raise ValueError("Processor has not been fitted.")
        return {
            "threshold": self.threshold_,
            "threshold_type": self.threshold_type,
            "daily_target": self.daily_target,
            "ema_span": self.ema_span,
            "lazy": self.lazy,
        }
