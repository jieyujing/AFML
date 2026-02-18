"""
Dollar Bars Processor for Financial Machine Learning (Polars Optimized).

This module implements Dollar Bars generation using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import polars as pl
from polars import DataFrame, LazyFrame

from .base import ProcessorMixin


class DollarBarsProcessor(ProcessorMixin):
    """
    Processor for generating dollar bars from tick data using Polars.

    Dollar bars aggregate ticks based on a dollar amount threshold, providing
    more statistically reliable bars than time-based sampling.

    This implementation uses Polars for improved performance on large datasets,
    with support for lazy evaluation and multi-threaded operations.

    This processor uses DYNAMIC THRESHOLDING by default. It calculates a
    daily threshold based on the EMA of daily dollar volume to maintain
    the target number of bars per day regardless of market activity.

    Attributes:
        daily_target: Target number of bars per trading day
        ema_span: Span for EMA calculation
        threshold_: Computed dollar threshold (current/default) after fitting
        _daily_thresholds: Dictionary mapping dates to dynamic thresholds

    Example:
        >>> processor = DollarBarsProcessor(daily_target=4)
        >>> dollar_bars = processor.fit_transform(raw_df)

        >>> # For large datasets, use lazy mode
        >>> processor = DollarBarsProcessor(daily_target=4, lazy=True)
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
        Initialize the DollarBarsProcessor.

        Args:
            daily_target: Target number of bars per trading day (default: 4)
            ema_span: Span for EMA calculation (default: 20)
            lazy: Whether to use lazy evaluation (default: False)
        """
        super().__init__()
        self.daily_target = daily_target
        self.ema_span = ema_span
        self.lazy = lazy
        self.threshold_: Optional[float] = None
        self._daily_thresholds: Optional[Dict[str, float]] = None

    @staticmethod
    def _from_pandas(df: Any) -> Union[DataFrame, LazyFrame]:
        if isinstance(df, (DataFrame, LazyFrame)):
            return df
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        return df

    def _validate_schema(self, df: Union[DataFrame, LazyFrame]) -> None:
        schema = df.collect_schema() if isinstance(df, LazyFrame) else df.schema
        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required - set(schema.names())
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _ensure_amount_column(self, df: LazyFrame) -> LazyFrame:
        schema = df.collect_schema()
        if "amount" not in schema.names():
            return df.with_columns(
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
                ).alias("amount")
            )
        return df

    def _ensure_amount_column_eager(self, df: DataFrame) -> DataFrame:
        if "amount" not in df.columns:
            return df.with_columns(
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
                ).alias("amount")
            )
        return df

    def fit(
        self,
        df: Union[DataFrame, LazyFrame, Any],
        y: Optional[Any] = None,
    ) -> "DollarBarsProcessor":
        """
        Calculate dynamic threshold parameters from the input data.

        This method computes the daily dollar volume, applies an EMA,
        and sets daily thresholds based on the previous day's EMA.

        Args:
            df: DataFrame with 'datetime' and price/volume columns (Polars or Pandas)
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        df = self._from_pandas(df)
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()
        self._validate_schema(df_lazy)
        df_lazy = self._ensure_amount_column(df_lazy)

        # Calculate daily stats (collect only daily aggregates)
        df_daily = (
            df_lazy.with_columns(pl.col("datetime").dt.date().alias("date"))
            .group_by("date")
            .agg(pl.col("amount").sum().alias("daily_amount"))
            .sort("date")  # Sort needed for EMA
            .collect(streaming=True)
        )

        # Proceed with Eager DataFrame for EMA calculation (cheap on daily data)
        df_daily = df_daily.with_columns(
            pl.col("daily_amount")
            .replace(0, None)
            .ewm_mean(span=self.ema_span, adjust=False)
            .alias("ema_amt")
        )

        # Calculate threshold with shift
        df_daily = df_daily.with_columns(
            (pl.col("ema_amt").shift(1) / self.daily_target).alias("threshold")
        )

        # Calculate global mean for initial threshold
        global_mean = df_daily.filter(pl.col("daily_amount") > 0)["daily_amount"].mean()
        start_threshold = global_mean / self.daily_target

        # Fill NaN/Null thresholds with start_threshold
        df_daily = df_daily.with_columns(
            pl.col("threshold").fill_null(start_threshold).fill_nan(start_threshold)
        )

        self._daily_thresholds = dict(zip(df_daily["date"], df_daily["threshold"]))
        self.threshold_ = start_threshold

        return self

    def transform(
        self,
        df: Union[DataFrame, LazyFrame, Any],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Generate dollar bars from raw tick data using dynamic thresholds.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns (Polars or Pandas)

        Returns:
            DataFrame or LazyFrame with dollar bars
        """
        if self._daily_thresholds is None:
            raise ValueError("Processor has not been fitted. Call fit() first.")

        df = self._from_pandas(df)
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()
        self._validate_schema(df_lazy)
        df_lazy = self._ensure_amount_column(df_lazy)

        # Add date column for threshold mapping
        df_lazy = df_lazy.with_columns(pl.col("datetime").dt.date().alias("date"))

        # Create expression for threshold lookup
        date_to_threshold = self._daily_thresholds

        df_lazy = df_lazy.with_columns(
            pl.col("date")
            .replace(date_to_threshold, default=self.threshold_)
            .alias("dynamic_threshold")
        )

        # Fill any remaining NaN with default threshold
        df_lazy = df_lazy.with_columns(
            pl.col("dynamic_threshold").fill_nan(self.threshold_)
        )

        # Normalize amount and cumulative sum
        df_lazy = df_lazy.with_columns(
            (pl.col("amount") / pl.col("dynamic_threshold")).alias("norm_amount")
        ).with_columns(pl.col("norm_amount").cum_sum().alias("cum_norm_amount"))

        # Assign bar IDs
        df_lazy = df_lazy.with_columns(
            pl.col("cum_norm_amount").floor().cast(pl.Int64).alias("bar_id")
        )

        # Aggregate bars
        dollar_bars_lazy = (
            df_lazy.group_by("bar_id")
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

        if self.lazy:
            return dollar_bars_lazy
        else:
            return dollar_bars_lazy.collect(streaming=True)

    def transform_chunked(
        self,
        data_source: Union[str, LazyFrame],
        chunk_size: int = 50_000_000,
    ) -> DataFrame:
        """
        Generate dollar bars from a massive parquet file or LazyFrame using chunked processing.
        Uses dynamic thresholds based on the date of the ticks.

        Args:
            data_source: Path to the parquet file (or glob), OR a pre-configured LazyFrame.
            chunk_size: Number of rows per chunk (default: 50M)

        Returns:
            DataFrame with dollar bars (columns: datetime, open, high, low, close, volume, amount)
        """
        if self._daily_thresholds is None:
            raise ValueError("Processor has not been fitted. Call fit() first.")

        threshold = self.threshold_
        all_bar_dfs: list[DataFrame] = []

        # Determine total rows and prepare lazy frame
        if isinstance(data_source, (str, _Path := __import__("pathlib").Path)):
            path_obj = _Path(str(data_source))
            if path_obj.is_dir():
                scan_path = str(path_obj / "*.parquet")
            else:
                scan_path = str(data_source)
            lf = pl.scan_parquet(scan_path)
        elif isinstance(data_source, LazyFrame):
            lf = data_source
        else:
            raise TypeError("data_source must be str, Path, or LazyFrame")

        # Get total rows for progress reporting
        total_rows = lf.select(pl.len()).collect().item()
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        print(
            f"    Total rows: {total_rows:,}, chunk_size: {chunk_size:,}, chunks: {n_chunks}"
        )

        # Running state across chunks
        cum_offset = 0.0  # Cumulative amount from all previous chunks
        rows_processed = 0

        for chunk_idx in range(n_chunks):
            # Read a chunk using slice (only materializes chunk_size rows)
            chunk = lf.slice(rows_processed, chunk_size).collect()

            if len(chunk) == 0:
                break

            chunk = self._ensure_amount_column_eager(chunk)

            # --- Dynamic Threshold Logic ---
            # We need date for lookup
            chunk = chunk.with_columns(pl.col("datetime").dt.date().alias("date"))

            # Map thresholds (replace is faster than join for simple scalar map)
            chunk = chunk.with_columns(
                pl.col("date")
                .replace(self._daily_thresholds, default=threshold)
                .alias("threshold_val")
            )

            # Normalize amount: how many "bars" is this tick worth?
            chunk = chunk.with_columns(
                (pl.col("amount") / pl.col("threshold_val")).alias("process_amt")
            )

            # Compute cumulative sum of "processed amount" (normalized units)
            # cum_offset matches the units of process_amt
            chunk = chunk.with_columns(
                (pl.col("process_amt").cum_sum() + cum_offset).alias("cum_process")
            )

            cum_offset = chunk["cum_process"].max()

            # Assign bar IDs
            chunk = chunk.with_columns(
                pl.col("cum_process").floor().cast(pl.Int64).alias("bar_id")
            )

            # Aggregate bars within this chunk (vectorized)
            chunk_bars = (
                chunk.group_by("bar_id")
                .agg(
                    pl.col("datetime").last().alias("datetime"),
                    pl.col("open").first().alias("open"),
                    pl.col("high").max().alias("high"),
                    pl.col("low").min().alias("low"),
                    pl.col("close").last().alias("close"),
                    pl.col("volume").sum().alias("volume"),
                    pl.col("amount").sum().alias("amount"),
                )
                .sort("bar_id")
            )

            all_bar_dfs.append(chunk_bars)

            rows_processed += len(chunk)

            if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
                # Count bars accumulated so far
                total_bars = sum(len(b) for b in all_bar_dfs)
                print(
                    f"    Chunk {chunk_idx + 1}/{n_chunks}: "
                    f"{rows_processed:,}/{total_rows:,} rows, "
                    f"~{total_bars:,} bar segments so far"
                )

        # Merge all chunk results — bars that span chunk boundaries need re-aggregation
        print("    Merging bar segments across chunk boundaries...")
        combined = pl.concat(all_bar_dfs)

        # Re-aggregate by bar_id to merge split bars
        dollar_bars = (
            combined.group_by("bar_id")
            .agg(
                pl.col("datetime").last().alias("datetime"),
                pl.col("open").first().alias("open"),
                pl.col("high").max().alias("high"),
                pl.col("low").min().alias("low"),
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("amount").sum().alias("amount"),
            )
            .sort("bar_id")
            .drop("bar_id")
        )

        print(f"    Chunked processing complete: {len(dollar_bars):,} bars total")

        return dollar_bars

    def fit_transform(
        self,
        df: Union[DataFrame, LazyFrame],
        y: Optional[Any] = None,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Fit and transform in one step using dynamic thresholds.

        Args:
            df: Input DataFrame
            y: Ignored

        Returns:
            Dollar bars DataFrame
        """
        self.fit(df)
        return self.transform(df)

    def get_threshold_info(self) -> Dict[str, Any]:
        """
        Get information about the computed threshold.

        Returns:
            Dict with threshold information

        Raises:
            ValueError: If processor has not been fitted
        """
        if self._daily_thresholds is None:
            raise ValueError("Processor has not been fitted.")
        return {
            "threshold": self.threshold_,
            "daily_target": self.daily_target,
            "ema_span": self.ema_span,
            "lazy": self.lazy,
        }
