"""
Dollar Bars Processor for Financial Machine Learning (Polars Optimized).

This module implements Dollar Bars generation using Polars for improved
performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
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

    Attributes:
        daily_target: Target number of bars per trading day
        ema_span: Span for EMA calculation (used in dynamic mode)
        threshold_: Computed dollar threshold after fitting
        threshold_type: Either 'fixed' or 'dynamic'

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
    ) -> "DollarBarsProcessor":
        """
        Calculate threshold parameters from the input data.

        Args:
            df: DataFrame with 'datetime', 'open', 'high', 'low', 'close', 'volume' columns
            y: Ignored (for sklearn compatibility)

        Returns:
            self
        """
        # Ensure we work with LazyFrame for aggregation to save memory
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()

        # Consistently calculate amount
        if "amount" not in df_lazy.collect_schema().names():
            df_lazy = df_lazy.with_columns(
                (
                    (
                        (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0
                    ) * pl.col("volume")
                ).alias("amount")
            )

        # Calculate average daily volume
        df_daily = (
            df_lazy
            .with_columns(pl.col("datetime").dt.date().alias("date"))
            .group_by("date")
            .agg(pl.col("amount").sum().alias("daily_amount"))
            .filter(pl.col("daily_amount") > 0)
            .collect(streaming=True)  # Collect only the daily stats
        )

        avg_daily_volume = df_daily["daily_amount"].mean()

        # Store fixed threshold
        self.threshold_ = avg_daily_volume / self.daily_target
        self.threshold_type = "fixed"

        return self

    def fit_dynamic(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> "DollarBarsProcessor":
        """
        Fit using dynamic threshold based on EMA of daily volume.

        Args:
            df: DataFrame with 'datetime' and price/volume columns

        Returns:
            self
        """
        # Ensure we work with LazyFrame
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()

        # Consistently calculate amount
        if "amount" not in df_lazy.collect_schema().names():
            df_lazy = df_lazy.with_columns(
                (
                    (
                        (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0
                    ) * pl.col("volume")
                ).alias("amount")
            )

        # Calculate daily stats (collect only daily aggregates)
        df_daily = (
            df_lazy
            .with_columns(pl.col("datetime").dt.date().alias("date"))
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

        # Fill NaN thresholds with start_threshold
        df_daily = df_daily.with_columns(pl.col("threshold").fill_nan(start_threshold))
        
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
        
        # Ensure we work with LazyFrame
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()

        # Consistently calculate amount
        schema_names = df_lazy.collect_schema().names()
        if "amount" not in schema_names:
            df_lazy = df_lazy.with_columns(
                (
                    (
                        (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0
                    ) * pl.col("volume")
                ).alias("amount")
            )

        # Cumulative sum and bar assignment (Lazy)
        df_lazy = df_lazy.with_columns(pl.col("amount").cum_sum().alias("cum_amount"))

        # Assign bar IDs
        df_lazy = df_lazy.with_columns(
            ((pl.col("cum_amount") / threshold).floor().cast(pl.Int64)).alias("bar_id")
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
            # Use streaming collect for memory safety on large datasets
            return dollar_bars_lazy.collect(streaming=True)

    def transform_chunked(
        self,
        parquet_path: str,
        chunk_size: int = 50_000_000,
    ) -> DataFrame:
        """
        Generate dollar bars from a massive parquet file using chunked processing.

        This avoids the OOM issue caused by cum_sum() over billions of rows.
        We process data in chunks, compute cum_sum within each chunk (offset by
        the running total from previous chunks), assign bar_ids, and aggregate
        using vectorized Polars group_by. Partial bars at chunk boundaries are
        carried forward and merged.

        Args:
            parquet_path: Path to the parquet file (or glob for directory)
            chunk_size: Number of rows per chunk (default: 50M)

        Returns:
            DataFrame with dollar bars (columns: datetime, open, high, low, close, volume, amount)
        """
        if self.threshold_ is None:
            raise ValueError("Processor has not been fitted. Call fit() first.")

        threshold = self.threshold_
        all_bar_dfs: list[DataFrame] = []

        from pathlib import Path as _Path

        path_obj = _Path(parquet_path)
        if path_obj.is_dir():
            scan_path = str(path_obj / "*.parquet")
        else:
            scan_path = parquet_path

        # Get total rows for progress reporting
        total_rows = pl.scan_parquet(scan_path).select(pl.len()).collect().item()
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        print(f"    Total rows: {total_rows:,}, chunk_size: {chunk_size:,}, chunks: {n_chunks}")

        # Running state across chunks
        cum_offset = 0.0  # Cumulative amount from all previous chunks
        rows_processed = 0

        for chunk_idx in range(n_chunks):
            # Read a chunk using slice (only materializes chunk_size rows)
            chunk = (
                pl.scan_parquet(scan_path)
                .slice(rows_processed, chunk_size)
                .collect()
            )

            if len(chunk) == 0:
                break

            # Ensure 'amount' column exists
            if "amount" not in chunk.columns:
                chunk = chunk.with_columns(
                    (
                        (
                            (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0
                        ) * pl.col("volume")
                    ).alias("amount")
                )

            # Compute cumulative sum within chunk, offset by previous chunks
            chunk = chunk.with_columns(
                (pl.col("amount").cum_sum() + cum_offset).alias("cum_amount")
            )

            # Update offset for next chunk
            chunk_total = chunk["amount"].sum()
            cum_offset += chunk_total

            # Assign bar IDs (global, because cum_amount includes offset)
            chunk = chunk.with_columns(
                (pl.col("cum_amount") / threshold).floor().cast(pl.Int64).alias("bar_id")
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
            del chunk  # Free memory

            if (chunk_idx + 1) % 5 == 0 or (chunk_idx + 1) == n_chunks:
                # Count bars accumulated so far
                total_bars = sum(len(b) for b in all_bar_dfs)
                print(
                    f"    Chunk {chunk_idx + 1}/{n_chunks}: "
                    f"{rows_processed:,}/{total_rows:,} rows, "
                    f"~{total_bars:,} bar segments so far"
                )

        # Merge all chunk results — bars that span chunk boundaries need re-aggregation
        # The same bar_id may appear in consecutive chunks (the bar was split)
        print("    Merging bar segments across chunk boundaries...")
        combined = pl.concat(all_bar_dfs)
        del all_bar_dfs

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

        del combined
        print(f"    Chunked processing complete: {len(dollar_bars):,} bars total")

        return dollar_bars

    def _transform_dynamic(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Generate dynamic dollar bars with daily EMA threshold."""
        # Ensure we work with LazyFrame
        df_lazy = df if isinstance(df, LazyFrame) else df.lazy()

        # Add date column for threshold mapping
        df_lazy = df_lazy.with_columns(pl.col("datetime").dt.date().alias("date"))

        # Map dynamic thresholds
        if self._daily_thresholds is None:
            raise ValueError("Dynamic thresholds not fitted. Call fit_dynamic() first.")

        # Create expression for threshold lookup
        date_to_threshold = self._daily_thresholds
        
        df_lazy = df_lazy.with_columns(
            pl.col("date")
            .replace(date_to_threshold, default=self.threshold_)
            .alias("dynamic_threshold")
        )

        # Fill any remaining NaN with default threshold
        df_lazy = df_lazy.with_columns(pl.col("dynamic_threshold").fill_nan(self.threshold_))

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
