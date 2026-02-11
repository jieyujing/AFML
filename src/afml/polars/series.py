"""
Polars Series Utilities for AFML.

This module provides utility functions and classes for working with
Polars Series in the AFML context.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import polars as pl
from polars import Series


class PolarsSeriesUtils:
    """Utility class for Polars Series operations specific to financial data."""

    @staticmethod
    def ensure_series(
        data: Union[Series, pl.DataFrame, Any],
        *,
        name: Optional[str] = None,
    ) -> Series:
        """
        Ensure the input is a Polars Series.

        Args:
            data: Input data (Series, DataFrame column, pandas Series, or list)
            name: Name for the Series if creating new

        Returns:
            Polars Series

        Raises:
            TypeError: If data cannot be converted to Series
        """
        if isinstance(data, Series):
            if name is not None:
                return data.alias(name)
            return data

        if isinstance(data, pl.DataFrame):
            if name is None:
                raise ValueError("Must specify 'name' when extracting from DataFrame")
            return data[name]

        if hasattr(data, "to_polars"):
            series = data.to_polars()
            if name is not None:
                return series.alias(name)
            return series

        try:
            if name is not None:
                return Series(values=data, name=name)
            return Series(values=data)
        except Exception as e:
            raise TypeError(f"Cannot convert {type(data)} to Polars Series: {e}")

    @staticmethod
    def series_from_pandas(
        pandas_series: Any,
        *,
        name: Optional[str] = None,
    ) -> Series:
        """
        Convert pandas Series to Polars Series.

        Args:
            pandas_series: pandas Series to convert
            name: Optional new name for the Series

        Returns:
            Polars Series
        """
        result = pl.from_pandas(pandas_series)

        if name is not None:
            return result.alias(name)
        return result

    @staticmethod
    def pct_change(
        series: Series,
        *,
        periods: int = 1,
    ) -> Series:
        """
        Calculate percentage change between elements.

        Args:
            series: Input Series
            periods: Number of periods to shift

        Returns:
            Series with percentage changes
        """
        return series.pct_change(periods=periods)

    @staticmethod
    def log_return(series: Series) -> Series:
        """
        Calculate logarithmic returns.

        Args:
            series: Price series

        Returns:
            Series with log returns
        """
        return series.log().diff()

    @staticmethod
    def rolling_mean(
        series: Series,
        window: int,
        *,
        min_periods: Optional[int] = None,
    ) -> Series:
        """
        Calculate rolling mean.

        Args:
            series: Input Series
            window: Window size
            min_periods: Minimum periods required (defaults to window)

        Returns:
            Series with rolling means
        """
        if min_periods is None:
            min_periods = window
        return series.rolling_mean(window_size=window, min_periods=min_periods)

    @staticmethod
    def rolling_std(
        series: Series,
        window: int,
        *,
        min_periods: Optional[int] = None,
    ) -> Series:
        """
        Calculate rolling standard deviation.

        Args:
            series: Input Series
            window: Window size
            min_periods: Minimum periods required (defaults to window)

        Returns:
            Series with rolling std
        """
        if min_periods is None:
            min_periods = window
        return series.rolling_std(window_size=window, min_periods=min_periods)

    @staticmethod
    def rolling_max(
        series: Series,
        window: int,
        *,
        min_periods: Optional[int] = None,
    ) -> Series:
        """
        Calculate rolling maximum.

        Args:
            series: Input Series
            window: Window size
            min_periods: Minimum periods required (defaults to window)

        Returns:
            Series with rolling max
        """
        if min_periods is None:
            min_periods = window
        return series.rolling_max(window_size=window, min_periods=min_periods)

    @staticmethod
    def rolling_min(
        series: Series,
        window: int,
        *,
        min_periods: Optional[int] = None,
    ) -> Series:
        """
        Calculate rolling minimum.

        Args:
            series: Input Series
            window: Window size
            min_periods: Minimum periods required (defaults to window)

        Returns:
            Series with rolling min
        """
        if min_periods is None:
            min_periods = window
        return series.rolling_min(window_size=window, min_periods=min_periods)

    @staticmethod
    def ewm_mean(
        series: Series,
        span: int,
        *,
        adjust: bool = True,
    ) -> Series:
        """
        Calculate exponentially weighted mean.

        Args:
            series: Input Series
            span: Span parameter for EWMA
            adjust: Whether to adjust weights

        Returns:
            Series with EWMA values
        """
        return series.ewm_mean(span=span, adjust=adjust)

    @staticmethod
    def ewm_std(
        series: Series,
        span: int,
        *,
        adjust: bool = True,
    ) -> Series:
        """
        Calculate exponentially weighted standard deviation.

        Args:
            series: Input Series
            span: Span parameter for EWMA
            adjust: Whether to adjust weights

        Returns:
            Series with EWMA std
        """
        return series.ewm_std(span=span, adjust=adjust)

    @staticmethod
    def ewm_vol(
        series: Series,
        span: int,
        *,
        adjust: bool = True,
    ) -> Series:
        """
        Calculate exponentially weighted volatility.

        Args:
            series: Input Series (typically returns)
            span: Span parameter for EWMA
            adjust: Whether to adjust weights

        Returns:
            Series with EWMA volatility
        """
        return series.ewm_std(span=span, adjust=adjust)

    @staticmethod
    def cumulative_sum(series: Series) -> Series:
        """
        Calculate cumulative sum.

        Args:
            series: Input Series

        Returns:
            Series with cumulative sums
        """
        return series.cum_sum()

    @staticmethod
    def cumulative_max(series: Series) -> Series:
        """
        Calculate cumulative maximum.

        Args:
            series: Input Series

        Returns:
            Series with cumulative max
        """
        return series.cum_max()

    @staticmethod
    def cumulative_min(series: Series) -> Series:
        """
        Calculate cumulative minimum.

        Args:
            series: Input Series

        Returns:
            Series with cumulative min
        """
        return series.cum_min()

    @staticmethod
    def diff(
        series: Series,
        *,
        periods: int = 1,
    ) -> Series:
        """
        Calculate first-order differences.

        Args:
            series: Input Series
            periods: Number of periods to shift

        Returns:
            Series with differences
        """
        return series.diff(n=periods)

    @staticmethod
    def rank(
        series: Series,
        *,
        method: str = "average",
        descending: bool = False,
    ) -> Series:
        """
        Calculate rank of elements.

        Args:
            series: Input Series
            method: Ranking method ('average', 'min', 'max', 'dense', 'ordinal')
            descending: Rank in descending order

        Returns:
            Series with ranks
        """
        return series.rank(method=method, descending=descending)

    @staticmethod
    def quantile(
        series: Series,
        quantile: float,
        *,
        interpolation: str = "nearest",
    ) -> float:
        """
        Calculate quantile of the series.

        Args:
            series: Input Series
            quantile: Quantile value between 0 and 1
            interpolation: Interpolation method

        Returns:
            Quantile value
        """
        return series.quantile(quantile, interpolation=interpolation)

    @staticmethod
    def z_score(
        series: Series,
        *,
        window: Optional[int] = None,
    ) -> Series:
        """
        Calculate z-score (standardized values).

        Args:
            series: Input Series
            window: Optional rolling window for rolling z-score

        Returns:
            Series with z-scores
        """
        if window is not None:
            mean = series.rolling_mean(window_size=window)
            std = series.rolling_std(window_size=window)
        else:
            mean = series.mean()
            std = series.std()

        return (series - mean) / std

    @staticmethod
    def correlation(
        series1: Series,
        series2: Series,
        *,
        window: Optional[int] = None,
    ) -> Union[float, Series]:
        """
        Calculate correlation between two series.

        Args:
            series1: First Series
            series2: Second Series
            window: Optional rolling window for rolling correlation

        Returns:
            Correlation value or rolling correlation Series
        """
        if window is not None:
            return (series1 * series2).rolling_mean(window_size=window) - (
                series1.rolling_mean(window_size=window)
                * series2.rolling_mean(window_size=window)
            )
        return series1.corr(series2)

    @staticmethod
    def count_nans(series: Series) -> int:
        """
        Count number of null/NaN values.

        Args:
            series: Input Series

        Returns:
            Count of null values
        """
        return series.null_count()

    @staticmethod
    def fill_nans(
        series: Series,
        value: Optional[Any] = None,
        *,
        method: Optional[str] = None,
    ) -> Series:
        """
        Fill null/NaN values.

        Args:
            series: Input Series
            value: Value to fill with (if method is None)
            method: Fill method ('forward', 'backward', 'mean', etc.)

        Returns:
            Series with filled values
        """
        if method is not None:
            return series.fill_null(strategy=method)
        if value is not None:
            return series.fill_null(value)
        return series

    @staticmethod
    def replace_inf(
        series: Series,
        value: Any,
    ) -> Series:
        """
        Replace infinite values with a given value.

        Args:
            series: Input Series
            value: Value to replace inf with

        Returns:
            Series with inf replaced
        """
        import numpy as np

        return series.map_batches(
            lambda x: np.where(np.isinf(x), value, x),
            return_dtype=series.dtype,
        )

    @staticmethod
    def is_finite(series: Series) -> Series:
        """
        Create boolean mask for finite values.

        Args:
            series: Input Series

        Returns:
            Boolean Series
        """
        return series.is_finite()

    @staticmethod
    def sign(series: Series) -> Series:
        """
        Get sign of values (-1, 0, or 1).

        Args:
            series: Input Series

        Returns:
            Series with signs
        """
        return series.sign()

    @staticmethod
    def abs(series: Series) -> Series:
        """
        Get absolute values.

        Args:
            series: Input Series

        Returns:
            Series with absolute values
        """
        return series.abs()

    @staticmethod
    def clip(
        series: Series,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
    ) -> Series:
        """
        Clip values to bounds.

        Args:
            series: Input Series
            lower: Lower bound
            upper: Upper bound

        Returns:
            Series with clipped values
        """
        if lower is not None and upper is not None:
            return series.clip(lower, upper)
        elif lower is not None:
            return series.clip_lower(lower)
        elif upper is not None:
            return series.clip_upper(upper)
        return series


def ensure_series(
    data: Union[Series, pl.DataFrame, Any],
    *,
    name: Optional[str] = None,
) -> Series:
    """
    Ensure the input is a Polars Series.

    This is a convenience function that wraps PolarsSeriesUtils.ensure_series.

    Args:
        data: Input data (Series, DataFrame column, pandas Series, or list)
        name: Name for the Series if creating new

    Returns:
        Polars Series

    Raises:
        TypeError: If data cannot be converted to Series
    """
    return PolarsSeriesUtils.ensure_series(data, name=name)


def series_from_pandas(
    pandas_series: Any,
    *,
    name: Optional[str] = None,
) -> Series:
    """
    Convert pandas Series to Polars Series.

    Args:
        pandas_series: pandas Series to convert
        name: Optional new name for the Series

    Returns:
        Polars Series
    """
    return PolarsSeriesUtils.series_from_pandas(pandas_series, name=name)
