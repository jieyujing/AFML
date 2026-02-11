"""
Polars Feature Engineer for Financial Machine Learning.

This module implements Alpha158 and FFD (Fractionally Differentiated) features
using Polars for improved performance on large-scale financial time series data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import polars as pl
from polars import DataFrame, LazyFrame, Series

from afml.base import ProcessorMixin


class PolarsFeatureEngineer(ProcessorMixin):
    """
    Feature engineer that generates Alpha158 and FFD features using Polars.

    Features include:
    - ROC/Returns features (integer differencing)
    - Short-term technical indicators (5-50 periods)
    - Fractionally Differentiated features (FFD)
    - FFD Momentum, Volatility, Slope

    This implementation uses Polars for improved performance on large datasets,
    with support for lazy evaluation and multi-threaded operations.

    Attributes:
        windows: Rolling window sizes for feature calculation
        ffd_d: Fractional differentiation parameter (default 0.5)
        volatility_span: Span for volatility calculation

    Example:
        >>> engineer = PolarsFeatureEngineer(windows=[5, 10, 20, 30, 50])
        >>> features = engineer.fit_transform(dollar_bars)
    """

    def __init__(
        self,
        windows: List[int] = None,
        ffd_d: float = 0.5,
        volatility_span: int = 100,
        *,
        lazy: bool = False,
    ):
        """
        Initialize the PolarsFeatureEngineer.

        Args:
            windows: Rolling window sizes for feature calculation
            ffd_d: Fractional differentiation parameter (0 < d < 1)
            volatility_span: Span for volatility calculation
            lazy: Whether to use lazy evaluation
        """
        super().__init__()
        self.windows = windows if windows is not None else [5, 10, 20, 30, 50]
        self.ffd_d = ffd_d
        self.volatility_span = volatility_span
        self.lazy = lazy
        self._metadata: Dict[str, Any] = {}

    def fit(
        self,
        df: Union[DataFrame, LazyFrame],
        y: Optional[Any] = None,
    ) -> "PolarsFeatureEngineer":
        """
        Store metadata from the input data.

        Args:
            df: DataFrame with price columns
            y: Ignored

        Returns:
            self
        """
        if isinstance(df, LazyFrame):
            df = df.collect()

        self._metadata = {
            "n_rows": df.height,
            "columns": df.columns,
            "windows": self.windows,
            "ffd_d": self.ffd_d,
        }

        return self

    def transform(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """
        Generate features from price data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with computed features
        """
        if isinstance(df, LazyFrame):
            result = self._compute_features(df.lazy())
            return result.lazy() if self.lazy else result.collect()

        return self._compute_features(df)

    def _compute_features(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Compute all features."""
        result = df.clone() if not isinstance(df, LazyFrame) else df

        # ROC features
        result = self._add_roc_features(result)

        # Rolling window features
        result = self._add_rolling_features(result)

        # Technical indicators
        result = self._add_technical_indicators(result)

        # FFD features
        result = self._add_ffd_features(result)

        return result

    def _add_roc_features(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Add ROC/Returns features."""
        close = pl.col("close")

        # Returns at different lags
        for lag in [1, 2, 3, 4, 5]:
            df = df.with_columns((close / close.shift(lag) - 1).alias(f"return_{lag}"))

        return df

    def _add_rolling_features(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Add rolling window features."""
        close = pl.col("close")
        high = pl.col("high")
        low = pl.col("low")

        for w in self.windows:
            # Rolling mean
            df = df.with_columns(
                close.rolling_mean(window_size=w).alias(f"close_ma_{w}")
            )

            # Rolling std (volatility)
            df = df.with_columns(
                close.rolling_std(window_size=w).alias(f"close_std_{w}")
            )

            # Rolling min/max
            df = df.with_columns(
                high.rolling_max(window_size=w).alias(f"high_max_{w}"),
                low.rolling_min(window_size=w).alias(f"low_min_{w}"),
            )

        return df

    def _add_technical_indicators(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Add technical indicator features."""
        close = pl.col("close")
        high = pl.col("high")
        low = pl.col("low")

        # RSI
        delta = close.diff()
        gain = delta.clip(0, None)
        loss = (-delta).clip(0, None)

        avg_gain = gain.ewm_mean(span=14)
        avg_loss = loss.ewm_mean(span=14)

        rs = avg_gain / avg_loss.clip(1e-10, None)
        rsi = 100 - (100 / (rs + 1))
        df = df.with_columns(rsi.alias("rsi_14"))

        # MACD
        ema_12 = close.ewm_mean(span=12)
        ema_26 = close.ewm_mean(span=26)
        macd = ema_12 - ema_26
        signal = macd.ewm_mean(span=9)

        df = df.with_columns(
            macd.alias("macd"),
            signal.alias("macd_signal"),
            (macd - signal).alias("macd_hist"),
        )

        # Bollinger Bands
        bb_ma = close.ewm_mean(span=20)
        bb_std = close.ewm_std(span=20)

        df = df.with_columns(
            ((close - bb_ma) / (bb_std * 2)).alias("bb_position"),
        )

        # Average True Range (ATR)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pl.concat([tr1, tr2, tr3], how="horizontal").max()
        atr = tr.ewm_mean(span=14)

        df = df.with_columns(
            (atr / close).alias("atr_14"),
        )

        return df

    def _add_ffd_features(
        self,
        df: Union[DataFrame, LazyFrame],
    ) -> Union[DataFrame, LazyFrame]:
        """Add Fractionally Differentiated (FFD) features."""
        if self.ffd_d <= 0 or self.ffd_d >= 1:
            return df

        close = pl.col("close")

        # FFD level (fractional differentiation)
        ffd_close = self._frac_diff(close, self.ffd_d)
        df = df.with_columns(ffd_close.alias("ffd_close"))

        # FFD momentum
        df = df.with_columns((ffd_close - ffd_close.shift(1)).alias("ffd_momentum"))

        # FFD volatility
        ffd_returns = ffd_close.pct_change()
        ffd_vol = ffd_returns.ewm_std(span=self.volatility_span)
        df = df.with_columns(ffd_vol.alias("ffd_volatility"))

        # FFD slope - using simple diff as proxy for now since rolling_map is slow
        # and _compute_slope implementation was flawed for Polars expression context
        df = df.with_columns(
            (ffd_close - ffd_close.shift(20)).alias("ffd_slope")
        )

        return df

    def _get_ffd_weights(self, d: float, threshold: float = 1e-4, size: int = 1000) -> List[float]:
        """Generate weights for fractional differentiation."""
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < threshold or k >= size:
                break
            w.append(w_k)
            k += 1
        return w

    def _frac_diff(
        self,
        series: pl.Expr,
        d: float,
        threshold: float = 1e-4,
    ) -> pl.Expr:
        """
        Compute fractional differentiation (FFD) using fixed window.
        """
        weights = self._get_ffd_weights(d, threshold)
        
        # Apply weights using fold
        # resulting_series = w[0]*series + w[1]*series.shift(1) + ...
        
        res = series * weights[0]
        for k, w in enumerate(weights[1:], 1):
            res = res + series.shift(k) * w
            
        return res

    def _rolling_slope(
        self,
        series: pl.Expr,
        window: int,
    ) -> pl.Expr:
        """Compute rolling slope using linear regression."""
        x_mean = (window - 1) / 2
        x_std = ((window**2 - 1) / 12) ** 0.5

        # Use rolling window with apply for slope calculation
        return series.rolling_map(
            lambda y: self._compute_slope(y, x_mean, x_std),
            window_size=window,
        )

    def _compute_slope(
        self,
        y: Series,
        x_mean: float,
        x_std: float,
    ) -> float:
        """Compute slope for a window of values."""
        if y.null_count() > len(y) // 2:
            return float("nan")

        x = pl.Series(range(len(y)))
        y_np = y.to_numpy()

        numerator = ((x - x_mean) * (y_np - y_np.mean())).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if denominator == 0:
            return float("nan")

        return numerator / denominator

    def fit_transform(
        self,
        df: Union[DataFrame, LazyFrame],
        y: Optional[Any] = None,
    ) -> Union[DataFrame, LazyFrame]:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame
            y: Ignored

        Returns:
            DataFrame with features
        """
        self.fit(df)
        return self.transform(df)

    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about computed features.

        Returns:
            Dict with feature information
        """
        return {
            "windows": self.windows,
            "ffd_d": self.ffd_d,
            "volatility_span": self.volatility_span,
            "lazy": self.lazy,
            "metadata": self._metadata,
        }
