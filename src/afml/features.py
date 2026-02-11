"""
Feature Engineer for Financial Machine Learning.

This module implements Alpha158 and Fractionally Differentiated (FFD) features
for financial machine learning models.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Optional, List

from .base import ProcessorMixin


class FeatureEngineer(ProcessorMixin):
    """
    Generates features for financial machine learning.

    Implements:
    - Alpha158: Candlestick and rolling technical features
    - FFD: Fractionally differentiated features for stationarity
    - Market Regime: Volatility, autocorrelation, entropy features

    Attributes:
        windows: Rolling window sizes for feature calculation
        selected_features_: List of selected feature names after fitting
        d_values_: Dictionary of optimal d values for FFD features

    Example:
        >>> engineer = FeatureEngineer(windows=[5, 10, 20, 30, 50])
        >>> features = engineer.fit_transform(df, labels)
    """

    def __init__(
        self,
        windows: List[int] = [5, 10, 20, 30, 50],
        ffd_check_stationarity: bool = True,
        ffd_d: float = 0.4,
    ):
        """
        Initialize the FeatureEngineer.

        Args:
            windows: List of rolling window sizes
            ffd_check_stationarity: Whether to find optimal d for FFD
            ffd_d: Fixed d value if not checking stationarity
        """
        super().__init__()
        self.windows = windows
        self.ffd_check_stationarity = ffd_check_stationarity
        self.ffd_d = ffd_d
        self.selected_features_: Optional[List[str]] = None
        self.d_values_: dict = {}

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Compute feature statistics and optionally select features.

        Args:
            df: DataFrame with OHLCV columns
            y: Labels for feature selection

        Returns:
            self
        """
        self.selected_features_ = list(df.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all configured features.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns

        Returns:
            DataFrame with generated features
        """
        features = pd.DataFrame(index=df.index)

        # Alpha158 features
        alpha_features = self._generate_alpha158(df)
        features = pd.concat([features, alpha_features], axis=1)

        # FFD features
        ffd_features = self._generate_ffd(df)
        features = pd.concat([features, ffd_features], axis=1)

        # Market regime features
        regime_features = self._generate_regime_features(df)
        features = pd.concat([features, regime_features], axis=1)

        # Normalize
        features = self._normalize_features(features)

        return features

    def _generate_alpha158(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Alpha158 candlestick and rolling features."""
        o = df["open"]
        h = df["high"]
        low = df["low"]
        c = df["close"]
        # vol defined but unused in this method - that's okay
        features = pd.DataFrame(index=df.index)

        # K-bar features
        features["KMID"] = (c - o) / o
        features["KLEN"] = (h - low) / o
        features["KMID2"] = (c - o) / (h - low + 1e-12)
        features["KUP"] = (h - np.maximum(o, c)) / o
        features["KUP2"] = (h - np.maximum(o, c)) / (h - low + 1e-12)
        features["KLOW"] = (np.minimum(o, c) - low) / o
        features["KLOW2"] = (np.minimum(o, c) - low) / (h - low + 1e-12)
        features["KSFT"] = (2 * c - h - low) / o
        features["KSFT2"] = (2 * c - h - low) / (h - low + 1e-12)

        # Rolling features
        for w in self.windows:
            rolling = self._rolling_features(df, w)
            features = pd.concat([features, rolling], axis=1)

        return features

    def _rolling_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Generate rolling features for a given window."""
        c = df["close"]
        h = df["high"]
        low = df["low"]
        vol = df["volume"]
        features = pd.DataFrame(index=df.index)

        w = window  # Create alias for formatted strings

        # Trend/Momentum
        features[f"ROC{w}"] = c.shift(w) / c
        features[f"MA{w}"] = c.rolling(w).mean() / c
        features[f"STD{w}"] = c.rolling(w).std() / c
        features[f"BETA{w}"] = self._rolling_slope(c, w) / c
        features[f"RSQR{w}"] = self._rolling_rsquare(c, w)
        features[f"RESI{w}"] = self._rolling_residual(c, w) / c

        # Price Level
        features[f"MAX{w}"] = h.rolling(w).max() / c
        features[f"MIN{w}"] = low.rolling(w).min() / c
        features[f"QTLU{w}"] = c.rolling(w).quantile(0.8) / c
        features[f"QTLD{w}"] = c.rolling(w).quantile(0.2) / c

        min_low = low.rolling(w).min()
        max_high = h.rolling(w).max()
        features[f"RSV{w}"] = (c - min_low) / (max_high - min_low + 1e-12)

        # Time-based
        features[f"IMAX{w}"] = h.rolling(w).apply(
            lambda x: (w - 1 - np.argmax(x)) / w, raw=True
        )
        features[f"IMIN{w}"] = low.rolling(w).apply(
            lambda x: (w - 1 - np.argmin(x)) / w, raw=True
        )
        features[f"IMXD{w}"] = features[f"IMAX{w}"] - features[f"IMIN{w}"]

        # Price Movement
        price_change = c - c.shift(1)
        features[f"CNTP{w}"] = (c > c.shift(1)).rolling(w).mean()
        features[f"CNTN{w}"] = (c < c.shift(1)).rolling(w).mean()
        features[f"CNTD{w}"] = features[f"CNTP{w}"] - features[f"CNTN{w}"]

        pos_change = np.maximum(price_change, 0)
        neg_change = np.maximum(-price_change, 0)
        abs_change = np.abs(price_change)

        features[f"SUMP{w}"] = pos_change.rolling(w).sum() / (
            abs_change.rolling(w).sum() + 1e-12
        )
        features[f"SUMN{w}"] = neg_change.rolling(w).sum() / (
            abs_change.rolling(w).sum() + 1e-12
        )
        features[f"SUMD{w}"] = features[f"SUMP{w}"] - features[f"SUMN{w}"]

        # Volume
        features[f"VMA{w}"] = vol.rolling(w).mean() / (vol + 1e-12)
        features[f"VSTD{w}"] = vol.rolling(w).std() / (vol + 1e-12)

        vol_change = vol - vol.shift(1)
        pos_vol_change = np.maximum(vol_change, 0)
        neg_vol_change = np.maximum(-vol_change, 0)
        abs_vol_change = np.abs(vol_change)

        features[f"VSUMP{w}"] = pos_vol_change.rolling(w).sum() / (
            abs_vol_change.rolling(w).sum() + 1e-12
        )
        features[f"VSUMN{w}"] = neg_vol_change.rolling(w).sum() / (
            abs_vol_change.rolling(w).sum() + 1e-12
        )
        features[f"VSUMD{w}"] = features[f"VSUMP{w}"] - features[f"VSUMN{w}"]

        # Correlation
        log_vol = np.log(vol + 1)
        features[f"CORR{w}"] = c.rolling(w).corr(log_vol)

        price_ret = c / c.shift(1) - 1
        vol_ret = np.log(vol / vol.shift(1) + 1)
        features[f"CORD{w}"] = price_ret.rolling(w).corr(vol_ret)

        abs_ret = np.abs(price_ret)
        weighted_vol = abs_ret * vol
        features[f"WVMA{w}"] = weighted_vol.rolling(w).std() / (
            weighted_vol.rolling(w).mean() + 1e-12
        )

        return features

    def _generate_ffd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Fractionally Differentiated features."""
        features = pd.DataFrame(index=df.index)
        content_cols = ["close", "volume"]

        # FFD features
        for col in content_cols:
            if col not in df.columns:
                continue

            series = np.log(df[col] + 1e-6)

            if self.ffd_check_stationarity:
                d = self._find_min_d(series)
            else:
                d = self.ffd_d

            self.d_values_[col] = d

            ffd_series = self._frac_diff_ffd(series, d)
            base_name = f"FFD_{col.upper()}_D{int(d * 100)}"
            features[base_name] = ffd_series

            for w in self.windows:
                features[f"{base_name}_MA{w}"] = ffd_series.rolling(w).mean()
                features[f"{base_name}_STD{w}"] = ffd_series.rolling(w).std()
                features[f"{base_name}_SLOPE{w}"] = self._rolling_slope(ffd_series, w)

        return features

    def _generate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate market regime features."""
        close = df["close"]
        log_ret = np.log(close / close.shift(1)).fillna(0)
        features = pd.DataFrame(index=df.index)

        for w in [20, 50, 100]:
            features[f"REGIME_VOL_{w}"] = log_ret.rolling(w).std()
            features[f"REGIME_AC1_{w}"] = log_ret.rolling(w).apply(
                self._autocorr_lag1, raw=True
            )
            features[f"REGIME_ENT_{w}"] = log_ret.rolling(w).apply(
                self._shannon_entropy, raw=True
            )

        return features

    def _frac_diff_ffd(
        self, series: pd.Series, d: float, thres: float = 1e-4
    ) -> pd.Series:
        """Apply Fractional Differentiation."""
        w = self._get_weights_ffd(d, thres, len(series))
        width = len(w) - 1

        if width == 0:
            return series

        output = series.rolling(window=len(w)).apply(
            lambda x: np.dot(x, w)[0], raw=True
        )

        return output

    def _get_weights_ffd(self, d: float, thres: float, lim: int) -> np.ndarray:
        """Calculate FFD weights."""
        w, k = [1.0], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
            if k >= lim:
                break
        return np.array(w[::-1]).reshape(-1, 1)

    def _find_min_d(self, series: pd.Series) -> float:
        """Find minimum d for stationarity."""
        possible_ds = np.linspace(0, 1, 11)
        for d in possible_ds:
            if d == 0:
                if self._check_stationarity(series):
                    return 0.0
                continue
            diff_series = self._frac_diff_ffd(series, d).dropna()
            if self._check_stationarity(diff_series):
                return d
        return 1.0

    def _check_stationarity(self, series: pd.Series) -> bool:
        """Run ADF test for stationarity."""
        if len(series) < 20:
            return False
        try:
            result = adfuller(series, maxlag=1, regression="c", autolag=None)
            return result[1] < 0.05
        except Exception:
            return False

    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling slope."""

        def slope(x):
            if len(x) < 2:
                return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]

        return series.rolling(window).apply(slope, raw=True)

    def _rolling_rsquare(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling R-squared."""

        def rsquare(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            y_pred = slope * y + intercept
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            if ss_tot < 1e-12:
                return 1.0
            return 1 - ss_res / ss_tot

        return series.rolling(window).apply(rsquare, raw=True)

    def _rolling_residual(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling residual."""

        def residual(x):
            if len(x) < 2:
                return np.nan
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            return x[-1] - (slope * (len(x) - 1) + intercept)

        return series.rolling(window).apply(residual, raw=True)

    def _autocorr_lag1(self, x: np.ndarray) -> float:
        """Calculate lag-1 autocorrelation."""
        if len(x) < 2:
            return 0.0
        var = np.var(x)
        if var < 1e-9:
            return 0.0
        mean = np.mean(x)
        cov = np.mean((x[:-1] - mean) * (x[1:] - mean))
        return cov / var

    def _shannon_entropy(self, x: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy."""
        if len(x) < 2:
            return 0.0
        try:
            hist, _ = np.histogram(x, bins=bins, density=True)
            p = hist / hist.sum()
            p = p[p > 0]
            if len(p) == 0:
                return 0.0
            return -np.sum(p * np.log(p))
        except Exception:
            return 0.0

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Standardize features."""
        features = features.ffill().fillna(0)
        mean = features.mean()
        std = features.std().replace(0, 1)
        return (features - mean) / std

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        if self.selected_features_ is None:
            raise ValueError("Must fit transformer first.")
        return self.selected_features_
