"""
Alpha158 Feature Engineering & Fractional Differentiation for Financial Machine Learning

This module implements:
1. Qlib's Alpha158 feature set (adapted for Dollar Bars)
2. Fractional Differentiation (FFD) based on AFML Chapter 5
3. Structural Break features

Reference:
- Qlib: https://github.com/microsoft/qlib
- AFML Chapter 5: Fractionally Differentiated Features
- AFML Chapter 19: Feature Engineering
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from statsmodels.tsa.stattools import adfuller


class FracDiffFeatureGenerator:
    """
    Implements Fractional Differentiation (FFD) to generate stationary features
    while preserving maximum memory (AFML Chapter 5).
    """

    def __init__(
        self, 
        check_stationarity: bool = True,
        windows: list[int] = [5, 10, 20, 30, 50],
        normalize: bool = True
    ):
        self.check_stationarity = check_stationarity
        self.windows = windows
        self.normalize = normalize
        self.d_values = {}

    def generate(self, df: pd.DataFrame, content_cols: List[str] = ["close", "volume"]) -> pd.DataFrame:
        """
        Generate fractionally differentiated features for specified columns.
        Applies Log transform before differencing for price and volume.
        """
        features = pd.DataFrame(index=df.index)

        print(f"   [FFD] Optimizing d values for: {content_cols}")
        
        for col in content_cols:
            if col not in df.columns:
                continue
                
            # 1. Log transform (standard practice for price/volume)
            series = np.log(df[col] + 1e-6)  # constant to avoid log(0)
            
            # 2. Find optimal d
            d = self._find_min_d(series) if self.check_stationarity else 0.4
            self.d_values[col] = d
            print(f"      -> {col}: d={d:.2f}")

            # 3. Apply FFD (Raw Stationary Series)
            # This is the "Level" feature that preserves memory
            ffd_series = self.frac_diff_ffd(series, d)
            base_feature_name = f"FFD_{col.upper()}_D{int(d*100)}"
            features[base_feature_name] = ffd_series
            
            # 4. Generate Rolling Features on the FFD Series
            # This replaces ROC/Momentum with FFD-based Momentum
            for w in self.windows:
                # Trend: Moving Average of FFD series
                features[f"{base_feature_name}_MA{w}"] = ffd_series.rolling(w).mean()
                
                # Volatility: Standard Deviation of FFD series
                features[f"{base_feature_name}_STD{w}"] = ffd_series.rolling(w).std()
                
                # Drift/Slope: Slope of FFD series (Local Trend)
                features[f"{base_feature_name}_SLOPE{w}"] = self._rolling_slope(ffd_series, w)

        # 5. Normalize FFD features (Standardize)
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def get_weights_ffd(self, d: float, thres: float, lim: int) -> np.ndarray:
        """
        Calculate weights for Fractional Differentiation (Fixed Window).
        Ref: AFML Chapter 5, Section 5.5
        """
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
            if k >= lim:
                break
        # Weights are applied in reverse order (w_0 * x_t + w_1 * x_{t-1} ...)
        # The window should be such that w[0] matches current price
        return np.array(w[::-1]).reshape(-1, 1)

    def frac_diff_ffd(self, series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
        """
        Apply Fractional Differentiation (Fixed Window).
        """
        # 1. Get weights
        # lim is set to len(series) to ensure we cover enough, but actually loop stops by thres
        w = self.get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        
        if width == 0:
            return series
            
        print(f"      (window size: {width} bars)")

        # 2. Apply weights via rolling window
        # Note: This effectively computes sum(w[i] * x[t-i])
        output = series.rolling(window=len(w)).apply(
            lambda x: np.dot(x, w)[0], 
            raw=True
        )
        
        return output

    def _find_min_d(self, series: pd.Series) -> float:
        """Find minimum differencing order d in [0, 1] that passes ADF test."""
        possible_ds = np.linspace(0, 1, 11)
        for d in possible_ds:
            if d == 0:
                if self._check_stationarity(series): return 0.0
                continue
            diff_series = self.frac_diff_ffd(series, d).dropna()
            if self._check_stationarity(diff_series): return d
        return 1.0

    def _check_stationarity(self, series: pd.Series) -> bool:
        """Run Augmented Dickey-Fuller test (p < 0.05)."""
        if len(series) < 20: return False
        try:
            result = adfuller(series, maxlag=1, regression='c', autolag=None)
            return result[1] < 0.05
        except Exception:
            return False

    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear regression slope optimized for FFD series."""
        def slope(x):
            if len(x) < 2: return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(window).apply(slope, raw=True)

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.ffill().fillna(0)
        mean = features.mean()
        std = features.std().replace(0, 1)
        return (features - mean) / std


class Alpha158FeatureGenerator:
    """
    Standalone implementation of Qlib Alpha158 feature set.
    """

    def __init__(
        self,
        windows: list[int] = [5, 10, 20, 30, 50],
        normalize: bool = True,
    ):
        self.windows = windows
        self.normalize = normalize

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all Alpha158 features."""
        # Validate input
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        features = pd.DataFrame(index=df.index)

        # 1. K-Bar features (candlestick patterns)
        kbar = self._kbar_features(df)
        features = pd.concat([features, kbar], axis=1)

        # 2. Rolling features for each window
        for window in self.windows:
            rolling = self._rolling_features(df, window)
            features = pd.concat([features, rolling], axis=1)

        # 3. Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)

        return features

    def _kbar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        features = pd.DataFrame(index=df.index)

        features["KMID"] = (c - o) / o
        features["KLEN"] = (h - l) / o
        features["KMID2"] = (c - o) / (h - l + 1e-12)
        features["KUP"] = (h - np.maximum(o, c)) / o
        features["KUP2"] = (h - np.maximum(o, c)) / (h - l + 1e-12)
        features["KLOW"] = (np.minimum(o, c) - l) / o
        features["KLOW2"] = (np.minimum(o, c) - l) / (h - l + 1e-12)
        features["KSFT"] = (2 * c - h - l) / o
        features["KSFT2"] = (2 * c - h - l) / (h - l + 1e-12)

        return features

    def _rolling_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        features = pd.DataFrame(index=df.index)
        w = window

        # === Trend/Momentum ===
        features[f"ROC{w}"] = c.shift(w) / c
        features[f"MA{w}"] = c.rolling(w).mean() / c
        features[f"STD{w}"] = c.rolling(w).std() / c
        features[f"BETA{w}"] = self._rolling_slope(c, w) / c
        features[f"RSQR{w}"] = self._rolling_rsquare(c, w)
        features[f"RESI{w}"] = self._rolling_residual(c, w) / c

        # === Price Level ===
        features[f"MAX{w}"] = h.rolling(w).max() / c
        features[f"MIN{w}"] = l.rolling(w).min() / c
        features[f"QTLU{w}"] = c.rolling(w).quantile(0.8) / c
        features[f"QTLD{w}"] = c.rolling(w).quantile(0.2) / c
        
        min_low = l.rolling(w).min()
        max_high = h.rolling(w).max()
        features[f"RSV{w}"] = (c - min_low) / (max_high - min_low + 1e-12)

        # === Time-based ===
        features[f"IMAX{w}"] = h.rolling(w).apply(lambda x: (w - 1 - np.argmax(x)) / w, raw=True)
        features[f"IMIN{w}"] = l.rolling(w).apply(lambda x: (w - 1 - np.argmin(x)) / w, raw=True)
        features[f"IMXD{w}"] = features[f"IMAX{w}"] - features[f"IMIN{w}"]

        # === Price Movement ===
        price_change = c - c.shift(1)
        features[f"CNTP{w}"] = (c > c.shift(1)).rolling(w).mean()
        features[f"CNTN{w}"] = (c < c.shift(1)).rolling(w).mean()
        features[f"CNTD{w}"] = features[f"CNTP{w}"] - features[f"CNTN{w}"]
        
        pos_change = np.maximum(price_change, 0)
        neg_change = np.maximum(-price_change, 0)
        abs_change = np.abs(price_change)
        
        features[f"SUMP{w}"] = pos_change.rolling(w).sum() / (abs_change.rolling(w).sum() + 1e-12)
        features[f"SUMN{w}"] = neg_change.rolling(w).sum() / (abs_change.rolling(w).sum() + 1e-12)
        features[f"SUMD{w}"] = features[f"SUMP{w}"] - features[f"SUMN{w}"]

        # === Volume ===
        features[f"VMA{w}"] = v.rolling(w).mean() / (v + 1e-12)
        features[f"VSTD{w}"] = v.rolling(w).std() / (v + 1e-12)
        
        vol_change = v - v.shift(1)
        pos_vol_change = np.maximum(vol_change, 0)
        neg_vol_change = np.maximum(-vol_change, 0)
        abs_vol_change = np.abs(vol_change)

        features[f"VSUMP{w}"] = pos_vol_change.rolling(w).sum() / (abs_vol_change.rolling(w).sum() + 1e-12)
        features[f"VSUMN{w}"] = neg_vol_change.rolling(w).sum() / (abs_vol_change.rolling(w).sum() + 1e-12)
        features[f"VSUMD{w}"] = features[f"VSUMP{w}"] - features[f"VSUMN{w}"]

        # === Correlation ===
        log_vol = np.log(v + 1)
        features[f"CORR{w}"] = c.rolling(w).corr(log_vol)
        
        price_ret = c / c.shift(1) - 1
        vol_ret = np.log(v / v.shift(1) + 1)
        features[f"CORD{w}"] = price_ret.rolling(w).corr(vol_ret)
        
        abs_ret = np.abs(price_ret)
        weighted_vol = abs_ret * v
        features[f"WVMA{w}"] = weighted_vol.rolling(w).std() / (weighted_vol.rolling(w).mean() + 1e-12)

        return features

    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        def slope(x):
            if len(x) < 2: return np.nan
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        return series.rolling(window).apply(slope, raw=True)

    def _rolling_rsquare(self, series: pd.Series, window: int) -> pd.Series:
        def rsquare(x):
            if len(x) < 2: return np.nan
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            y_pred = slope * y + intercept
            ss_res = np.sum((x - y_pred) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            if ss_tot < 1e-12: return 1.0
            return 1 - ss_res / ss_tot
        return series.rolling(window).apply(rsquare, raw=True)

    def _rolling_residual(self, series: pd.Series, window: int) -> pd.Series:
        def residual(x):
            if len(x) < 2: return np.nan
            y = np.arange(len(x))
            slope, intercept = np.polyfit(y, x, 1)
            return x[-1] - (slope * (len(x) - 1) + intercept)
        return series.rolling(window).apply(residual, raw=True)

    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.ffill().fillna(0)
        mean = features.mean()
        std = features.std().replace(0, 1)
        return (features - mean) / std


class MarketRegimeFeatureGenerator:
    """
    Generates features capturing market regimes (AFML Chapter 18 & 19):
    1. Volatility (Risk)
    2. Serial Correlation (Trendiness/Efficiency)
    3. Market Entropy (Information content/Complexity)
    """
    
    def __init__(
        self, 
        windows: list[int] = [20, 50, 100]
    ):
        self.windows = windows

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        # Pre-compute log returns for calculation
        close = df["close"]
        log_ret = np.log(close / close.shift(1)).fillna(0)
        
        print(f"   [Regime] Generating features for windows: {self.windows}")
        
        for w in self.windows:
            # 1. Volatility (Realized Volatility)
            # Annualized volatility assuming daily bars? 
            # We stick to raw std per window for ML features (normalization comes later)
            features[f"REGIME_VOL_{w}"] = log_ret.rolling(w).std()
            
            # 2. Serial Correlation (Autocorrelation Lag 1)
            # Measures efficiency: 0 = Random walk, +ve = Trend, -ve = Mean Reversion
            features[f"REGIME_AC1_{w}"] = log_ret.rolling(w).apply(
                lambda x: self._autocorr_lag1(x), raw=True
            )
            
            # 3. Market Entropy (Shannon Entropy of returns)
            features[f"REGIME_ENT_{w}"] = log_ret.rolling(w).apply(
                lambda x: self._shannon_entropy(x), raw=True
            )
            
            # 4. Hurst Exponent (simplified rolling proxy)
            # Full Hurst is expensive. We can use Volatility scaling as proxy:
            # H ~ log(Range) / log(Interval) or similar.
            # Here we skip it to keep it fast, AC1 and Entropy cover similar ground.

        return features

    def _autocorr_lag1(self, x: np.ndarray) -> float:
        """Calculate lag-1 autocorrelation safely."""
        if len(x) < 2: return 0.0
        # Manual calculation is faster than pd.Series.autocorr inside rolling
        var = np.var(x)
        if var < 1e-9: return 0.0
        mean = np.mean(x)
        # (x_t - mu)(x_{t-1} - mu)
        cov = np.mean((x[:-1] - mean) * (x[1:] - mean))
        return cov / var

    def _shannon_entropy(self, x: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Shannon Entropy of the distribution of returns.
        High entropy = Random/Noisy
        Low entropy = Deterministic/Structured
        """
        if len(x) < 2: return 0.0
        try:
            # Discretize into bins
            hist, _ = np.histogram(x, bins=bins, density=True)
            # Convert to probabilities (approximate)
            p = hist / hist.sum()
            # Remove zeros for log
            p = p[p > 0]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log(p))
        except:
            return 0.0


def main():
    """Generate comprehensive features for labeled dollar bars."""
    print("=" * 80)
    print("Feature Engineering: Alpha158 + FFD Momentum + Market Regime")
    print("=" * 80)

    # 1. Load labeled dollar bars
    print("\n1. Loading labeled dollar bars...")
    try:
        df = pd.read_csv("dollar_bars_labeled.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'dollar_bars_labeled.csv' not found.")
        return

    print(f"   Loaded {len(df)} labeled bars")

    # 2. Generate Alpha158 features (Keeping legacy for comparison)
    print("\n2. Generating Alpha158 features...")
    alpha_gen = Alpha158FeatureGenerator()
    features_alpha = alpha_gen.generate(df)
    print(f"   Alpha158: {len(features_alpha.columns)} features")

    # 3. Generate Fractional Differentiation features (replacing high-diff features)
    print("\n3. Generating FFD-based Momentum features...")
    # Using check_stationarity=True for rigorous implementation
    # Adding normalization and windows
    ffd_gen = FracDiffFeatureGenerator(
        check_stationarity=True,
        windows=[5, 10, 20, 30, 50],
        normalize=True
    )
    features_ffd = ffd_gen.generate(df, content_cols=["close", "volume"])
    print(f"   FFD: {len(features_ffd.columns)} features (including rolling stats)")

    # 4. Generate Market Regime features (NEW)
    print("\n4. Generating Market Regime features...")
    regime_gen = MarketRegimeFeatureGenerator(windows=[20, 50, 100])
    features_regime = regime_gen.generate(df)
    print(f"   Regime: {len(features_regime.columns)} features")

    # Merge features
    features = pd.concat([features_alpha, features_ffd, features_regime], axis=1)

    # 5. Combine with labels
    print("\n5. Combining features with labels...")
    result = features.copy()
    result["label"] = df["label"]
    result["ret"] = df["ret"]

    # Drop rows with NaN
    original_len = len(result)
    result = result.dropna()
    print(f"   Dropped {original_len - len(result)} rows (warmup period)")
    print(f"   Final dataset: {len(result)} samples")

    # 6. Save results
    output_combined = "features_labeled.csv"
    result.to_csv(output_combined)
    print(f"\n   ✓ Saved combined dataset to: {output_combined}")

    # 7. Detailed Statistics
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("-" * 80)
    
    # Feature counts
    print(f"Total Features:     {len(features.columns)}")
    print(f"  - Alpha158:       {len(features_alpha.columns)}")
    print(f"  - FFD (Stationary): {len(features_ffd.columns)}")
    print(f"  - Regime (State):   {len(features_regime.columns)}")
    
    # FFD Parameter Summary
    print("\nOptimized FFD Parameters (Memory Preservation):")
    for col, d in ffd_gen.d_values.items():
        # Get weight window size for this d
        w = ffd_gen.get_weights_ffd(d, 1e-4, len(df))
        print(f"  {col.upper():<10} | d = {d:.2f} | Memory Window: {len(w):>4} bars")

    # Feature-Label Correlation
    print("\nTop 15 Features by Label Correlation:")
    correlations = result.corr()["label"].abs().sort_values(ascending=False)
    # Exclude 'label' and 'ret' from the correlation list
    top_corr = correlations.drop(["label", "ret"], errors="ignore").head(15)
    
    print(f"{'Feature Name':<30} | Abs Correlation")
    print("-" * 50)
    for name, corr in top_corr.items():
        print(f"{name:<30} | {corr:.4f}")

    # Data Quality Check
    print("\nData Distribution Summary (Labels):")
    label_counts = result["label"].value_counts().sort_index()
    total_samples = len(result)
    for label, count in label_counts.items():
        print(f"  Class {label:>2}: {count:>6} samples ({count/total_samples*100:>5.1f}%)")

    print("\n" + "=" * 80)
    print("✓ Feature Engineering Complete!")

if __name__ == "__main__":
    main()
