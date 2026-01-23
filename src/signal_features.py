"""
Signal-Based Features for Financial Machine Learning

Convert traditional technical signals into continuous features for ML models.
Instead of using signals as binary trading rules, we extract their strength
and confidence as features.

Implements:
1. Moving Average Signals (Crossover strength, Distance)
2. Bollinger Bands (Position, Bandwidth, %B)
3. RSI (Level, Divergence)
4. MACD (Signal line, Histogram)
5. ATR (Volatility regime)

Reference:
- AFML Chapter 19: Feature Engineering
- Murphy (1999): Technical Analysis of Financial Markets
"""

import pandas as pd
import numpy as np
import os
from typing import List


class SignalFeatureGenerator:
    """
    Generate continuous features from technical signals.

    Philosophy: Instead of "MA crosses up -> BUY", we generate:
    - Distance between MAs (how strong is the divergence?)
    - Slope of MA (how fast is the trend accelerating?)
    - Days since last cross (how fresh is the signal?)
    """

    def __init__(
        self,
        ma_windows: List[int] = [10, 20, 50],
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_window: int = 14,
    ):
        """
        Args:
            ma_windows: Moving average windows
            bb_window: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            rsi_window: RSI period
            macd_fast: MACD fast EMA period
            macd_slow: MACD slow EMA period
            macd_signal: MACD signal line period
            atr_window: ATR period
        """
        self.ma_windows = ma_windows
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_window = atr_window

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signal-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with signal features
        """
        print(f"   [Signal Features] Generating features...")

        features = pd.DataFrame(index=df.index)
        close = df["close"]

        # 1. Moving Average Features
        print(f"      -> MA Features (windows: {self.ma_windows})")
        ma_features = self._ma_features(close)
        features = pd.concat([features, ma_features], axis=1)

        # 2. Bollinger Bands Features
        print(f"      -> Bollinger Bands (window: {self.bb_window}, std: {self.bb_std})")
        bb_features = self._bollinger_features(close)
        features = pd.concat([features, bb_features], axis=1)

        # 3. RSI Features
        print(f"      -> RSI (window: {self.rsi_window})")
        rsi_features = self._rsi_features(close)
        features = pd.concat([features, rsi_features], axis=1)

        # 4. MACD Features
        print(
            f"      -> MACD (fast: {self.macd_fast}, slow: {self.macd_slow}, signal: {self.macd_signal})"
        )
        macd_features = self._macd_features(close)
        features = pd.concat([features, macd_features], axis=1)

        # 5. ATR Features
        print(f"      -> ATR (window: {self.atr_window})")
        atr_features = self._atr_features(df)
        features = pd.concat([features, atr_features], axis=1)

        return features

    def _ma_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate MA-based features (not just crossover signals).

        Features:
        - MA Distance: (Close - MA) / Close (normalized distance)
        - MA Slope: Rate of change of MA
        - MA Cross Strength: Distance between fast and slow MA
        """
        features = pd.DataFrame(index=close.index)

        # Calculate MAs
        mas = {}
        for w in self.ma_windows:
            mas[w] = close.rolling(w).mean()

        # 1. Distance from each MA
        for w in self.ma_windows:
            features[f"MA_DIST_{w}"] = (close - mas[w]) / close

        # 2. MA Slope (normalized)
        for w in self.ma_windows:
            ma_change = mas[w].diff()
            features[f"MA_SLOPE_{w}"] = ma_change / close

        # 3. MA Cross Strength (for pairs)
        # Example: MA10 vs MA20, MA20 vs MA50
        if len(self.ma_windows) >= 2:
            for i in range(len(self.ma_windows) - 1):
                fast = self.ma_windows[i]
                slow = self.ma_windows[i + 1]
                # Positive = Fast above Slow (Bullish), Negative = Bearish
                features[f"MA_CROSS_{fast}_{slow}"] = (mas[fast] - mas[slow]) / close

        return features

    def _bollinger_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate Bollinger Bands features.

        Features:
        - %B: Position within bands (0 = lower, 0.5 = middle, 1 = upper)
        - Bandwidth: (Upper - Lower) / Middle (volatility measure)
        - Distance from bands
        """
        features = pd.DataFrame(index=close.index)

        # Calculate Bollinger Bands
        ma = close.rolling(self.bb_window).mean()
        std = close.rolling(self.bb_window).std()
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        # 1. %B (Position within bands)
        # %B = (Close - Lower) / (Upper - Lower)
        # %B > 1: Above upper band
        # 0 < %B < 1: Within bands
        # %B < 0: Below lower band
        features["BB_PERCENT_B"] = (close - lower) / (upper - lower + 1e-9)

        # 2. Bandwidth (Normalized volatility)
        features["BB_BANDWIDTH"] = (upper - lower) / (ma + 1e-9)

        # 3. Distance from upper/lower bands
        features["BB_DIST_UPPER"] = (upper - close) / close
        features["BB_DIST_LOWER"] = (close - lower) / close

        return features

    def _rsi_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate RSI features.

        Features:
        - RSI: Relative Strength Index
        - RSI Centered: RSI - 50 (centered around 0)
        - RSI Extreme: Distance from overbought/oversold zones
        """
        features = pd.DataFrame(index=close.index)

        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(self.rsi_window).mean()
        avg_loss = loss.rolling(self.rsi_window).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        # 1. Raw RSI
        features["RSI"] = rsi

        # 2. RSI Centered (easier for ML to learn)
        features["RSI_CENTERED"] = rsi - 50

        # 3. RSI Extreme (distance from overbought/oversold)
        # Positive = Overbought territory, Negative = Oversold territory
        features["RSI_EXTREME_UPPER"] = np.maximum(rsi - 70, 0)
        features["RSI_EXTREME_LOWER"] = np.minimum(rsi - 30, 0)

        return features

    def _macd_features(self, close: pd.Series) -> pd.DataFrame:
        """
        Generate MACD features.

        Features:
        - MACD Line
        - Signal Line
        - MACD Histogram
        - MACD Slope
        """
        features = pd.DataFrame(index=close.index)

        # Calculate MACD
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # Normalize by price
        features["MACD_LINE"] = macd_line / close
        features["MACD_SIGNAL"] = signal_line / close
        features["MACD_HISTOGRAM"] = macd_histogram / close

        # MACD slope (rate of change)
        features["MACD_SLOPE"] = macd_histogram.diff() / close

        return features

    def _atr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ATR (Average True Range) features.

        ATR measures volatility - useful for regime detection.

        Features:
        - ATR (raw)
        - ATR normalized by price
        - ATR percentile (relative volatility)
        """
        features = pd.DataFrame(index=df.index)

        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR (Exponential Moving Average of TR)
        atr = tr.ewm(span=self.atr_window, adjust=False).mean()

        # 1. ATR normalized by price
        features["ATR_NORM"] = atr / close

        # 2. ATR percentile (rolling 100 bars)
        features["ATR_PERCENTILE"] = atr.rolling(100).apply(
            lambda x: (x[-1] <= x).sum() / len(x), raw=True
        )

        # 3. ATR trend (is volatility increasing?)
        features["ATR_SLOPE"] = (atr - atr.shift(5)) / close

        return features


def main():
    """Generate signal-based features for dollar bars."""
    print("=" * 80)
    print("Signal-Based Feature Engineering")
    print("=" * 80)

    # 1. Load dollar bars
    print("\n1. Loading dollar bars...")
    try:
        df = pd.read_csv(os.path.join("data", "output", "dynamic_dollar_bars.csv"), index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'dynamic_dollar_bars.csv' not found.")
        return

    print(f"   Loaded {len(df)} bars")

    # 2. Generate signal features
    print("\n2. Generating signal-based features...")
    signal_gen = SignalFeatureGenerator()
    features_signal = signal_gen.generate(df)

    print(f"   Generated {len(features_signal.columns)} signal features")

    # 3. Save results
    print("\n3. Saving signal features...")
    features_signal.to_csv(os.path.join("data", "output", "features_signal.csv"))
    print("   ✓ Saved to: features_signal.csv")

    # 4. Display statistics
    print("\n" + "=" * 80)
    print("SIGNAL FEATURE SUMMARY")
    print("-" * 80)
    print(f"Total Features: {len(features_signal.columns)}")
    print("\nFeature Preview:")
    print(features_signal.describe())

    print("\n" + "=" * 80)
    print("✓ Signal Feature Engineering Complete!")


if __name__ == "__main__":
    main()
