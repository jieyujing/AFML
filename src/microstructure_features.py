"""
Microstructure Features for Financial Machine Learning

Implements advanced market microstructure metrics from AFML Chapter 19:
1. VPIN (Volume-Synchronized Probability of Informed Trading)
2. Kyle's Lambda (Price Impact per Unit Volume)
3. Amihud's Lambda (Illiquidity Ratio)
4. Roll Measure (Bid-Ask Spread Estimator)
5. Corwin-Schultz Spread Estimator

References:
- AFML Chapter 19: Microstructural Features
- Easley et al. (2012): Flow Toxicity
- Kyle (1985): Price Impact Model
- Amihud (2002): Illiquidity Measure
"""

import pandas as pd
import numpy as np
import os
from typing import List


class MicrostructureFeatureGenerator:
    """
    Generate market microstructure features that capture:
    - Information asymmetry (VPIN)
    - Price impact (Kyle's Lambda)
    - Liquidity (Amihud's Lambda, Roll Measure)
    - Transaction costs (Spread estimators)
    """

    def __init__(
        self,
        vpin_buckets: int = 50,
        windows: List[int] = [20, 50, 100],
    ):
        """
        Args:
            vpin_buckets: Number of volume buckets for VPIN calculation
            windows: Rolling windows for aggregated metrics
        """
        self.vpin_buckets = vpin_buckets
        self.windows = windows

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all microstructure features.

        Args:
            df: DataFrame with OHLCV data (Dollar Bars)

        Returns:
            DataFrame with microstructure features
        """
        print(f"   [Microstructure] Generating features...")

        features = pd.DataFrame(index=df.index)

        # 1. VPIN (Volume-Synchronized Probability of Informed Trading)
        print(f"      -> VPIN (Buckets: {self.vpin_buckets})")
        features["VPIN"] = self._calculate_vpin(df)

        # 2. Kyle's Lambda (Price Impact)
        print(f"      -> Kyle's Lambda")
        features["KYLE_LAMBDA"] = self._calculate_kyle_lambda(df)

        # 3. Amihud's Lambda (Illiquidity)
        print(f"      -> Amihud's Lambda")
        features["AMIHUD_LAMBDA"] = self._calculate_amihud_lambda(df)

        # 4. Roll Measure (Effective Spread Estimator)
        print(f"      -> Roll Measure")
        features["ROLL_MEASURE"] = self._calculate_roll_measure(df)

        # 5. Corwin-Schultz Spread (High-Low Spread Estimator)
        print(f"      -> Corwin-Schultz Spread")
        features["CORWIN_SCHULTZ_SPREAD"] = self._calculate_corwin_schultz_spread(df)

        # 6. Generate Rolling Aggregations
        print(f"      -> Rolling aggregations (windows: {self.windows})")
        for w in self.windows:
            # VPIN
            features[f"VPIN_MA{w}"] = features["VPIN"].rolling(w).mean()
            features[f"VPIN_STD{w}"] = features["VPIN"].rolling(w).std()

            # Kyle's Lambda
            features[f"KYLE_MA{w}"] = features["KYLE_LAMBDA"].rolling(w).mean()

            # Amihud's Lambda
            features[f"AMIHUD_MA{w}"] = features["AMIHUD_LAMBDA"].rolling(w).mean()

        return features

    def _calculate_vpin(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate VPIN (Volume-Synchronized Probability of Informed Trading).

        VPIN measures order flow toxicity by estimating the fraction of
        informed trading volume. High VPIN indicates high information asymmetry.

        Reference: Easley et al. (2012) - "Flow Toxicity and Liquidity in a High Frequency World"

        Algorithm:
        1. Classify volume as buy/sell based on price direction
        2. Bucket volume into fixed-size buckets
        3. Calculate |Buy Volume - Sell Volume| / Total Volume per bucket
        4. VPIN = Rolling average of the imbalance ratio

        Args:
            df: DataFrame with close and volume

        Returns:
            VPIN series
        """
        close = df["close"]
        volume = df["volume"]

        # Step 1: Classify trades (Tick Rule)
        # If price up -> Buy-initiated, If price down -> Sell-initiated
        price_change = close.diff()
        buy_volume = volume.copy()
        sell_volume = volume.copy()

        buy_volume[price_change < 0] = 0  # Only count as buy if price went up
        sell_volume[price_change > 0] = 0  # Only count as sell if price went down

        # For unchanged price, split volume 50/50
        unchanged = price_change == 0
        buy_volume[unchanged] = volume[unchanged] / 2
        sell_volume[unchanged] = volume[unchanged] / 2

        # Step 2: Bucket volume (use cumulative volume buckets)
        total_volume = volume.sum()
        bucket_size = total_volume / self.vpin_buckets

        # Create bucket assignments
        cumulative_volume = volume.cumsum()
        bucket_id = (cumulative_volume / bucket_size).astype(int)

        # Step 3: Calculate order imbalance per bucket
        df_temp = pd.DataFrame(
            {
                "bucket": bucket_id,
                "buy_vol": buy_volume,
                "sell_vol": sell_volume,
                "total_vol": volume,
            }
        )

        # Group by bucket and calculate imbalance
        bucket_stats = df_temp.groupby("bucket").agg(
            {"buy_vol": "sum", "sell_vol": "sum", "total_vol": "sum"}
        )
        bucket_stats["imbalance"] = np.abs(
            bucket_stats["buy_vol"] - bucket_stats["sell_vol"]
        )
        bucket_stats["vpin_bucket"] = bucket_stats["imbalance"] / (
            bucket_stats["total_vol"] + 1e-9
        )

        # Step 4: Map back to original index and calculate rolling VPIN
        # For each bar, assign the VPIN of its bucket
        vpin_series = bucket_id.map(bucket_stats["vpin_bucket"]).fillna(0)

        # Smooth with rolling average (50 bars as per Easley et al.)
        vpin_smooth = vpin_series.rolling(50, min_periods=1).mean()

        return vpin_smooth

    def _calculate_kyle_lambda(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Kyle's Lambda (Price Impact Coefficient).

        Kyle's Lambda measures the price impact per unit of order flow:
        λ = ΔPrice / ΔVolume

        High lambda indicates high price impact (low liquidity).

        Reference: Kyle (1985) - Continuous Auctions and Insider Trading

        We estimate using rolling regression:
        |ΔP| = λ * V + ε

        Args:
            df: DataFrame with close and volume

        Returns:
            Kyle's Lambda series
        """
        close = df["close"]
        volume = df["volume"]

        # Calculate price change and volume
        price_change = close.diff().abs()  # Absolute price change
        # Ensure volume > 0
        vol_safe = volume.replace(0, np.nan)

        # Simple estimate: |ΔP| / V
        # More sophisticated: Rolling regression
        kyle_lambda = (price_change / vol_safe).replace([np.inf, -np.inf], np.nan)

        # Smooth with rolling median (more robust than mean)
        kyle_lambda_smooth = kyle_lambda.rolling(50, min_periods=1).median()

        return kyle_lambda_smooth

    def _calculate_amihud_lambda(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Amihud's Illiquidity Measure.

        Amihud's Lambda = |Return| / Dollar Volume

        High Amihud Lambda indicates low liquidity (high price impact per dollar traded).

        Reference: Amihud (2002) - "Illiquidity and Stock Returns"

        Args:
            df: DataFrame with close, volume, amount

        Returns:
            Amihud's Lambda series
        """
        close = df["close"]
        amount = df["amount"]  # Dollar volume

        # Calculate absolute return
        returns = close.pct_change().abs()

        # Amihud = |Return| / Dollar Volume
        # Ensure dollar volume > 0
        dollar_vol_safe = amount.replace(0, np.nan)

        amihud = (returns / dollar_vol_safe).replace([np.inf, -np.inf], np.nan)

        # Smooth with rolling median
        amihud_smooth = amihud.rolling(50, min_periods=1).median()

        return amihud_smooth

    def _calculate_roll_measure(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Roll Measure (Effective Spread Estimator).

        Roll's measure estimates the effective bid-ask spread from
        price changes alone (no bid-ask data needed).

        Roll Measure = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))

        Negative covariance indicates bid-ask bounce.

        Reference: Roll (1984) - "A Simple Implicit Measure of the Effective Bid-Ask Spread"

        Args:
            df: DataFrame with close prices

        Returns:
            Roll Measure series
        """
        close = df["close"]

        # Calculate price changes
        price_change = close.diff()

        # Calculate rolling covariance of price changes with lag 1
        # Cov(ΔP_t, ΔP_{t-1})
        def roll_cov(x):
            if len(x) < 2:
                return np.nan
            # x is the price change series
            # We need Cov(x[:-1], x[1:])
            return np.cov(x[:-1], x[1:])[0, 1]

        rolling_cov = price_change.rolling(50).apply(roll_cov, raw=True)

        # Roll Measure = 2 * sqrt(-Cov)
        # If Cov > 0, there's no bid-ask bounce, set to 0
        roll_measure = 2 * np.sqrt(np.maximum(-rolling_cov, 0))

        return roll_measure

    def _calculate_corwin_schultz_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Corwin-Schultz High-Low Spread Estimator.

        Uses the high and low prices to estimate the bid-ask spread.

        Reference: Corwin and Schultz (2012) - "A Simple Way to Estimate Bid-Ask Spreads"

        Algorithm:
        S = (2 * (e^β - 1)) / (1 + e^β)
        where β = E[(log(H/L))^2] / 2

        Args:
            df: DataFrame with high and low prices

        Returns:
            Corwin-Schultz spread series
        """
        high = df["high"]
        low = df["low"]

        # Calculate high-low ratio
        hl_ratio = np.log(high / low)

        # Rolling mean of squared log HL ratio
        beta = (hl_ratio**2).rolling(50).mean()

        # Spread estimate
        # S = (2 * (e^β - 1)) / (1 + e^β) = (e^β - 1) / e^β * 2 / (1 + e^β)
        # Simplified: S ≈ sqrt(2 * β) for small β (approximation)
        spread = np.sqrt(2 * beta)

        return spread


def main():
    """Generate microstructure features for dollar bars."""
    print("=" * 80)
    print("Microstructure Feature Engineering")
    print("=" * 80)

    # 1. Load dollar bars
    print("\n1. Loading dollar bars...")
    try:
        df = pd.read_csv(os.path.join("data", "output", "dynamic_dollar_bars.csv"), index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: 'dynamic_dollar_bars.csv' not found.")
        return

    print(f"   Loaded {len(df)} bars")

    # 2. Generate microstructure features
    print("\n2. Generating microstructure features...")
    micro_gen = MicrostructureFeatureGenerator(vpin_buckets=50, windows=[20, 50, 100])
    features_micro = micro_gen.generate(df)

    print(f"   Generated {len(features_micro.columns)} microstructure features")

    # 3. Save results
    print("\n3. Saving microstructure features...")
    features_micro.to_csv(os.path.join("data", "output", "features_microstructure.csv"))
    print("   ✓ Saved to: features_microstructure.csv")

    # 4. Display statistics
    print("\n" + "=" * 80)
    print("MICROSTRUCTURE FEATURE SUMMARY")
    print("-" * 80)
    print(f"Total Features: {len(features_micro.columns)}")
    print("\nFeature Preview:")
    print(features_micro.describe())

    print("\n" + "=" * 80)
    print("✓ Microstructure Feature Engineering Complete!")


if __name__ == "__main__":
    main()
