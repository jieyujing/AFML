"""
Performance Benchmark: Pandas vs Polars for AFML Pipeline.

This script compares the performance of pandas-based and
Polars-based implementations for key operations.
"""

import time
import numpy as np
import pandas as pd
import polars as pl


def generate_pandas_data(n_rows: int = 100000) -> pd.DataFrame:
    """Generate sample OHLCV pandas DataFrame."""
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=n_rows, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    open_prices = close + np.random.randn(n_rows) * 0.1
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n_rows) * 0.1)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n_rows) * 0.1)
    volume = np.random.randint(1000, 10000, n_rows).astype(float)

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def generate_polars_data(n_rows: int = 100000) -> pl.DataFrame:
    """Generate sample OHLCV Polars DataFrame."""
    np.random.seed(42)

    dates = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2024, 1, 1),
        interval="1h",
        eager=True,
    )[:n_rows]

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    open_prices = close + np.random.randn(n_rows) * 0.1
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n_rows) * 0.1)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n_rows) * 0.1)
    volume = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def benchmark_pandas_dollar_bars(df: pd.DataFrame, n_runs: int = 3) -> float:
    """Benchmark pandas dollar bars."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        multiplier = 300.0
        avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
        df_copy = df.copy()
        df_copy["amount"] = avg_price * df_copy["volume"] * multiplier

        daily_amount = df_copy.set_index("datetime").resample("D")["amount"].sum()
        daily_amount = daily_amount[daily_amount > 0]
        threshold = daily_amount.mean() / 4

        cum_amount = df_copy["amount"].cumsum()
        group_ids = (cum_amount.shift(1).fillna(0) // threshold).astype(int)

        result = (
            df_copy.groupby(group_ids)
            .agg(
                {
                    "datetime": "last",
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "amount": "sum",
                }
            )
            .reset_index(drop=True)
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def benchmark_polars_dollar_bars(df: pl.DataFrame, n_runs: int = 3) -> float:
    """Benchmark Polars dollar bars."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        multiplier = 300.0
        df_with_amount = df.with_columns(
            (
                (
                    (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close"))
                    / 4.0
                )
                * pl.col("volume")
                * multiplier
            ).alias("amount")
        )

        df_daily = (
            df_with_amount.sort("datetime")
            .group_by_dynamic("datetime", every="1d")
            .agg(pl.col("amount").sum().alias("daily_amount"))
            .filter(pl.col("daily_amount") > 0)
        )

        threshold = df_daily["daily_amount"].mean() / 4

        df_cum = df_with_amount.with_columns(
            pl.col("amount").cum_sum().alias("cum_amount")
        )

        result = (
            df_cum.with_columns(
                ((pl.col("cum_amount") / threshold).floor().cast(pl.Int64)).alias(
                    "bar_id"
                )
            )
            .group_by("bar_id")
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

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def benchmark_pandas_features(df: pd.DataFrame, n_runs: int = 3) -> float:
    """Benchmark pandas feature engineering."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        close = df["close"]
        high = df["high"]
        low = df["low"]

        windows = [5, 10, 20, 30, 50]

        for w in windows:
            df[f"close_ma_{w}"] = close.rolling(w).mean()
            df[f"close_std_{w}"] = close.rolling(w).std()
            df[f"high_max_{w}"] = high.rolling(w).max()
            df[f"low_min_{w}"] = low.rolling(w).min()

        df["rsi_14"] = close.rolling(14).apply(
            lambda x: (
                100
                - (
                    100
                    / (
                        1
                        + x.diff().clip(lower=0).mean()
                        / (-x.diff()).clip(lower=0).mean()
                    )
                )
            )
        )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def benchmark_polars_features(df: pl.DataFrame, n_runs: int = 3) -> float:
    """Benchmark Polars feature engineering."""
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()

        close = pl.col("close")
        high = pl.col("high")
        low = pl.col("low")
        windows = [5, 10, 20, 30, 50]

        result = df.clone()

        for w in windows:
            result = result.with_columns(
                close.rolling_mean(window_size=w).alias(f"close_ma_{w}"),
                close.rolling_std(window_size=w).alias(f"close_std_{w}"),
                high.rolling_max(window_size=w).alias(f"high_max_{w}"),
                low.rolling_min(window_size=w).alias(f"low_min_{w}"),
            )

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return np.mean(times)


def benchmark_pandas_operations(df: pd.DataFrame, n_runs: int = 3) -> dict:
    """Benchmark various pandas operations."""
    results = {}

    # Filter
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        filtered = df[df["close"] > df["close"].mean()]
        times.append(time.perf_counter() - start)
    results["pandas_filter"] = np.mean(times)

    # Groupby
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        grouped = df.groupby(df["datetime"].dt.hour).agg({"close": ["mean", "std"]})
        times.append(time.perf_counter() - start)
    results["pandas_groupby"] = np.mean(times)

    return results


def benchmark_polars_operations(df: pl.DataFrame, n_runs: int = 3) -> dict:
    """Benchmark various Polars operations."""
    results = {}

    # Filter
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        filtered = df.filter(pl.col("close") > pl.col("close").mean())
        times.append(time.perf_counter() - start)
    results["polars_filter"] = np.mean(times)

    # Groupby
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        grouped = df.group_by(df["datetime"].dt.hour()).agg(
            [
                pl.col("close").mean().alias("close_mean"),
                pl.col("close").std().alias("close_std"),
            ]
        )
        times.append(time.perf_counter() - start)
    results["polars_groupby"] = np.mean(times)

    return results


def run_benchmarks(n_rows: int = 50000, n_runs: int = 3) -> dict:
    """Run all benchmarks."""
    print("=" * 60)
    print("Performance Benchmark: Pandas vs Polars")
    print("=" * 60)
    print(f"\nDataset size: {n_rows:,} rows")
    print(f"Runs per benchmark: {n_runs}")
    print()

    # Generate data
    print("Generating data...")
    pandas_df = generate_pandas_data(n_rows)
    polars_df = generate_polars_data(n_rows)
    print(
        f"  Pandas DataFrame: {pandas_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )
    print(f"  Polars DataFrame: {polars_df.estimated_size() / 1024**2:.2f} MB")
    print()

    results = {}

    # Dollar bars benchmark
    print("Benchmarking Dollar Bars generation...")
    pandas_time = benchmark_pandas_dollar_bars(pandas_df, n_runs)
    polars_time = benchmark_polars_dollar_bars(polars_df, n_runs)
    speedup = pandas_time / polars_time
    print(f"  Pandas: {pandas_time:.3f}s")
    print(f"  Polars: {polars_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    results["dollar_bars"] = {
        "pandas": pandas_time,
        "polars": polars_time,
        "speedup": speedup,
    }
    print()

    # Features benchmark
    print("Benchmarking Feature Engineering...")
    pandas_time = benchmark_pandas_features(pandas_df, n_runs)
    polars_time = benchmark_polars_features(polars_df, n_runs)
    speedup = pandas_time / polars_time
    print(f"  Pandas: {pandas_time:.3f}s")
    print(f"  Polars: {polars_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    results["features"] = {
        "pandas": pandas_time,
        "polars": polars_time,
        "speedup": speedup,
    }
    print()

    # General operations
    print("Benchmarking General Operations...")
    pandas_ops = benchmark_pandas_operations(pandas_df, n_runs)
    polars_ops = benchmark_polars_operations(polars_df, n_runs)

    for op in ["filter", "groupby"]:
        pandas_time = pandas_ops.get(f"pandas_{op}", 0)
        polars_time = polars_ops.get(f"polars_{op}", 0)
        speedup = pandas_time / polars_time if polars_time > 0 else 0
        print(
            f"  {op}: Pandas={pandas_time:.4f}s, Polars={polars_time:.4f}s, Speedup={speedup:.2f}x"
        )
        results[f"operations_{op}"] = {
            "pandas": pandas_time,
            "polars": polars_time,
            "speedup": speedup,
        }
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Average Polars Speedup: {np.mean([r['speedup'] for r in results.values()]):.2f}x"
    )
    print(
        f"Memory Savings: ~{100 * (1 - polars_df.estimated_size() / pandas_df.memory_usage(deep=True).sum()):.1f}%"
    )
    print()

    return results


if __name__ == "__main__":
    results = run_benchmarks(n_rows=50000, n_runs=3)

    # Save results
    import json

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to benchmark_results.json")
