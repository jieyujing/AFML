"""
Generate sample data for Polars pipeline testing.
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta


def generate_sample_data(
    n_rows: int = 10000,
    output_path: str = "data/daily_bars.csv",
) -> pl.DataFrame:
    """Generate sample OHLCV data for testing."""
    print(f"Generating sample data with {n_rows:,} rows...")

    np.random.seed(42)

    # Generate datetime range (hourly data for ~1 year)
    start_date = datetime(2020, 1, 1)
    end_date = start_date + timedelta(hours=n_rows)

    dates = pl.datetime_range(
        start=start_date,
        end=end_date,
        interval="1h",
        eager=True,
    )[:n_rows]

    # Generate price data with realistic movements
    returns = np.random.randn(n_rows) * 0.002  # ~2% daily vol
    close = 100 + np.cumsum(returns) * 10

    # Generate OHLC from close
    intraday_vol = np.abs(np.random.randn(n_rows)) * 0.001
    open_prices = close + np.random.randn(n_rows) * 0.5
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n_rows)) * 0.3
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n_rows)) * 0.3
    volume = np.random.randint(1000, 10000, n_rows).astype(float)

    df = pl.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    print(f"  Generated {len(df)} rows")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"  Close price: {df['close'].min():.2f} to {df['close'].max():.2f}")

    df.write_csv(output_path)
    print(f"  Saved to: {output_path}")

    return df


if __name__ == "__main__":
    import sys

    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/daily_bars.csv"

    generate_sample_data(n_rows=n_rows, output_path=output_path)
