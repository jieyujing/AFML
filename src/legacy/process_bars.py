import pandas as pd
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

from afml import DollarBarsProcessor


def load_and_prep_data(filepath):
    """
    Load data and calculate dollar amount.
    IF contract multiplier is 300.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Calculate approximate dollar amount per bar
    # Using average price of the bar * volume * multiplier
    multiplier = 300.0
    avg_price = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    df["amount"] = avg_price * df["volume"] * multiplier

    return df


def aggregate_bars(group):
    """
    Aggregates a group of 1m bars into a single bar.
    """
    return pd.Series(
        {
            "datetime": group["datetime"].iloc[-1],  # Timestamp of the end of the bar
            "open": group["open"].iloc[0],
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": group["close"].iloc[-1],
            "volume": group["volume"].sum(),
            "amount": group["amount"].sum(),
        }
    )


def generate_fixed_dollar_bars(df, daily_target=4):
    """
    Generate dollar bars with a fixed threshold based on average daily volume.
    """
    print("Generating Fixed Dollar Bars...")

    # 1. Calculate Average Daily Dollar Volume
    df_daily = df.set_index("datetime").resample("D")["amount"].sum()
    # Filter out days with 0 volume (holidays/weekends) for accurate average
    df_daily = df_daily[df_daily > 0]
    avg_daily_volume = df_daily.mean()

    threshold = avg_daily_volume / daily_target
    print(f"  Fixed Threshold: {threshold:,.2f} (Target: {daily_target} bars/day)")

    # 2. Vectorized grouping
    # cumsum of amount
    cum_amount = df["amount"].cumsum()

    # assign group ids
    # The group ID increments every time cumulative amount passes a multiple of threshold
    # We use shift(1) so that the trade that crosses the threshold is INCLUDED in the current bar
    group_ids = (cum_amount.shift(1).fillna(0) // threshold).astype(int)

    # 3. Aggregate
    # We group by the calculated ID
    dollar_bars = df.groupby(group_ids).apply(aggregate_bars, include_groups=False)
    dollar_bars = dollar_bars.reset_index(drop=True)

    return dollar_bars, threshold


def generate_dynamic_dollar_bars(df, daily_target=4, ema_span=20):
    """
    Generate dollar bars where the threshold updates daily based on EMA of daily volume.
    """
    print("Generating Dynamic Dollar Bars (EMA)...")

    # 1. Calculate Daily Stats
    daily_stats = (
        df.set_index("datetime")
        .resample("D")["amount"]
        .sum()
        .to_frame(name="daily_amt")
    )

    # 2. Calculate EMA Threshold
    # We use shift(1) because we must determine today's threshold based on PAST data (yesterday's EMA)
    # to avoid look-ahead bias.
    daily_stats["ema_amt"] = (
        daily_stats["daily_amt"]
        .replace(0, np.nan)
        .ewm(span=ema_span, adjust=False)
        .mean()
    )
    daily_stats["threshold"] = daily_stats["ema_amt"].shift(1) / daily_target

    # Fill NaN at start (use first available valid threshold or global mean if needed)
    # For the very first days, we might fallback to the global mean or the first day's volume
    global_mean = daily_stats["daily_amt"][daily_stats["daily_amt"] > 0].mean()
    start_threshold = global_mean / daily_target
    daily_stats["threshold"] = daily_stats["threshold"].fillna(start_threshold)

    # Map threshold back to minute bars
    # We need to map by Date
    df = df.copy()
    df["date"] = df["datetime"].dt.date

    # Create a mapping dictionary for speed
    daily_stats["date"] = daily_stats.index.date
    threshold_map = daily_stats.set_index("date")["threshold"].to_dict()

    df["dynamic_threshold"] = df["date"].map(threshold_map)
    df["dynamic_threshold"] = df["dynamic_threshold"].fillna(
        start_threshold
    )  # Safety fill

    # 3. Generate Bars with Dynamic Threshold
    # Since threshold changes, we can't use simple cumsum // threshold.
    # We use a normalized cumsum approach:
    # We accumulate (amount / current_threshold). When this normalized sum crosses integer boundaries, we bar.

    df["norm_amount"] = df["amount"] / df["dynamic_threshold"]
    df["cum_norm_amount"] = df["norm_amount"].cumsum()

    group_ids = (df["cum_norm_amount"].shift(1).fillna(0)).astype(int)

    dollar_bars = df.groupby(group_ids).apply(aggregate_bars, include_groups=False)
    dollar_bars = dollar_bars.reset_index(drop=True)

    return dollar_bars


def analyze_normality(df, name):
    """
    Calculate Log Returns and run JB Test.
    """
    # Log Returns
    # dropna() to remove the first NaN
    returns = np.log(df["close"] / df["close"].shift(1)).dropna()

    # JB Test
    # The Jarque-Bera test tests whether the sample data has the skewness and kurtosis matching a normal distribution.
    # Statistic: The test statistic.
    # p-value: The p-value for the hypothesis test.
    jb_stat, p_value = stats.jarque_bera(returns)

    # Standard stats
    kurtosis = returns.kurtosis()
    skew = returns.skew()

    return {
        "name": name,
        "count": len(returns),
        "jb_stat": jb_stat,
        "p_value": p_value,
        "kurtosis": kurtosis,
        "skew": skew,
    }


def save_to_csv(df, output_path, bar_type="dollar_bars"):
    """
    Save dollar bars to CSV in AFML-compatible format.

    Args:
        df: DataFrame with OHLCV data
        output_path: Path to save the CSV file
        bar_type: Type of bars (for logging purposes)

    The output format follows AFML conventions:
    - datetime as index (not a column)
    - OHLCV columns: open, high, low, close, volume
    - Optional: amount column for dollar volume
    """
    print(f"\nSaving {bar_type} to {output_path}...")

    # Create a copy to avoid modifying the original
    df_save = df.copy()

    # Set datetime as index for AFML compatibility
    df_save = df_save.set_index("datetime")

    # Select and order columns for AFML workflows
    # Standard OHLCV format with additional amount column
    columns_to_save = ["open", "high", "low", "close", "volume", "amount"]
    df_save = df_save[columns_to_save]

    # Save to CSV with datetime index
    df_save.to_csv(output_path)

    print(f"  ✓ Saved {len(df_save)} bars")
    print(f"  ✓ Date range: {df_save.index[0]} to {df_save.index[-1]}")
    print(f"  ✓ Columns: {list(df_save.columns)}")

    return output_path


def plot_random_sample(df, sample_size=None, bar_type="Dollar Bars"):
    """
    Plot bars with candlestick chart and volume.

    Args:
        df: DataFrame with OHLCV data and datetime column
        sample_size: Number of bars to display (None = all data)
        bar_type: Type of bars (for title)
    """
    print(f"\nGenerating visualization for {bar_type}...")

    # Use all data if sample_size is None
    if sample_size is None:
        df_sample = df.copy()
        print(f"  Plotting all {len(df_sample)} bars...")
    else:
        # Ensure we have enough data
        if len(df) < sample_size:
            sample_size = len(df)
            start_idx = 0
        else:
            # Random start position
            start_idx = random.randint(0, len(df) - sample_size)

        # Extract sample
        df_sample = df.iloc[start_idx : start_idx + sample_size].copy()
        print(f"  Plotting random sample of {len(df_sample)} bars...")

    # Create figure with 2 subplots (price and volume)
    # Adjust figure size based on data size
    fig_width = max(14, len(df_sample) / 30)  # Scale width with data
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Plot candlesticks
    for idx, row in df_sample.iterrows():
        # Determine color
        color = "green" if row["close"] >= row["open"] else "red"

        # Draw high-low line
        ax1.plot(
            [idx, idx], [row["low"], row["high"]], color=color, linewidth=1, alpha=0.8
        )

        # Draw body rectangle
        body_height = abs(row["close"] - row["open"])
        body_bottom = min(row["open"], row["close"])

        rect = Rectangle(
            (idx - 0.3, body_bottom),
            0.6,
            body_height,
            facecolor=color,
            edgecolor=color,
            alpha=0.8,
        )
        ax1.add_patch(rect)

    # Format price chart
    ax1.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"{bar_type} - Random Sample ({sample_size} bars)\n"
        f"Period: {df_sample['datetime'].iloc[0]} to {df_sample['datetime'].iloc[-1]}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(-1, sample_size)

    # Plot volume bars
    colors = [
        "green" if row["close"] >= row["open"] else "red"
        for _, row in df_sample.iterrows()
    ]
    ax2.bar(
        range(len(df_sample)), df_sample["volume"], color=colors, alpha=0.6, width=0.8
    )

    # Format volume chart
    ax2.set_ylabel("Volume", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Bar Index", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add statistics box
    stats_text = (
        f"Sample Statistics:\n"
        f"Bars: {len(df_sample)}\n"
        f"Price Range: {df_sample['low'].min():.2f} - {df_sample['high'].max():.2f}\n"
        f"Avg Volume: {df_sample['volume'].mean():.0f}\n"
        f"Total Amount: ${df_sample['amount'].sum():,.0f}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save figure
    output_image = f"{bar_type.lower().replace(' ', '_')}_sample.png"
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"  ✓ Chart saved to: {output_image}")

    # Show plot
    plt.show()

    return output_image


def main():
    file_path = "IF9999.CCFX-2022-1-1-To-2026-01-20-1m.csv"

    # 1. Load
    df_time = load_and_prep_data(file_path)
    print(f"Loaded {len(df_time)} minute bars.")

    # Using OO DollarBarsProcessor (Dynamic Mode)
    print("\nUsing DollarBarsProcessor (Dynamic Mode)...")
    processor = DollarBarsProcessor(daily_target=4, ema_span=20)
    df_dynamic = processor.fit_transform_dynamic(df_time)
    print(f"Generated {len(df_dynamic)} dynamic dollar bars.")
    info = processor.get_threshold_info()
    print(f"  Threshold type: {info['threshold_type']}")
    print(f"  Starting threshold: {info['threshold']:,.2f}")

    # Also generate fixed for comparison
    df_fixed, fixed_thresh = generate_fixed_dollar_bars(df_time, daily_target=4)
    print(f"Generated {len(df_fixed)} fixed dollar bars.")

    # 4. Save Dynamic Dollar Bars to CSV (AFML format)
    output_file = "dynamic_dollar_bars.csv"
    save_to_csv(df_dynamic, output_file, bar_type="Dynamic Dollar Bars")

    # 5. Visualize All Data
    plot_random_sample(df_dynamic, sample_size=None, bar_type="Dynamic Dollar Bars")

    # 6. Analysis
    print("\n" + "=" * 60)
    print(
        f"{'Type':<20} | {'Count':<8} | {'JB Stat':<12} | {'Kurtosis':<10} | {'Skew':<10}"
    )
    print("-" * 60)

    datasets = [
        (df_time, "Time Bars (1m)"),
        (df_fixed, "Fixed Dollar Bars"),
        (df_dynamic, "Dynamic Dollar Bars"),
    ]

    for df, name in datasets:
        res = analyze_normality(df, name)
        print(
            f"{res['name']:<20} | {res['count']:<8} | {res['jb_stat']:<12.2f} | {res['kurtosis']:<10.2f} | {res['skew']:<10.2f}"
        )

    print("=" * 60)
    print("\nConclusion:")

    time_jb = analyze_normality(df_time, "Time")["jb_stat"]
    fixed_jb = analyze_normality(df_fixed, "Fixed")["jb_stat"]
    dyn_jb = analyze_normality(df_dynamic, "Dynamic")["jb_stat"]

    best_jb = min(time_jb, fixed_jb, dyn_jb)

    if best_jb < time_jb:
        print(
            "Dollar bars successfully reduced the JB statistic, indicating a distribution closer to normal."
        )
        if fixed_jb < dyn_jb:
            print(f"Fixed Threshold performed best (JB: {fixed_jb:.2f}).")
        else:
            print(f"Dynamic Threshold performed best (JB: {dyn_jb:.2f}).")
    else:
        print("Dollar bars did not improve normality in this specific case (uncommon).")


if __name__ == "__main__":
    main()
