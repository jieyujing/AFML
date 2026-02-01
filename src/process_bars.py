import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import os


def load_and_prep_data(filepath, multiplier=300.0):
    """
    Load data and calculate dollar amount using Polars.
    """
    print(f"Loading data from {filepath}...")
    try:
        # Load with Polars
        df = pl.read_csv(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    # Convert datetime
    df = df.with_columns(
        pl.col("datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
    ).sort("datetime")

    # Calculate approximate dollar amount per bar
    # Using average price of the bar * volume * multiplier
    # Expression: (open + high + low + close) / 4.0 * volume * multiplier
    df = df.with_columns(
        (
            ((pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0)
            * pl.col("volume")
            * multiplier
        ).alias("amount")
    )

    return df


def aggregate_bars_polars(df_lazy, group_col_name="group_id"):
    """
    Aggregates a group of 1m bars into a single bar using Polars expressions.
    """
    return (
        df_lazy.group_by(group_col_name)
        .agg(
            [
                pl.col("datetime").last(),
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("amount").sum(),
            ]
        )
        .sort("datetime")
    )


def generate_fixed_dollar_bars(df, daily_target=4):
    """
    Generate dollar bars with a fixed threshold based on average daily volume.
    """
    print("Generating Fixed Dollar Bars...")

    # 1. Calculate Average Daily Dollar Volume
    # Resample by Day
    df_daily = (
        df.group_by_dynamic("datetime", every="1d")
        .agg(pl.col("amount").sum())
        .filter(pl.col("amount") > 0)
    )
    
    avg_daily_volume = df_daily["amount"].mean()

    threshold = avg_daily_volume / daily_target
    print(f"  Fixed Threshold: {threshold:,.2f} (Target: {daily_target} bars/day)")

    # 2. Vectorized grouping
    # cumsum of amount
    cum_amount = df["amount"].cum_sum()

    # assign group ids
    # The group ID increments every time cumulative amount passes a multiple of threshold
    # We use shift(1) so that the trade that crosses the threshold is INCLUDED in the current bar
    group_ids = (cum_amount.shift(1).fill_null(0) / threshold).cast(pl.Int64)

    # Add group_id to dataframe
    df_with_groups = df.with_columns(group_ids.alias("group_id"))

    # 3. Aggregate
    # Use Lazy for optimization on aggregation
    dollar_bars = aggregate_bars_polars(df_with_groups.lazy(), "group_id").collect()

    return dollar_bars, threshold


def generate_dynamic_dollar_bars(df, daily_target=4, ema_span=20):
    """
    Generate dollar bars where the threshold updates daily based on EMA of daily volume.
    """
    print("Generating Dynamic Dollar Bars (EMA)...")

    # 1. Calculate Daily Stats
    daily_stats = (
        df.group_by_dynamic("datetime", every="1d")
        .agg(pl.col("amount").sum().alias("daily_amt"))
    )

    # 2. Calculate EMA Threshold
    # Polars ewm_mean
    # We use shift(1) because we must determine today's threshold based on PAST data
    
    # Calculate global mean for fallback
    global_mean = daily_stats.filter(pl.col("daily_amt") > 0)["daily_amt"].mean()
    start_threshold = global_mean / daily_target

    daily_stats = daily_stats.with_columns(
        pl.col("daily_amt")
        .replace(0, None) # replace 0 with null for safety if needed, or keep 0
        .ewm_mean(span=ema_span, adjust=False)
        .alias("ema_amt")
    )
    
    daily_stats = daily_stats.with_columns(
        (pl.col("ema_amt").shift(1) / daily_target).alias("threshold")
    )
    
    # Fill nulls
    daily_stats = daily_stats.with_columns(
        pl.col("threshold").fill_null(start_threshold)
    )

    # Map threshold back to minute bars
    # We need to extract Date from datetime to join
    df = df.with_columns(pl.col("datetime").dt.date().alias("date"))
    daily_stats = daily_stats.with_columns(pl.col("datetime").dt.date().alias("date"))
    
    # Join to get dynamic threshold
    # select only date and threshold from stats to avoid collision
    stats_map = daily_stats.select(["date", "threshold"])
    
    df = df.join(stats_map, on="date", how="left")
    
    # Fill any remaining nulls (safety)
    df = df.with_columns(
        pl.col("threshold").fill_null(start_threshold).alias("dynamic_threshold")
    )

    # 3. Generate Bars with Dynamic Threshold
    # Normalized cumsum approach
    df = df.with_columns(
        (pl.col("amount") / pl.col("dynamic_threshold")).alias("norm_amount")
    )
    
    cum_norm_amount = df["norm_amount"].cum_sum()
    group_ids = (cum_norm_amount.shift(1).fill_null(0)).cast(pl.Int64)
    
    df_with_groups = df.with_columns(group_ids.alias("group_id"))

    dollar_bars = aggregate_bars_polars(df_with_groups.lazy(), "group_id").collect()

    return dollar_bars


def analyze_normality(df, name):
    """
    Calculate Log Returns and run JB Test. 
    Accepts Polars DataFrame.
    """
    # Convert to pandas/numpy for scipy stats
    if isinstance(df, pl.DataFrame):
        close = df["close"].to_numpy()
    else:
        close = df["close"]
        
    returns = np.log(close[1:] / close[:-1])
    # remove nans/inf
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    if len(returns) < 2:
        return {
            "name": name,
            "count": 0,
            "jb_stat": 0,
            "p_value": 0,
            "kurtosis": 0,
            "skew": 0,
        }

    jb_stat, p_value = stats.jarque_bera(returns)
    kurtosis = stats.kurtosis(returns)
    skew = stats.skew(returns)

    return {
        "name": name,
        "count": len(returns),
        "jb_stat": jb_stat,
        "p_value": p_value,
        "kurtosis": kurtosis,
        "skew": skew,
    }

def analyze_stationarity(df, name):
    """
    Run Augmented Dickey-Fuller (ADF) test for stationarity.
    Accepts Polars DataFrame.
    """
    if isinstance(df, pl.DataFrame):
        close = df["close"].to_numpy()
    else:
        close = df["close"]

    returns = np.log(close[1:] / close[:-1])
    returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

    # ADF Test
    try:
        adf_result = adfuller(returns)
        adf_stat = adf_result[0]
        p_value = adf_result[1]
        crit_values = adf_result[4]
    except Exception as e:
        print(f"ADF Error for {name}: {e}")
        adf_stat = 0
        p_value = 1
        crit_values = {'1%': 0}

    return {
        "name": name,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "crit_1%": crit_values['1%']
    }

def save_to_csv(df, output_path, bar_type="dollar_bars"):
    """
    Save dollar bars to CSV in AFML-compatible format.
    Accepts Polars DataFrame.
    """
    print(f"\nSaving {bar_type} to {output_path}...")

    # Select columns
    # Polars writes CSV very fast
    columns_to_save = ["datetime", "open", "high", "low", "close", "volume", "amount"]
    
    # Ensure columns exist
    existing_cols = [c for c in columns_to_save if c in df.columns]
    
    df_save = df.select(existing_cols)
    
    # Save
    df_save.write_csv(output_path)

    print(f"  ✓ Saved {len(df_save)} bars")
    print(f"  ✓ Date range: {df_save['datetime'][0]} to {df_save['datetime'][-1]}")
    print(f"  ✓ Columns: {df_save.columns}")

    return output_path

def plot_random_sample(df, sample_size=None, bar_type="Dollar Bars"):
    """
    Plot bars with candlestick chart and volume.
    Accepts Polars DataFrame, converts to Pandas for plotting.
    """
    print(f"\nGenerating visualization for {bar_type}...")
    
    # Convert to pandas for plotting compatibility
    # If df is too large, we should sample inside Polars first
    
    total_len = len(df)

    if sample_size is None or sample_size >= total_len:
        df_pd = df.to_pandas()
        print(f"  Plotting all {len(df_pd)} bars...")
    else:
        # Random start
        start_idx = random.randint(0, total_len - sample_size)
        df_pd = df.slice(start_idx, sample_size).to_pandas()
        print(f"  Plotting random sample of {len(df_pd)} bars...")

    # The rest is identical to the original pandas plotting code
    # Ensure datetime is compatible
    if not pd.api.types.is_datetime64_any_dtype(df_pd["datetime"]):
         df_pd["datetime"] = pd.to_datetime(df_pd["datetime"])

    # Create figure with 2 subplots (price and volume)
    fig_width = max(14, len(df_pd) / 30)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Optimization: Iterate over dataframe is slow, but for plotting < 1000 bars it's acceptable.
    # Otherwise we should use vectorized matplotlib calls (vlines/hlines).
    # Keeping original iterative logic for visual fidelity with current implementation.
    
    for idx, row in df_pd.iterrows():
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
            body_height if body_height > 0 else 0.00001, # Avoid zero height warning
            facecolor=color,
            edgecolor=color,
            alpha=0.8,
        )
        ax1.add_patch(rect)

    ax1.set_ylabel("Price", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"{bar_type} - Random Sample ({len(df_pd)} bars)\n"
        f"Period: {df_pd['datetime'].iloc[0]} to {df_pd['datetime'].iloc[-1]}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlim(-1, len(df_pd))

    # Plot volume bars
    colors = [
        "green" if row["close"] >= row["open"] else "red"
        for _, row in df_pd.iterrows()
    ]
    ax2.bar(
        range(len(df_pd)), df_pd["volume"], color=colors, alpha=0.6, width=0.8
    )

    ax2.set_ylabel("Volume", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Bar Index", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Add statistics box
    stats_text = (
        f"Sample Statistics:\n"
        f"Bars: {len(df_pd)}\n"
        f"Price Range: {df_pd['low'].min():.2f} - {df_pd['high'].max():.2f}\n"
        f"Avg Volume: {df_pd['volume'].mean():.0f}\n"
        f"Total Amount: ${df_pd['amount'].sum():,.0f}"
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

    # Ensure output directory exists
    output_dir = "visual_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Save figure
    filename = f"{bar_type.lower().replace(' ', '_')}_sample.png"
    output_image = os.path.join(output_dir, filename)
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"  ✓ Chart saved to: {output_image}")

    return output_image

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Dollar Bars from Time Bars.")
    parser.add_argument("file_path", nargs="?", default="IF9999.CCFX-2020-1-1-To-2026-01-22-1m.csv", help="Path to input CSV file")
    parser.add_argument("--multiplier", type=float, default=300.0, help="Contract multiplier (default: 300.0 for IF)")
    
    args = parser.parse_args()

    # 1. Load (Polars)
    df_time = load_and_prep_data(args.file_path, multiplier=args.multiplier)
    print(f"Loaded {len(df_time)} minute bars.")

    # 2. Generate Fixed Dollar Bars
    df_fixed, fixed_thresh = generate_fixed_dollar_bars(df_time, daily_target=4)
    print(f"Generated {len(df_fixed)} fixed dollar bars.")

    # 3. Generate Dynamic Dollar Bars
    df_dynamic = generate_dynamic_dollar_bars(df_time, daily_target=4, ema_span=20)
    print(f"Generated {len(df_dynamic)} dynamic dollar bars.")

    # 4. Save Dynamic Dollar Bars
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dynamic_dollar_bars.csv")
    save_to_csv(df_dynamic, output_file, bar_type="Dynamic Dollar Bars")

    # 5. Visualize All Data
    plot_random_sample(df_dynamic, sample_size=300, bar_type="Dynamic Dollar Bars")

    # 6. Analysis
    print("\n" + "=" * 80)
    print(f"{ 'Type':<20} | {'Count':<8} | {'JB Stat':<12} | {'Skew':<6} | {'ADF Stat':<10} | {'p-value':<8}")
    print("-" * 80)

    datasets = [
        (df_time, "Time Bars (1m)"),
        (df_fixed, "Fixed Dollar Bars"),
        (df_dynamic, "Dynamic Dollar Bars"),
    ]

    for df, name in datasets:
        norm_res = analyze_normality(df, name)
        stat_res = analyze_stationarity(df, name)
        
        is_stationary = stat_res['p_value'] < 0.05
        p_val_str = f"{stat_res['p_value']:.4f}" + ("*" if is_stationary else "")
        
        print(
            f"{norm_res['name']:<20} | {norm_res['count']:<8} | "
            f"{norm_res['jb_stat']:<12.2f} | {norm_res['skew']:<6.2f} | "
            f"{stat_res['adf_stat']:<10.2f} | {p_val_str:<8}"
        )

    print("=" * 80)
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
            print(
                f"In this dataset, Fixed Threshold performed best (JB: {fixed_jb:.2f})."
            )
        else:
            print(
                f"In this dataset, Dynamic Threshold performed best (JB: {dyn_jb:.2f})."
            )
    else:
        print("Dollar bars did not improve normality in this specific case (uncommon).")
        
    print("\nStationarity Check (* indicates p < 0.05):")
    dyn_stat = analyze_stationarity(df_dynamic, "Dynamic")
    if dyn_stat['p_value'] < 0.05:
        print(f"Dynamic Dollar Bars are Stationary (p={dyn_stat['p_value']:.4f}). Ready for labeling.")
    else:
        print(f"WARNING: Dynamic Dollar Bars are Non-Stationary (p={dyn_stat['p_value']:.4f}).")
        print("Suggestion: Apply Fractional Differentiation (FFD) in the Feature Engineering stage.")


if __name__ == "__main__":
    main()