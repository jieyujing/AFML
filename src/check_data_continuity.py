import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_continuity(parquet_dir: str):
    print("Listing files...")
    # Scan parquet for memory efficiency or load into memory if < 100GB
    # For speed and given we likely have monthly chunks, scanning is safer.
    q = pl.scan_parquet(f"{parquet_dir}/*.parquet")

    print("Aggregating trade counts by hour...")
    # 1. Check for macro-level gaps: Count of trades per hour
    df_counts = (
        q.sort("timestamp")
        .group_by_dynamic("timestamp", every="1h")
        .agg(pl.count().alias("trade_count"))
        .collect()
    )

    # Convert to pandas for plotting
    pdf_counts = df_counts.to_pandas()

    print("Plotting trade counts...")
    plt.figure(figsize=(15, 6))
    plt.plot(pdf_counts["timestamp"], pdf_counts["trade_count"])
    plt.title("Trades per Hour (Continuity Check)")
    plt.xlabel("Date")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.tight_layout()
    output_dir = Path("visual_analysis")
    output_dir.mkdir(exist_ok=True)
    output_img = output_dir / "trade_continuity.png"
    plt.savefig(output_img)
    print(f"Saved plot to {output_img}")

    # === Summary Report ===
    print("\n" + "="*40)
    print("DATA CONTINUITY SUMMARY")
    print("="*40)
    
    if not pdf_counts.empty:
        total_trades = pdf_counts["trade_count"].sum()
        start_time = pdf_counts["timestamp"].min()
        end_time = pdf_counts["timestamp"].max()
        
        # Calculate duration
        duration = end_time - start_time
        total_hours_span = duration.total_seconds() / 3600
        # Add 1 to include the last hour bucket fully if it exists
        if total_hours_span == 0:
            total_hours_span = 1
            
        actual_hours_count = len(pdf_counts)
        
        # Coverage efficiency
        # Note: Crypto is 24/7, so expected hours ~= total_hours_span
        # We round total_hours_span for comparison as grid is hourly
        expected_hours = int(total_hours_span) + 1
        missing_hours = expected_hours - actual_hours_count
        coverage_pct = (actual_hours_count / expected_hours) * 100 if expected_hours > 0 else 0

        print(f"Total Trades:       {total_trades:,.0f}")
        print(f"Time Range:         {start_time} to {end_time}")
        print(f"Total Span:         {expected_hours} hours ({duration.days} days)")
        print(f"Actual Data Hours:  {actual_hours_count} hours")
        print(f"Missing Hours:      {missing_hours} hours (approx)")
        print(f"Data Coverage:      {coverage_pct:.2f}%")
        print("="*40)

        if coverage_pct < 99.0:
            print("WARNING: Significant data gaps detected! Check the plot.")
        else:
            print("SUCCESS: Data continuity looks good.")
    else:
        print("ERROR: No data found.")

if __name__ == "__main__":
    visualize_continuity("data/BTCUSDT/parquet_db")
