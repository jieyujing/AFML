
import pytest
import polars as pl
import numpy as np
from datetime import datetime, timedelta, date
from afml import DollarBarsProcessor
import os

def test_transform_chunked_dynamic():
    """Test that transform_chunked correctly applies dynamic thresholds."""
    
    # 1. Setup Data: 2 days. 
    # Day 1: Low volume. Day 2: High volume (10x).
    # If using fixed threshold (based on average), Day 2 would have 10x bars of Day 1.
    # If using dynamic threshold, Day 2 threshold should adjust up, keeping bar count roughly similar (depending on EMA).
    
    # Parameters
    n_days = 2
    rows_per_day = 1000
    daily_target = 10
    
    start_time = datetime(2025, 1, 1)
    
    data = []
    
    # Day 1: Volume = 100 per tick. Total Day 1 = 100,000.
    for i in range(rows_per_day):
        data.append({
            "datetime": start_time + timedelta(seconds=i),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0,
            "volume": 100, # Amount ~ 100 * 100 = 10,000
        })
        
    # Day 2: Volume = 1000 per tick. Total Day 2 = 1,000,000.
    day2_start = start_time + timedelta(days=1)
    for i in range(rows_per_day):
        data.append({
            "datetime": day2_start + timedelta(seconds=i),
            "open": 100.0, "high": 105.0, "low": 95.0, "close": 100.0,
            "volume": 1000, # Amount ~ 100 * 1000 = 100,000
        })
        
    df = pl.DataFrame(data)
    filename = "test_dynamic.parquet"
    df.write_parquet(filename)
    
    print(f"Created {filename} with varying volume regimes.")
    
    try:
        # 2. Fit Dynamic
        # We need a small EMA span so it adapts quickly
        processor = DollarBarsProcessor(
            daily_target=daily_target, 
            ema_span=2, # Fast adaptation
            lazy=True
        )
        
        processor.fit(df)
        print("Fitted dynamic thresholds:")
        for d, t in processor._daily_thresholds.items():
            print(f"  {d}: {t:,.0f}")
            
        # Check that thresholds follow expected logic (Lagging EMA)
        dates = sorted(list(processor._daily_thresholds.keys()))
        t1 = processor._daily_thresholds[dates[0]] # Day 1 (Global Mean based)
        t2 = processor._daily_thresholds[dates[1]] # Day 2 (Day 1 Volume based)
        
        print(f"\nThreshold Day 1: {t1:,.0f}")
        print(f"Threshold Day 2: {t2:,.0f}")

        # Day 1 volume was low (10M), Day 2 volume was high (100M).
        # t2 is based on Day 1 volume, so it should be LOW (~1M).
        # t1 is based on global mean (avg of 10M and 100M ~ 55M), so it should be HIGH (~5.5M).
        assert t1 > t2, "Day 1 threshold (global mean) should be higher than Day 2 (based on low Day 1 volume)"

        # 3. Transform Chunked
        bars = processor.transform_chunked(filename, chunk_size=500)
        
        print(f"\nGenerated {len(bars)} bars.")
        
        # Analyze bars
        bars = bars.with_columns(pl.col("datetime").dt.date().alias("date"))
        
        day1_bars = bars.filter(pl.col("date") == dates[0])
        day2_bars = bars.filter(pl.col("date") == dates[1])
        
        avg_amt_1 = day1_bars["amount"].mean() if len(day1_bars) > 0 else 0
        avg_amt_2 = day2_bars["amount"].mean() if len(day2_bars) > 0 else 0
        
        print(f"\nAvg Bar Amount Day 1: {avg_amt_1:,.0f} (Expected ~{t1:,.0f})")
        print(f"Avg Bar Amount Day 2: {avg_amt_2:,.0f} (Expected ~{t2:,.0f})")
        
        # Verify that bar sizes roughly match the dynamic threshold for that day
        # Allow some margin of error due to discretization
        assert abs(avg_amt_1 - t1) < t1 * 0.5, "Day 1 bar size should track Day 1 threshold"
        assert abs(avg_amt_2 - t2) < t2 * 0.5, "Day 2 bar size should track Day 2 threshold"
        
        print("\nSUCCESS: Dynamic threshold logic verified in transform_chunked.")
        
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_transform_chunked_dynamic()
