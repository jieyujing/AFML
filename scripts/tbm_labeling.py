import pandas as pd
import numpy as np
import os
import time

from afmlkit.feature.core.volatility import ewms
from afmlkit.label.kit import TBMLabel
from afmlkit.bar.data_model import TradesData

def compute_tbm_labels():
    FULL_DATA_PATH = "outputs/dollar_bars/dollar_bars_freq20.csv"
    CUSUM_EVENTS_PATH = "outputs/dollar_bars/cusum_sampled_bars.csv"
    OUTPUT_PATH = "outputs/dollar_bars/tbm_labels.csv"

    if not os.path.exists(FULL_DATA_PATH) or not os.path.exists(CUSUM_EVENTS_PATH):
        print("Error: Input files not found.")
        return

    print("Loading full dollar bars...")
    df_full = pd.read_csv(FULL_DATA_PATH)
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

    # Computing volatility exactly like CUSUM filter
    print("Computing log returns and volatility for full data...")
    prices = df_full['close'].values.astype(np.float64)
    log_ret = np.empty(len(prices), dtype=np.float64)
    log_ret[0] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret[1:] = np.log(prices[1:] / prices[:-1])
    
    # Using the same span=50 as cusum filter
    volatility = ewms(log_ret, span=50)
    
    # Assign back to dataframe aligned with timestamps
    df_full['volatility'] = volatility
    
    # Create TradesData
    print("Formatting TradesData object...")
    # Convert exactly to nanoseconds integers for the underlying array
    ts = df_full['timestamp'].values.astype(np.int64)
    px = df_full['close'].values.astype(np.float64)
    # Using volume as qty approximation for TradesData interface
    qty = df_full['volume'].values.astype(np.float64)
    trade_ids = np.arange(len(df_full))
    
    trades = TradesData(ts, px, qty, id=trade_ids, timestamp_unit='ns', preprocess=False)
    
    print("Loading CUSUM events...")
    df_events = pd.read_csv(CUSUM_EVENTS_PATH)
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
    
    df_full = df_full.set_index('timestamp')
    df_events = df_events.set_index('timestamp')
    
    # Map the volatility into the events based on timestamp
    df_events['volatility'] = df_full.loc[df_events.index, 'volatility']
    
    # Filter Nans if any
    df_events = df_events.dropna(subset=['volatility'])
    
    print(f"Instantiating TBMLabel with {len(df_events)} events...")
    # horizontal_barriers=(bottom, top)
    tbm = TBMLabel(
        features=df_events,
        target_ret_col='volatility',
        min_ret=0.0,
        horizontal_barriers=(1.0, 1.0),
        vertical_barrier=pd.Timedelta(hours=24),
        is_meta=False,
        min_close_time=pd.Timedelta(seconds=1)
    )
    
    print("Computing TBM Labels...")
    st = time.time()
    feats, out = tbm.compute_labels(trades)
    print(f"Elapsed time: {time.time() - st:.4f}s")
    
    print(out.head())
    
    print("\nDistribution of labels:")
    print(out['bin'].value_counts())
    
    print(f"\nComputing sample weights...")
    weights = tbm.compute_weights(trades)
    out = out.join(weights)
    
    print(f"\nSaving TBM output to {OUTPUT_PATH}...")
    out.to_csv(OUTPUT_PATH)
    print("Done!")

if __name__ == "__main__":
    compute_tbm_labels()
