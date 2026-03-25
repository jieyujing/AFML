import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the root directory to path to allow importing afml_visual_guides
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.util_plotting_lib import (
    plot_dollar_vs_tick_bars,
    plot_cusum_filter_events,
    plot_tbm_bounds_single_case,
    plot_all_tbm_labels,
    plot_sample_uniqueness_and_concurrency
)

def run_visualizations():
    print("Loading data...")
    try:
        dollar_bars = pd.read_csv('outputs/dollar_bars/dollar_bars_freq20.csv', index_col='timestamp', parse_dates=True)
        cusum_sampled = pd.read_csv('outputs/dollar_bars/cusum_sampled_bars.csv', index_col='timestamp', parse_dates=True)
        tbm_labels = pd.read_csv('outputs/dollar_bars/tbm_labels.csv', index_col='timestamp', parse_dates=True)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    os.makedirs('outputs/visualizations', exist_ok=True)

    # 1. Dollar Bars vs Tick Bars Frequency
    # Let's mock tick bars since we don't have them in csv, maybe by just adding random datetimes
    print("\n1. Generating Tick Bars vs Dollar Bars Generation Frequency plot...")
    tick_datetimes = pd.date_range(start=dollar_bars.index.min(), end=dollar_bars.index.max(), periods=len(dollar_bars)*5)
    
    plot_dollar_vs_tick_bars(
        tick_datetimes=tick_datetimes,
        dollar_datetimes=dollar_bars.index,
        resample_rule='D',
        save_path='outputs/visualizations/dollar_vs_tick_bars.png'
    )
    print("Saved outputs/visualizations/dollar_vs_tick_bars.png")

    # 2. CUSUM Filter Sampled Events on Price Series
    print("\n2. Generating CUSUM Filter Events plot...")
    # Just take a subset of data to make the plot readable
    # IMPORTANT: Start from the first CUSUM event to ensure events are visible
    start_date = cusum_sampled.index[0] - pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=30)
    subset_price = dollar_bars.loc[start_date:end_date, 'close']
    subset_events = cusum_sampled.loc[(cusum_sampled.index >= start_date) & (cusum_sampled.index <= end_date)].index
    
    plot_cusum_filter_events(
        price_series=subset_price,
        event_indices=subset_events,
        save_path='outputs/visualizations/cusum_filter_events.png'
    )
    print("Saved outputs/visualizations/cusum_filter_events.png")

    # 3. Triple Barrier Method (Single Case)
    print("\n3. Generating Triple Barrier Method plot...")
    try:
        # Pick the first completed label to show
        event_time = tbm_labels.index[10] # Skip first few to have some history
        end_time = pd.to_datetime(tbm_labels.loc[event_time, 't1'])
        
        # 为了展示 100 根线，我们先截取一个足够大的范围
        # 找到 event_time 附近的索引并向后截取更多
        idx_start = dollar_bars.index.get_indexer([event_time], method='nearest')[0]
        local_price = dollar_bars.iloc[max(0, idx_start-50):min(len(dollar_bars), idx_start+150)]['close']
        
        # Estimate target up/down from return
        pt0 = dollar_bars.loc[event_time, 'close']
        ret = tbm_labels.loc[event_time, 'ret']
        
        # It's an estimation for visualization
        hit_time = end_time  # In our simplify logic hit time is t1
        
        target_up = pt0 * 1.05 # arbitrary for viz 
        target_down = pt0 * 0.95 # arbitrary for viz
        
        if ret > 0:
            target_up = pt0 * (1 + ret)
        elif ret < 0:
            target_down = pt0 * (1 + ret)

        plot_tbm_bounds_single_case(
            price_series=local_price,
            t0=event_time,
            t3=end_time + pd.Timedelta(hours=1), # extend t3 a bit
            target_up=target_up,
            target_down=target_down,
            t_i=hit_time,
            save_path='outputs/visualizations/tbm_bounds_single_case.png'
        )
        print("Saved outputs/visualizations/tbm_bounds_single_case.png")
    except Exception as e:
        print(f"Failed to generate TBM plot: {e}")

    # 3b. Triple Barrier Method (All Events)
    print("\n3b. Generating All TBM Labels plot...")
    try:
        plot_all_tbm_labels(
            price_series=dollar_bars['close'],
            labels_df=tbm_labels,
            save_path='outputs/visualizations/tbm_all_labels.png'
        )
        print("Saved outputs/visualizations/tbm_all_labels.png")
    except Exception as e:
        print(f"Failed to generate All TBM plot: {e}")

    # 4. Concurrent Labels and Sample Uniqueness
    print("\n4. Generating Concurrent Labels and Sample Uniqueness plot...")
    try:
        # Select first 500 samples
        subset_tbm = tbm_labels.iloc[:500]
        t_events = pd.Series(subset_tbm.index, index=subset_tbm.index)
        t1 = pd.Series(pd.to_datetime(subset_tbm['t1']).values, index=subset_tbm.index)
        
        # We mock sample weights for the visualization if they don't exist
        if 'wI' in subset_tbm.columns:
            weights = subset_tbm['wI']
        else:
            weights = pd.Series(np.random.uniform(0.1, 1.0, len(subset_tbm)), index=subset_tbm.index)

        plot_sample_uniqueness_and_concurrency(
            t_events=t_events,
            t1=t1,
            sample_weights=weights,
            save_path='outputs/visualizations/sample_concurrency_uniqueness.png'
        )
        print("Saved outputs/visualizations/sample_concurrency_uniqueness.png")
    except Exception as e:
        print(f"Failed to generate Uniqueness plot: {e}")

if __name__ == '__main__':
    run_visualizations()
