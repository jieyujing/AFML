import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
from afmlkit.bar.kit import DynamicDollarBarKit
from afmlkit.bar.data_model import TradesData
from afmlkit.utils.log import get_logger

logger = get_logger(__name__)

def generate_mock_data(n_days=30):
    """Generate mock trade data if no file is provided"""
    logger.info("Generating mock trades data...")
    dates = pd.date_range(start='2023-01-01', periods=n_days * 1000, freq='1min')
    prices = np.random.lognormal(mean=0, sigma=0.01, size=len(dates))
    prices = 100 * np.exp(np.cumsum(prices - np.mean(prices))) # Random walk
    volumes = np.random.lognormal(mean=0, sigma=1, size=len(dates)) * 10

    df = pd.DataFrame({
        'timestamp': dates.astype('int64'),
        'price': prices,
        'amount': volumes
    })
    return df

def run_evaluation(data_path: str = None, start_time: str = None, end_time: str = None):
    # 1. Load Data
    if data_path:
        logger.info(f"Loading data from {data_path}...")
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
            if start_time:
                df = df[df['timestamp'] >= pd.Timestamp(start_time).value]
            if end_time:
                df = df[df['timestamp'] <= pd.Timestamp(end_time).value]
            if 'amount' not in df.columns and 'volume' in df.columns:
                df = df.rename(columns={'volume': 'amount'})
            trades = TradesData(ts=df['timestamp'].values, px=df['price'].values, qty=df['amount'].values)
        elif data_path.endswith('.h5'):
            trades = TradesData.load_trades_h5(data_path, start_time=start_time, end_time=end_time)
            # Ensure view range is exactly as requested, though load_trades_h5 fetches groups intersecting the time range
            if start_time or end_time:
                s = start_time if start_time else trades.data['timestamp'].min()
                e = end_time if end_time else trades.data['timestamp'].max()
                trades.set_view_range(s, e)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            if start_time:
                df = df[df['timestamp'] >= pd.Timestamp(start_time).value]
            if end_time:
                df = df[df['timestamp'] <= pd.Timestamp(end_time).value]
            if 'amount' not in df.columns and 'volume' in df.columns:
                df = df.rename(columns={'volume': 'amount'})
            trades = TradesData(ts=df['timestamp'].values, px=df['price'].values, qty=df['amount'].values)
        else:
            raise ValueError("Unsupported data format. Please use parquet, h5, or csv.")
    else:
        df = generate_mock_data()
        trades = TradesData(
            ts=df['timestamp'].values, 
            px=df['price'].values, 
            qty=df['amount'].values
        )

    # 2. Target Frequencies
    target_frequencies = [4, 6, 10, 20, 50]
    
    results = []
    optimal_freq = None
    min_jb_score = float('inf')
    best_series = None

    logger.info("Starting Dynamic Dollar Bar generation and evaluation...")

    for freq in target_frequencies:
        logger.info(f"Processing Target Frequency: {freq} bars/day")
        # Initialize Kit
        kit = DynamicDollarBarKit(trades, target_daily_bars=freq, ewma_span=20)
        
        # Build bars
        bars_df = kit.build_ohlcv()
        
        # Standardize check
        if bars_df.empty or len(bars_df) < 5:
            logger.warning(f"Not enough bars generated for freq={freq}.")
            continue

        # Calculate Log Returns
        # Drop first to get valid return
        log_ret = np.log(bars_df['close']).diff().dropna()
        
        if len(log_ret) < 2:
            continue

        # 3. Statistical Evaluation
        # Jarque-Bera Test for Normality (Lower score = closer to normal)
        jb_stat, jb_p_value = jarque_bera(log_ret)
        
        # First-order Autocorrelation
        autocorr = log_ret.autocorr(lag=1)
        
        # Record results
        results.append({
            'Target Freq': freq,
            'Actual Bars': len(bars_df),
            'JB Stat': jb_stat,
            'JB p-value': jb_p_value,
            'Autocorr (lag-1)': autocorr
        })
        
        # Selection Criterion: Primarily lowest JB Stat (closest to normal distribution)
        # (Could also factor autocorrelation as a combined score penalty)
        if jb_stat < min_jb_score:
            min_jb_score = jb_stat
            optimal_freq = freq
            best_series = bars_df
            best_threshold_curve = pd.Series(kit.thresholds_array, index=pd.to_datetime(trades.data['timestamp']))

    res_df = pd.DataFrame(results)
    print("\n--- Evaluation Results ---")
    print(res_df.to_string(index=False))
    
    if optimal_freq is None:
        logger.error("Failed to find optimal frequency.")
        return

    print(f"\n=> Optimal Target Frequency Chosen: {optimal_freq} bars/day")

    # 4. Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Plot 1: JB Stat across frequencies
    axes[0].bar([str(f) for f in res_df['Target Freq']], res_df['JB Stat'], color='skyblue', edgecolor='black')
    axes[0].set_title("Jarque-Bera Statistic by Target Frequency (Lower is better/more normal)")
    axes[0].set_ylabel("JB Statistic")
    axes[0].set_xlabel("Target Daily Bars")
    
    # Highlight the optimal
    best_idx = res_df[res_df['Target Freq'] == optimal_freq].index[0]
    axes[0].patches[best_idx].set_facecolor('salmon')

    # Plot 2: Best Bar Series (Close Price)
    axes[1].plot(best_series.index, best_series['close'], color='blue', label='Bar Close Price')
    axes[1].set_title(f"Optimal Dynamic Dollar Bars (Target = {optimal_freq} per day)")
    axes[1].set_ylabel("Price")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Dynamic Threshold Evolution
    # Resample strictly to avoid dense plotting
    daily_threshold = best_threshold_curve.resample('D').mean()
    axes[2].plot(daily_threshold.index, daily_threshold.values, color='green', label='EWMA Daily Bar Threshold')
    axes[2].set_title("Dynamic EWMA Threshold Tracking")
    axes[2].set_ylabel("Dollar Volume Threshold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_dollar_bars_evaluation.png')
    logger.info("Plot saved to dynamic_dollar_bars_evaluation.png")
    # plt.show() # Uncomment if running interactively

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate optimal Dynamic Dollar Bars")
    parser.add_argument("--data", type=str, default=None, help="Path to raw trade data (parquet/h5/csv)")
    parser.add_argument("--start", type=str, default=None, help="Start date (e.g. 2023-01-01)")
    parser.add_argument("--end", type=str, default=None, help="End date (e.g. 2023-03-01)")
    args = parser.parse_args()
    
    run_evaluation(args.data, args.start, args.end)
