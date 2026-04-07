"""
06b_meta_training_table.py - Build event-candidate long table for unified meta model.

Each row = one (event_time, candidate) pair from the 6 selected primary candidates.
Label: meta_y = 1 if TBM outcome matches primary_side, else 0.

CRITICAL: meta_training_table contains post-event-only columns (meta_y, label_end_time).
NEVER use it for live inference. Use meta_inference_table instead.
"""
import os
import re
import numpy as np
import pandas as pd
from afmlkit.sampling import cusum_filter
from afmlkit.label.tbm import triple_barrier
from afmlkit.feature.core.ma import ewma

OUTPUT_DIR = "strategies/AL9999/output/meta_model"
BARS_PATH = "strategies/AL9999/output/bars/dollar_bars_target15.parquet"
CALIB_PATH = "strategies/AL9999/output/primary_search/cusum_calibration.parquet"
CAND_PATH = "strategies/AL9999/output/primary_search/top_candidates.parquet"

# Top 6 candidates from primary search
TOP_CANDIDATES = [
    "rate=0.15_fast=5_slow=20_vb=10",   # candidate_id=0
    "rate=0.15_fast=8_slow=20_vb=10",   # candidate_id=1
    "rate=0.15_fast=5_slow=60_vb=10",   # candidate_id=2
    "rate=0.15_fast=5_slow=20_vb=20",   # candidate_id=3
    "rate=0.1_fast=5_slow=60_vb=30",    # candidate_id=4
    "rate=0.05_fast=8_slow=30_vb=10",   # candidate_id=5
]


def get_candidate_events(bars: pd.DataFrame, combo_id: str, k_lookup: pd.DataFrame) -> pd.DataFrame:
    """Generate CUSUM events and DMA side for one candidate combo.

    FIX #2: cusum_filter returns log_ret-array indices (0..len-2).
    Adding +1 maps to the bar where the return is realized.
    """
    m = re.match(r"rate=(\d+\.\d+)_fast=(\d+)_slow=(\d+)_vb=(\d+)", combo_id)
    if m is None:
        raise ValueError(f"Invalid combo_id format: {combo_id}")
    rate = float(m[1])
    fast = int(m[2])
    slow = int(m[3])
    vb = int(m[4])

    # Get k value for this CUSUM rate
    k = k_lookup.loc[k_lookup['rate'] == rate, 'k'].iloc[0]

    # CUSUM events: returns indices into log_ret array
    close = bars['close'].values
    log_returns = np.diff(np.log(close))           # length = len(close) - 1
    event_ret_indices = cusum_filter(log_returns, np.array([k]))

    # FIX #2: +1 maps log_ret index -> bar index where return is realized
    event_bar_indices = event_ret_indices + 1

    # DMA side at event bar
    fast_ma = ewma(close, span=fast)
    slow_ma = ewma(close, span=slow)
    event_fast = fast_ma[event_bar_indices]
    event_slow = slow_ma[event_bar_indices]
    primary_side = np.where(event_fast > event_slow, 1, -1).astype(np.int8)

    # FIX #2: use event_bar_indices (not event_ret_indices) for timestamps/prices
    event_times = bars.index[event_bar_indices]

    return pd.DataFrame({
        'event_time': event_times,
        'combo_id': combo_id,
        'candidate_id': TOP_CANDIDATES.index(combo_id),
        'primary_side': primary_side,
        'cusum_rate': rate,
        'fast_window': fast,
        'slow_window': slow,
        'vertical_bars': vb,
        'bar_idx': event_bar_indices,   # store bar index (not return index)
        'price': close[event_bar_indices],
    })


def compute_tbm_for_candidate(
    bars: pd.DataFrame,
    events_df: pd.DataFrame,
    pt_sl_sigma: float = 1.0,
) -> pd.DataFrame:
    """Compute TBM outcomes for all events of one candidate."""
    close = bars['close'].values.astype(np.float64)
    timestamps = bars.index.values.astype(np.int64)

    event_idxs = events_df['bar_idx'].values.astype(np.int64)
    sides = events_df['primary_side'].values.astype(np.int8)
    vb = events_df['vertical_bars'].iloc[0]

    # Filter events too close to bars end
    valid_mask = event_idxs + vb < len(bars) - 1
    event_idxs = event_idxs[valid_mask]
    sides = sides[valid_mask]
    event_times = events_df.loc[valid_mask, 'event_time'].values

    # Volatility target at event bars
    vol = bars['close'].pct_change().ewm(span=20).std().values
    targets = vol[event_idxs].astype(np.float64)

    # FIX: Dollar bars are irregular in time. Fixed 4 hours is wrong.
    # To enforce a BAR HORIZON (vb bars), we pass pseudo-timestamps where 1 bar = 1 second.
    # Then vertical_barrier=float(vb) operates exactly as vb bars horizon.
    pseudo_timestamps = np.arange(len(bars), dtype=np.int64) * int(1e9)

    labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
        timestamps=pseudo_timestamps,
        close=close,
        event_idxs=event_idxs,
        targets=targets,
        profit_loss_barriers=(pt_sl_sigma, pt_sl_sigma),
        vertical_barrier=float(vb),
        min_close_time_sec=0.0,
        side=sides,
        min_ret=0.0,
    )

    # Resolve exit timestamps
    exit_idxs = np.where(touch_idxs != -1, touch_idxs, np.minimum(event_idxs + vb, len(bars) - 1))
    label_end_time = pd.to_datetime(bars.index[exit_idxs.astype(np.int64)])

    # tbm_outcome: +1 = hit profit barrier (correct), -1 = hit stop (wrong), 0 = timeout
    tbm_outcome = np.where(labels == 1, sides, -sides)
    tbm_outcome = np.where(labels == 0, 0, tbm_outcome)

    result = events_df[valid_mask].copy().reset_index(drop=True)
    result['label_end_time'] = label_end_time
    result['tbm_outcome'] = tbm_outcome.astype(np.int8)
    result['meta_y'] = (tbm_outcome == result['primary_side'].values).astype(np.int8)
    result['tbm_pt_sigma'] = pt_sl_sigma
    result['tbm_sl_sigma'] = pt_sl_sigma
    result['is_timeout'] = (labels == 0).astype(np.int8)
    result['ret'] = rets.astype(np.float32)
    result['touch_type'] = np.where(labels == 1, 'pt', np.where(labels == -1, 'sl', 'timeout'))
    return result


def build_meta_training_table() -> pd.DataFrame:
    """Build event-candidate long table + inference-only table."""
    print("[Step 1] Loading bars and calibration...")
    bars = pd.read_parquet(BARS_PATH)
    k_lookup = pd.read_parquet(CALIB_PATH)
    candidates = pd.read_parquet(CAND_PATH)
    print(f"  Bars: {len(bars)} | Calibration: {len(k_lookup)} rows")

    missing = set(TOP_CANDIDATES) - set(candidates['combo_id'])
    if missing:
        raise ValueError(f"CRITICAL: Candidates {missing} not found in top_candidates.parquet")

    all_rows = []
    for combo_id in TOP_CANDIDATES:
        print(f"  Processing {combo_id}...")
        events = get_candidate_events(bars, combo_id, k_lookup)
        tbm_result = compute_tbm_for_candidate(bars, events)
        all_rows.append(tbm_result)

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sort_values('event_time').reset_index(drop=True)

    # event_id: hash of (symbol, event_time)
    #
    # v1 CONVENTION (not a strict ground-truth claim):
    # Rows with the same event_time are treated as belonging to the same "event cluster"
    # for deduplication purposes. This is a simplification: in reality, different CUSUM rates
    # may trigger at slightly different bars even at the same wall-clock time, so the
    # "same event_time" across candidates does NOT mean "same underlying market event."
    #
    # This convention enables:
    #   - deduplication via groupby('event_time')  (not 'event_id')
    #   - ensemble feature computation (n_fired per timestamp)
    #
    # Explicit warning: event_id is a stable audit identifier derived from (symbol, event_time),
    # not a strict unique identifier of a market event generation mechanism.
    df['event_id'] = pd.util.hash_pandas_object(
        pd.DataFrame({'symbol': 'AL9999', 'event_time': df['event_time'].astype(str)})
    ).astype(str)

    # CRITICAL: split into training table and inference table
    # Inference table must NOT contain any post-event-only columns
    INFERENCE_COLS = [
        'event_time', 'event_id', 'combo_id', 'candidate_id',
        'primary_side', 'cusum_rate', 'fast_window', 'slow_window',
        'vertical_bars', 'bar_idx', 'price',
        # NOTE: These are added in feature engineering:
        # shock_size, ret_1, vol_jump, ewm_vol, vol_percentile,
        # time_since_prev_event_h, ma_gap_pct, fast_slope, slow_slope,
        # price_to_slow, session_bucket, dow,
        # n_fired, n_long, n_short, conflict_flag, agreement_ratio,
        # is_majority_side, consensus_side
    ]
    inference_df = df[[c for c in INFERENCE_COLS if c in df.columns]].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "meta_training_table.parquet")
    df.to_parquet(train_path, index=False)
    print(f"  Saved training table: {train_path} ({len(df)} rows)")

    inf_path = os.path.join(OUTPUT_DIR, "meta_inference_table.parquet")
    inference_df.to_parquet(inf_path, index=False)
    print(f"  Saved inference table: {inf_path} ({len(inference_df)} rows)")

    # Diagnostics
    print(f"\n  Label distribution: meta_y=1: {df['meta_y'].sum()}, meta_y=0: {(1-df['meta_y']).sum()}")
    print(f"  Base rate: {df['meta_y'].mean():.4f}")
    print(f"  Timeout rate: {df['is_timeout'].mean():.4f}")
    print(f"  Events with >1 candidate: {(df.groupby('event_id').size() > 1).sum()}")

    return df


if __name__ == "__main__":
    df = build_meta_training_table()
    print("\nColumns:", df.columns.tolist())
    print(df[['event_time','combo_id','candidate_id','primary_side','meta_y','label_end_time']].head(6))
    # Verify: same event_time should appear for different candidates
    multi = df.groupby('event_time').filter(lambda g: len(g) > 1)
    print(f'\nEvents with multiple candidates: {multi["event_time"].nunique()}')
    print(multi[['event_time','combo_id','candidate_id','primary_side','meta_y']].head(9))
    # Verify distribution
    print(f"\nEvents per event_time distribution:")
    print(df.groupby('event_time').size().describe())
    print(df.groupby('event_time').size().value_counts().sort_index())
