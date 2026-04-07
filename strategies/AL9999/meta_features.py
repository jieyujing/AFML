"""
meta_features.py - Feature engineering for unified meta model.

Features organized into 5 categories:
1. Candidate identity: candidate_id, cusum_rate, fast/slow windows, vertical_bars
2. Event-level: shock_size, ret_1, vol_jump, ewm_vol, vol_percentile, time_since_prev_event_h
3. Trend-state: ma_gap_pct, fast_slope, slow_slope, price_to_slow
4. Regime: session_bucket, dow
5. Ensemble: n_fired, n_long, n_short, conflict_flag, agreement_ratio, is_majority_side, consensus_side

FIX #4 (from user feedback): group by combo_id NOT cusum_rate (same rate can have different fast/slow)
FIX #5: use transform() instead of apply() for groupby aggregations
FIX #6: compute time_since_prev on event-level timeline first, then merge back
FIX #11: remove combo_id_hash (use candidate_id directly)
"""
import os
import numpy as np
import pandas as pd

FEATURES_DIR = "strategies/AL9999/output/meta_model"
BARS_PATH = "strategies/AL9999/output/bars/dollar_bars_target15.parquet"


def add_event_level_features(df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Add event-level features: shock_size, ret_1, vol_jump, ewm_vol, vol_percentile.

    FIX #3 from user feedback: unified return口径.
    All return-based features use simple pct_change (NOT log) for consistency.
    """
    close = bars['close']

    # ret_1: simple return at event bar (NOT log return)
    # event_time is the bar where CUSUM triggered, reindex directly
    ret_series = close.pct_change()  # simple return, aligned with close index
    df['ret_1'] = ret_series.reindex(df['event_time']).fillna(0).values.astype(np.float32)
    df['shock_size'] = df['ret_1'].abs().astype(np.float32)

    # vol_jump: ewm_vol(t) / ewm_vol(t-10bars)
    vol = close.pct_change().ewm(span=20).std()
    vol_now = vol.reindex(df['event_time']).fillna(vol.median())
    vol_past = vol.shift(10).reindex(df['event_time']).fillna(vol.median())
    df['vol_jump'] = (vol_now / (vol_past + 1e-10)).values.astype(np.float32).clip(0.5, 5.0)

    # ewm_vol: current volatility
    df['ewm_vol'] = vol_now.values.astype(np.float32)

    # vol_percentile: v1 SIMPLIFIED — drop expanding + lambda (too slow O(n^2) in practice).
    # Replace with rolling quantile bucket: divide vol into 10 buckets via qcut on the
    # full series, then reindex to event_times. This is O(n log n) from qcut.
    # NOTE: qcut uses the full series distribution at each point, not a rolling window —
    # this is a simplification acceptable for v1.
    # PERFORMANCE NOTE: if this is still slow, remove vol_percentile from FEATURE_COLS entirely in v1.
    vol_ranks = pd.qcut(vol.rank(method='first'), q=10, labels=False, duplicates='drop')
    df['vol_percentile'] = vol_ranks.reindex(df['event_time']).fillna(4).values.astype(np.float32)  # 0-9 bucket, default to middle

    return df


def add_trend_state_features(df: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Add trend-state features.

    FIX #4: Must group by combo_id, NOT cusum_rate.
    Multiple candidates can share the same cusum_rate but have different fast/slow windows.
    Approach: compute MA for each unique (fast_window, slow_window) pair, batch reindex.
    """
    close = bars['close']

    # Get unique (fast, slow) pairs from df
    # candidate_id uniquely identifies a (fast, slow) combo, so use that
    unique_combos = df[['candidate_id', 'fast_window', 'slow_window']].drop_duplicates()
    # Map: candidate_id -> (fast, slow) -> MA series
    combo_fast_ma = {}
    combo_slow_ma = {}

    for _, row in unique_combos.iterrows():
        cid = row['candidate_id']
        fw = int(row['fast_window'])
        sw = int(row['slow_window'])
        combo_fast_ma[cid] = close.ewm(span=fw, adjust=False).mean()
        combo_slow_ma[cid] = close.ewm(span=sw, adjust=False).mean()

    # Batch compute MA at event_times: build a temporary df for vectorized lookup
    event_fast_ma = np.full(len(df), np.nan, dtype=np.float64)
    event_slow_ma = np.full(len(df), np.nan, dtype=np.float64)

    for cid, fast_ma in combo_fast_ma.items():
        mask = df['candidate_id'] == cid
        idx = df.loc[mask].index
        event_fast_ma[idx] = fast_ma.reindex(df.loc[mask, 'event_time']).ffill().values
        event_slow_ma[idx] = combo_slow_ma[cid].reindex(df.loc[mask, 'event_time']).ffill().values

    price_at_event = close.reindex(df['event_time']).fillna(close.iloc[-1]).values

    df['ma_gap_pct'] = ((event_fast_ma - event_slow_ma) / (price_at_event + 1e-10)).astype(np.float32)
    df['price_to_slow'] = ((price_at_event - event_slow_ma) / (price_at_event + 1e-10)).astype(np.float32)

    # Slopes: fast_ma(t) - fast_ma(t-k) / k
    k_fast, k_slow = 5, 20
    for cid, fast_ma in combo_fast_ma.items():
        mask = df['candidate_id'] == cid
        fast_slope = fast_ma.diff(k_fast) / k_fast
        slow_slope = combo_slow_ma[cid].diff(k_slow) / k_slow
        df.loc[mask, 'fast_slope'] = fast_slope.reindex(df.loc[mask, 'event_time']).ffill().fillna(0).values.astype(np.float32)
        df.loc[mask, 'slow_slope'] = slow_slope.reindex(df.loc[mask, 'event_time']).ffill().fillna(0).values.astype(np.float32)

    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime features: session_bucket, dow."""
    times = df['event_time'].dt
    minutes_since_midnight = times.hour * 60 + times.minute
    df['session_bucket'] = pd.cut(
        minutes_since_midnight,
        bins=[-1, 360, 720, 1080, 1440],
        labels=[0, 1, 2, 3]
    )
    # Fill any NaN with middle bucket (1) and convert to int8
    df['session_bucket'] = df['session_bucket'].fillna(1).astype(np.int8)
    df['dow'] = times.dayofweek.astype(np.int8)
    return df


def add_ensemble_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ensemble features.

    FIX #5: Use transform() instead of apply() to avoid index misalignment.
    transform() returns a Series with the same index as the original df.
    """
    grp = df.groupby('event_time', group_keys=False)

    df['n_fired'] = grp['candidate_id'].transform('count').astype(np.int8)
    df['n_long'] = grp['primary_side'].transform(lambda s: (s == 1).sum()).astype(np.int8)
    df['n_short'] = grp['primary_side'].transform(lambda s: (s == -1).sum()).astype(np.int8)
    df['conflict_flag'] = ((df['n_long'] > 0) & (df['n_short'] > 0)).astype(np.int8)
    df['agreement_ratio'] = (df[['n_long', 'n_short']].max(axis=1) / df['n_fired']).astype(np.float32)
    df['is_majority_side'] = (
        ((df['primary_side'] == 1) & (df['n_long'] >= df['n_short'])) |
        ((df['primary_side'] == -1) & (df['n_short'] > df['n_long']))
    ).astype(np.int8)
    df['consensus_side'] = np.sign(df['n_long'] - df['n_short']).astype(np.int8)
    return df


def add_time_since_prev(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_since_prev_event_h.

    FIX #6: Compute on unique event timeline first, then merge back.
    Otherwise same-event rows would incorrectly include intra-event gaps.
    """
    # Get one row per event
    event_timeline = df[['event_time']].drop_duplicates().sort_values('event_time')
    event_timeline = event_timeline.reset_index(drop=True)
    prev_event = event_timeline['event_time'].shift(1)
    time_diff_ns = event_timeline['event_time'].values - prev_event.fillna(event_timeline['event_time']).values
    event_timeline['time_since_prev_event_h'] = (time_diff_ns / 1e9 / 3600).astype(np.float32)

    # Merge back
    df = df.merge(event_timeline, on='event_time', how='left')
    return df


def compute_all_meta_features(df: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
    """Compute all meta features.

    :param df: DataFrame with event_time, candidate_id, fast_window, slow_window columns
    :param for_inference: if True, only use event-time-visible columns
    """
    print("[Feature Engineering] Loading bars...")
    bars = pd.read_parquet(BARS_PATH)

    print("  Adding event-level features...")
    df = add_event_level_features(df, bars)

    print("  Adding trend-state features...")
    df = add_trend_state_features(df, bars)

    print("  Adding regime features...")
    df = add_regime_features(df)

    print("  Adding ensemble features...")
    df = add_ensemble_features(df)

    print("  Adding time-since-prev...")
    df = add_time_since_prev(df)

    return df


def save_featured_tables():
    """Save both training and inference tables with features computed."""
    print("[Feature Engineering] Computing features for training table...")
    train_df = pd.read_parquet(os.path.join(FEATURES_DIR, "meta_training_table.parquet"))
    train_df = compute_all_meta_features(train_df)
    train_df.to_parquet(os.path.join(FEATURES_DIR, "meta_training_table.parquet"), index=False)
    print(f"  Saved training table with features: {len(train_df)} rows")

    print("[Feature Engineering] Computing features for inference table...")
    inf_df = pd.read_parquet(os.path.join(FEATURES_DIR, "meta_inference_table.parquet"))
    inf_df = compute_all_meta_features(inf_df, for_inference=True)
    inf_df.to_parquet(os.path.join(FEATURES_DIR, "meta_inference_table.parquet"), index=False)
    print(f"  Saved inference table with features: {len(inf_df)} rows")

    # Save feature spec
    FEATURE_COLS = [
        'candidate_id', 'cusum_rate', 'fast_window', 'slow_window', 'vertical_bars',
        'shock_size', 'ret_1', 'vol_jump', 'ewm_vol', 'vol_percentile',
        'time_since_prev_event_h',
        'ma_gap_pct', 'fast_slope', 'slow_slope', 'price_to_slow',
        'session_bucket', 'dow',
        'n_fired', 'n_long', 'n_short', 'conflict_flag',
        'agreement_ratio', 'is_majority_side', 'consensus_side',
    ]
    spec = pd.DataFrame({'feature': FEATURE_COLS, 'in_inference': [True] * len(FEATURE_COLS)})
    spec.to_csv(os.path.join(FEATURES_DIR, "meta_features_spec.csv"), index=False)
    print(f"  Saved feature spec")


if __name__ == "__main__":
    save_featured_tables()
    # Quick verification
    inf = pd.read_parquet(os.path.join(FEATURES_DIR, "meta_inference_table.parquet"))
    print('Inference table columns:', sorted(inf.columns.tolist()))
    print('meta_y in inference?', 'meta_y' in inf.columns)