# Unified Meta Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified Meta Model at the event-candidate level: one meta classifier trained on an event-candidate long table (6 candidates × events), using purging/embargo walk-forward validation, LightGBM with sigmoid calibration, and event-level deduplication.

**Architecture:** Replace per-candidate meta models with a single event-candidate panel where each row = (event_time, candidate). A single LightGBM classifier learns candidate identity + event context + ensemble signals. Purging/embargo prevents label horizon overlap leakage. Inference deduplicates per-event to output one decision per event.

**Tech Stack:** Python/NumPy/Pandas, LightGBM, scikit-learn (CalibratedClassifierCV, TimeSeriesSplit), matplotlib, shap

---

## File Structure

```
strategies/AL9999/
├── 06b_meta_training_table.py   # NEW: build event-candidate long table
├── meta_features.py          # NEW: feature engineering for meta
├── unified_meta_model.py      # NEW: unified meta model training + inference
├── output/
│   └── meta_model/               # NEW: all artifacts
│       ├── meta_training_table.parquet      # training + evaluation only
│       ├── meta_inference_table.parquet     # inference-only features (no meta_y, label_end_time)
│       ├── meta_features_spec.csv
│       ├── meta_splits.json
│       ├── meta_model_lgbm_v1.txt
│       ├── meta_calibrator_v1.pkl
│       ├── meta_oos_report.xlsx
│       ├── meta_inference_output.parquet
│       └── inference_api_spec.md
```

**Critical data separation:** `meta_training_table.parquet` contains columns that only exist post-event (meta_y, label_end_time, ret, touch_type) — NEVER use this at inference. `meta_inference_table.parquet` contains only event-time observable features.

---

### Task 1: Build Event-Candidate Long Table

**Files:**
- Create: `strategies/AL9999/06b_meta_training_table.py`
- Input: `strategies/AL9999/output/bars/dollar_bars_target15.parquet`, `strategies/AL9999/output/primary_search/cusum_calibration.parquet`, `strategies/AL9999/output/primary_search/top_candidates.parquet`
- Output: `strategies/AL9999/output/meta_model/meta_training_table.parquet`, `strategies/AL9999/output/meta_model/meta_inference_table.parquet`

**Before writing code, verify cusum_filter indexing:**

```python
# quick sanity check (run interactively first)
from afmlkit.sampling import cusum_filter
import numpy as np
close = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
log_ret = np.diff(np.log(close))
# log_ret[i] = return from bar i to bar i+1
# cusum_filter returns indices into log_ret array
indices = cusum_filter(log_ret, np.array([0.01]))
# indices=3 means CUSUM triggered on log_ret[3] = return from bar 3 to bar 4
# event timestamp should be bars.index[3+1] = bars.index[4]
# if you use bars.index[indices] directly you get bars.index[3] = OFF BY ONE
print("CUSUM triggered at log_ret index:", indices)
print("Correct event bar index:         ", indices + 1)
```

This confirms: **always use `event_indices + 1`** when mapping back to bars timestamps or prices.

- [ ] **Step 1: Write the skeleton with correct TOP_CANDIDATES (no typo)**

```python
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

# FIX #1: corrected typo "fast_fast" -> "fast"
TOP_CANDIDATES = [
    "rate=0.15_fast=5_slow=20_vb=10",   # candidate_id=0
    "rate=0.15_fast=8_slow=20_vb=10",   # candidate_id=1
    "rate=0.15_fast=5_slow=60_vb=10",   # candidate_id=2
    "rate=0.15_fast=5_slow=20_vb=20",   # candidate_id=3
    "rate=0.1_fast=5_slow=60_vb=30",    # candidate_id=4
    "rate=0.05_fast=8_slow=30_vb=10",   # candidate_id=5
]
```

- [ ] **Step 2: Write get_candidate_events with CUSUM off-by-one fix**

```python
def get_candidate_events(bars: pd.DataFrame, combo_id: str, k_lookup: pd.DataFrame) -> pd.DataFrame:
    """Generate CUSUM events and DMA side for one candidate combo.

    FIX #2: cusum_filter returns log_ret-array indices (0..len-2).
    Adding +1 maps to the bar where the return is realized.
    """
    m = re.match(r"rate=(\d+\.\d+)_fast=(\d+)_slow=(\d+)_vb=(\d+)", combo_id)
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
```

- [ ] **Step 3: Write compute_tbm_for_candidate**

```python
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
```

- [ ] **Step 4: Write build_meta_training_table (with inference table split)**

```python
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
```

- [ ] **Step 5: Run and verify**

Run: `python -c "
import sys; sys.path.insert(0, '.')
from strategies.AL9999.config import *
from strategies.AL9999.__init__ import *
exec(open('strategies/AL9999/06b_meta_training_table.py').read().split('if __name__')[0])
df = build_meta_training_table()
print(df.columns.tolist())
print(df[['event_time','combo_id','candidate_id','primary_side','meta_y','label_end_time']].head(6))
# Verify: same event_time should appear for different candidates
multi = df.groupby('event_time').filter(lambda g: len(g) > 1)
print(f'\nEvents with multiple candidates: {multi[\"event_time\"].nunique()}')
print(multi[['event_time','combo_id','candidate_id','primary_side','meta_y']].head(9))
"`
Expected: Each candidate generates its own event set (different CUSUM rates → different event counts). The 6 candidates do NOT all fire at the same times. So `df.groupby('event_time').size()` distribution should show 1-6 rows per unique event_time, NOT a fixed 6. Verify with:
```python
print(df.groupby('event_time').size().describe())
print(df.groupby('event_time').size().value_counts().sort_index())
```

- [ ] **Step 6: Commit**

```bash
git add strategies/AL9999/06b_meta_training_table.py
git commit -m "feat(AL9999): add event-candidate long table builder for unified meta model"
```

---

### Task 2: Feature Engineering for Meta Model

**Files:**
- Create: `strategies/AL9999/meta_features.py`
- Input: `strategies/AL9999/output/meta_model/meta_training_table.parquet`, `strategies/AL9999/output/meta_model/meta_inference_table.parquet`, `strategies/AL9999/output/bars/dollar_bars_target15.parquet`
- Test: `tests/strategies/test_meta_features.py`

- [ ] **Step 1: Write the feature engineering module with all fixes**

```python
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
        idx = df.loc[mask, 'event_time'].index
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
    df['session_bucket'] = pd.cut(
        times.hour * 60 + times.minute,
        bins=[0, 360, 720, 1080, 1440],
        labels=[0, 1, 2, 3]
    ).astype(np.int8)
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
```

- [ ] **Step 2: Write tests**

```python
# tests/strategies/test_meta_features.py
import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

# FIX #blocker2: Cannot use normal import because filename starts with digit (06c_).
# Use dynamic loading instead.
_spec = importlib.util.spec_from_file_location(
    "meta_features_module",
    Path("strategies/AL9999/meta_features.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

add_ensemble_features = _mod.add_ensemble_features
add_trend_state_features = _mod.add_trend_state_features
compute_all_meta_features = _mod.compute_all_meta_features

def test_ensemble_features_transform():
    """FIX #5: Verify transform() gives correct per-event values for all rows."""
    df = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01 10:00'] * 3 + ['2020-01-01 11:00'] * 3),
        'candidate_id': [0, 1, 2, 0, 1, 2],
        'primary_side': [1, 1, -1, 1, -1, -1],
    })
    df = add_ensemble_features(df)

    # Event 10:00 - 3 fired, 2 long, 1 short, agreement_ratio=2/3
    ev1 = df[df['event_time'] == '2020-01-01 10:00']
    assert ev1['n_fired'].iloc[0] == 3
    assert ev1['n_long'].iloc[0] == 2
    assert ev1['conflict_flag'].iloc[0] == 1
    assert abs(ev1['agreement_ratio'].iloc[0] - 2/3) < 1e-3
    # Every row for event 10:00 should have same n_fired/n_long/etc (not NaN or wrong)
    assert ev1['n_fired'].isna().sum() == 0

    # Event 11:00 - 3 fired, 1 long, 2 short
    ev2 = df[df['event_time'] == '2020-01-01 11:00']
    assert ev2['n_short'].iloc[0] == 2
    assert ev2['conflict_flag'].iloc[0] == 1

def test_trend_state_per_candidate():
    """FIX #4: Verify different candidates get different MA values."""
    # Two candidates with same cusum_rate but different fast/slow
    df = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01 10:00'] * 2),
        'candidate_id': [0, 1],   # different candidates
        'cusum_rate': [0.15, 0.15],  # same rate
        'fast_window': [5, 8],   # different fast
        'slow_window': [20, 20],  # same slow
    })
    bars = pd.DataFrame({
        'close': np.linspace(100, 110, 20),
    }, index=pd.date_range('2020-01-01', periods=20, freq='h'))
    bars.index.name = 'timestamp'

    df = add_trend_state_features(df, bars)
    # Different fast_window should give different ma_gap_pct
    assert df.loc[df['candidate_id']==0, 'ma_gap_pct'].iloc[0] != df.loc[df['candidate_id']==1, 'ma_gap_pct'].iloc[0]
```

- [ ] **Step 3: Run tests**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_meta_features.py -v`
Expected: PASS

- [ ] **Step 4: Compute features on actual data**

Run: `python -c "
import sys; sys.path.insert(0, '.')
from strategies.AL9999.config import *
from strategies.AL9999.__init__ import *
exec(open('strategies/AL9999/meta_features.py').read().split('if __name__')[0])
save_featured_tables()
inf = pd.read_parquet('strategies/AL9999/output/meta_model/meta_inference_table.parquet')
print('Inference table columns:', sorted(inf.columns.tolist()))
print('meta_y in inference?', 'meta_y' in inf.columns)
"`

- [ ] **Step 5: Commit**

```bash
git add strategies/AL9999/meta_features.py tests/strategies/test_meta_features.py
git commit -m "feat(AL9999): add meta model feature engineering with correct per-candidate MA and transform()"
```

---

### Task 3: Purging/Embargo and Walk-Forward Folds

**Files:**
- Create: `strategies/AL9999/unified_meta_model.py` (part 1: splits + purging helpers)
- Modify: `strategies/AL9999/config.py` (add META_MODEL_CONFIG)
- Test: `tests/strategies/test_purging.py`

- [ ] **Step 1: Write corrected purging function**

```python
# FIX #7: purge_by_label_overlap — clarify logic
def purge_by_label_overlap(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Remove training samples whose label_end_time falls inside the validation time window.

    The training set must not contain samples whose outcome is determined by
    information that overlaps with the validation period (information leakage).

    Rule: keep only samples where label_end_time <= val_start.
    Samples resolved before validation begins are safe to train on.

    This is a conservative (aggressive) purge. A less aggressive version
    could use: label_end_time < val_start (strictly before) or
    label_end_time < val_end with additional embargo.
    """
    if 'label_end_time' not in train_df.columns:
        return train_df

    before_val = train_df['label_end_time'] <= val_start
    is_nan = train_df['label_end_time'].isna()

    n_removed = len(train_df) - (before_val | is_nan).sum()
    if n_removed > 0:
        print(f"    Purging: removed {n_removed} rows overlapping validation [{val_start} ~ {val_end}]")

    return train_df[before_val | is_nan].copy()
```

- [ ] **Step 2: Write corrected embargo function**

```python
# FIX #5 from user feedback: embargo_after rewritten around val_start.
# In expanding window walk-forward, train is ALWAYS before val.
# The correct embargo target is val_start — remove train samples near val_start
# to create a temporal gap that prevents leakage from train tail into val.
# val_end is NOT used because in expanding window there is no train data after val_end
# within the same fold.
def embargo_after(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    embargo_delta: pd.Timedelta,
) -> pd.DataFrame:
    """
    Remove training samples whose event_time is within embargo_delta of val_start.

    Creates a temporal gap between training and validation sets to prevent
    information leakage from the training tail "bleeding" into validation.

    In expanding window: train = [earliest .. val_start - embargo], val = [val_start .. val_end]

    This replaces the incorrect previous version that used val_end as reference.
    """
    if 'event_time' not in train_df.columns:
        return train_df

    # Embargo boundary: samples must be at least embargo_delta BEFORE val_start
    embargo_boundary = val_start - embargo_delta

    before_embargo = train_df['event_time'] < embargo_boundary

    n_removed = len(train_df) - before_embargo.sum()
    if n_removed > 0:
        print(f"    Embargo: removed {n_removed} rows within {embargo_delta} of validation start ({val_start})")

    return train_df[before_embargo].copy()
```

- [ ] **Step 3: Write corrected walk-forward folds**

```python
# NOTE: this is time-uniform, NOT event-count-uniform — see docstring
def build_time_fraction_walk_forward_folds(
    event_times: pd.Series,
    n_folds: int = 6,
    val_horizon_days: int = 90,
    min_train_events: int = 50,
) -> list:
    """
    Build walk-forward expanding window folds using time-fraction boundaries.

    NOTE (FIX #7 from user feedback): this is time-uniform, NOT event-count-uniform.
    If event density varies significantly over time, fold sample sizes will vary.
    This is acceptable as a first pass; for stricter balance, consider event-count
    quantile-based boundaries in a future iteration.

    :param event_times: Series of event timestamps (sorted)
    :param n_folds: Number of validation windows
    :param val_horizon_days: Duration of each validation window in days
    :param min_train_events: Minimum number of training events per fold
    :returns: List of (train_idx, val_idx, fold_id, val_start, val_end, train_end) tuples
    """
    # FIX #blocker1: convert to pd.Timestamp to ensure .date() method is available
    # event_times.unique() returns numpy.datetime64 which has no .date() method
    sorted_times = pd.to_datetime(event_times.sort_values().unique())
    n_total = len(sorted_times)

    val_duration = pd.Timedelta(days=val_horizon_days)

    # Determine fold boundaries by time fractions (not event-count fractions)
    # Fold i: train = [earliest .. time_at_frac_i), val = [time_at_frac_i .. time_at_frac_i + val_duration)
    # This gives time-uniform folds; sample counts may vary
    fold_boundaries = []
    for fold_id in range(n_folds + 1):
        frac = fold_id / n_folds
        time_at_frac = sorted_times[0] + (sorted_times[-1] - sorted_times[0]) * frac
        fold_boundaries.append(time_at_frac)

    folds = []
    for fold_id in range(n_folds):
        train_end = fold_boundaries[fold_id]
        val_start = fold_boundaries[fold_id + 1]
        val_end = val_start + val_duration

        if val_end > sorted_times[-1]:
            continue  # skip if validation extends beyond data

        train_mask = event_times < train_end
        val_mask = (event_times >= val_start) & (event_times < val_end)

        n_train_rows = train_mask.sum()
        n_val_rows = val_mask.sum()

        n_train_events = event_times[train_mask].nunique()
        n_val_events = event_times[val_mask].nunique()

        if n_train_events < min_train_events or n_val_events < 10:
            continue  # skip folds with too few unique events

        train_idx = event_times[train_mask].index
        val_idx = event_times[val_mask].index

        folds.append((train_idx, val_idx, fold_id, val_start, val_end, train_end))
        # FIX #blocker1: pd.Timestamp ensures .date() method is available
        print(f"  Fold {fold_id}: train_rows={n_train_rows}, train_events={n_train_events}, "
              f"val_rows={n_val_rows}, val_events={n_val_events} | "
              f"val=[{pd.Timestamp(val_start).date()} ~ {pd.Timestamp(val_end).date()}]")

    return folds
```

- [ ] **Step 4: Write tests for purging and embargo**

```python
# tests/strategies/test_purging.py
import numpy as np
import pandas as pd
import pytest
from strategies.AL9999.config import META_MODEL_CONFIG
import sys
sys.path.insert(0, '.')

def test_purge_removes_overlapping():
    """Purging removes rows whose label_end_time overlaps with validation interval."""
    train = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-20', '2020-02-05', '2020-03-01']),
        'label_end_time': pd.to_datetime(['2020-01-05', '2020-01-25', '2020-02-15', '2020-03-01', '2020-04-01']),
        'value': [1, 2, 3, 4, 5],
    })
    val_start = pd.Timestamp('2020-02-01')
    val_end = pd.Timestamp('2020-03-01')

    # label_end_time <= val_start → KEEP
    # row 0: 2020-01-05 <= 2020-02-01 → KEEP
    # row 1: 2020-01-25 <= 2020-02-01 → KEEP
    # row 2: 2020-02-15 > 2020-02-01 → PURGE (overlaps)
    # row 3: 2020-03-01 > 2020-02-01 → PURGE (overlaps, exactly at val_end)
    # row 4: 2020-04-01 > 2020-02-01 → PURGE

    exec(open('strategies/AL9999/unified_meta_model.py').read().split('def fit_lgbm')[0])  # load helpers
    purged = purge_by_label_overlap(train, val_start, val_end)
    assert len(purged) == 2, f"Expected 2 rows, got {len(purged)}"
    assert purged['value'].tolist() == [1, 2]

def test_embargo_removes_train_tail():
    """Embargo removes training rows whose event_time is too close to val_start."""
    # In expanding window: val_start = 2020-03-01
    # embargo = 20 days → embargo_boundary = 2020-03-01 - 20d = 2020-02-09
    # Keep rows where event_time < 2020-02-09
    train = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01', '2020-02-05', '2020-02-15', '2020-02-20', '2020-03-01']),
        'value': [1, 2, 3, 4, 5],
    })
    val_start = pd.Timestamp('2020-03-01')
    embargo = pd.Timedelta(days=20)

    # embargo_boundary = 2020-03-01 - 20 days = 2020-02-09
    # rows 3 (2020-02-20) and 4 (2020-03-01) >= 2020-02-09 → REMOVE

    exec(open('strategies/AL9999/unified_meta_model.py').read().split('def fit_lgbm')[0])  # load helpers
    embargoed = embargo_after(train, val_start, embargo)
    # Only rows 1 (2020-01-01) and 2 (2020-02-05) are < 2020-02-09 → KEEP
    assert embargoed['value'].tolist() == [1, 2]

def test_embargo_all_far_enough():
    """All training rows far enough from val_start → none removed."""
    train = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01', '2020-01-15', '2020-01-30']),
        'value': [1, 2, 3],
    })
    val_start = pd.Timestamp('2020-03-01')
    embargo = pd.Timedelta(days=20)

    # embargo_boundary = 2020-02-09
    # all rows are < 2020-02-09 → KEEP all
    exec(open('strategies/AL9999/unified_meta_model.py').read().split('def fit_lgbm')[0])  # load helpers
    embargoed = embargo_after(train, val_start, embargo)
    assert embargoed['value'].tolist() == [1, 2, 3]
```

- [ ] **Step 5: Run tests**

Run: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_purging.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Add META_MODEL_CONFIG to config.py**

```python
# In config.py, add:
META_MODEL_CONFIG = {
    'n_folds': 6,
    'val_horizon_days': 90,
    'embargo_days': 30,   # conservative: max(vertical_bars) * bar_duration ≈ 30 days
    'min_train_events': 50,
    'lgbm_params': {
        'objective': 'binary',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'min_data_in_leaf': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    },
    'early_stopping_rounds': 200,
    'precision_threshold': 0.65,
    'calibration_method': 'sigmoid',  # or 'isotonic' when sample_size > 1000
}
```

- [ ] **Step 7: Commit**

```bash
git add strategies/AL9999/config.py tests/strategies/test_purging.py
git commit -m "feat(AL9999): add purging/embargo with corrected val_end logic and stable walk-forward folds"
```

---

### Task 4: LightGBM Training with Walk-Forward CV + Proper Calibration

**Files:**
- Modify: `strategies/AL9999/unified_meta_model.py` (add training + calibration)
- Test: `tests/strategies/test_meta_training.py`

**FIX #9 (from user feedback): Use LGBMClassifier directly, NOT a Booster wrapper.**
The Booster object is NOT sklearn-compatible. LGBMClassifier is natively compatible
with CalibratedClassifierCV.

- [ ] **Step 1: Write the feature column definitions and LGBMClassifier-based training**

```python
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
)
import joblib
import json
import openpyxl

META_MODEL_DIR = "strategies/AL9999/output/meta_model"

FEATURE_COLS = [
    # Candidate identity
    'candidate_id', 'cusum_rate', 'fast_window', 'slow_window', 'vertical_bars',
    # Event-level
    'shock_size', 'ret_1', 'vol_jump', 'ewm_vol', 'vol_percentile',
    'time_since_prev_event_h',
    # Trend-state
    'ma_gap_pct', 'fast_slope', 'slow_slope', 'price_to_slow',
    # Regime
    'session_bucket', 'dow',
    # Ensemble
    'n_fired', 'n_long', 'n_short', 'conflict_flag',
    'agreement_ratio', 'is_majority_side',
]

CATEGORICAL_FEATURES = ['candidate_id', 'session_bucket', 'dow']

def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]

def compute_primary_baselines(val_df: pd.DataFrame) -> dict:
    """Compute three baseline precisions for comparing with meta model.

    Three baselines for comparing meta model precision:

    1. row_precision: fraction of meta_y=1 across all rows (naive row-level baseline)

    2. event_precision (OPTIMISTIC): for each event, check if ANY candidate succeeded.
    This is optimistic because it treats an event as "correct" if at least one of
    its candidate rows happened to be correct — even if that candidate was not the
    one the meta model would have selected.

    3. event_majority_precision (REALISTIC): for each event, check if the
    majority-side candidate succeeded. Better comparison for per_event_max_p deduplication
    because it simulates "select by majority side" as a baseline strategy.

    Use event_majority_precision as the primary comparison baseline for precision_lift.
    """
    y_row = val_df['meta_y']
    row_prec = y_row.mean()

    # OPTIMISTIC: event correct if any candidate succeeded
    ev_prec = val_df.groupby('event_time')['meta_y'].max().mean()

    # REALISTIC: event correct if majority-side candidate succeeded
    def majority_side_success(g):
        majority = 1 if (g['primary_side'] == 1).sum() > (g['primary_side'] == -1).sum() else -1
        return g.loc[g['primary_side'] == majority, 'meta_y'].mean() if len(g.loc[g['primary_side'] == majority]) > 0 else 0

    ev_majority_prec = val_df.groupby('event_time').apply(majority_side_success).mean()

    return {
        'row_precision': row_prec,
        'event_precision': ev_prec,        # optimistic
        'event_majority_precision': ev_majority_prec,  # realistic
    }
```

- [ ] **Step 2: Write the walk-forward CV training loop**

```python
def fit_lgbm_walk_forward(
    df: pd.DataFrame,
    n_folds: int = 6,
    val_horizon_days: int = 90,
    embargo_delta: pd.Timedelta = pd.Timedelta(days=30),
) -> tuple:
    """
    Walk-forward CV with purging + embargo.
    Uses LGBMClassifier (sklearn-compatible) so CalibratedClassifierCV works natively.

    :returns: (oof_df, fold_reports, final_model_for_inference)
    """
    from strategies.AL9999.config import META_MODEL_CONFIG

    df = df.sort_values('event_time').reset_index(drop=True)
    event_times = df['event_time']
    feature_cols = get_feature_columns(df)
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    folds = build_time_fraction_walk_forward_folds(
        event_times, n_folds=n_folds,
        val_horizon_days=val_horizon_days,
        min_train_events=META_MODEL_CONFIG['min_train_events'],
    )

    oof_rows = []
    fold_reports = []
    # FIX #blocker2: initialize before loop to avoid UnboundLocalError when folds is empty
    final_model_for_inference = None

    for fold_i, (train_idx, val_idx, fold_id, val_start, val_end, train_end) in enumerate(folds):
        print(f"\n  === Fold {fold_id} ===")
        print(f"    Train: {str(train_end)[:10]} ({len(train_idx)} rows) | Val: {str(val_start)[:10]} ~ {str(val_end)[:10]}")

        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()

        # Purging + embargo
        train_df = purge_by_label_overlap(train_df, val_start, val_end)
        train_df = embargo_after(train_df, val_start, embargo_delta)

        X_tr = train_df[feature_cols].fillna(0).astype(np.float32)
        y_tr = train_df['meta_y'].values.astype(int)
        X_va = val_df[feature_cols].fillna(0).astype(np.float32)
        y_va = val_df['meta_y'].values.astype(int)

        print(f"    After purge/embargo: train={len(X_tr)} rows | val={len(X_va)} rows")

        # Sample weight: 1/n_fired to reduce intra-event double counting
        w_tr = (1.0 / train_df['n_fired'].values).astype(np.float32)

        # Primary baselines
        baselines = compute_primary_baselines(val_df)

        # LGBMClassifier (sklearn-compatible, works with CalibratedClassifierCV)
        # FIX #8 from user feedback: categorical_feature must be in fit(), not __init__()
        model = LGBMClassifier(
            **META_MODEL_CONFIG['lgbm_params'],
            n_estimators=2000,
        )

        # CalibratedClassifierCV: FIX #3 - skip calibration when too small
        # cv=1 is not recommended; instead skip calibration entirely when training set < 200
        # FIX #4: ensemble=False by default — single estimator is easier to save/load/explain
        calibration_method = META_MODEL_CONFIG.get('calibration_method', 'sigmoid')
        n_train = len(X_tr)
        if n_train >= 300:
            cal_cv = 3
            cal_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=cal_cv,
                ensemble=False,  # v1 default: single estimator is easier to save/load/explain
            )
        elif n_train >= 200:
            cal_cv = 2
            cal_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=cal_cv,
                ensemble=False,
            )
        else:
            # Too small for calibration — skip and use raw model
            print(f"    Warning: train set ({n_train}) < 200, skipping calibration")
            cal_model = None

        # FIX #8: pass categorical_feature via set_params(), not fit() kwargs
        # This ensures CalibratedClassifierCV safely passes it down
        model.set_params(categorical_feature=[c for c in cat_features if c in X_tr.columns])

        if cal_model is not None:
            cal_model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
            )
        else:
            # Fit raw model directly (no calibration)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
            )
            cal_model = model  # use raw model when calibration skipped

        # Predict on validation (works for both calibrated and raw model)
        p_va = cal_model.predict_proba(X_va)[:, 1]

        # Add probabilities back to val_df
        val_df = val_df.copy().reset_index(drop=True)
        val_df['p_meta'] = p_va

        # Row-level metrics
        row_auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else np.nan
        row_ap = average_precision_score(y_va, p_va)
        row_brier = brier_score_loss(y_va, p_va)
        row_precision = precision_score(y_va, (p_va >= 0.5).astype(int), zero_division=0)

        # Event-level metrics: per_event_max_p deduplication
        idx_best = val_df.groupby('event_time')['p_meta'].idxmax()
        deduped = val_df.loc[idx_best]

        y_dedup = deduped['meta_y'].values
        p_dedup = deduped['p_meta'].values
        ev_precision = precision_score(y_dedup, (p_dedup >= 0.5).astype(int), zero_division=0)
        ev_recall = recall_score(y_dedup, (p_dedup >= 0.5).astype(int), zero_division=0)
        ev_auc = roc_auc_score(y_dedup, p_dedup) if len(np.unique(y_dedup)) > 1 else np.nan

        print(f"    Row: AUC={row_auc:.4f} AP={row_ap:.4f} | Event: Prec={ev_precision:.4f} Rec={ev_recall:.4f}")

        # Per-fold baselines
        baselines_dedup = compute_primary_baselines(val_df)

        fold_reports.append({
            'fold_id': fold_id,
            'train_end': str(train_end)[:10],
            'val_start': str(val_start)[:10],
            'val_end': str(val_end)[:10],
            'n_train_rows': len(X_tr),
            'n_val_rows': len(X_va),
            'n_val_events': val_df['event_time'].nunique(),
            'base_rate': y_va.mean(),
            # Primary baselines (for comparison with meta)
            'primary_row_precision': baselines['row_precision'],
            'primary_event_precision': baselines['event_precision'],
            'primary_event_majority_precision': baselines['event_majority_precision'],
            # Meta metrics
            'meta_row_precision': row_precision,
            'meta_row_auc': row_auc,
            'meta_row_ap': row_ap,
            'meta_row_brier': row_brier,
            'meta_event_precision': ev_precision,
            'meta_event_recall': ev_recall,
            'meta_event_auc': ev_auc,
            'precision_lift_vs_row': ev_precision / baselines['row_precision'] if baselines['row_precision'] > 0 else np.nan,
            'precision_lift_vs_event': ev_precision / baselines['event_majority_precision'] if baselines['event_majority_precision'] > 0 else np.nan,
            'best_iteration': cal_model.best_iteration_ if hasattr(cal_model, 'best_iteration_') else 0,
        })

        # Save OOF
        oof_row = val_df[['event_time', 'event_id', 'combo_id', 'candidate_id',
                           'primary_side', 'meta_y', 'p_meta', 'fold_id']].copy()
        oof_rows.append(oof_row)

        # FIX #blocker2: update after each successful fold training
        final_model_for_inference = cal_model

    # FIX #blocker2: check if any fold was successfully trained
    if final_model_for_inference is None or len(oof_rows) == 0:
        raise ValueError(
            "No valid walk-forward folds were produced. "
            "Reduce min_train_events / val_horizon_days / embargo_days, or inspect event density."
        )

    oof_df = pd.concat(oof_rows, ignore_index=True)
    return oof_df, fold_reports, final_model_for_inference
```

- [ ] **Step 2: Write the main training script**

```python
def main():
    print("=" * 70)
    print("  AL9999 Unified Meta Model Training")
    print("=" * 70)

    os.makedirs(META_MODEL_DIR, exist_ok=True)

    # Step 1: Load training table
    print("\n[Step 1] Loading meta training table...")
    df = pd.read_parquet(os.path.join(META_MODEL_DIR, "meta_training_table.parquet"))
    print(f"  Loaded {len(df)} rows, {df['event_time'].nunique()} unique events")

    # Step 2: Walk-forward CV
    print("\n[Step 2] Walk-forward CV with purging/embargo...")
    embargo_delta = pd.Timedelta(days=META_MODEL_CONFIG['embargo_days'])
    # FIX #blocker3: unpack 3 values including final_model_for_inference
    oof_df, fold_reports, final_model_for_inference = fit_lgbm_walk_forward(
        df,
        n_folds=META_MODEL_CONFIG['n_folds'],
        val_horizon_days=META_MODEL_CONFIG['val_horizon_days'],
        embargo_delta=embargo_delta,
    )

    # Step 3: Save OOF
    print("\n[Step 3] Saving OOF predictions...")
    oof_df.to_parquet(os.path.join(META_MODEL_DIR, "meta_oof_signals.parquet"), index=False)
    print(f"  OOF saved: {len(oof_df)} rows")

    # Step 4: Save fold reports (parquet)
    reports_df = pd.DataFrame(fold_reports)
    reports_df.to_parquet(os.path.join(META_MODEL_DIR, "meta_fold_reports.parquet"), index=False)

    # Step 5: Generate xlsx OOS report
    print("\n[Step 5] Generating OOS report (xlsx)...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OOS Report"

    headers = [
        'Fold', 'Train End', 'Val Start', 'Val End',
        'Train Rows', 'Val Rows', 'Val Events',
        'Base Rate',
        # event_precision = optimistic (any candidate wins), event_majority_precision = realistic
        'Primary Event Prec (Optimistic)', 'Primary Event Prec (Realistic)', 'Meta Event Prec', 'Precision Lift',
        'Filtered Recall', 'Event AUC',
        'Row AUC', 'Row AP', 'Row Brier',
    ]
    ws.append(headers)
    for r in fold_reports:
        ws.append([
            r['fold_id'], r['train_end'], r['val_start'], r['val_end'],
            r['n_train_rows'], r['n_val_rows'], r['n_val_events'],
            f"{r['base_rate']:.4f}",
            f"{r['primary_event_precision']:.4f}",  # optimistic
            f"{r['primary_event_majority_precision']:.4f}",  # realistic
            f"{r['meta_event_precision']:.4f}",
            f"{r['precision_lift_vs_event']:.4f}",  # vs realistic
            f"{r['meta_event_recall']:.4f}",
            f"{r['meta_event_auc']:.4f}",
            f"{r['meta_row_auc']:.4f}",
            f"{r['meta_row_ap']:.4f}",
            f"{r['meta_row_brier']:.4f}",
        ])

    wb.save(os.path.join(META_MODEL_DIR, "meta_oos_report.xlsx"))
    print(f"  OOS report saved: meta_oos_report.xlsx")

    # Step 6: Save feature importance
    feature_cols = get_feature_columns(df)
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    final_model = LGBMClassifier(**META_MODEL_CONFIG['lgbm_params'], n_estimators=500)
    final_model.set_params(categorical_feature=[c for c in cat_features if c in feature_cols])
    final_model.fit(df[feature_cols].fillna(0).astype(np.float32), df['meta_y'])
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(META_MODEL_DIR, "meta_feature_importance.csv"), index=False)

    # Step 7: Save splits config for reproducibility
    splits_config = {
        'n_folds': META_MODEL_CONFIG['n_folds'],
        'val_horizon_days': META_MODEL_CONFIG['val_horizon_days'],
        'embargo_days': META_MODEL_CONFIG['embargo_days'],
        'min_train_events': META_MODEL_CONFIG['min_train_events'],
        'feature_cols': feature_cols,
    }
    with open(os.path.join(META_MODEL_DIR, "meta_splits.json"), 'w') as f:
        json.dump(splits_config, f, indent=2)

    print("\n" + "=" * 70)
    print("  Unified Meta Model Training Complete")
    print("=" * 70)
    print(f"\nOOS Summary:")
    for r in fold_reports:
        print(f"  Fold {r['fold_id']}: Meta Prec={r['meta_event_precision']:.4f} vs Primary(realistic)={r['primary_event_majority_precision']:.4f} | Lift(vs realistic)={r['precision_lift_vs_event']:.4f}")
```

- [ ] **Step 3: Run training**

Run: `NUMBA_DISABLE_JIT=1 python strategies/AL9999/unified_meta_model.py`
Expected: Creates meta_oos_report.xlsx, meta_fold_reports.parquet, meta_feature_importance.csv, meta_splits.json

- [ ] **Step 4: Commit**

```bash
git add strategies/AL9999/unified_meta_model.py
git commit -m "feat(AL9999): add unified meta model with LGBMClassifier, walk-forward purging, and sigmoid calibration"
```

---

### Task 5: Inference Pipeline and Decision Output

**Files:**
- Modify: `strategies/AL9999/unified_meta_model.py` (add inference functions)
- Create: `strategies/AL9999/output/meta_model/inference_api_spec.md`

**FIX #13 (from user feedback): Build inference MUST use meta_inference_table.parquet, NOT meta_training_table.parquet.**
The training table contains meta_y, label_end_time, ret, touch_type — none of which are available at live inference time.

- [ ] **Step 1: Write threshold selection function**

```python
def select_threshold_by_precision(
    val_df: pd.DataFrame,
    target_precision: float = 0.65,
    dedupe_mode: str = 'per_event_max_p',
) -> tuple[float, float, float]:
    """
    Select threshold that achieves target precision on validation set.

    :returns: (threshold, actual_precision, n_take_events)
    """
    val_df = val_df.copy().reset_index(drop=True)
    idx_best = val_df.groupby('event_time')['p_meta'].idxmax()
    deduped = val_df.loc[idx_best]
    y_true = deduped['meta_y']
    p_vals = deduped['p_meta']

    best_thresh, best_prec, best_n = 0.5, 0.0, 0
    for thresh in np.arange(0.50, 0.91, 0.01):
        pred = (p_vals >= thresh).astype(int)
        n_take = pred.sum()
        if n_take == 0:
            continue
        prec = precision_score(y_true, pred, zero_division=0)
        if prec >= target_precision and n_take > best_n:
            best_thresh, best_prec, best_n = thresh, prec, n_take

    # Fallback: use 0.5 if target not achievable
    if best_thresh == 0.5 and best_prec < target_precision:
        pred = (p_vals >= 0.5).astype(int)
        best_thresh = 0.5
        best_prec = precision_score(y_true, pred, zero_division=0)
        best_n = pred.sum()

    return best_thresh, best_prec, best_n
```

- [ ] **Step 2: Write deduplication function**

```python
def deduplicate_per_event(
    df: pd.DataFrame,
    p_col: str = 'p_meta',
    mode: str = 'per_event_max_p',
) -> pd.DataFrame:
    """
    Deduplicate to one row per event.

    :param df: DataFrame with event-candidate rows
    :param p_col: probability column
    :param mode: 'per_event_max_p' — take candidate with highest meta probability
                 'per_event_first' — take first candidate (deterministic)
    """
    if mode == 'per_event_max_p':
        idx_best = df.groupby('event_time')[p_col].idxmax()
        return df.loc[idx_best].copy()
    elif mode == 'per_event_first':
        return df.groupby('event_time').first().reset_index()
    else:
        raise ValueError(f"Unknown mode: {mode}")
```

- [ ] **Step 3: Write inference output builder**

```python
# FIX #13: NEVER use meta_training_table for inference
def build_inference_output(
    inference_df: pd.DataFrame,  # meta_inference_table.parquet, NOT meta_training_table
    oof_df: pd.DataFrame,         # used for threshold calibration only
    threshold: float,
    model,                         # fitted CalibratedClassifierCV
    dedupe_mode: str = 'per_event_max_p',
) -> pd.DataFrame:
    """
    Build inference output from the inference-only feature table.

    CRITICAL: inference_df must come from meta_inference_table.parquet.
    It contains only event-time-visible features. It must NOT contain
    meta_y, label_end_time, ret, touch_type, tbm_outcome.
    """
    if 'meta_y' in inference_df.columns:
        raise ValueError("FATAL: inference_df contains meta_y! Use meta_inference_table.parquet only.")

    inference_df = inference_df.copy().reset_index(drop=True)
    feature_cols = get_feature_columns(inference_df)
    X = inference_df[feature_cols].fillna(0).astype(np.float32)

    # Predict
    p_meta = model.predict_proba(X)[:, 1]
    inference_df['p_meta'] = p_meta
    inference_df['threshold'] = threshold

    # Deduplicate
    deduped = deduplicate_per_event(inference_df, p_col='p_meta', mode=dedupe_mode)

    # Build output schema
    output = pd.DataFrame({
        'event_time': deduped['event_time'],
        'event_id': deduped['event_id'],
        'chosen_combo_id': deduped['combo_id'],
        'side': deduped['primary_side'],
        'p_meta': deduped['p_meta'].astype(np.float32),
        'threshold': threshold,
        'decision': np.where(deduped['p_meta'] >= threshold, 'TAKE', 'SKIP'),
    })
    output['model_version'] = 'meta_model_v1'
    return output
```

- [ ] **Step 4: Write and save inference API spec**

```markdown
# Meta Model Inference API Spec

## Data Separation (CRITICAL)

| Table | Purpose | Contains meta_y? |
|---|---|---|
| meta_training_table.parquet | Training + evaluation | YES — NEVER use at inference |
| meta_inference_table.parquet | Live inference | NO — safe to use |

## Inference Usage

```python
from strategies.AL9999.unified_meta_model import (
    build_inference_output,
    select_threshold_by_precision,
)
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Load artifacts
cal_model = joblib.load("strategies/AL9999/output/meta_model/meta_calibrator_v1.pkl")
inf_df = pd.read_parquet("strategies/AL9999/output/meta_model/meta_inference_table.parquet")
oof_df = pd.read_parquet("strategies/AL9999/output/meta_model/meta_oof_signals.parquet")

# Select threshold on OOF
threshold, prec, n = select_threshold_by_precision(oof_df, target_precision=0.65)
print(f"Threshold={threshold:.2f} achieves precision={prec:.4f} on {n} events")

# Build inference
output = build_inference_output(inf_df, oof_df, threshold, cal_model)
output.to_parquet("strategies/AL9999/output/meta_model/meta_inference_output.parquet", index=False)
```

## Output Schema

| Field | Type | Description |
|---|---|---|
| event_time | datetime64[ns] | Event timestamp |
| event_id | string | Unique event identifier |
| chosen_combo_id | string | Selected candidate |
| side | int8 | +1 (long) or -1 (short) |
| p_meta | float32 | Calibrated meta probability |
| threshold | float32 | Decision threshold used |
| decision | string | "TAKE" or "SKIP" |
| model_version | string | "meta_model_v1" |

## Decision Rules

1. Deduplication: `per_event_max_p` — one row per event, highest probability wins
2. Threshold: selected by target precision (default 0.65) on OOS validation
3. Decision: TAKE if p_meta >= threshold, else SKIP
```

- [ ] **Step 5: Add inference to main script and run**

Add to main() in unified_meta_model.py:

```python
    # Step 8: Save final model (from last fold) for inference
    print("\n[Step 8] Saving final model...")
    # FIX #blocker3: use final_model_for_inference returned from fit_lgbm_walk_forward
    joblib.dump(final_model_for_inference, os.path.join(META_MODEL_DIR, "meta_calibrator_v1.pkl"))
    print(f"  Saved calibrator: meta_calibrator_v1.pkl")

    # Step 9: Inference output
    print("\n[Step 9] Building inference output...")
    inf_df = pd.read_parquet(os.path.join(META_MODEL_DIR, "meta_inference_table.parquet"))

    # Verify no label columns leak into inference table
    assert 'meta_y' not in inf_df.columns, "FATAL: meta_y in inference table!"
    assert 'label_end_time' not in inf_df.columns, "FATAL: label_end_time in inference table!"

    # Select threshold on OOF
    threshold, prec, n = select_threshold_by_precision(oof_df, target_precision=0.65)
    print(f"  Selected threshold={threshold:.2f} → precision={prec:.4f} on {n} OOS events")

    # Build inference output with the SAVED model
    # FIX #blocker3: use final_model_for_inference instead of undefined cal_model
    output = build_inference_output(inf_df, oof_df, threshold, final_model_for_inference)
    output.to_parquet(os.path.join(META_MODEL_DIR, "meta_inference_output.parquet"), index=False)
    n_take = (output['decision'] == 'TAKE').sum()
    print(f"  Inference output: {len(output)} events, {n_take} TAKE")

    # NOTE: inference_api_spec.md is created separately by the agent after review.
    # Do NOT write it from this script (INFERENCE_API_SPEC was never defined here).
```

- [ ] **Step 6: Commit**

```bash
git add strategies/AL9999/unified_meta_model.py
git commit -m "feat(AL9999): add inference pipeline with strict data separation and threshold selection"
```

---

### Task 6: SHAP Explainability (Lower Priority — Run After Tasks 1-5 are Stable)

**Files:**
- Modify: `strategies/AL9999/unified_meta_model.py` (add SHAP functions)
- Output: `strategies/AL9999/output/meta_model/meta_shap_summary.png`, `meta_reason_tags.parquet`

**Note (from user feedback):** SHAP is not a blocking dependency. Run Tasks 1-5 first and verify training/calibration/inference are stable before adding SHAP.

- [ ] **Step 1: Write SHAP explanation functions**

```python
import shap

FEATURE_TAG_MAP = {
    'n_fired': 'high_ensemble_signal',
    'agreement_ratio': 'strong_consensus',
    'ewm_vol': 'high_vol_regime',
    'ma_gap_pct': 'strong_trend',
    'fast_slope': 'fast_ma_momentum',
    'slow_slope': 'slow_ma_momentum',
    'conflict_flag': 'intra_event_conflict',
    'vol_jump': 'volatility_jump',
    'ret_1': 'strong_return_bar',
    'price_to_slow': 'far_from_slow_ma',
    'candidate_id': 'top_recall_candidate',
    'cusum_rate': 'high_freq_candidate',
}

def compute_shap_for_sample(model, X_sample: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Compute SHAP values for a sample using TreeExplainer."""
    explainer = shap.TreeExplainer(model.estimator_ if hasattr(model, 'estimator_') else model)
    shap_values = explainer.shap_values(X_sample[feature_cols])
    # shap_values shape: (n_samples, n_features) for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # take positive class
    return shap_values

def generate_reason_tags(shap_values: np.ndarray, feature_cols: list, top_k: int = 3) -> list:
    """Generate human-readable reason tags from top SHAP contributions."""
    tags = []
    for i in range(len(shap_values)):
        row = shap_values[i]
        top_indices = np.argsort(np.abs(row))[-top_k:][::-1]
        row_tags = []
        for idx in top_indices:
            feat_name = feature_cols[idx]
            tag = FEATURE_TAG_MAP.get(feat_name, feat_name)
            direction = 'positive' if row[idx] > 0 else 'negative'
            row_tags.append(f"{tag}({direction})")
        tags.append('; '.join(row_tags))
    return tags

def save_shap_summary(model, X_sample: pd.DataFrame, feature_cols: list, output_dir: str):
    """Save SHAP summary plot."""
    shap_values = compute_shap_for_sample(model, X_sample, feature_cols)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample[feature_cols], show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "meta_shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  SHAP summary saved")

    # Importance from mean |SHAP|
    imp = pd.DataFrame({
        'feature': feature_cols,
        'shap_importance': np.abs(shap_values).mean(axis=0),
    }).sort_values('shap_importance', ascending=False)
    imp.to_csv(os.path.join(output_dir, "meta_feature_importance_shap.csv"), index=False)
    return imp
```

- [ ] **Step 2: Run SHAP on sample (only after Tasks 1-5 verified)**

```python
# Add to main() after Step 7:
# Step 9: SHAP explainability (only if training is stable)
print("\n[Step 9] Computing SHAP explanations...")
X_sample = df[feature_cols].fillna(0).astype(np.float32).sample(min(2000, len(df)), random_state=42)
# FIX #blocker2: use final_model_for_inference, not undefined cal_model
shap_imp = save_shap_summary(final_model_for_inference, X_sample, feature_cols, META_MODEL_DIR)
print("\n  SHAP Top 10 Features:")
for _, row in shap_imp.head(10).iterrows():
    print(f"    {row['feature']}: {row['shap_importance']:.4f}")
```

- [ ] **Step 3: Commit**

```bash
git add strategies/AL9999/unified_meta_model.py
git commit -m "feat(AL9999): add SHAP explainability with reason tags"
```

---

## Self-Review

### Issue-by-Issue Fix Verification

| # | Issue | Fix Applied |
|---|---|---|
| 1 | TOP_CANDIDATES typo "fast_fast" | Corrected to "rate=0.15_fast=5_slow=60_vb=10" |
| 2 | CUSUM off-by-one: `bars.index[event_indices]` | Added +1: `event_bar_indices = event_ret_indices + 1`; verified with interactive test |
| 3 | add_trend_state_features grouping by cusum_rate | Changed to group by candidate_id (unique fast/slow per candidate) |
| 4 | reindex(row['event_time'], ...).iloc[-1] in loop | Batch compute per candidate_id then vectorized assign to df |
| 5 | groupby.apply() returns wrong-index Series | Changed to `transform()` with same-index return |
| 6 | purge_by_label_overlap ignores val_end | Clarified: only uses val_start (conservative purge); val_end is parameter for doc only |
| 7 | embargo_after uses train's own label_end_time.max() | Rewritten: embargo_boundary = val_start - embargo_delta (train tail cutoff, not val_end) |
| 8 | build_walk_forward_folds unstable boundaries | Fixed: time-fraction-based boundaries; added min_train_events constraint |
| 9 | CalibratedClassifierCV with broken LGBMWrapper | Fixed: use LGBMClassifier directly (native sklearn API) |
| 10 | predict_proba shape (n,) vs (n,2) | Fixed by using LGBMClassifier (returns (n,2) natively) |
| 11 | primary_precision definition wrong | Fixed: added row_precision, event_precision, event_majority_precision as three baselines |
| 12 | build_inference_output uses training table | Fixed: strict data separation — assert meta_y not in inference_df; use meta_inference_table.parquet |
| 13 | time_since_prev_event_h wrong (intra-event gap counted) | Fixed: compute on unique event timeline first, merge back |
| 14 | combo_id_hash % 31 arbitrary | Removed; use candidate_id directly as categorical |
| 15 (user 2) | Step 5 expected "6 rows per event" is wrong | Fixed: different CUSUM rates → different event sets; distribution is 1-6 per event |
| 16 (user 2) | event_id includes bar_idx (wrong) | Fixed: event_id = hash(symbol + event_time) only; same event across candidates gets same id |
| 17 (user 2) | ret_1 has two implementations (log + simple) | Fixed: unified to simple pct_change; removed dead ret_at_event code |
| 18 (user 2) | vol_percentile rolling+apply is O(n×window) slow | Fixed: replaced with qcut rank O(n log n); NOTE: qcut uses full series distribution, not rolling window (v1 simplification) |
| 19 (user 2) | CalibratedClassifierCV cv=1 when len<200 invalid | Fixed: n_train<200 → skip calibration entirely (cal_model=None); >=200 → cv=2; >=300 → cv=3 |
| 20 (user 2) | CalibratedClassifierCV ensemble= unspecified | Fixed: explicitly set ensemble=False (single estimator is easier to save/load/explain) |
| 21 (user 2) | categorical_feature in __init__ not fit() | Fixed: use set_params(categorical_feature=...) before fit() |
| 22 (user 3) | `.date()` on numpy.datetime64 throws AttributeError | Fixed: `pd.to_datetime(sorted_times)` + `pd.Timestamp()` wrapper for all `.date()` calls |
| 23 (user 3) | `cal_model` undefined when 0 folds produced | Fixed: initialize `final_model_for_inference = None` before loop; check after loop |
| 24 (user 3) | inference_api_spec.md import path wrong | Fixed: `from strategies.AL9999.unified_meta_model import ...` |
| 25 (user 3) | SHAP example uses undefined `cal_model` | Fixed: `save_shap_summary(final_model_for_inference, ...)` |
| 26 (user 3) | docstring returns `(oof_df, calibrators_per_fold, fold_reports)` mismatch | Fixed: `:returns: (oof_df, fold_reports, final_model_for_inference)` |

### Spec Coverage Checklist

| Spec Section | Task | Status |
|---|---|---|
| Section 2: meta_training_table schema | Task 1 | ✅ |
| Section 3: Candidate identity features | Task 2 | ✅ |
| Section 3: Event-level features | Task 2 | ✅ |
| Section 3: Trend-state features | Task 2 | ✅ |
| Section 3: Regime features | Task 2 | ✅ |
| Section 3: Ensemble features | Task 2 | ✅ |
| Section 4: meta_y = 1(tbm_outcome == primary_side) | Task 1 | ✅ |
| Section 5: Purging/embargo walk-forward | Task 3 | ✅ (corrected val_end logic) |
| Section 5: Walk-forward expanding window | Task 3 | ✅ |
| Section 5: OOS xlsx report with per-fold metrics | Task 4 | ✅ |
| Section 6: Logistic baseline + LightGBM | Task 4 | ✅ |
| Section 6: Sigmoid calibration | Task 4 | ✅ (LGBMClassifier fix) |
| Section 6: SHAP explainability | Task 6 | ✅ (deferred) |
| Section 7: per_event_max_p deduplication | Task 5 | ✅ |
| Section 7: Threshold selection by target precision | Task 5 | ✅ |
| Section 7: Decision output (TAKE/SKIP) | Task 5 | ✅ |
| Section 7: Strict training/inference data separation | Task 5 | ✅ (critical fix) |
| Section 8: Output artifacts | Tasks 1-6 | ✅ |

### Placeholder Scan
No placeholders. All code blocks contain actual implementation details. All file paths are exact. All commands show expected output.

### Type Consistency
- `candidate_id`: int 0-5 (consistent across all tasks)
- `meta_y`: int (0/1) from numpy, stored as int8 equivalent (consistent)
- `p_meta`: float32 probability from LGBMClassifier.predict_proba[:, 1] (consistent)
- `primary_side`: int8 (+1/-1) (consistent)
- `tbm_outcome`: int +1/-1/0 (hit_pt/hit_sl/timeout) (consistent)
- `combo_id`: string like "rate=0.15_fast=5_slow=20_vb=10" (consistent)
- `embargo_delta`: pd.Timedelta (consistent)

### Key Execution Order
1. **Task 1 first** — builds the training/inference tables
2. **Task 2 second** — adds features to both tables
3. **Tasks 3-4 third** — purging helpers + training with LGBMClassifier
4. **Task 5 fourth** — inference with data-separation guarantees
5. **Task 6 last** — SHAP only after everything else is verified

### Critical Test Before Proceeding to Next Task
- Task 1 verify: `df.groupby('event_time').size().describe()` shows 1-6 rows per event (not fixed 6). Same event_time across different candidates should share the same `event_id`.
- Task 2 verify: `df.loc[df['candidate_id']==0, 'ma_gap_pct']` differs from `df.loc[df['candidate_id']==1, 'ma_gap_pct']` even when cusum_rate is the same
- Task 3 verify: `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_purging.py -v` all pass (3 embargo tests updated for val_start-based logic)
- Task 4 verify: xlsx report shows precision_lift > 1.0 on majority of folds

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-07-unified-meta-model.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**


```python
# Option 1: Subagent-driven
from superpowers import subagent
subagent.dispatch("Task 1: Build training/inference tables")

# Option 2: Inline execution
from superpowers import executing_plans
executing_plans.execute_tasks(plan_path="docs/superpowers/plans/2026-04-07-unified-meta-model.md")