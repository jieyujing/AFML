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