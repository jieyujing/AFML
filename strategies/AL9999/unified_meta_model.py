"""
unified_meta_model.py - Unified meta model training + inference.

A single LightGBM classifier trained on an event-candidate panel where each row = (event_time, candidate).

Key features:
- Purging/embargo walk-forward validation prevents label horizon overlap leakage
- LGBMClassifier with sigmoid calibration
- Event-level deduplication at inference
"""
import os
import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    # Quick test
    print("Testing purging and embargo functions...")
    train = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-20', '2020-02-05', '2020-03-01']),
        'label_end_time': pd.to_datetime(['2020-01-05', '2020-01-25', '2020-02-15', '2020-03-01', '2020-04-01']),
        'value': [1, 2, 3, 4, 5],
    })
    val_start = pd.Timestamp('2020-02-01')
    val_end = pd.Timestamp('2020-03-01')

    purged = purge_by_label_overlap(train, val_start, val_end)
    print(f"Purged: {len(purged)} rows (expected 2)")
    assert len(purged) == 2

    embargoed = embargo_after(train, val_start, pd.Timedelta(days=20))
    print(f"Embargoed: {len(embargoed)} rows (expected 2)")
    assert len(embargoed) == 2

    print("All tests passed!")