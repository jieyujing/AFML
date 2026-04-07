# tests/strategies/test_purging.py
import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

# Load the module dynamically
_spec = importlib.util.spec_from_file_location(
    "unified_meta_model",
    Path("strategies/AL9999/unified_meta_model.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

purge_by_label_overlap = _mod.purge_by_label_overlap
embargo_after = _mod.embargo_after


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
    embargoed = embargo_after(train, val_start, embargo)
    assert embargoed['value'].tolist() == [1, 2, 3]