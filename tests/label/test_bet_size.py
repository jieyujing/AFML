import numpy as np
import pandas as pd
import pytest
from afmlkit.label.bet_size import get_signal_size, discretize_size, get_size_change_signals

def test_get_signal_size_zero_division():
    # Test boundary limits to prevent ZeroDivisionError
    size_p1 = get_signal_size(1.0, 1)
    size_p0 = get_signal_size(0.0, 1)
    
    assert not np.isnan(size_p1)
    assert not np.isinf(size_p1)
    # The normal CDF of a huge number will be basically 1.
    assert np.isclose(size_p1, 1.0, atol=1e-3)
    
    assert not np.isnan(size_p0)
    assert not np.isinf(size_p0)
    assert np.isclose(size_p0, -1.0, atol=1e-3)

def test_get_signal_size_mapping_properties():
    # Neutral probability should result in zero size
    assert np.isclose(get_signal_size(0.5, 1), 0.0)
    
    # Check side application
    p = 0.8
    size_long = get_signal_size(p, 1)
    size_short = get_signal_size(p, -1)
    assert size_long > 0
    assert size_short < 0
    assert np.isclose(size_long, -size_short)

def test_discretize_size_logic():
    # step_size = 0.1
    # 0.04 -> 0.0
    # 0.06 -> 0.1
    # 0.14 -> 0.1
    # 0.16 -> 0.2
    assert np.isclose(discretize_size(0.04, 0.1), 0.0)
    assert np.isclose(discretize_size(0.06, 0.1), 0.1)
    assert np.isclose(discretize_size(0.14, 0.1), 0.1)
    assert np.isclose(discretize_size(0.16, 0.1), 0.2)
    assert np.isclose(discretize_size(-0.04, 0.1), 0.0)
    assert np.isclose(discretize_size(-0.06, 0.1), -0.1)

def test_get_size_change_signals():
    sizes = pd.Series([0.0, 0.0, 0.1, 0.1, 0.2, 0.1, 0.1])
    changes = get_size_change_signals(sizes)
    # Start at 0, no prev size, but shift(1).fillna(0) means 0 vs 0 -> false
    assert changes.iloc[0] == False
    assert changes.iloc[1] == False
    assert changes.iloc[2] == True  # 0.0 to 0.1
    assert changes.iloc[3] == False # 0.1 to 0.1
    assert changes.iloc[4] == True  # 0.1 to 0.2
    assert changes.iloc[5] == True  # 0.2 to 0.1
    assert changes.iloc[6] == False # 0.1 to 0.1

    # Should raise ValueError if step_size <= 0
    with pytest.raises(ValueError):
        discretize_size(0.5, 0.0)

def test_get_concurrent_sizes():
    from afmlkit.label.bet_size import get_concurrent_sizes
    # t0: signal A starts, size = 1.0, ends t2
    # t1: signal B starts, size = 0.5, ends t3
    # t2: signal C starts, size = 0.0, ends t4
    
    # Grid: t0, t1, t2, t3, t4
    times = pd.date_range("2023-01-01 10:00", periods=5, freq="h")
    t0, t1, t2, t3, t4 = times
    
    t_events = pd.DatetimeIndex([t0, t1, t2])
    t_exits = pd.Series([t2, t3, t4])
    sizes = np.array([1.0, 0.5, 0.0])
    
    avg_sizes = get_concurrent_sizes(sizes, t_events, t_exits)
    
    # Check times index matches Grid
    assert len(avg_sizes) == 5
    
    # At t0: A is active -> sum=1.0, count=1 -> 1.0
    assert np.isclose(avg_sizes.loc[t0], 1.0)
    
    # At t1: A, B active -> sum=1.5, count=2 -> 0.75
    assert np.isclose(avg_sizes.loc[t1], 0.75)
    
    # At t2: A exits, C enters. B, C active -> sum=0.5, count=2 -> 0.25
    assert np.isclose(avg_sizes.loc[t2], 0.25)
    
    # At t3: B exits. C active -> sum=0.0, count=1 -> 0.0
    assert np.isclose(avg_sizes.loc[t3], 0.0)
    
    # At t4: C exits. None active -> 0.0
    assert np.isclose(avg_sizes.loc[t4], 0.0)
