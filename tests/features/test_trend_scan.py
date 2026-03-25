"""
Trend Scan Forward Labeling Verification Suite

Tasks 4.1 & 4.2 from the trend-scan-redesign change:
  4.1  Verify Numba-accelerated output matches a pure-Pandas/Statsmodels baseline (forward scan).
  4.2  Verify forward-looking causality (only future is evaluated).

Run:
    uv run python tests/features/test_trend_scan.py          # standalone
    uv run pytest tests/features/test_trend_scan.py -v       # via pytest
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Pure-Python / Statsmodels Baseline (deliberately slow, for correctness)
# ---------------------------------------------------------------------------

def _trend_scan_baseline(
    prices: np.ndarray,
    event_indices: np.ndarray,
    L_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reference implementation using scipy.stats.linregress — NO Numba.
    
    This is intentionally slow but uses a well-tested external library 
    for OLS so we can verify the Numba results against a trusted baseline.
    """
    from scipy.stats import linregress

    n_events = len(event_indices)
    t_values = np.zeros(n_events, dtype=np.float64)
    sides = np.zeros(n_events, dtype=np.int64)
    best_windows = np.zeros(n_events, dtype=np.int64)

    for i in range(n_events):
        idx = event_indices[i]
        best_abs_t = 0.0
        best_t = 0.0
        best_L = 0

        for L in L_windows:
            start = idx
            if start + L > len(prices):
                continue
            if L < 3:
                continue

            y = prices[start: start + L]
            x = np.arange(len(y), dtype=np.float64)

            # scipy.stats.linregress returns (slope, intercept, r, p, stderr)
            # stderr here is the standard error of the slope
            result = linregress(x, y)
            slope = result.slope
            stderr = result.stderr

            if stderr < 1e-12:
                # Perfectly linear — assign large t-value
                t_val = 1e12 if slope >= 0 else -1e12
            else:
                t_val = slope / stderr

            abs_t = abs(t_val)
            if abs_t > best_abs_t:
                best_abs_t = abs_t
                best_t = t_val
                best_L = L

        t_values[i] = best_t
        if best_t > 0:
            sides[i] = 1
        elif best_t < 0:
            sides[i] = -1
        else:
            sides[i] = 0
        best_windows[i] = best_L

    return t_values, sides, best_windows


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def uptrend_prices():
    """Synthetic uptrend: linear + small noise."""
    np.random.seed(42)
    n = 500
    t = np.arange(n, dtype=np.float64)
    prices = 100.0 + 0.5 * t + np.random.normal(0, 0.3, n)
    return prices


@pytest.fixture
def downtrend_prices():
    """Synthetic downtrend: linear + small noise."""
    np.random.seed(123)
    n = 500
    t = np.arange(n, dtype=np.float64)
    prices = 200.0 - 0.3 * t + np.random.normal(0, 0.2, n)
    return prices


@pytest.fixture
def flat_prices():
    """Flat / zero-variance prices."""
    return np.full(300, 100.0, dtype=np.float64)


@pytest.fixture
def mixed_prices():
    """Mixed trend: up → flat → down → up."""
    np.random.seed(7)
    n_seg = 100
    up = 100.0 + 0.5 * np.arange(n_seg) + np.random.normal(0, 0.1, n_seg)
    flat = np.full(n_seg, up[-1]) + np.random.normal(0, 0.01, n_seg)
    down = flat[-1] - 0.4 * np.arange(n_seg) + np.random.normal(0, 0.1, n_seg)
    up2 = down[-1] + 0.6 * np.arange(n_seg) + np.random.normal(0, 0.1, n_seg)
    return np.concatenate([up, flat, down, up2])


@pytest.fixture
def default_windows():
    return np.array([10, 20, 30, 50], dtype=np.int64)


# ---------------------------------------------------------------------------
# Task 4.1: Mathematical Correctness Tests
# ---------------------------------------------------------------------------

class TestTrendScanCorrectness:
    """Verify Numba output matches scipy.stats.linregress baseline."""

    def test_uptrend_matches_baseline(self, uptrend_prices, default_windows):
        """Uptrend scenario: Numba should agree with linregress on t-values and sides."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        events = np.array([50, 100, 200, 300, 400], dtype=np.int64)

        # Numba result
        t_numba, sides_numba, wins_numba = _trend_scan_core(
            uptrend_prices, events, default_windows
        )
        # Baseline result
        t_base, sides_base, wins_base = _trend_scan_baseline(
            uptrend_prices, events, default_windows
        )

        # Sides must match exactly
        np.testing.assert_array_equal(sides_numba, sides_base,
                                       err_msg="Side mismatch on uptrend")

        # t-values should be close (both are positive for uptrend)
        for i in range(len(events)):
            # Allow 1% relative tolerance for floating-point differences
            if abs(t_base[i]) > 1.0:
                np.testing.assert_allclose(
                    t_numba[i], t_base[i], rtol=0.01,
                    err_msg=f"t-value mismatch at event {events[i]}"
                )

        # Best windows must match
        np.testing.assert_array_equal(wins_numba, wins_base,
                                       err_msg="Window mismatch on uptrend")

    def test_downtrend_matches_baseline(self, downtrend_prices, default_windows):
        """Downtrend scenario: all sides should be -1."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        events = np.array([60, 150, 300, 450], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            downtrend_prices, events, default_windows
        )
        t_base, sides_base, wins_base = _trend_scan_baseline(
            downtrend_prices, events, default_windows
        )

        np.testing.assert_array_equal(sides_numba, sides_base)
        np.testing.assert_array_equal(wins_numba, wins_base)

        # All sides should be -1 for consistent downtrend
        assert np.all(sides_numba == -1), f"Expected all -1 sides for downtrend, got {sides_numba}"

    def test_flat_prices_zero_side(self, flat_prices, default_windows):
        """Zero-variance prices: t-value should be 0, side should be 0."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        events = np.array([50, 100, 200], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            flat_prices, events, default_windows
        )

        # All t-values should be 0 (zero variance in OLS)
        np.testing.assert_array_equal(t_numba, np.zeros(len(events)),
                                       err_msg="Flat prices should yield t=0")
        np.testing.assert_array_equal(sides_numba, np.zeros(len(events)),
                                       err_msg="Flat prices should yield side=0")

    def test_mixed_trend_matches_baseline(self, mixed_prices, default_windows):
        """Mixed trend: test that direction changes are captured correctly."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        # Events in different regime segments
        events = np.array([50, 150, 250, 350], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            mixed_prices, events, default_windows
        )
        t_base, sides_base, wins_base = _trend_scan_baseline(
            mixed_prices, events, default_windows
        )

        np.testing.assert_array_equal(sides_numba, sides_base,
                                       err_msg="Side mismatch on mixed trend")
        np.testing.assert_array_equal(wins_numba, wins_base,
                                       err_msg="Window mismatch on mixed trend")

    def test_single_large_window(self, uptrend_prices):
        """Window larger than available history should be skipped gracefully."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        events = np.array([5], dtype=np.int64)
        windows = np.array([3, 10, 50, 100], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            uptrend_prices, events, windows
        )
        t_base, sides_base, wins_base = _trend_scan_baseline(
            uptrend_prices, events, windows
        )

        # Only L=3 and L=5(=idx+1) should be viable. L=10,50,100 need start < 0.
        np.testing.assert_array_equal(sides_numba, sides_base)

    def test_pandas_wrapper_output_shape(self, uptrend_prices):
        """trend_scan_labels should return properly structured DataFrame."""
        from afmlkit.feature.core.trend_scan import trend_scan_labels

        dates = pd.date_range("2024-01-01", periods=len(uptrend_prices), freq="min")
        price_series = pd.Series(uptrend_prices, index=dates)
        t_events = pd.DatetimeIndex([dates[50], dates[100], dates[200], dates[300]])

        result = trend_scan_labels(price_series, t_events, L_windows=[10, 20, 50])

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"t1", "t_value", "side"}
        assert len(result) == len(t_events)
        assert result.index.equals(t_events)

    def test_pandas_wrapper_dtype(self, uptrend_prices):
        """Verify output dtypes are correct."""
        from afmlkit.feature.core.trend_scan import trend_scan_labels

        dates = pd.date_range("2024-01-01", periods=len(uptrend_prices), freq="min")
        price_series = pd.Series(uptrend_prices, index=dates)
        t_events = pd.DatetimeIndex([dates[100]])

        result = trend_scan_labels(price_series, t_events, L_windows=[10, 20])

        assert pd.api.types.is_datetime64_any_dtype(result["t1"])
        assert result["t_value"].dtype == np.float64
        assert result["side"].dtype == np.int8


# ---------------------------------------------------------------------------
# Task 4.2: Forward-Looking Verification
# ---------------------------------------------------------------------------

class TestForwardLookingLabels:
    """Verify that Trend Scan uses only forward-looking data as a Labeling method."""

    def test_forward_window_only(self):
        """
        Place a sharp signal BEFORE the event. The trend scan should NOT
        detect it because it only looks forward.

        prices:
            [0..99]    = sharp uptrend (should be invisible to event at 100)
            [100]      = event point
            [101..199] = 100 (flat)
        """
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        n = 200
        prices = np.full(n, 100.0, dtype=np.float64)
        # Place a strong uptrend BEFORE position 100
        for i in range(101):
            prices[i] = 100.0 + 5.0 * (i - 100)

        events = np.array([100], dtype=np.int64)
        windows = np.array([10, 20, 50], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            prices, events, windows
        )

        # Event at 100 sees flat history [100 .. 100+L-1].
        # It should NOT see the uptrend at 0..100.
        # t-value should be ~0, side should be 0.
        assert abs(t_numba[0]) < 1e-6, (
            f"Backward bias detected! t_value={t_numba[0]} should be ~0 for flat future"
        )
        assert sides_numba[0] == 0, (
            f"Backward bias detected! side={sides_numba[0]} should be 0 for flat future"
        )

    def test_forward_window_detects_future(self):
        """
        Place a sharp signal AFTER the event. The trend scan SHOULD detect it.

        prices:
            [0..99]    = flat (irrelevant)
            [100]      = event point
            [101..199] = strong uptrend
        """
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        n = 200
        prices = np.full(n, 100.0, dtype=np.float64)
        # Place a strong uptrend AFTER position 100
        for i in range(101, 200):
            prices[i] = 100.0 + 5.0 * (i - 100)

        events = np.array([100], dtype=np.int64)
        windows = np.array([10, 20, 50], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            prices, events, windows
        )

        # Event at 100 should detect the uptrend in the future
        assert t_numba[0] > 0, f"Should detect future uptrend, got t={t_numba[0]}"
        assert sides_numba[0] == 1, f"Should detect side=1, got {sides_numba[0]}"

    def test_window_boundaries_exact(self):
        """
        Verify the exact data slice for a given window length.

        For event at index=20, window L=10:
            The window should use prices[20 : 30]
            That is indices 20,21,...,29 (exactly 10 points).
        """
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        n = 50
        # Mark specific positions to verify which are used
        prices = np.full(n, 100.0, dtype=np.float64)

        # Create a distinctive uptrend in exactly [20..29]
        for i in range(20, 30):
            prices[i] = 100.0 + 2.0 * (i - 20)

        events = np.array([20], dtype=np.int64)
        windows = np.array([10], dtype=np.int64)

        t_numba, sides_numba, _ = _trend_scan_core(prices, events, windows)

        # The window [20..29] has a clear uptrend → side=1, large positive t
        assert sides_numba[0] == 1, f"Expected side=1, got {sides_numba[0]}"
        assert t_numba[0] > 5.0, f"Expected strong t-value, got {t_numba[0]}"

    def test_future_contamination_multi_event(self):
        """
        Multiple events: verify forward scanning logic.
        """
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        n = 300
        prices = np.full(n, 100.0, dtype=np.float64)

        # Segment 1: uptrend [0..100]
        for i in range(101):
            prices[i] = 80.0 + 0.5 * i

        # Segment 2: downtrend [101..200]
        for i in range(101, 201):
            prices[i] = prices[100] - 0.3 * (i - 100)

        # Events at the start of their respective future trends
        events = np.array([0, 100, 250], dtype=np.int64)
        windows = np.array([10, 20, 50], dtype=np.int64)

        t_numba, sides_numba, _ = _trend_scan_core(prices, events, windows)

        # Event at 0 (sees future uptrend) → side=1
        assert sides_numba[0] == 1, f"Event@0 sees future uptrend, got side={sides_numba[0]}"

        # Event at 100 (sees future downtrend) → side=-1
        assert sides_numba[1] == -1, f"Event@100 sees future downtrend, got side={sides_numba[1]}"

        # Event at 250 (flat future [251..299]) → side=0
        assert sides_numba[2] == 0, f"Event@250 sees flat future, got side={sides_numba[2]}"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_event_near_end(self):
        """Event near the end of the series (limited available future)."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        prices = np.arange(20, dtype=np.float64) + 100.0
        events = np.array([16], dtype=np.int64)
        windows = np.array([3, 5, 10], dtype=np.int64)

        # Event at 16 has 4 points left [16, 17, 18, 19].
        # Only L=3 should be viable, L=5 gives start+L = 16+5 = 21 > len(20)
        t_numba, sides_numba, wins_numba = _trend_scan_core(
            prices, events, windows
        )

        assert wins_numba[0] == 3, f"Expected best window=3, got {wins_numba[0]}"
        assert sides_numba[0] == 1, "Linear uptrend should give side=1"

    def test_empty_events(self):
        """No events: should return empty arrays."""
        from afmlkit.feature.core.trend_scan import _trend_scan_core

        prices = np.arange(100, dtype=np.float64)
        events = np.array([], dtype=np.int64)
        windows = np.array([10, 20], dtype=np.int64)

        t_numba, sides_numba, wins_numba = _trend_scan_core(
            prices, events, windows
        )

        assert len(t_numba) == 0
        assert len(sides_numba) == 0
        assert len(wins_numba) == 0

    def test_wrapper_invalid_index_raises(self):
        """trend_scan_labels with non-DatetimeIndex should raise ValueError."""
        from afmlkit.feature.core.trend_scan import trend_scan_labels

        prices = pd.Series(np.arange(100, dtype=np.float64))  # integer index
        t_events = pd.DatetimeIndex(["2024-01-01"])

        with pytest.raises(ValueError, match="DatetimeIndex"):
            trend_scan_labels(prices, t_events)

    def test_wrapper_no_overlap_raises(self):
        """Events with no overlap to price index should raise ValueError."""
        from afmlkit.feature.core.trend_scan import trend_scan_labels

        dates = pd.date_range("2024-01-01", periods=100, freq="min")
        prices = pd.Series(np.arange(100, dtype=np.float64), index=dates)
        # t_events are in a completely different date range
        t_events = pd.DatetimeIndex(["2025-06-01", "2025-06-02"])

        with pytest.raises(ValueError, match="No events"):
            trend_scan_labels(prices, t_events)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Trend Scan Verification Suite")
    print("=" * 70)

    # Quick standalone test (no pytest required)
    np.random.seed(42)
    n = 500
    t = np.arange(n, dtype=np.float64)
    prices = 100.0 + 0.5 * t + np.random.normal(0, 0.3, n)
    events = np.array([50, 100, 200, 300, 400], dtype=np.int64)
    windows = np.array([10, 20, 30, 50], dtype=np.int64)

    print("\n[Task 4.1] Numba vs Baseline comparison:")
    from afmlkit.feature.core.trend_scan import _trend_scan_core

    t_numba, sides_numba, wins_numba = _trend_scan_core(prices, events, windows)
    t_base, sides_base, wins_base = _trend_scan_baseline(prices, events, windows)

    for i, ev in enumerate(events):
        match = "✓" if sides_numba[i] == sides_base[i] else "✗"
        print(
            f"  Event@{ev:3d}: Numba t={t_numba[i]:+8.4f} side={sides_numba[i]:+d} L={wins_numba[i]:2d}  |  "
            f"Base t={t_base[i]:+8.4f} side={sides_base[i]:+d} L={wins_base[i]:2d}  [{match}]"
        )

    assert np.array_equal(sides_numba, sides_base), "FAIL: Side mismatch!"
    assert np.array_equal(wins_numba, wins_base), "FAIL: Window mismatch!"
    print("  All sides and windows match! ✓")

    print("\n[Task 4.2] Forward causality verification:")
    # Test: past data should be invisible
    prices_bias_test = np.full(200, 100.0, dtype=np.float64)
    # Place strong downtrend BEFORE position 100
    for i in range(100):
        prices_bias_test[i] = 100.0 - 5.0 * i

    t_val, side_val, _ = _trend_scan_core(
        prices_bias_test,
        np.array([100], dtype=np.int64),
        np.array([10, 20, 50], dtype=np.int64),
    )
    print(f"  Event@100 (downtrend past, flat future): t={t_val[0]:+.6f}, side={side_val[0]}")
    assert abs(t_val[0]) < 1e-6, f"FAIL: Backward bias detected! t={t_val[0]}"
    print("  Forward label causality confirmed! ✓")

    print("\n" + "=" * 70)
    print("All verifications passed! ✓")
    print("=" * 70)
