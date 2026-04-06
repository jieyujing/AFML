"""
tests/strategies/test_al9999_tsfresh_feature_engineering.py

Unit tests for tsfresh-style feature engineering module.
Tests transform functions, feature naming, and feature extraction.
"""
import sys
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest


def _load_module():
    """Load the 02b_tsfresh_feature_engineering module using importlib."""
    # Add project root to sys.path so module can import from strategies
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    module_path = (
        project_root
        / "strategies"
        / "AL9999"
        / "02b_tsfresh_feature_engineering.py"
    )
    spec = importlib.util.spec_from_file_location("al9999_tsfresh_feature_engineering", module_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load module once at module level
module = _load_module()

apply_raw = module.apply_raw
apply_pct_change = module.apply_pct_change
apply_fracdiff = module.apply_fracdiff
apply_zscore = module.apply_zscore
build_feature_name = module.build_feature_name
extract_features_from_slice = module.extract_features_from_slice
TSFRESH_FEATURE_FUNCS = module.TSFRESH_FEATURE_FUNCS


class TestApplyRaw:
    """Test apply_raw transform."""

    def test_apply_raw_returns_same_series(self):
        """apply_raw should return the series unchanged."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_raw(s)
        np.testing.assert_array_equal(result.values, s.values)

    def test_apply_raw_with_nan(self):
        """apply_raw should preserve NaN values."""
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        result = apply_raw(s)
        np.testing.assert_array_equal(result.values, s.values)


class TestApplyPctChange:
    """Test apply_pct_change transform."""

    def test_apply_pct_change_basic(self):
        """Test basic percentage change calculation."""
        s = pd.Series([100.0, 101.0, 103.0, 102.0, 105.0])
        result = apply_pct_change(s)
        expected = s.pct_change().dropna().values
        np.testing.assert_allclose(result.dropna().values, expected, rtol=1e-3)

    def test_apply_pct_change_first_is_nan(self):
        """First element should be NaN after pct_change."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_pct_change(s)
        assert np.isnan(result.iloc[0])

    def test_apply_pct_change_periods(self):
        """Test with custom periods."""
        s = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0])
        result = apply_pct_change(s, periods=2)
        expected = s.pct_change(2).dropna().values
        np.testing.assert_allclose(result.dropna().values, expected, rtol=1e-3)


class TestApplyFracdiff:
    """Test apply_fracdiff transform."""

    def test_apply_fracdiff_d0_returns_original(self):
        """When d=0, should return original series."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_fracdiff(s, d=0.0)
        np.testing.assert_allclose(result.values, s.values, rtol=1e-5)

    def test_apply_fracdiff_basic(self):
        """Test fractional differentiation with positive d."""
        # Need sufficiently long series for frac_diff_ffd weights
        # For d=0.1, thres=1e-4, weights are ~500 elements
        np.random.seed(42)
        s = pd.Series(np.cumsum(np.random.randn(1000) * 0.01 + 0.001))
        result = apply_fracdiff(s, d=0.1, thres=1e-4)
        # frac_diff_ffd uses convolution with mode='valid', so result is shorter
        assert len(result) < len(s)
        assert not np.all(np.isnan(result))

    def test_apply_fracdiff_auto_d(self):
        """Test auto d optimization (d=None)."""
        s = pd.Series(np.random.randn(50) * 0.01 + 0.001)
        result = apply_fracdiff(s, d=None, thres=1e-4)
        assert len(result) == len(s)


class TestApplyZscore:
    """Test apply_zscore transform."""

    def test_apply_zscore_basic(self):
        """Test basic z-score normalization."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_zscore(s, window=3)
        # At index 2: rolling mean=(1+2+3)/3=2, rolling_std > 0
        # So (3-2)/std should be approximately 1
        assert abs(result.iloc[2]) < 1e-10 or abs(result.iloc[2] - 1.0) < 0.1

    def test_apply_zscore_nan_handling(self):
        """First window-1 elements should be NaN."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_zscore(s, window=3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])

    def test_apply_zscore_valid_values(self):
        """Values within window should be normalized."""
        s = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])  # Constant series
        result = apply_zscore(s, window=3)
        # Constant series should have NaN due to 0 std
        assert np.isnan(result.iloc[2:]).any() or np.all(np.isnan(result))


class TestBuildFeatureName:
    """Test build_feature_name function."""

    def test_build_feature_name_raw(self):
        """Test raw transform naming."""
        assert build_feature_name("close", "raw", "mean") == "feat_close_raw_mean"
        assert build_feature_name("volume", "raw", "skewness") == "feat_volume_raw_skewness"

    def test_build_feature_name_pct_change(self):
        """Test pct_change transform naming."""
        assert build_feature_name("close", "pct_change", "mean") == "feat_close_pct_mean"
        assert build_feature_name("volume", "pct_change", "skewness") == "feat_volume_pct_skewness"

    def test_build_feature_name_fracdiff(self):
        """Test fracdiff transform naming."""
        assert build_feature_name("close", "fracdiff", "mean") == "feat_close_fd_mean"
        assert build_feature_name("log_close", "fracdiff", "kurtosis") == "feat_log_close_fd_kurtosis"

    def test_build_feature_name_zscore(self):
        """Test zscore transform naming with window."""
        assert build_feature_name("close", "zscore", "mean", window=10) == "feat_close_z10_mean"
        assert build_feature_name("close", "zscore", "mean", window=20) == "feat_close_z20_mean"
        assert build_feature_name("volume", "zscore", "std", window=40) == "feat_volume_z40_std"


class TestExtractFeaturesFromSlice:
    """Test extract_features_from_slice function."""

    def test_extract_features_minimal_input(self):
        """Test with less than 3 values returns all NaN."""
        slice_values = np.array([1.0, 2.0])
        func_names = ["mean", "skewness"]
        result = extract_features_from_slice(slice_values, func_names)
        assert np.isnan(result["mean"])
        assert np.isnan(result["skewness"])

    def test_extract_features_mean(self):
        """Test mean feature calculation."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_features_from_slice(slice_values, ["mean"])
        assert result["mean"] == pytest.approx(3.0)

    def test_extract_features_median(self):
        """Test median feature calculation."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_features_from_slice(slice_values, ["median"])
        assert result["median"] == pytest.approx(3.0)

    def test_extract_features_standard_deviation(self):
        """Test standard deviation calculation (ddof=1)."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_features_from_slice(slice_values, ["standard_deviation"])
        # std([1,2,3,4,5], ddof=1) ≈ 1.581
        assert result["standard_deviation"] == pytest.approx(1.58113883, abs=1e-5)

    def test_extract_features_skewness(self):
        """Test skewness calculation."""
        # Symmetric distribution has skewness ≈ 0
        slice_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = extract_features_from_slice(slice_values, ["skewness"])
        assert abs(result["skewness"]) < 0.01

    def test_extract_features_kurtosis(self):
        """Test kurtosis calculation."""
        # Normal distribution has kurtosis ≈ 0 (excess)
        np.random.seed(42)
        slice_values = np.random.randn(100)
        result = extract_features_from_slice(slice_values, ["kurtosis"])
        assert abs(result["kurtosis"]) < 1.0

    def test_extract_features_minimum(self):
        """Test minimum feature."""
        slice_values = np.array([5.0, 2.0, 8.0, 1.0, 3.0])
        result = extract_features_from_slice(slice_values, ["minimum"])
        assert result["minimum"] == pytest.approx(1.0)

    def test_extract_features_maximum(self):
        """Test maximum feature."""
        slice_values = np.array([5.0, 2.0, 8.0, 1.0, 3.0])
        result = extract_features_from_slice(slice_values, ["maximum"])
        assert result["maximum"] == pytest.approx(8.0)

    def test_extract_features_abs_energy(self):
        """Test absolute energy (sum of squares)."""
        slice_values = np.array([1.0, 2.0, 3.0])
        result = extract_features_from_slice(slice_values, ["abs_energy"])
        # 1^2 + 2^2 + 3^2 = 14
        assert result["abs_energy"] == pytest.approx(14.0)

    def test_extract_features_mean_change(self):
        """Test mean change (average of diff)."""
        slice_values = np.array([1.0, 3.0, 6.0, 10.0])
        result = extract_features_from_slice(slice_values, ["mean_change"])
        # diff = [2, 3, 4], mean = 3
        assert result["mean_change"] == pytest.approx(3.0)

    def test_extract_features_mean_abs_change(self):
        """Test mean absolute change."""
        slice_values = np.array([1.0, 3.0, 2.0, 5.0])
        result = extract_features_from_slice(slice_values, ["mean_abs_change"])
        # diff = [2, -1, 3], abs = [2, 1, 3], mean = 2
        assert result["mean_abs_change"] == pytest.approx(2.0)

    def test_extract_features_count_above_mean(self):
        """Test count above mean."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_features_from_slice(slice_values, ["count_above_mean"])
        # mean = 3.0, values above: 4.0, 5.0 → count = 2
        assert result["count_above_mean"] == pytest.approx(2.0)

    def test_extract_features_count_below_mean(self):
        """Test count below mean."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = extract_features_from_slice(slice_values, ["count_below_mean"])
        # mean = 3.0, values below: 1.0, 2.0 → count = 2
        assert result["count_below_mean"] == pytest.approx(2.0)

    def test_extract_features_first_location_of_maximum(self):
        """Test first location of maximum (normalized)."""
        slice_values = np.array([1.0, 5.0, 5.0, 3.0])  # max=5.0 at idx 1 (first)
        result = extract_features_from_slice(slice_values, ["first_location_of_maximum"])
        # First max at index 1, normalized: 1/4 = 0.25
        assert result["first_location_of_maximum"] == pytest.approx(0.25)

    def test_extract_features_first_location_of_minimum(self):
        """Test first location of minimum (normalized)."""
        slice_values = np.array([3.0, 1.0, 1.0, 5.0])  # min=1.0 at idx 1 (first)
        result = extract_features_from_slice(slice_values, ["first_location_of_minimum"])
        # First min at index 1, normalized: 1/4 = 0.25
        assert result["first_location_of_minimum"] == pytest.approx(0.25)

    def test_extract_features_last_location_of_maximum(self):
        """Test last location of maximum (normalized)."""
        slice_values = np.array([1.0, 5.0, 5.0, 5.0])  # max=5.0 at idx 3 (last)
        result = extract_features_from_slice(slice_values, ["last_location_of_maximum"])
        # Last max at index 3, normalized: 3/4 = 0.75
        assert result["last_location_of_maximum"] == pytest.approx(0.75)

    def test_extract_features_last_location_of_minimum(self):
        """Test last location of minimum (normalized)."""
        slice_values = np.array([1.0, 1.0, 3.0, 5.0])  # min=1.0 at idx 1 (last)
        result = extract_features_from_slice(slice_values, ["last_location_of_minimum"])
        # Last min at index 1, normalized: 1/4 = 0.25
        assert result["last_location_of_minimum"] == pytest.approx(0.25)

    def test_extract_features_all_16(self):
        """Test all 16 features together."""
        slice_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        func_names = list(TSFRESH_FEATURE_FUNCS.keys())
        result = extract_features_from_slice(slice_values, func_names)
        # All 16 features should be present
        assert len(result) == 16
        # Check mean is correct
        assert result["mean"] == pytest.approx(5.5)


class TestTSFRESHFeatureFuncs:
    """Test that all 16 feature functions are accessible via TSFRESH_FEATURE_FUNCS."""

    def test_all_16_features_present(self):
        """Verify all 16 features are in TSFRESH_FEATURE_FUNCS."""
        expected_features = [
            "mean", "median", "standard_deviation", "skewness", "kurtosis",
            "minimum", "maximum", "abs_energy", "mean_change", "mean_abs_change",
            "count_above_mean", "count_below_mean",
            "first_location_of_maximum", "first_location_of_minimum",
            "last_location_of_maximum", "last_location_of_minimum",
        ]
        for feat in expected_features:
            assert feat in TSFRESH_FEATURE_FUNCS, f"Missing feature: {feat}"

    def test_feature_functions_callable(self):
        """Verify all feature functions are callable."""
        test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        for name, func in TSFRESH_FEATURE_FUNCS.items():
            result = func(test_array)
            assert np.isfinite(result) or np.isnan(result), f"Feature {name} returned invalid value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])