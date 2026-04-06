"""Tests for CUSUM calibrator module."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load the module using importlib to avoid packaging issues
_module_path = (
    PROJECT_ROOT
    / "strategies"
    / "AL9999"
    / "primary_factory"
    / "cusum_calibrator.py"
)
_spec = importlib.util.spec_from_file_location("cusum_calibrator", _module_path)
_cusum_calibrator_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cusum_calibrator_module)
calibrate_cusum_rates = _cusum_calibrator_module.calibrate_cusum_rates


def test_calibrate_cusum_rates():
    """Test CUSUM rate calibration with synthetic data."""
    # Generate synthetic bars: 1000 rows, close = cumprod(randn+1)
    np.random.seed(42)
    close = pd.Series(np.cumprod(1 + np.random.randn(1000) * 0.01))
    close.index = pd.date_range('2020-01-01', periods=1000, freq='h')
    bars = pd.DataFrame({'close': close})

    # Use k_min=0.001 since synthetic data needs lower threshold to achieve 5%/10% rates
    result = calibrate_cusum_rates(bars, target_rates=[0.05, 0.10], k_min=0.001, tol=1e-3)

    assert len(result) == 2
    assert result['rate'].tolist() == [0.05, 0.10]
    assert all(result['k'] > 0)
    assert all(result['actual_rate'] > 0)
    # Verify calibration accuracy
    assert all(np.abs(result['actual_rate'] - result['rate']) < 1e-3)


def test_calibrate_cusum_rates_single_rate():
    """Test with a single target rate."""
    np.random.seed(42)
    close = pd.Series(np.cumprod(1 + np.random.randn(500) * 0.01))
    close.index = pd.date_range('2020-01-01', periods=500, freq='h')
    bars = pd.DataFrame({'close': close})

    result = calibrate_cusum_rates(bars, target_rates=[0.15], k_min=0.001, tol=1e-3)

    assert len(result) == 1
    assert result['rate'].iloc[0] == 0.15
    assert result['k'].iloc[0] > 0
    assert result['actual_rate'].iloc[0] > 0
    assert np.abs(result['actual_rate'].iloc[0] - 0.15) < 1e-3


def test_calibrate_cusum_rates_output_columns():
    """Test that output DataFrame has correct columns."""
    np.random.seed(42)
    close = pd.Series(np.cumprod(1 + np.random.randn(500) * 0.01))
    close.index = pd.date_range('2020-01-01', periods=500, freq='h')
    bars = pd.DataFrame({'close': close})

    result = calibrate_cusum_rates(bars, target_rates=[0.05, 0.10, 0.15], k_min=0.001)

    expected_columns = ['rate', 'k', 'actual_rate', 'n_events']
    assert list(result.columns) == expected_columns
    assert result['n_events'].min() > 0  # At least some events detected