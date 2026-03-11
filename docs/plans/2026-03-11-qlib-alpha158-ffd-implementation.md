# Qlib Alpha158 FFD Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Implement Phase 1 of Qlib Alpha158 feature engineering with FFD (Fractional Differentiation) transformation, creating a new `alpha158_features.py` module that produces `ffd_*` prefixed features alongside existing features.

**Architecture:** Modular function design with `compute_ffd_base()` as the foundation, then separate functions for volatility, MA, rank, and volume features. All functions accept FFD-transformed series as input. Integration via optional `alpha158` config flag in existing `feature_calculator.py`.

**Tech Stack:** Python, pandas, numpy, statsmodels (ADF test), afmlkit.feature.core.frac_diff, Streamlit (UI)

---

## Task 1: Verify afmlkit Core Imports

**Files:**
- Verify: `afmlkit/feature/core/frac_diff.py`
- Verify: `afmlkit/feature/core/ma.py`
- Verify: `afmlkit/feature/core/volatility.py`

**Step 1: Test all required imports**

```bash
cd D:\PycharmProjects\AFMLKIT
uv run python -c "
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d, get_weights_ffd
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.volatility import ewms
import pandas as pd
import numpy as np
print('All imports successful')
"
```

**Step 2: Record any missing imports**

If any import fails, note the error and check the actual export structure in `afmlkit/feature/core/__init__.py`.

**Step 3: Verify frac_diff_ffd signature**

```bash
uv run python -c "
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
import inspect
print('frac_diff_ffd signature:', inspect.signature(frac_diff_ffd))
print('optimize_d signature:', inspect.signature(optimize_d))
"
```

Expected output:
```
frac_diff_ffd signature: (series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series
optimize_d signature: (series: pd.Series, thres: float = 1e-4, d_step: float = 0.05, max_d: float = 1.0) -> float
```

**Step 4: Commit (if any verification code was added)**

This is primarily a verification task. Only commit if you added helper code.

---

## Task 2: Create alpha158_features Module Skeleton

**Files:**
- Create: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Create empty module with docstring**

```python
# webapp/utils/alpha158_features.py
"""
Alpha158 Features - FFD-transformed feature engineering.

This module implements Qlib Alpha158-style features using Fractional Differentiation
(FFD) to create stationary, memory-preserving feature series.

Key design principles:
1. Apply FFD only to log_price to get stationary base series X_tilde
2. Use X_tilde directly for momentum features (no returns calculation)
3. Feed X_tilde into SMA/EMA/Rank operations (they don't increase differencing order)
4. Volume features use raw data (volume is already stationary)

All features use 'ffd_*' prefix for clear identification.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any

from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d
from afmlkit.feature.core.ma import ewma, sma
from afmlkit.feature.core.volatility import ewms


# Default configuration
DEFAULT_VOLATILITY_SPANS = [5, 10, 20]
DEFAULT_MA_WINDOWS = [5, 10, 20]
DEFAULT_EMA_WINDOWS = [5, 10]
DEFAULT_RANK_WINDOW = 20
DEFAULT_FFD_THRES = 1e-4
DEFAULT_FFD_D_STEP = 0.05
```

**Step 2: Run linting to verify module structure**

```bash
uv run ruff check webapp/utils/alpha158_features.py
```

Expected: No errors (only unused imports warning, which is fine for now)

**Step 3: Commit**

```bash
git add webapp/utils/alpha158_features.py
git commit -m "feat: add alpha158_features module skeleton"
```

---

## Task 3: Implement compute_ffd_base Function

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
import numpy as np
import pandas as np
import pytest
from webapp.utils.alpha158_features import compute_ffd_base


def test_compute_ffd_base_basic():
    """Test basic FFD base series computation."""
    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Check return types
    assert isinstance(ffd_series, pd.Series)
    assert isinstance(optimal_d, float)

    # Check optimal_d is in valid range
    assert 0.0 <= optimal_d <= 1.0

    # Check series length (should be <= input due to FFD lag)
    assert len(ffd_series) <= len(close)
    assert len(ffd_series) > 0

    # Check no NaN in result
    assert ffd_series.isna().sum() == 0


def test_compute_ffd_base_already_stationary():
    """Test FFD returns d=0 for already stationary series."""
    np.random.seed(42)
    stationary = pd.Series(np.random.randn(200))

    ffd_series, optimal_d = compute_ffd_base(stationary)

    # Should return d=0 or very small
    assert optimal_d < 0.1


def test_compute_ffd_base_custom_params():
    """Test FFD with custom threshold and d_step."""
    np.random.seed(42)
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100)

    ffd_series, optimal_d = compute_ffd_base(
        close,
        thres=1e-5,
        d_step=0.1
    )

    assert isinstance(ffd_series, pd.Series)
    assert 0.0 <= optimal_d <= 1.0


def test_compute_ffd_base_preserves_index():
    """Test FFD preserves datetime index."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='T')
    close = pd.Series(np.cumsum(np.random.randn(200)) + 100, index=dates)

    ffd_series, optimal_d = compute_ffd_base(close)

    # Index should be preserved (though first few rows may be dropped)
    assert isinstance(ffd_series.index, pd.DatetimeIndex)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_base_basic -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'webapp.utils.alpha158_features'" or "ImportError: cannot import name 'compute_ffd_base'"

**Step 3: Implement compute_ffd_base function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_ffd_base(
    close: pd.Series,
    thres: float = DEFAULT_FFD_THRES,
    d_step: float = DEFAULT_FFD_D_STEP
) -> Tuple[pd.Series, float]:
    """
    Compute FFD base series and return optimal d*.

    Applies fractional differentiation to log(close) using the Fixed-Width Window
    method. Automatically searches for the minimum d that makes the series stationary.

    Args:
        close: Closing price series (can be raw or log-transformed)
        thres: FFD weight truncation threshold (default: 1e-4)
        d_step: Step size for d search (default: 0.05)

    Returns:
        tuple: (ffd_log_price Series, optimal d value)

    Example:
        >>> close = pd.Series([100, 101, 102, ...])
        >>> ffd_series, d_star = compute_ffd_base(close)
    """
    # Ensure input is float64 for numerical stability
    close = close.astype(np.float64)

    # Compute log price if close is in raw form (heuristic: all values > 0)
    if close.min() > 0:
        log_price = np.log(close)
    else:
        log_price = close.copy()

    # Remove any existing NaN before FFD
    log_price = log_price.dropna()

    # Find optimal d* using ADF test
    optimal_d = optimize_d(log_price, thres=thres, d_step=d_step)

    # Apply FFD with optimal d
    log_price_named = pd.Series(
        log_price.values,
        index=log_price.index,
        name="log_close"
    )
    ffd_series = frac_diff_ffd(log_price_named, d=optimal_d, thres=thres)

    # Rename for clarity
    ffd_series.name = "ffd_log_price"

    return ffd_series, optimal_d
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_base_basic -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_base_already_stationary -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_base_custom_params -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_base_preserves_index -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_ffd_base with optimal d search"
```

---

## Task 4: Implement compute_ffd_volatility Function

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
from webapp.utils.alpha158_features import compute_ffd_volatility


def test_compute_ffd_volatility_basic():
    """Test FFD volatility feature computation."""
    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100), name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5, 10, 20])

    # Check all expected columns exist
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' in result.columns
    assert 'ffd_vol_ewm_5' in result.columns
    assert 'ffd_vol_ewm_10' in result.columns
    assert 'ffd_vol_ewm_20' in result.columns

    # Check data types
    assert all(result.dtypes == np.float64)


def test_compute_ffd_volatility_custom_spans():
    """Test FFD volatility with custom spans."""
    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100))

    result = compute_ffd_volatility(ffd_series, spans=[3, 7])

    assert 'ffd_vol_std_3' in result.columns
    assert 'ffd_vol_std_7' in result.columns
    assert 'ffd_vol_ewm_3' in result.columns
    assert 'ffd_vol_ewm_7' in result.columns


def test_compute_ffd_volatility_preserves_index():
    """Test FFD volatility preserves datetime index."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='T')
    ffd_series = pd.Series(np.random.randn(100), index=dates, name="ffd_log_price")

    result = compute_ffd_volatility(ffd_series, spans=[5])

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 100
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_volatility_basic -v
```

Expected: FAIL with "ImportError: cannot import name 'compute_ffd_volatility'"

**Step 3: Implement compute_ffd_volatility function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_ffd_volatility(
    ffd_series: pd.Series,
    spans: List[int] = None
) -> pd.DataFrame:
    """
    Compute volatility features based on FFD series.

    Calculates:
    - Rolling standard deviation (ffd_vol_std_{span})
    - EWM volatility (ffd_vol_ewm_{span})

    Args:
        ffd_series: FFD-transformed series (typically ffd_log_price)
        spans: List of span values for volatility calculation (default: [5, 10, 20])

    Returns:
        DataFrame: Contains ffd_vol_std_* and ffd_vol_ewm_* columns
    """
    if spans is None:
        spans = DEFAULT_VOLATILITY_SPANS

    result = pd.DataFrame(index=ffd_series.index)
    ffd_values = ffd_series.values.astype(np.float64)

    for span in spans:
        # Rolling standard deviation
        std_col = f"ffd_vol_std_{span}"
        result[std_col] = pd.Series(ffd_values, index=ffd_series.index).rolling(window=span).std()

        # EWM volatility (using log returns of FFD series)
        ewm_col = f"ffd_vol_ewm_{span}"
        ewm_vol = ewms(ffd_values, span)
        result[ewm_col] = pd.Series(ewm_vol, index=ffd_series.index)

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_volatility_basic -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_volatility_custom_spans -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_volatility_preserves_index -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_ffd_volatility for rolling and EWM volatility"
```

---

## Task 5: Implement compute_ffd_ma Function

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
from webapp.utils.alpha158_features import compute_ffd_ma


def test_compute_ffd_ma_basic():
    """Test FFD moving average feature computation."""
    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100), name="ffd_log_price")

    result = compute_ffd_ma(ffd_series, windows=[5, 10, 20])

    # Check all expected columns exist
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_ma_10' in result.columns
    assert 'ffd_ma_20' in result.columns
    assert 'ffd_ema_5' in result.columns
    assert 'ffd_ema_10' in result.columns
    assert 'ffd_ema_20' in result.columns


def test_compute_ffd_ma_custom_windows():
    """Test FFD MA with custom windows."""
    np.random.seed(42)
    ffd_series = pd.Series(np.random.randn(100))

    result = compute_ffd_ma(ffd_series, windows=[3, 7])

    assert 'ffd_ma_3' in result.columns
    assert 'ffd_ma_7' in result.columns
    assert 'ffd_ema_3' in result.columns
    assert 'ffd_ema_7' in result.columns


def test_compute_ffd_ma_preserves_index():
    """Test FFD MA preserves datetime index."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='T')
    ffd_series = pd.Series(np.random.randn(100), index=dates, name="ffd_log_price")

    result = compute_ffd_ma(ffd_series, windows=[5])

    assert isinstance(result.index, pd.DatetimeIndex)
    assert len(result) == 100
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_ma_basic -v
```

Expected: FAIL with "ImportError: cannot import name 'compute_ffd_ma'"

**Step 3: Implement compute_ffd_ma function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_ffd_ma(
    ffd_series: pd.Series,
    windows: List[int] = None,
    ema_windows: List[int] = None
) -> pd.DataFrame:
    """
    Compute moving average features based on FFD series.

    Calculates:
    - Simple moving average (ffd_ma_{window})
    - Exponential moving average (ffd_ema_{window})

    Args:
        ffd_series: FFD-transformed series
        windows: List of window sizes for SMA (default: [5, 10, 20])
        ema_windows: List of window sizes for EMA (default: same as windows)

    Returns:
        DataFrame: Contains ffd_ma_* and ffd_ema_* columns
    """
    if windows is None:
        windows = DEFAULT_MA_WINDOWS

    if ema_windows is None:
        ema_windows = windows

    result = pd.DataFrame(index=ffd_series.index)
    ffd_values = ffd_series.values.astype(np.float64)

    # Simple Moving Averages
    for window in windows:
        ma_col = f"ffd_ma_{window}"
        ma_values = sma(ffd_values, window=window)
        result[ma_col] = pd.Series(ma_values, index=ffd_series.index)

    # Exponential Moving Averages
    for window in ema_windows:
        ema_col = f"ffd_ema_{window}"
        ema_values = ewma(ffd_values, span=window)
        result[ema_col] = pd.Series(ema_values, index=ffd_series.index)

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_ma_basic -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_ma_custom_windows -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_ma_preserves_index -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_ffd_ma for SMA and EMA features"
```

---

## Task 6: Implement compute_ffd_rank Function

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
from webapp.utils.alpha158_features import compute_ffd_rank


def test_compute_ffd_rank_basic():
    """Test FFD rank feature computation."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='T')

    df = pd.DataFrame({
        'ffd_ma_5': pd.Series(np.random.randn(100), index=dates),
        'ffd_vol_std_10': pd.Series(np.random.randn(100), index=dates)
    })

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5', 'ffd_vol_std_10'], rank_window=20)

    # Check rank columns exist
    assert 'ffd_rank_ffd_ma_5_20' in result.columns
    assert 'ffd_rank_ffd_vol_std_10_20' in result.columns

    # Check values are in [0, 1] range
    assert result['ffd_rank_ffd_ma_5_20'].between(0, 1).all()
    assert result['ffd_rank_ffd_vol_std_10_20'].between(0, 1).all()


def test_compute_ffd_rank_single_feature():
    """Test FFD rank with single feature."""
    np.random.seed(42)
    df = pd.DataFrame({'ffd_ma_5': np.random.randn(50)})

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5'], rank_window=10)

    assert 'ffd_rank_ffd_ma_5_10' in result.columns
    assert len(result) == 50


def test_compute_ffd_rank_preserves_original():
    """Test FFD rank doesn't modify original columns."""
    np.random.seed(42)
    df = pd.DataFrame({
        'ffd_ma_5': np.random.randn(50),
        'other_col': np.random.randn(50)
    })

    result = compute_ffd_rank(df, feature_cols=['ffd_ma_5'], rank_window=10)

    # Original columns preserved
    assert 'ffd_ma_5' in result.columns
    assert 'other_col' in result.columns
    assert 'ffd_rank_ffd_ma_5_10' in result.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_rank_basic -v
```

Expected: FAIL with "ImportError: cannot import name 'compute_ffd_rank'"

**Step 3: Implement compute_ffd_rank function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_ffd_rank(
    df: pd.DataFrame,
    feature_cols: List[str] = None,
    rank_window: int = DEFAULT_RANK_WINDOW
) -> pd.DataFrame:
    """
    Compute rolling rank features (percentile rank within rolling window).

    This implements temporal rank: where does current value sit relative to
    the past `rank_window` observations? Single-stock alternative to cross-sectional rank.

    Args:
        df: DataFrame containing base features
        feature_cols: List of feature columns to rank (default: all ffd_* columns)
        rank_window: Rolling window size for percentile rank (default: 20)

    Returns:
        DataFrame: Original columns + ffd_rank_{feature}_{window} columns
    """
    result = df.copy()

    if feature_cols is None:
        # Default: rank all ffd_* prefixed columns
        feature_cols = [col for col in df.columns if col.startswith('ffd_')]

    for col in feature_cols:
        if col not in df.columns:
            continue

        rank_col = f"ffd_rank_{col}_{rank_window}"
        # Percentile rank: 0 = lowest in window, 1 = highest in window
        result[rank_col] = df[col].rolling(window=rank_window, min_periods=1).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_rank_basic -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_rank_single_feature -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_ffd_rank_preserves_original -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_ffd_rank for temporal percentile rank"
```

---

## Task 7: Implement Volume and Price Features

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
from webapp.utils.alpha158_features import compute_volume_features


def test_compute_volume_features_basic():
    """Test volume feature computation."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='T')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100),
        'volume': np.random.exponential(1000, 100)
    }, index=dates)

    result = compute_volume_features(df)

    # Check expected columns
    assert 'ffd_vwap' in result.columns
    assert 'ffd_amount' in result.columns
    assert 'ffd_amplification' in result.columns


def test_compute_volume_amount():
    """Test amount = close * volume."""
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0],
        'volume': [1000, 1100, 1200]
    })

    result = compute_volume_features(df)

    expected_amount = df['close'] * df['volume']
    assert result['ffd_amount'].iloc[1:].equals(expected_amount.iloc[1:])
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_volume_features_basic -v
```

Expected: FAIL

**Step 3: Implement compute_volume_features function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume and price-based features.

    These features use raw OHLCV data (not FFD-transformed) because:
    - Volume is already a stationary flow variable
    - Price levels are needed for VWAP, amount calculations

    Calculates:
    - ffd_vwap: Volume-weighted average price (rolling)
    - ffd_amount: Dollar amount (close * volume)
    - ffd_amplification: Price range relative to open

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame: Contains ffd_vwap, ffd_amount, ffd_amplification columns
    """
    result = pd.DataFrame(index=df.index)

    # VWAP - rolling 5-period volume-weighted average price
    if all(col in df.columns for col in ['close', 'volume']):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        result['ffd_vwap'] = (typical_price * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()

        # Amount (dollar volume)
        result['ffd_amount'] = df['close'] * df['volume']

        # Amplification (range / open)
        if 'open' in df.columns:
            result['ffd_amplification'] = (df['high'] - df['low']) / df['open']

    return result
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_volume_features_basic -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_volume_amount -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_volume_features for VWAP, amount, amplification"
```

---

## Task 8: Implement compute_alpha158_features Entry Function

**Files:**
- Modify: `webapp/utils/alpha158_features.py`
- Test: `tests/webapp/test_alpha158_features.py`

**Step 1: Write failing test**

Add to `tests/webapp/test_alpha158_features.py`:

```python
from webapp.utils.alpha158_features import compute_alpha158_features


def test_compute_alpha158_features_complete():
    """Test complete Alpha158 feature computation."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='T')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    result, metadata = compute_alpha158_features(df)

    # Check base FFD feature
    assert 'ffd_log_price' in result.columns

    # Check volatility features
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' in result.columns

    # Check MA features
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_ma_10' in result.columns
    assert 'ffd_ma_20' in result.columns

    # Check volume features
    assert 'ffd_vwap' in result.columns
    assert 'ffd_amount' in result.columns

    # Check metadata
    assert 'optimal_d' in metadata
    assert 0.0 <= metadata['optimal_d'] <= 1.0
    assert 'feature_columns' in metadata


def test_compute_alpha158_features_with_config():
    """Test Alpha158 with custom config."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='T')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    config = {
        'volatility': {'spans': [5, 10]},
        'ma': {'windows': [5, 10]},
        'rank': {'enabled': True, 'window': 20}
    }

    result, metadata = compute_alpha158_features(df, config=config)

    # Check custom spans
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_vol_std_10' in result.columns
    assert 'ffd_vol_std_20' not in result.columns  # Not in config

    # Check rank features enabled
    assert 'ffd_rank_ffd_ma_5_20' in result.columns


def test_compute_alpha158_features_missing_columns():
    """Test Alpha158 gracefully handles missing columns."""
    np.random.seed(42)
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    })

    # Should not raise error, just skip features requiring missing columns
    result, metadata = compute_alpha158_features(df)

    # Core features should still be computed
    assert 'ffd_log_price' in result.columns
    assert 'ffd_ma_5' in result.columns
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_alpha158_features_complete -v
```

Expected: FAIL with "ImportError: cannot import name 'compute_alpha158_features'"

**Step 3: Implement compute_alpha158_features function**

Add to `webapp/utils/alpha158_features.py`:

```python
def compute_alpha158_features(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compute complete Alpha158-style features (FFD-transformed version).

    This is the main entry point that orchestrates all feature computation:
    1. Compute FFD base series (ffd_log_price) with optimal d*
    2. Compute volatility features based on FFD series
    3. Compute MA/EMA features based on FFD series
    4. Compute volume features from raw OHLCV
    5. Compute rolling rank features
    6. Clean NaN rows and return metadata

    Args:
        df: Input DataFrame with OHLCV columns (must have 'close', optionally 'open/high/low/volume')
        config: Feature configuration dictionary with keys:
            - volatility: {spans: [5, 10, 20]}
            - ma: {windows: [5, 10, 20], ema_windows: [5, 10]}
            - rank: {enabled: True, window: 20}
            - volume: {enabled: True}
            - ffd: {thres: 1e-4, d_step: 0.05}

    Returns:
        tuple: (feature DataFrame, metadata dict)

    Metadata contains:
        - optimal_d: FFD optimal d* value
        - feature_columns: List of computed feature column names
        - rows_before_clean: Row count before NaN cleanup
        - rows_after_clean: Row count after NaN cleanup
        - config: Actual configuration used
    """
    config = config or {}
    metadata = {}

    # Validate input
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    df = df.sort_index().copy()

    # --- Step 1: Compute FFD base series ---
    ffd_config = config.get('ffd', {})
    thres = ffd_config.get('thres', DEFAULT_FFD_THRES)
    d_step = ffd_config.get('d_step', DEFAULT_FFD_D_STEP)

    ffd_series, optimal_d = compute_ffd_base(
        df['close'],
        thres=thres,
        d_step=d_step
    )

    # Start building result DataFrame
    result = pd.DataFrame(index=df.index)
    result['ffd_log_price'] = ffd_series

    # --- Step 2: Compute volatility features ---
    vol_config = config.get('volatility', {})
    vol_spans = vol_config.get('spans', DEFAULT_VOLATILITY_SPANS)

    vol_features = compute_ffd_volatility(ffd_series, spans=vol_spans)
    result = pd.concat([result, vol_features], axis=1)

    # --- Step 3: Compute MA features ---
    ma_config = config.get('ma', {})
    ma_windows = ma_config.get('windows', DEFAULT_MA_WINDOWS)
    ema_windows = ma_config.get('ema_windows', DEFAULT_EMA_WINDOWS)

    ma_features = compute_ffd_ma(ffd_series, windows=ma_windows, ema_windows=ema_windows)
    result = pd.concat([result, ma_features], axis=1)

    # --- Step 4: Add momentum feature (FFD series itself) ---
    result['ffd_mom'] = ffd_series

    # --- Step 5: Compute volume features ---
    volume_config = config.get('volume', {})
    if volume_config.get('enabled', True):
        if all(col in df.columns for col in ['close', 'volume']):
            vol_features = compute_volume_features(df)
            result = pd.concat([result, vol_features], axis=1)

    # --- Step 6: Compute rank features ---
    rank_config = config.get('rank', {})
    if rank_config.get('enabled', False):
        rank_window = rank_config.get('window', DEFAULT_RANK_WINDOW)
        ma_cols = [f"ffd_ma_{w}" for w in ma_windows if f"ffd_ma_{w}" in result.columns]
        result = compute_ffd_rank(result, feature_cols=ma_cols, rank_window=rank_window)

    # --- Step 7: Collect metadata ---
    metadata['optimal_d'] = optimal_d
    metadata['feature_columns'] = [col for col in result.columns if col != 'ffd_log_price']
    metadata['rows_before_clean'] = len(result)

    # --- Step 8: Clean NaN rows ---
    result = result.dropna()
    metadata['rows_after_clean'] = len(result)
    metadata['rows_dropped'] = metadata['rows_before_clean'] - metadata['rows_after_clean']
    metadata['config'] = config

    return result, metadata
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_alpha158_features_complete -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_alpha158_features_with_config -v
uv run pytest tests/webapp/test_alpha158_features.py::test_compute_alpha158_features_missing_columns -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add webapp/utils/alpha158_features.py tests/webapp/test_alpha158_features.py
git commit -m "feat: implement compute_alpha158_features entry function"
```

---

## Task 9: Integrate with feature_calculator.py

**Files:**
- Modify: `webapp/utils/feature_calculator.py`
- Test: `tests/webapp/test_feature_engineering_integration.py`

**Step 1: Add alpha158 import and config option**

Read current `webapp/utils/feature_calculator.py` to find the right insertion point.

Add after existing imports:

```python
# Optional Alpha158 features
# from webapp.utils.alpha158_features import compute_alpha158_features
```

**Step 2: Modify compute_all_features to support alpha158**

In `compute_all_features`, add after existing feature computation:

```python
# --- Alpha158 features (optional) ---
alpha158_config = config.get('alpha158', {})
if alpha1158_config.get('enabled', False):
    from webapp.utils.alpha158_features import compute_alpha158_features

    alpha158_df, alpha158_meta = compute_alpha158_features(df, config=alpha158_config)

    # Merge features (with ffd_* prefix to avoid collisions)
    for col in alpha158_df.columns:
        if col not in features_df.columns:
            features_df[col] = alpha158_df[col]

    # Update metadata
    metadata['alpha158_enabled'] = True
    metadata['alpha158_columns'] = alpha158_df.columns.tolist()
    metadata['alpha158_optimal_d'] = alpha158_meta.get('optimal_d')
```

**Step 3: Update purge_nan_rows to handle merged features**

The existing `purge_nan_rows` logic at the end will handle NaN cleanup for merged features.

**Step 4: Write integration test**

Add to `tests/webapp/test_feature_engineering_integration.py`:

```python
from webapp.utils.feature_calculator import compute_all_features


def test_compute_all_features_with_alpha158():
    """Test compute_all_features with Alpha158 enabled."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='T')

    df = pd.DataFrame({
        'open': np.linspace(100, 110, 200),
        'high': np.linspace(101, 111, 200),
        'low': np.linspace(99, 109, 200),
        'close': np.linspace(100, 110, 200),
        'volume': np.random.exponential(1000, 200)
    }, index=dates)

    config = {
        'volatility': {'spans': [10]},
        'alpha158': {
            'enabled': True,
            'volatility': {'spans': [5, 10]},
            'ma': {'windows': [5, 10]},
            'rank': {'enabled': True, 'window': 20}
        }
    }

    result, metadata = compute_all_features(df, config=config)

    # Check existing features
    assert 'vol_ewm_10' in result.columns

    # Check Alpha158 features
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_rank_ffd_ma_5_20' in result.columns

    # Check metadata
    assert metadata.get('alpha158_enabled') is True
    assert 'alpha158_optimal_d' in metadata
```

**Step 5: Run tests**

```bash
uv run pytest tests/webapp/test_feature_engineering_integration.py::test_compute_all_features_with_alpha158 -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add webapp/utils/feature_calculator.py tests/webapp/test_feature_engineering_integration.py
git commit -m "feat: integrate Alpha158 features into compute_all_features"
```

---

## Task 10: Update UI Configuration Options

**Files:**
- Modify: `webapp/pages/03_feature_engineering.py`

**Step 1: Read current file to understand structure**

Already read earlier. Find the configuration section in "1. 特征配置" step.

**Step 2: Add Alpha158 configuration section**

In the "1. 特征配置" step, add after existing config:

```python
# Alpha158 配置
with st.expander("Alpha158 特征 (FFD 改造版)", expanded=False):
    alpha158_enabled = st.checkbox(
        "启用 Alpha158 特征",
        value=False,
        help="启用后将计算 FFD 变换的 Alpha158 风格特征，与现有特征并存"
    )

    if alpha158_enabled:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**波动率窗口**")
            alpha158_vol_spans = st.text_input(
                "波动率 spans (逗号分隔)",
                value="5,10,20"
            )

        with col2:
            st.markdown("**均线窗口**")
            alpha158_ma_windows = st.text_input(
                "MA windows (逗号分隔)",
                value="5,10,20"
            )

        alpha158_rank_enabled = st.checkbox(
            "启用时序 Rank 特征",
            value=True
        )

        alpha158_rank_window = st.number_input(
            "Rank 窗口",
            min_value=5,
            max_value=60,
            value=20
        )
```

**Step 3: Update config saving logic**

In the "保存特征配置" button handler, add:

```python
if alpha158_enabled:
    feature_config['alpha158'] = {
        'enabled': True,
        'volatility': {
            'spans': [int(x.strip()) for x in alpha158_vol_spans.split(',') if x.strip().isdigit()]
        },
        'ma': {
            'windows': [int(x.strip()) for x in alpha158_ma_windows.split(',') if x.strip().isdigit()]
        },
        'rank': {
            'enabled': alpha158_rank_enabled,
            'window': alpha158_rank_window
        }
    }
```

**Step 4: Update display in "2. 特征计算" step**

In the calculation summary section, add:

```python
if metadata.get('alpha158_enabled'):
    st.markdown("**Alpha158 特征:**")
    st.write(f"  - 最优 d*: {metadata.get('alpha158_optimal_d', 'N/A'):.4f}")
    st.write(f"  - 特征数量：{len(metadata.get('alpha158_columns', []))}")
```

**Step 5: Manual test**

```bash
cd webapp
uv run streamlit run app.py
```

Navigate to 特征工程 page, enable Alpha158 features, and verify calculation works.

**Step 6: Commit**

```bash
git add webapp/pages/03_feature_engineering.py
git commit -m "feat: add Alpha158 configuration UI options"
```

---

## Task 11: Run Full Test Suite and Code Quality Checks

**Files:**
- All modified files

**Step 1: Run all Alpha158 tests**

```bash
uv run pytest tests/webapp/test_alpha158_features.py -v
```

Expected: All tests PASS

**Step 2: Run all webapp tests**

```bash
uv run pytest tests/webapp/ -v
```

Expected: All tests PASS

**Step 3: Run linting**

```bash
uv run ruff check webapp/utils/alpha158_features.py webapp/utils/feature_calculator.py webapp/pages/03_feature_engineering.py
```

**Step 4: Run formatting**

```bash
uv run ruff format webapp/utils/alpha158_features.py webapp/utils/feature_calculator.py webapp/pages/03_feature_engineering.py
```

**Step 5: Commit final changes**

```bash
git commit -am "chore: final linting and formatting"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `webapp/FEATURE_ENGINEERING_GUIDE.md`

**Step 1: Add Alpha158 section to guide**

Add to `webapp/FEATURE_ENGINEERING_GUIDE.md` after existing feature lists:

```markdown
## Alpha158 特征（FFD 改造版）

### 底座序列
- `ffd_log_price` - 分数阶差分对数价格（自动优化 d* 参数）

### 波动率特征
- `ffd_vol_std_{span}` - 滚动标准差
- `ffd_vol_ewm_{span}` - EWM 波动率

### 均线特征
- `ffd_ma_{window}` - 简单移动平均
- `ffd_ema_{window}` - 指数移动平均

### 动量特征
- `ffd_mom` - FFD 序列本身（替代传统收益率动量）

### 量价特征
- `ffd_vwap` - 成交量加权平均价
- `ffd_amount` - 成交金额
- `ffd_amplification` - 振幅

### 时序排序特征
- `ffd_rank_{feature}_{window}` - 滚动百分位排序

## Alpha158 配置示例

```yaml
alpha158:
  enabled: true
  volatility:
    spans: [5, 10, 20]
  ma:
    windows: [5, 10, 20]
  rank:
    enabled: true
    window: 20
```
```

**Step 2: Commit**

```bash
git add webapp/FEATURE_ENGINEERING_GUIDE.md
git commit -m "docs: add Alpha158 feature documentation"
```

---

## Task 13: Final Verification

**Step 1: Verify end-to-end flow**

```bash
cd webapp
uv run streamlit run app.py --server.headless true
```

**Step 2: Test complete workflow**

1. Navigate to 特征工程 page
2. Enable Alpha158 features in configuration
3. Click "保存特征配置"
4. Click "开始计算特征"
5. Verify ffd_* features appear in output
6. Verify metadata shows optimal_d

**Step 3: Final commit**

```bash
git commit -am "chore: final verification cleanup"
```

---

## Completion Checklist

- [ ] Task 1: Verify afmlkit core imports
- [ ] Task 2: Create alpha158_features module skeleton
- [ ] Task 3: Implement compute_ffd_base function
- [ ] Task 4: Implement compute_ffd_volatility function
- [ ] Task 5: Implement compute_ffd_ma function
- [ ] Task 6: Implement compute_ffd_rank function
- [ ] Task 7: Implement compute_volume_features function
- [ ] Task 8: Implement compute_alpha158_features entry function
- [ ] Task 9: Integrate with feature_calculator.py
- [ ] Task 10: Update UI configuration options
- [ ] Task 11: Run full test suite and code quality checks
- [ ] Task 12: Update documentation
- [ ] Task 13: Final verification

---

## Dependencies

```
afmlkit.feature.core.frac_diff
├── frac_diff_ffd
└── optimize_d

webapp.utils.alpha158_features (new)
├── compute_ffd_base
├── compute_ffd_volatility
├── compute_ffd_ma
├── compute_ffd_rank
├── compute_volume_features
└── compute_alpha158_features

webapp.utils.feature_calculator (modified)
└── compute_all_features (alpha158 integration)

webapp.pages.03_feature_engineering (modified)
└── Alpha158 configuration UI
```

---

## Testing Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `tests/webapp/test_alpha158_features.py` | 15+ unit tests | To run |
| `tests/webapp/test_feature_engineering_integration.py` | Integration tests | To run |
| Manual Streamlit test | E2E workflow | To run |
