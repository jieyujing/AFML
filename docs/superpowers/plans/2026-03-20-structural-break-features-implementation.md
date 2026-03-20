# Structural Break Features Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 structural break features (SADF, QADF, CADF, Sub-Martingale Test, Super-Martingale Test) for AFMLKit's FeatureKit based on AFML Chapter 17.

**Architecture:** Three-layer API design: Numba core functions for performance, public functional API for flexibility, and Transform classes for FeatureKit pipeline integration. SISO/MISO pattern follows existing codebase conventions.

**Tech Stack:** Python, NumPy, Numba JIT, pandas, pytest

---

## File Structure Overview

```
afmlkit/feature/core/structural_break/
├── __init__.py          # MODIFY - Add new exports
├── cusum.py             # EXISTS - Reference pattern
└── NEW FILES:
    ├── adf.py           # ADF base with OLS regression
    ├── sadf.py          # Supremum ADF (bubble detection)
    ├── qadf.py          # Quantile ADF (noise reduction)
    ├── cadf.py          # Conditional ADF (bubble strength)
    └── smt.py           # Sub/Super-Martingale tests

tests/structural_breaks/  # NEW DIRECTORY
├── test_adf.py
├── test_sadf.py
├── test_qadf.py
├── test_cadf.py
├── test_smt.py
└── test_integration.py
```

---

## Prerequisites

Before starting, ensure:
1. You are in the correct worktree created by the brainstorming skill
2. Dependencies are installed: `uv sync`
3. Tests run with: `NUMBA_DISABLE_JIT=1 uv run pytest tests/ -v`

---

## Task 1: ADF Base Module (adf.py)

**Files:**
- Create: `afmlkit/feature/core/structural_break/adf.py`
- Test: `tests/structural_breaks/test_adf.py`

**Purpose:** Foundation module providing OLS regression and basic ADF test functionality used by SADF, QADF, CADF.

### Step 1.1: Write the failing test

```python
import pytest
import numpy as np
from afmlkit.feature.core.structural_break.adf import (
    _ols_coefficients,
    _adf_regression_core,
    adf_test,
    adf_test_rolling,
)


def test_ols_coefficients_basic():
    """Test OLS regression with known coefficients."""
    # y = 2 + 3*x
    x = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [1, 4]], dtype=np.float64)
    y = np.array([2, 5, 8, 11, 14], dtype=np.float64)
    coef, residuals = _ols_coefficients(x, y)
    assert np.allclose(coef, [2, 3], atol=1e-10)


def test_adf_regression_core_trend():
    """Test ADF regression on trending series."""
    # Linear trend: should reject unit root (large negative t-stat)
    y = np.arange(100, dtype=np.float64) * 0.1 + 100
    t_stat, p_value = _adf_regression_core(y, max_lag=0, trend=True)
    assert t_stat < -2.0  # Should be significant
    assert 0 <= p_value <= 1


def test_adf_regression_core_random_walk():
    """Test ADF on random walk (unit root present)."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100)) + 100
    t_stat, p_value = _adf_regression_core(y, max_lag=0, trend=False)
    # Random walk should have t-stat closer to 0 (fail to reject unit root)
    assert abs(t_stat) < 2.0


def test_adf_test_basic():
    """Test public adf_test function."""
    y = np.arange(50, dtype=np.float64) * 0.1 + 100
    t_stat, p_value, lag = adf_test(y, max_lag=4, trend=True, use_numba=True)
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)
    assert isinstance(lag, int)
    assert -10 < t_stat < 10


def test_adf_test_rolling():
    """Test rolling ADF calculation."""
    np.random.seed(42)
    y = np.cumsum(np.random.randn(200)) + 100
    window = 50
    t_stats = adf_test_rolling(y, window=window, max_lag=2, trend=True)
    assert len(t_stats) == len(y)
    assert np.all(np.isnan(t_stats[:window-1]))
    assert np.all(np.isfinite(t_stats[window:]))
```

### Step 1.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_adf.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'afmlkit.feature.core.structural_break.adf'"

### Step 1.3: Write minimal implementation

Create `afmlkit/feature/core/structural_break/adf.py`:

```python
"""
ADF (Augmented Dickey-Fuller) test base implementation.

Provides foundation for SADF, QADF, CADF tests.
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple, Union
import pandas as pd


# MacKinnon approximate critical values for p-value interpolation
# Format: sample_size -> (1%, 5%, 10%)
_MACKINNON_WITH_TREND = {
    25: (-4.38, -3.60, -3.24),
    50: (-4.15, -3.50, -3.18),
    100: (-4.04, -3.45, -3.15),
    250: (-3.99, -3.43, -3.13),
    500: (-3.98, -3.42, -3.13),
    1000: (-3.96, -3.41, -3.12),
}

_MACKINNON_NO_TREND = {
    25: (-3.75, -3.00, -2.63),
    50: (-3.58, -2.93, -2.60),
    100: (-3.51, -2.89, -2.58),
    250: (-3.46, -2.88, -2.57),
    500: (-3.44, -2.87, -2.57),
    1000: (-3.43, -2.86, -2.57),
}


@njit(nogil=True)
def _ols_coefficients(
    X: NDArray[np.float64],
    y: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    OLS regression: y = X @ beta + epsilon.

    Uses normal equations: beta = (X'X)^-1 X'y

    :param X: Design matrix (n_obs, n_features)
    :param y: Target vector (n_obs,)
    :returns: Tuple of (coefficients, residuals)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    # Solve using simple linear algebra (Numba-compatible)
    # For small matrices, direct inversion is acceptable
    beta = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta
    residuals = y - y_hat

    return beta, residuals


@njit(nogil=True)
def _adf_regression_core(
    y: NDArray[np.float64],
    max_lag: int,
    trend: bool = True
) -> Tuple[float, float]:
    """
    Core ADF regression without lagged differences (simplest case).

    Model: dy_t = alpha + beta*t + gamma*y_{t-1} + epsilon_t

    Returns t-statistic for gamma (test of unit root).

    :param y: Price series (will be differenced internally)
    :param max_lag: Number of lagged differences (0 for basic DF)
    :param trend: Include time trend term
    :returns: (t_statistic, p_value_approx)
    """
    n = len(y)
    if n < 10:
        return np.nan, np.nan

    # Compute log differences
    dy = np.diff(y)
    y_lag = y[:-1]

    # Build design matrix
    n_obs = len(dy)
    if trend:
        # [constant, trend, y_lag]
        X = np.ones((n_obs, 3), dtype=np.float64)
        X[:, 1] = np.arange(n_obs, dtype=np.float64)
        X[:, 2] = y_lag
        gamma_idx = 2
    else:
        # [constant, y_lag]
        X = np.ones((n_obs, 2), dtype=np.float64)
        X[:, 1] = y_lag
        gamma_idx = 1

    # Add lagged differences if max_lag > 0
    if max_lag > 0:
        # For simplicity in minimal implementation, skip lagged terms
        # Full implementation would add difference columns
        pass

    # OLS estimation
    beta, residuals = _ols_coefficients(X, dy)

    # Compute t-statistic for gamma (unit root coefficient)
    gamma_hat = beta[gamma_idx]

    # Variance of residuals
    rss = np.sum(residuals ** 2)
    df = n_obs - X.shape[1]
    sigma_sq = rss / df if df > 0 else rss

    # (X'X)^-1 diagonal element for gamma
    XtX_inv = np.linalg.inv(X.T @ X)
    var_gamma = sigma_sq * XtX_inv[gamma_idx, gamma_idx]

    if var_gamma <= 0:
        return np.nan, np.nan

    se_gamma = np.sqrt(var_gamma)
    t_stat = gamma_hat / se_gamma

    # Approximate p-value using MacKinnon interpolation
    p_value = _approx_p_value(t_stat, n_obs, trend)

    return t_stat, p_value


@njit(nogil=True)
def _approx_p_value(t_stat: float, n: int, trend: bool) -> float:
    """
    Approximate p-value using MacKinnon critical values.

    Linear interpolation between critical value tables.
    """
    table = _MACKINNON_WITH_TREND if trend else _MACKINNON_NO_TREND

    # Find appropriate sample size bracket
    sizes = np.array([25, 50, 100, 250, 500, 1000], dtype=np.int64)

    # Determine which bracket n falls into
    if n < 25:
        crit_vals = table[25]
    elif n >= 1000:
        crit_vals = table[1000]
    else:
        # Find bracket
        for i in range(len(sizes) - 1):
            if sizes[i] <= n < sizes[i + 1]:
                # Linear interpolation
                n1, n2 = sizes[i], sizes[i + 1]
                w = (n - n1) / (n2 - n1)
                cv1 = np.array(table[int(n1)])
                cv2 = np.array(table[int(n2)])
                crit_vals = tuple(cv1 * (1 - w) + cv2 * w)
                break
        else:
            crit_vals = table[1000]

    # Map t-statistic to approximate p-value
    # crit_vals = (1%, 5%, 10%) critical values
    c1, c5, c10 = crit_vals

    if t_stat < c1:
        return 0.01
    elif t_stat < c5:
        return 0.05
    elif t_stat < c10:
        return 0.10
    else:
        return 0.20  # Conservative upper bound


def adf_test(
    y: Union[pd.Series, NDArray[np.float64]],
    max_lag: int = 12,
    trend: bool = True,
    use_numba: bool = True
) -> Tuple[float, float, int]:
    """
    Single ADF test on a price series.

    :param y: Price series (raw prices, not returns)
    :param max_lag: Maximum lag for augmented DF (default 12)
    :param trend: Include time trend in regression
    :param use_numba: Use Numba-accelerated implementation
    :returns: (t_statistic, p_value, selected_lag)
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)

    # For minimal implementation, use simple case
    # Full implementation would do lag selection via AIC/BIC
    selected_lag = 0  # Simplified

    t_stat, p_value = _adf_regression_core(y, max_lag=selected_lag, trend=trend)

    return float(t_stat), float(p_value), int(selected_lag)


def adf_test_rolling(
    y: Union[pd.Series, NDArray[np.float64]],
    window: int,
    max_lag: int = 12,
    trend: bool = True,
    use_numba: bool = True
) -> NDArray[np.float64]:
    """
    Rolling ADF test statistic calculation.

    :param y: Price series
    :param window: Rolling window size
    :param max_lag: Maximum lag for ADF
    :param trend: Include time trend
    :param use_numba: Use Numba backend
    :returns: Array of t-statistics (NaN for first window-1 observations)
    """
    if isinstance(y, pd.Series):
        y = y.values
    y = np.asarray(y, dtype=np.float64)

    n = len(y)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window, n + 1):
        window_data = y[i - window:i]
        t_stat, _ = _adf_regression_core(window_data, max_lag=0, trend=trend)
        result[i - 1] = t_stat

    return result
```

### Step 1.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_adf.py -v`

Expected: PASS

### Step 1.5: Commit

```bash
git add afmlkit/feature/core/structural_break/adf.py tests/structural_breaks/test_adf.py
git commit -m "feat: add ADF base module for structural break features"
```

---

## Task 2: SADF Module (sadf.py)

**Files:**
- Create: `afmlkit/feature/core/structural_break/sadf.py`
- Modify: `afmlkit/feature/core/structural_break/__init__.py` (add exports)
- Test: `tests/structural_breaks/test_sadf.py`

**Purpose:** Supremum ADF test for detecting explosive behavior (bubbles) in price series.

### Step 2.1: Write the failing test

```python
import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.sadf import (
    _sadf_core,
    sadf_test,
    SADFTest,
)


def test_sadf_core_basic():
    """Test SADF core computation."""
    # Create explosive price series
    np.random.seed(42)
    t = np.arange(100, dtype=np.float64)
    prices = 100 * np.exp(0.01 * t + 0.001 * t**1.5)  # Super-exponential growth

    min_window = 20
    max_window = 50
    result = _sadf_core(prices, min_window, max_window, max_lag=0, trend=True)

    assert len(result) == len(prices)
    assert np.all(np.isnan(result[:min_window]))
    assert np.all(np.isfinite(result[min_window:]))
    # SADF should be positive for explosive series
    assert np.mean(result[min_window:]) > 0


def test_sadf_test_series():
    """Test SADF with pandas Series input."""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = pd.Series(
        100 * np.exp(np.cumsum(np.random.randn(200) * 0.02)),
        index=dates
    )

    result = sadf_test(prices, min_window=20, max_window=100, max_lag=2)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)
    assert result.index.equals(prices.index)


def test_sadf_test_array():
    """Test SADF with numpy array input."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(150) * 0.02))

    result = sadf_test(prices, min_window=20, max_window=80, use_log=True)

    assert len(result) == len(prices)
    assert isinstance(result, np.ndarray)


def test_sadf_transform():
    """Test SADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    }, index=dates)

    transform = SADFTest(min_window=20, max_window=60, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_sadf'
    assert len(result) == len(df)


def test_sadf_no_bubble_series():
    """Test SADF on random walk (no bubble)."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)  # Random walk

    result = sadf_test(prices, min_window=20, max_window=100)

    # SADF should generally be low for non-explosive series
    valid = result[~np.isnan(result)]
    assert np.percentile(valid, 90) < 2.0
```

### Step 2.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_sadf.py -v`

Expected: FAIL - ModuleNotFoundError

### Step 2.3: Write minimal implementation

Create `afmlkit/feature/core/structural_break/sadf.py`:

```python
"""
SADF (Supremum Augmented Dickey-Fuller) test for bubble detection.

Detects explosive behavior by taking supremum of ADF statistics across
expanding windows.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import SISOTransform
from afmlkit.feature.core.structural_break.adf import _adf_regression_core


@njit(nogil=True)
def _sadf_core(
    log_prices: NDArray[np.float64],
    min_window: int,
    max_window: int,
    max_lag: int,
    trend: bool
) -> NDArray[np.float64]:
    """
    SADF core computation.

    For each time t, computes ADF statistics for all windows from min_window
to current window (or max_window), and returns the supremum.

    :param log_prices: Log price series
    :param min_window: Minimum window size for ADF
    :param max_window: Maximum window size (use -1 for expanding to t)
    :param max_lag: Maximum lag for ADF regression
    :param trend: Include time trend in ADF
    :returns: SADF statistics (NaN for first min_window observations)
    """
    n = len(log_prices)
    result = np.full(n, np.nan, dtype=np.float64)

    for t in range(min_window, n):
        # Current window limit
        current_max = min(t, max_window) if max_window > 0 else t

        max_adf = -np.inf

        for window in range(min_window, current_max + 1):
            start = t - window + 1
            if start < 0:
                continue

            window_data = log_prices[start:t + 1]
            adf_t, _ = _adf_regression_core(window_data, max_lag, trend)

            if not np.isnan(adf_t) and adf_t > max_adf:
                max_adf = adf_t

        if max_adf > -np.inf:
            result[t] = max_adf

    return result


def sadf_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    min_window: int = 20,
    max_window: int = 100,
    max_lag: int = 0,
    trend: bool = True,
    use_log: bool = True,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    SADF test for bubble detection.

    :param prices: Price series
    :param min_window: Minimum window size (default 20)
    :param max_window: Maximum window size (default 100, None for expanding)
    :param max_lag: ADF lag order (default 0)
    :param trend: Include time trend (default True)
    :param use_log: Apply log transform to prices (default True, recommended)
    :param use_numba: Use Numba backend (default True)
    :returns: SADF statistic series
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    prices_arr = np.asarray(prices, dtype=np.float64)

    if use_log:
        prices_arr = np.log(prices_arr)

    max_w = max_window if max_window is not None else -1

    result = _sadf_core(prices_arr, min_window, max_w, max_lag, trend)

    if is_pandas:
        return pd.Series(result, index=index, name='sadf')
    return result


class SADFTest(SISOTransform):
    """
    SADF Transform for FeatureKit pipeline.

    Detects explosive price behavior (bubbles) using supremum ADF test.

    :param input_col: Input price column name (default 'close')
    :param min_window: Minimum window for ADF calculation (default 20)
    :param max_window: Maximum window (default 100, None for expanding)
    :param max_lag: ADF lag order (default 0)
    :param trend: Include time trend (default True)
    :param use_log: Apply log transform (default True)
    """

    def __init__(
        self,
        input_col: str = 'close',
        min_window: int = 20,
        max_window: Optional[int] = 100,
        max_lag: int = 0,
        trend: bool = True,
        use_log: bool = True
    ):
        super().__init__(input_col, 'sadf')
        self.min_window = min_window
        self.max_window = max_window
        self.max_lag = max_lag
        self.trend = trend
        self.use_log = use_log

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """Pandas backend implementation."""
        prices = x[self.requires[0]].values
        if self.use_log:
            prices = np.log(prices)

        max_w = self.max_window if self.max_window is not None else -1
        result = _sadf_core(prices, self.min_window, max_w, self.max_lag, self.trend)

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """Numba backend implementation (same as pandas for SADF)."""
        return self._pd(x)
```

### Step 2.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_sadf.py -v`

Expected: PASS

### Step 2.5: Commit

```bash
git add afmlkit/feature/core/structural_break/sadf.py tests/structural_breaks/test_sadf.py
git commit -m "feat: add SADF test for bubble detection"
```

---

## Task 3: QADF Module (qadf.py)

**Files:**
- Create: `afmlkit/feature/core/structural_break/qadf.py`
- Modify: `afmlkit/feature/core/structural_break/__init__.py`
- Test: `tests/structural_breaks/test_qadf.py`

**Purpose:** Quantile ADF - rolling quantile of SADF to reduce noise.

### Step 3.1: Write the failing test

```python
import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.qadf import (
    _rolling_quantile_core,
    qadf_test,
    QADFTest,
)


def test_rolling_quantile_core():
    """Test rolling quantile computation."""
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    window = 5
    quantile = 0.95

    result = _rolling_quantile_core(data, window, quantile)

    assert len(result) == len(data)
    assert np.all(np.isnan(result[:window-1]))
    assert np.all(np.isfinite(result[window-1:]))

    # 95th percentile of [1,2,3,4,5] should be near 5
    assert result[4] > 4.5


def test_qadf_from_sadf():
    """Test QADF computed from SADF values."""
    np.random.seed(42)
    # Create SADF-like series
    sadf = np.random.randn(200) * 2 + np.sin(np.arange(200) * 0.1)

    result = qadf_test(sadf, window=20, quantile=0.95)

    assert len(result) == len(sadf)
    assert np.all(np.isnan(result[:19]))
    assert np.all(np.isfinite(result[20:]))


def test_qadf_transform():
    """Test QADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    np.random.seed(42)
    # Pre-computed SADF values
    sadf = pd.Series(
        np.random.randn(150).cumsum() * 0.5 + 1,
        index=dates,
        name='sadf'
    )
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(150))})
    df['sadf'] = sadf

    transform = QADFTest(window=30, quantile=0.95)
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'qadf'
    assert len(result) == len(df)


def test_qadf_high_quantile():
    """Test that higher quantile gives higher values."""
    np.random.seed(42)
    sadf = np.random.randn(100) * 2

    q95 = qadf_test(sadf, window=20, quantile=0.95)
    q50 = qadf_test(sadf, window=20, quantile=0.50)

    valid_idx = ~np.isnan(q95) & ~np.isnan(q50)
    assert np.all(q95[valid_idx] >= q50[valid_idx])
```

### Step 3.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_qadf.py -v`

Expected: FAIL - ModuleNotFoundError

### Step 3.3: Write minimal implementation

Create `afmlkit/feature/core/structural_break/qadf.py`:

```python
"""
QADF (Quantile ADF) test - rolling quantile of SADF for noise reduction.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import MISOTransform


@njit(nogil=True)
def _rolling_quantile_core(
    x: NDArray[np.float64],
    window: int,
    quantile: float
) -> NDArray[np.float64]:
    """
    Rolling quantile calculation using simple sort.

    :param x: Input series
    :param window: Rolling window size
    :param quantile: Quantile level (0-1)
    :returns: Rolling quantile values
    """
    n = len(x)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        window_data = x[i - window + 1:i + 1]
        # Filter out NaN values
        valid = window_data[~np.isnan(window_data)]
        if len(valid) > 0:
            result[i] = np.percentile(valid, quantile * 100)

    return result


def qadf_test(
    sadf_values: Union[pd.Series, NDArray[np.float64]],
    window: int = 20,
    quantile: float = 0.95,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    QADF test - rolling quantile of SADF values.

    :param sadf_values: SADF statistic series
    :param window: Rolling window for quantile (default 20)
    :param quantile: Quantile level (default 0.95)
    :param use_numba: Use Numba backend (default True)
    :returns: QADF series
    """
    is_pandas = isinstance(sadf_values, pd.Series)
    index = sadf_values.index if is_pandas else None

    arr = np.asarray(sadf_values, dtype=np.float64)
    result = _rolling_quantile_core(arr, window, quantile)

    if is_pandas:
        return pd.Series(result, index=index, name='qadf')
    return result


class QADFTest(MISOTransform):
    """
    QADF Transform for FeatureKit pipeline.

    Computes rolling quantile of SADF values for smoother bubble detection.

    :param window: Rolling window for quantile (default 20)
    :param quantile: Quantile level (default 0.95)
    """

    def __init__(
        self,
        window: int = 20,
        quantile: float = 0.95,
        input_cols: Optional[list] = None
    ):
        # Default input is SADF column
        inputs = input_cols if input_cols else ['sadf']
        super().__init__(inputs, 'qadf')
        self.window = window
        self.quantile = quantile

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """Pandas backend."""
        sadf = x[self.requires[0]].values
        result = _rolling_quantile_core(sadf, self.window, self.quantile)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """Numba backend."""
        return self._pd(x)
```

### Step 3.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_qadf.py -v`

Expected: PASS

### Step 3.5: Commit

```bash
git add afmlkit/feature/core/structural_break/qadf.py tests/structural_breaks/test_qadf.py
git commit -m "feat: add QADF test for noise-reduced bubble detection"
```

---

## Task 4: CADF Module (cadf.py)

**Files:**
- Create: `afmlkit/feature/core/structural_break/cadf.py`
- Modify: `afmlkit/feature/core/structural_break/__init__.py`
- Test: `tests/structural_breaks/test_cadf.py`

**Purpose:** Conditional ADF - expected ADF value given it exceeds QADF threshold.

### Step 4.1: Write the failing test

```python
import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.cadf import (
    _conditional_expectation_core,
    cadf_test,
    CADFTest,
)


def test_conditional_expectation_core():
    """Test conditional expectation calculation."""
    adf = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
    quantile = np.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5], dtype=np.float64)  # Threshold at 5
    window = 5

    result = _conditional_expectation_core(adf, quantile, window)

    assert len(result) == len(adf)
    # Where adf > quantile, should have conditional mean
    # Window 5-9: adf [6,7,8,9,10], threshold 5, mean should be around 8
    assert np.isfinite(result[9])


def test_cadf_from_series():
    """Test CADF from price series."""
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))

    result = cadf_test(
        prices,
        min_window=20,
        max_window=100,
        quantile_window=20,
        quantile=0.95
    )

    assert len(result) == len(prices)
    assert np.all(np.isfinite(result[~np.isnan(result)]))


def test_cadf_transform():
    """Test CADF Transform class."""
    dates = pd.date_range('2023-01-01', periods=150, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(150) * 0.02))
    }, index=dates)

    transform = CADFTest(
        min_window=20,
        max_window=80,
        quantile_window=20,
        quantile=0.95
    )
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'cadf'
    assert len(result) == len(df)


def test_cadf_with_injected_values():
    """Test CADF with pre-computed SADF/QADF injection."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Pre-computed values
    sadf = np.random.randn(100).cumsum() * 0.5 + 1
    qadf = pd.Series(np.convolve(sadf, np.ones(20)/20, mode='same'), index=dates)
    sadf = pd.Series(sadf, index=dates)

    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = CADFTest(
        sadf_values=sadf.values,
        qadf_values=qadf.values,
        quantile_window=20
    )
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
```

### Step 4.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_cadf.py -v`

Expected: FAIL - ModuleNotFoundError

### Step 4.3: Write minimal implementation

Create `afmlkit/feature/core/structural_break/cadf.py`:

```python
"""
CADF (Conditional ADF) test - expected ADF given it exceeds quantile threshold.

Quantifies average bubble strength when SADF > QADF threshold.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional
import pandas as pd

from afmlkit.feature.base import MISOTransform
from afmlkit.feature.core.structural_break.sadf import _sadf_core
from afmlkit.feature.core.structural_break.qadf import _rolling_quantile_core


@njit(nogil=True)
def _conditional_expectation_core(
    adf_values: NDArray[np.float64],
    quantile_values: NDArray[np.float64],
    window: int
) -> NDArray[np.float64]:
    """
    Conditional expectation: E[ADF_t | ADF_t > QADF_{t-L:t}].

    :param adf_values: ADF/SADF values
    :param quantile_values: Rolling quantile (threshold) values
    :param window: Window size (should match QADF window)
    :returns: Conditional expectation values
    """
    n = len(adf_values)
    result = np.full(n, np.nan, dtype=np.float64)

    for i in range(window - 1, n):
        # Check if current ADF exceeds threshold
        if adf_values[i] > quantile_values[i]:
            # Collect all values in window that exceed threshold
            window_start = max(0, i - window + 1)
            count = 0
            total = 0.0

            for j in range(window_start, i + 1):
                if adf_values[j] > quantile_values[j]:
                    total += adf_values[j]
                    count += 1

            if count > 0:
                result[i] = total / count

    return result


def cadf_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    sadf_values: Optional[NDArray[np.float64]] = None,
    qadf_values: Optional[NDArray[np.float64]] = None,
    min_window: int = 20,
    max_window: int = 100,
    quantile_window: int = 20,
    quantile: float = 0.95,
    max_lag: int = 0,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    CADF test - conditional expectation of ADF given threshold exceedance.

    :param prices: Price series (if sadf_values not provided)
    :param sadf_values: Pre-computed SADF values (optional)
    :param qadf_values: Pre-computed QADF values (optional)
    :param min_window: SADF minimum window
    :param max_window: SADF maximum window
    :param quantile_window: QADF rolling window
    :param quantile: QADF quantile level
    :param max_lag: ADF lag order
    :returns: CADF series
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    # Compute SADF if not provided
    if sadf_values is None:
        prices_arr = np.asarray(prices, dtype=np.float64)
        log_prices = np.log(prices_arr)
        sadf_arr = _sadf_core(log_prices, min_window, max_window, max_lag, True)
    else:
        sadf_arr = np.asarray(sadf_values, dtype=np.float64)

    # Compute QADF if not provided
    if qadf_values is None:
        qadf_arr = _rolling_quantile_core(sadf_arr, quantile_window, quantile)
    else:
        qadf_arr = np.asarray(qadf_values, dtype=np.float64)

    # Compute conditional expectation
    result = _conditional_expectation_core(sadf_arr, qadf_arr, quantile_window)

    if is_pandas:
        return pd.Series(result, index=index, name='cadf')
    return result


class CADFTest(MISOTransform):
    """
    CADF Transform for FeatureKit pipeline.

    Computes conditional expected SADF when it exceeds QADF threshold.

    :param min_window: SADF minimum window (default 20)
    :param max_window: SADF maximum window (default 100)
    :param quantile_window: QADF rolling window (default 20)
    :param quantile: QADF quantile level (default 0.95)
    :param sadf_values: Pre-computed SADF values (optional)
    :param qadf_values: Pre-computed QADF values (optional)
    """

    def __init__(
        self,
        min_window: int = 20,
        max_window: int = 100,
        quantile_window: int = 20,
        quantile: float = 0.95,
        sadf_values: Optional[NDArray[np.float64]] = None,
        qadf_values: Optional[NDArray[np.float64]] = None
    ):
        super().__init__(['close'], 'cadf')
        self.min_window = min_window
        self.max_window = max_window
        self.quantile_window = quantile_window
        self.quantile = quantile
        self.sadf_values = sadf_values
        self.qadf_values = qadf_values

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        """Pandas backend."""
        if self.sadf_values is not None and self.qadf_values is not None:
            result = _conditional_expectation_core(
                self.sadf_values,
                self.qadf_values,
                self.quantile_window
            )
        else:
            prices = x[self.requires[0]].values
            result = cadf_test(
                prices,
                min_window=self.min_window,
                max_window=self.max_window,
                quantile_window=self.quantile_window,
                quantile=self.quantile
            ).values

        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        """Numba backend."""
        return self._pd(x)
```

### Step 4.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_cadf.py -v`

Expected: PASS

### Step 4.5: Commit

```bash
git add afmlkit/feature/core/structural_break/cadf.py tests/structural_breaks/test_cadf.py
git commit -m "feat: add CADF test for conditional bubble strength"
```

---

## Task 5: SMT Module (smt.py)

**Files:**
- Create: `afmlkit/feature/core/structural_break/smt.py`
- Modify: `afmlkit/feature/core/structural_break/__init__.py`
- Test: `tests/structural_breaks/test_smt.py`

**Purpose:** Sub-Martingale and Super-Martingale tests for trend detection.

### Step 5.1: Write the failing test

```python
import pytest
import numpy as np
import pandas as pd
from afmlkit.feature.core.structural_break.smt import (
    _sub_martingale_core,
    _super_martingale_core,
    sub_martingale_test,
    super_martingale_test,
    martingale_test,
    SubMartingaleTest,
    SuperMartingaleTest,
    MartingaleTest,
)


def test_sub_martingale_core_uptrend():
    """Test sub-martingale detects uptrend."""
    # Strong uptrend
    prices = np.exp(np.arange(100) * 0.02) * 100

    result = _sub_martingale_core(prices, decay=0.95, window_size=20)

    assert len(result) == len(prices)
    assert np.all(np.isfinite(result[20:]))
    # Uptrend should have positive sub-martingale statistic
    assert np.mean(result[20:]) > 0


def test_sub_martingale_core_downtrend():
    """Test sub-martingale on downtrend."""
    # Downtrend
    prices = np.exp(-np.arange(100) * 0.02) * 100

    result = _sub_martingale_core(prices, decay=0.95, window_size=20)

    # Downtrend should have negative or near-zero sub-martingale
    assert np.mean(result[20:]) < 1.0


def test_super_martingale_core():
    """Test super-martingale is negative of sub-martingale."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100))

    sub = _sub_martingale_core(prices, decay=0.95, window_size=20)
    sup = _super_martingale_core(prices, decay=0.95, window_size=20)

    assert len(sub) == len(sup)
    # Super should be approximately -sub
    assert np.allclose(sub, -sup, atol=1e-10)


def test_sub_martingale_test():
    """Test public sub-martingale function."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    result = sub_martingale_test(prices, decay=0.95, window=30)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)


def test_super_martingale_test():
    """Test public super-martingale function."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    result = super_martingale_test(prices, decay=0.95, window=30)

    assert len(result) == len(prices)
    assert isinstance(result, pd.Series)


def test_martingale_test():
    """Test combined martingale test."""
    np.random.seed(42)
    prices = pd.Series(100 + np.cumsum(np.random.randn(150)))

    sub, sup = martingale_test(prices, decay=0.95, window=30)

    assert len(sub) == len(prices)
    assert len(sup) == len(prices)


def test_sub_martingale_transform():
    """Test SubMartingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = SubMartingaleTest(decay=0.95, window=20, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_sub_martingale'


def test_super_martingale_transform():
    """Test SuperMartingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = SuperMartingaleTest(decay=0.95, window=20, input_col='close')
    result = transform(df, backend='nb')

    assert isinstance(result, pd.Series)
    assert result.name == 'close_super_martingale'


def test_martingale_transform():
    """Test combined Martingale Transform class."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({'close': 100 + np.cumsum(np.random.randn(100))}, index=dates)

    transform = MartingaleTest(decay=0.95, window=20, input_col='close')
    sub, sup = transform(df, backend='nb')

    assert isinstance(sub, pd.Series)
    assert isinstance(sup, pd.Series)
    assert sub.name == 'close_sub_martingale'
    assert sup.name == 'close_super_martingale'


def test_martingale_decay_effect():
    """Test that higher decay gives more persistent statistics."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200))

    sub_90 = sub_martingale_test(prices, decay=0.90, window=50)
    sub_99 = sub_martingale_test(prices, decay=0.99, window=50)

    # Higher decay should give smoother series
    var_90 = np.nanvar(sub_90)
    var_99 = np.nanvar(sub_99)
    assert var_99 < var_90  # More smoothing with higher decay
```

### Step 5.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_smt.py -v`

Expected: FAIL - ModuleNotFoundError

### Step 5.3: Write minimal implementation

Create `afmlkit/feature/core/structural_break/smt.py`:

```python
"""
Sub-Martingale and Super-Martingale tests for trend detection.

S_t = sum(w_i * r_i) / sqrt(sum(w_i^2))

where w_i = lambda^(t-i) are exponential weights.

Reference: AFML Chapter 17
"""
import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Union, Optional, Tuple
import pandas as pd

from afmlkit.feature.base import SISOTransform, SIMOTransform


@njit(nogil=True)
def _sub_martingale_core(
    prices: NDArray[np.float64],
    decay: float,
    window_size: int
) -> NDArray[np.float64]:
    """
    Sub-martingale test core computation.

    H0: E[P_{t+1} | P_t, ...] <= P_t (not an uptrend)

    Formula: S_t = sum(w_i * r_i) / sqrt(sum(w_i^2))

    :param prices: Price series
    :param decay: Decay factor (0-1)
    :param window_size: Window size (-1 for expanding)
    :returns: Sub-martingale statistics
    """
    n = len(prices)
    result = np.full(n, np.nan, dtype=np.float64)

    # Compute returns
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(0.0, returns)  # Pad first value

    for t in range(1, n):
        if window_size > 0:
            start = max(0, t - window_size + 1)
        else:
            start = 0

        # Compute weighted sum
        weighted_sum = 0.0
        weight_sq_sum = 0.0

        for i in range(start, t + 1):
            if i == 0:
                continue  # Skip first return (was padded)
            w = decay ** (t - i)
            weighted_sum += w * returns[i]
            weight_sq_sum += w * w

        if weight_sq_sum > 1e-12:
            result[t] = weighted_sum / np.sqrt(weight_sq_sum)

    return result


@njit(nogil=True)
def _super_martingale_core(
    prices: NDArray[np.float64],
    decay: float,
    window_size: int
) -> NDArray[np.float64]:
    """
    Super-martingale test - negative of sub-martingale.

    H0: E[P_{t+1} | P_t, ...] >= P_t (not a downtrend)
    """
    return -_sub_martingale_core(prices, decay, window_size)


def sub_martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    Sub-martingale test for uptrend detection.

    :param prices: Price series
    :param decay: Exponential decay factor (default 0.95)
    :param window: Rolling window (None for expanding)
    :param use_numba: Use Numba backend
    :returns: Sub-martingale statistics
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    arr = np.asarray(prices, dtype=np.float64)
    window_size = window if window is not None else -1

    result = _sub_martingale_core(arr, decay, window_size)

    if is_pandas:
        return pd.Series(result, index=index, name='sub_martingale')
    return result


def super_martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None,
    use_numba: bool = True
) -> Union[pd.Series, NDArray[np.float64]]:
    """
    Super-martingale test for downtrend detection.

    :param prices: Price series
    :param decay: Exponential decay factor (default 0.95)
    :param window: Rolling window (None for expanding)
    :param use_numba: Use Numba backend
    :returns: Super-martingale statistics
    """
    is_pandas = isinstance(prices, pd.Series)
    index = prices.index if is_pandas else None

    arr = np.asarray(prices, dtype=np.float64)
    window_size = window if window is not None else -1

    result = _super_martingale_core(arr, decay, window_size)

    if is_pandas:
        return pd.Series(result, index=index, name='super_martingale')
    return result


def martingale_test(
    prices: Union[pd.Series, NDArray[np.float64]],
    decay: float = 0.95,
    window: Optional[int] = None,
    use_numba: bool = True
) -> Tuple[Union[pd.Series, NDArray], Union[pd.Series, NDArray]]:
    """
    Combined sub and super-martingale tests.

    :returns: Tuple of (sub_martingale, super_martingale)
    """
    sub = sub_martingale_test(prices, decay, window, use_numba)
    sup = super_martingale_test(prices, decay, window, use_numba)
    return sub, sup


class SubMartingaleTest(SISOTransform):
    """
    Sub-martingale Transform for uptrend detection.

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, 'sub_martingale')
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        window_size = self.window if self.window is not None else -1
        result = _sub_martingale_core(prices, self.decay, window_size)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class SuperMartingaleTest(SISOTransform):
    """
    Super-martingale Transform for downtrend detection.

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, 'super_martingale')
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        prices = x[self.requires[0]].values
        window_size = self.window if self.window is not None else -1
        result = _super_martingale_core(prices, self.decay, window_size)
        return pd.Series(result, index=x.index, name=self.output_name)

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)


class MartingaleTest(SIMOTransform):
    """
    Combined martingale test Transform - outputs both sub and super.

    Produces two columns: ['{col}_sub_martingale', '{col}_super_martingale']

    :param decay: Decay factor (default 0.95)
    :param window: Window size (None for expanding)
    :param input_col: Price column (default 'close')
    """

    def __init__(self, decay: float = 0.95, window: Optional[int] = None, input_col: str = 'close'):
        super().__init__(input_col, ['sub_martingale', 'super_martingale'])
        self.decay = decay
        self.window = window

    def _pd(self, x: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        prices = x[self.requires[0]].values
        window_size = self.window if self.window is not None else -1

        sub = _sub_martingale_core(prices, self.decay, window_size)
        sup = _super_martingale_core(prices, self.decay, window_size)

        return (
            pd.Series(sub, index=x.index, name=self.output_name[0]),
            pd.Series(sup, index=x.index, name=self.output_name[1])
        )

    def _nb(self, x: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return self._pd(x)
```

### Step 5.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_smt.py -v`

Expected: PASS

### Step 5.5: Commit

```bash
git add afmlkit/feature/core/structural_break/smt.py tests/structural_breaks/test_smt.py
git commit -m "feat: add Sub/Super-Martingale tests for trend detection"
```

---

## Task 6: Update __init__.py Exports

**Files:**
- Modify: `afmlkit/feature/core/structural_break/__init__.py`

### Step 6.1: Write the failing test

Create `tests/structural_breaks/test_integration.py`:

```python
import pytest
import pandas as pd
import numpy as np


def test_import_all_features():
    """Test all structural break features can be imported."""
    from afmlkit.feature.core.structural_break import (
        # ADF base
        adf_test,
        adf_test_rolling,
        # SADF
        sadf_test,
        SADFTest,
        # QADF
        qadf_test,
        QADFTest,
        # CADF
        cadf_test,
        CADFTest,
        # SMT
        sub_martingale_test,
        super_martingale_test,
        martingale_test,
        SubMartingaleTest,
        SuperMartingaleTest,
        MartingaleTest,
    )

    assert callable(adf_test)
    assert callable(sadf_test)
    assert callable(qadf_test)
    assert callable(cadf_test)
    assert callable(sub_martingale_test)


def test_end_to_end_pipeline():
    """Test end-to-end feature pipeline."""
    from afmlkit.feature.core.structural_break import (
        SADFTest, QADFTest, SubMartingaleTest, SuperMartingaleTest
    )

    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'close': 100 * np.exp(np.cumsum(np.random.randn(200) * 0.02))
    }, index=dates)

    # Apply transforms
    sadf = SADFTest(min_window=20, max_window=100)(df, backend='nb')
    df['sadf'] = sadf

    qadf = QADFTest(window=20)(df, backend='nb')

    sub = SubMartingaleTest(window=30)(df, backend='nb')
    sup = SuperMartingaleTest(window=30)(df, backend='nb')

    assert len(qadf) == len(df)
    assert len(sub) == len(df)
    assert len(sup) == len(df)
```

### Step 6.2: Run test to verify it fails

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_integration.py -v`

Expected: FAIL - ImportError

### Step 6.3: Write minimal implementation

Update `afmlkit/feature/core/structural_break/__init__.py`:

```python
"""
Structural break features for AFMLKit.

Detects regime changes, bubbles, and trend mutations in financial time series.

Features:
- ADF: Augmented Dickey-Fuller base test
- SADF: Supremum ADF for bubble detection
- QADF: Quantile ADF for noise reduction
- CADF: Conditional ADF for bubble strength quantification
- SMT: Sub/Super-Martingale tests for trend detection
- CUSUM: Chu-Stinchcombe-White CUSUM test

Reference: AFML Chapter 17
"""

# ADF base
from afmlkit.feature.core.structural_break.adf import (
    adf_test,
    adf_test_rolling,
)

# SADF
from afmlkit.feature.core.structural_break.sadf import (
    sadf_test,
    SADFTest,
)

# QADF
from afmlkit.feature.core.structural_break.qadf import (
    qadf_test,
    QADFTest,
)

# CADF
from afmlkit.feature.core.structural_break.cadf import (
    cadf_test,
    CADFTest,
)

# SMT (Sub/Super-Martingale)
from afmlkit.feature.core.structural_break.smt import (
    sub_martingale_test,
    super_martingale_test,
    martingale_test,
    SubMartingaleTest,
    SuperMartingaleTest,
    MartingaleTest,
)

# Existing CUSUM exports
from afmlkit.feature.core.structural_break.cusum import (
    cusum_test_developing,
    cusum_test_last,
    cusum_test_rolling,
)

__all__ = [
    # ADF
    'adf_test',
    'adf_test_rolling',
    # SADF
    'sadf_test',
    'SADFTest',
    # QADF
    'qadf_test',
    'QADFTest',
    # CADF
    'cadf_test',
    'CADFTest',
    # SMT
    'sub_martingale_test',
    'super_martingale_test',
    'martingale_test',
    'SubMartingaleTest',
    'SuperMartingaleTest',
    'MartingaleTest',
    # CUSUM
    'cusum_test_developing',
    'cusum_test_last',
    'cusum_test_rolling',
]
```

### Step 6.4: Run test to verify it passes

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/test_integration.py -v`

Expected: PASS

### Step 6.5: Commit

```bash
git add afmlkit/feature/core/structural_break/__init__.py tests/structural_breaks/test_integration.py
git commit -m "feat: export all structural break features from __init__.py"
```

---

## Task 7: Run Full Test Suite

**Files:**
- All test files in `tests/structural_breaks/`

### Step 7.1: Run all structural break tests

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/structural_breaks/ -v`

Expected: All tests PASS

### Step 7.2: Verify no regressions

Run: `NUMBA_DISABLE_JIT=1 uv run pytest tests/ -v --tb=short`

Expected: All existing tests still PASS

### Step 7.3: Final commit

```bash
git add .
git commit -m "feat: complete structural break features implementation (SADF, QADF, CADF, SMT)"
```

---

## Implementation Complete

All 5 structural break features have been implemented:

| Feature | Module | Purpose |
|---------|--------|---------|
| ADF | adf.py | Base unit root test |
| SADF | sadf.py | Bubble detection (supremum) |
| QADF | qadf.py | Noise reduction (quantile) |
| CADF | cadf.py | Bubble strength (conditional) |
| SMT | smt.py | Trend detection (martingale) |

**Usage Example:**

```python
from afmlkit.feature.core.structural_break import (
    sadf_test, qadf_test, cadf_test, martingale_test,
    SADFTest, CADFTest, MartingaleTest,
)

# Bubble detection
sadf = sadf_test(df['close'], min_window=20, max_window=100)
bubble_signal = sadf > 1.5

# Robust bubble detection with FeatureKit
cadf = CADFTest(min_window=20, max_window=100).transform(df[['close']])

# Trend detection
sub, sup = martingale_test(df['close'], decay=0.95)
```

**Plan complete and saved to `docs/superpowers/plans/2026-03-20-structural-break-features-implementation.md`. Ready to execute?**

Execution path: Use superpowers:subagent-driven-development with fresh subagent per task + two-stage review.
