# AGENTS Instructions

This document provides essential guidance for AI coding agents working in the FinMLKit repository.

---

## Build / Install / Lint / Test Commands

### Installation

```bash
# Install in editable mode with dev dependencies (REQUIRED before any work)
pip install -e .[dev]
```

### Testing

```bash
# Run full test suite (JIT disabled - recommended for CI/reliability)
NUMBA_DISABLE_JIT=1 pytest -q

# Run full test suite with coverage
NUMBA_DISABLE_JIT=1 pytest tests/ --cov=finmlkit --cov-report=xml --cov-report=term -v

# Run a SINGLE test file
pytest tests/bars/test_comp_ohlcv.py -v

# Run a SINGLE test function
pytest tests/bars/test_comp_ohlcv.py::test_comp_ohlcv -v

# Run all tests in a directory
pytest tests/features/ -v

# Helper scripts (create fresh venv and run tests)
./local_test.sh          # JIT enabled
./local_test_nojit.sh    # JIT disabled (CI-style)
```

### Linting / Formatting

```bash
# Flake8 linting
flake8 finmlkit/

# Black formatting (check)
black --check finmlkit/

# Black formatting (apply)
black finmlkit/
```

---

## Numba JIT Compilation Notes

**CRITICAL**: This codebase uses Numba for high-performance computation. Testing Numba functions requires special handling.

### JIT-Disabled Testing (Recommended)

- Set `NUMBA_DISABLE_JIT=1` environment variable before running tests
- CI runs with JIT disabled (`numba` is not compatible with mass testing)
- Always test with JIT disabled first for debugging

### JIT-Enabled Testing

- Test with JIT enabled locally to ensure production behavior
- First run includes compilation overhead; subsequent runs are fast
- Use `n_runs=11` pattern and exclude first run for benchmarks

### In-File JIT Disabling (Debug Only)

```python
import os
os.environ['NUMBA_DISABLE_JIT'] = "1"  # Must be before any numba imports
from finmlkit.bar.base import comp_bar_ohlcv
```

**IMPORTANT**: Remove/comment out JIT-disabling code before committing.

---

## Code Style Guidelines

### General Principles

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use descriptive variable names; avoid cryptic abbreviations
- Keep functions focused; prefer small, composable units
- Leave the working tree clean before creating PRs

### Imports

Order imports in three groups, separated by blank lines:

1. **Standard library** (alphabetical)
2. **Third-party** (alphabetical: `numpy`, `numba`, `pandas`, etc.)
3. **Local imports** (relative or absolute)

```python
# Standard library
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Sequence

# Third-party
import numpy as np
import pandas as pd
from numba import njit, prange
from numpy.typing import NDArray

# Local imports
from .data_model import TradesData
from finmlkit.utils.log import get_logger
```

### Type Hints

- Use explicit type hints for function signatures
- Use `numpy.typing.NDArray` for NumPy arrays with dtype specification
- Use `Optional[T]` for optional parameters
- Use `Union[A, B]` for multiple accepted types

```python
from typing import Tuple, Optional, Union, Sequence
from numpy.typing import NDArray
import numpy as np

def process_data(
    prices: NDArray[np.float64],
    volumes: NDArray[np.float64],
    threshold: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    ...
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Functions | `snake_case` | `comp_bar_ohlcv`, `build_ohlcv` |
| Classes | `PascalCase` | `BarBuilderBase`, `SISOTransform` |
| Private functions | `_leading_underscore` | `_time_bar_indexer`, `_validate_input` |
| Constants | `UPPER_SNAKE_CASE` | `DEFAULT_THRESHOLD` |
| Instance variables | `snake_case` | `self.trades_df`, `self._close_ts` |

### Numba Function Patterns

```python
from numba import njit, prange
from numba.typed import List as NumbaList

# Standard Numba function
@njit(nogil=True)
def _process_array(data: NDArray[np.float64]) -> NDArray[np.float64]:
    ...

# Parallel Numba function (use prange, not range)
@njit(nogil=True, parallel=True)
def _process_parallel(data: NDArray[np.float64]) -> NDArray[np.float64]:
    n = len(data)
    result = np.zeros(n, dtype=np.float64)
    for i in prange(n):  # prange enables parallelization
        result[i] = data[i] * 2.0
    return result
```

---

## Docstring Style

Use **reStructuredText** format for Sphinx documentation compatibility.

### Function Docstrings

```python
def compute_returns(prices: NDArray[np.float64], log: bool = True) -> NDArray[np.float64]:
    """
    Compute returns from a price series.

    :param prices: Array of price values.
    :param log: If True, compute log returns; otherwise simple returns.
    :returns: Array of return values.
    :raises ValueError: If prices array has fewer than 2 elements.

    .. note::
        The first element of the output will be NaN as there is no prior price.

    Example:
        >>> prices = np.array([100.0, 101.0, 99.0])
        >>> compute_returns(prices, log=True)
        array([nan, 0.0099..., -0.0202...])
    """
    ...
```

### Class Docstrings

```python
class BarBuilderBase(ABC):
    r"""Abstract base class for building bars from raw trades data.

    This class serves as a template for subclasses that implement specific
    bar sampling strategies (time, tick, volume, etc.).

    Key functionalities include:

    - :meth:`build_ohlcv`: Computes OHLCV, VWAP, trade count, and median trade size.
    - :meth:`build_directional_features`: Calculates buy/sell splits.

    Args:
        trades (TradesData): Object containing raw trades DataFrame.

    Raises:
        ValueError: If required columns are missing from trades data.

    See Also:
        :class:`finmlkit.bar.kit.TimeBarKit`: Concrete subclass for time bars.
    """
```

---

## Error Handling

### Input Validation

Validate inputs at function start with descriptive error messages:

```python
def process_trades(prices: NDArray, volumes: NDArray) -> NDArray:
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes arrays must have the same length.")
    if len(prices) < 2:
        raise ValueError("At least 2 prices are required.")
    ...
```

### Numba Error Handling

Numba functions have limited exception support. Validate outside when possible:

```python
def public_api(prices, volumes):
    # Validate BEFORE calling Numba function
    if len(prices) != len(volumes):
        raise ValueError("Array lengths must match.")
    return _numba_implementation(prices, volumes)

@njit(nogil=True)
def _numba_implementation(prices, volumes):
    # Minimal validation inside Numba
    ...
```

---

## Project Structure

```
finmlkit/
├── bar/           # Bar construction (time, tick, volume, dollar bars)
│   ├── base.py    # Core bar building functions and BarBuilderBase
│   ├── logic.py   # Bar indexer functions (Numba-accelerated)
│   ├── data_model.py  # Data structures (TradesData, FootprintData)
│   └── kit.py     # Concrete implementations (TimeBarKit, etc.)
├── feature/       # Feature engineering framework
│   ├── base.py    # Transform base classes (SISO, MISO, SIMO, MIMO)
│   ├── core/      # Core feature implementations
│   └── kit.py     # FeatureKit for batch computation
├── label/         # Labeling methods (Triple Barrier Method, etc.)
├── sampling/      # Sampling filters (CUSUM filter)
├── utils/         # Utilities (logging, helpers)
└── _version.py    # Version string

tests/
├── bars/          # Tests for bar module
├── features/      # Tests for feature module
├── labels/        # Tests for label module
├── sampling/      # Tests for sampling module
├── utils.py       # Test utilities
└── README.md      # Testing guidance
```

---

## Key Conventions

### DataFrame Index

- Time series DataFrames use **datetime index**
- Timestamps stored as nanoseconds (`int64`) internally
- Convert with `pd.to_datetime(timestamps, unit='ns')`

### Array Types

- Use `np.float64` for prices, volumes, returns
- Use `np.int64` for indices, counts
- Use `np.int8` for categorical/enum values (e.g., trade side: -1, 1)
- Use `np.float32` for memory-constrained aggregations

### Logger Usage

```python
from finmlkit.utils.log import get_logger

logger = get_logger(__name__)

logger.info("Processing started...")
logger.warning("Unusual condition detected")
logger.error("Failed to process data")
```

---

## Search Tools

Prefer `rg` (ripgrep) for searching. If missing, install via:
```bash
sudo apt-get update && sudo apt-get install -y ripgrep
```
Or fallback to: `git ls-files | xargs grep -n`

---

## Git Workflow

- Keep commits focused and atomic
- Write descriptive commit messages
- Ensure working tree is clean before PR creation
- Run tests locally before pushing: `NUMBA_DISABLE_JIT=1 pytest -q`

---

## References

- Primary methodology source: **Advances in Financial Machine Learning** by Marcos López de Prado
- Documentation: [finmlkit.readthedocs.io](https://finmlkit.readthedocs.io)
- Detailed testing guide: [tests/README.md](tests/README.md)
- Contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
