# AGENTS.md - AFML Project Guidelines

This file provides guidelines for AI coding agents working on the AFML (Advances in Financial Machine Learning) codebase.

## Project Overview

AFML implements quantitative finance algorithms based on Marcos López de Prado's book. The project uses Python with pandas, numpy, scikit-learn, and visualization libraries for financial ML research.

## Build, Lint, and Test Commands

### Primary Tool: uv

All commands should use `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Add dependencies
uv add <package>
uv add --dev <package>

# Run Python scripts
uv run python src/<module>.py

# Run ruff linter
uv run ruff check              # Check for issues
uv run ruff check --fix        # Auto-fix issues
uv run ruff check src/ tests/  # Check specific directories

# Run tests
uv run pytest                  # Run all tests
uv run pytest -v               # Verbose output
uv run pytest tests/           # Run specific test directory
uv run pytest -k <pattern>     # Run tests matching pattern
uv run pytest --cov            # With coverage
```

## Code Style Guidelines

### Imports

- Use standard library imports first, then third-party, then local
- Group imports by type (stdlib, pandas/numpy, sklearn, local)
- Example:

```python
import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

import sys
import os

from .local_module import helper_function
```

### Type Hints

- Use type hints for function signatures
- Prefer Optional[X] over Union[X, None]
- Use specific types where possible (List, Dict, Set, Tuple)
- Example:

```python
def process_data(
    df: pd.DataFrame,
    threshold: float = 0.5,
    columns: Optional[List[str]] = None,
) -> pd.Series:
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `feature_data`, `num_samples` |
| Functions | snake_case | `calculate_volatility()`, `get_events()` |
| Classes | PascalCase | `PurgedKFold`, `MetaLabeler` |
| Constants | UPPER_SNAKE_CASE | `DEFAULT_THRESHOLD`, `MAX_DEPTH` |
| Private methods | _snake_case | `_validate_input()`, `_compute_weights` |

### Function Organization

- Functions should have docstrings with Args and Returns sections
- Keep functions focused (single responsibility)
- Maximum ~100 lines per function; break down complex logic
- Use early returns for validation/edge cases
- Example structure:

```python
def main_function(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Brief description of what the function does.

    Args:
        data: Input DataFrame containing price bars
        params: Dictionary of configuration parameters

    Returns:
        DataFrame with computed features/labels

    Raises:
        ValueError: If required columns are missing
    """
    # Validation
    required_cols = ['close', 'volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")

    # Main logic with helper functions
    result = _compute_core_logic(data, params)
    result = _apply_transformations(result)

    return result
```

### Error Handling

- Use specific exception types (ValueError, KeyError, FileNotFoundError)
- Provide descriptive error messages with context
- Wrap I/O and external calls in try/except blocks
- Log errors before re-raising or returning defaults
- Example:

```python
try:
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
except FileNotFoundError:
    logger.error(f"Data file not found: {filepath}")
    raise
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise ValueError(f"Failed to load {filepath}") from e
```

### Logging and Output

- Use print() statements for script output (consistent with existing codebase)
- Prefix log messages with step numbers: `print("1. Loading data...")`
- Use consistent formatting for statistics output
- Save outputs to: `visual_analysis/`, `data/`, or root directory

### File Structure

```
AFML/
├── src/                    # Main source code
│   ├── *.py               # Core modules
│   └── afml/              # Submodules (dataengineering, features, etc.)
├── tests/                 # Test files (pytest)
├── visual_analysis/       # Output visualizations
├── data/                  # Data files
├── config/                # Configuration files
├── pyproject.toml         # Project configuration
└── AGENTS.md             # This file
```

## Module Dependencies

### Legacy Modules (src/)
These modules provide functional APIs for the pipeline:

- `process_bars.py`: Dollar Bars generation (functional style)
- `labeling.py`: Triple barrier labeling (functional style)
- `features.py`: Feature engineering (functional style)
- `sample_weights.py`: Uniqueness/decay weights (functional style)
- `cv_setup.py`: Cross-validation (functional style)
- `meta_labeling.py`: Two-layer ML (functional style)
- `bet_sizing.py`: Position sizing (utility module)

### OO Module (src/afml/)
New object-oriented API with sklearn-compatible interfaces:

```python
from afml import (
    DollarBarsProcessor,    # Generate dollar bars
    TripleBarrierLabeler,   # Apply triple barrier labeling
    FeatureEngineer,       # Generate features
    SampleWeightCalculator,# Calculate sample weights
    BetSizer,              # Calculate bet sizes
    MetaLabelingPipeline,  # Meta-labeling workflow
    PurgedKFoldCV,         # Cross-validation
)
```

All processors follow sklearn conventions:
- `fit(X)` - Fit the processor
- `transform(X)` - Transform data
- `fit_transform(X)` - Fit and transform in one step

Example usage:
```python
# OO-style pipeline
processor = DollarBarsProcessor(daily_target=4)
df_bars = processor.fit_transform(df)

labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
labeler.fit(df_bars["close"])
events = labeler.label(df_bars["close"], cusum_events)

engineer = FeatureEngineer(windows=[5, 10, 20, 30, 50])
features = engineer.fit_transform(df_bars)

weights = SampleWeightCalculator(decay=0.9).fit_transform(events, labels)
```

### Module Relationships

```
src/process_bars.py ─────────────────────────────────────► src/afml/dollar_bars.py
src/labeling.py ─────────────────────────────────────────► src/afml/labeling.py
src/features.py ─────────────────────────────────────────► src/afml/features.py
src/sample_weights.py ────────────────────────────────────► src/afml/sample_weights.py
src/cv_setup.py ─────────────────────────────────────────► src/afml/cv.py
src/meta_labeling.py ────────────────────────────────────► src/afml/meta_labeling.py
src/bet_sizing.py ───────────────────────────────────────► src/afml/bet_sizing.py
```

## Testing Guidelines

- Place tests in `tests/` directory
- Use pytest framework
- Test file naming: `test_<module>.py`
- Test function naming: `test_<function>_<scenario>`
- Mock I/O operations where possible
- Run `uv run pytest` before committing

## Financial ML Specifics

### Label Values

- Labels follow convention: `-1` (loss), `0` (neutral), `1` (profit)
- Primary model predicts direction; meta-model filters predictions
- Bet sizing uses probability distributions (CDF-based)

### Key Concepts

- **Triple Barrier**: Vertical (time), Horizontal TP/SL limits
- **Purged K-Fold**: CV that prevents label leakage
- **Sample Weights**: Uniqueness (no look-ahead), Time Decay
- **Meta-Labeling**: Secondary model filters primary predictions

## Common Workflows

### Running a Complete Pipeline

```bash
# 1. Process bars
uv run python src/process_bars.py

# 2. Generate labels
uv run python src/labeling.py

# 3. Create features
uv run python src/features.py

# 4. Compute sample weights
uv run python src/sample_weights.py

# 5. Run meta-labeling
uv run python src/meta_labeling.py
```

### Adding New Features

1. Add function to appropriate module in `src/`
2. Add type hints and docstring
3. Add unit tests in `tests/`
4. Run `uv run ruff check` for linting
5. Test with `uv run pytest`

## Important Notes

- **Memory Preservation**: Use FFD (Fractionally Differenced) instead of ROC when preserving price memory is needed
- **Look-Ahead Bias**: Never use future information in features/labels
- **Concurrency**: Account for overlapping samples using purged CV and uniqueness weights
- **Data Files**: Generated files are gitignored (.csv, .png, .json, etc.)

## File Patterns

- **Generated outputs**: `*.csv`, `*.png`, `*.json`, `*.pkl` (gitignored)
- **Configuration**: `pyproject.toml` (ruff, tool settings)
- **Scripts**: Run via `uv run python src/<script>.py`
