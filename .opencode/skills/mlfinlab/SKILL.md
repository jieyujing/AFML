---
name: mlfinlab
description: Expert knowledge of the Machine Learning Financial Laboratory (MLFinLab) library for advanced financial machine learning. Use when (1) Implementing financial data structures (dollar/volume/tick/imbalance/run bars), (2) Using financial machine learning algorithms (labeling, sample weights, cross-validation), (3) Building portfolio optimization or bet sizing strategies, (4) Working with microstructural features or structural breaks, or (5) Understanding MLFinLab's design patterns and API.
---

# MLFinLab - Machine Learning Financial Laboratory

MLFinLab implements advanced financial machine learning algorithms from leading research and industry practices. This skill provides comprehensive understanding of the library's architecture, APIs, and usage patterns.

## Core Modules

### Data Structures
Information-driven bars for financial data:
- **Standard bars**: Time, tick, volume, dollar bars
- **Imbalance bars**: Volume/tick/dollar imbalance bars (EMA/Const)
- **Run bars**: Volume/tick/dollar run bars (EMA/Const)

See `references/api_reference/` for complete API documentation.

### Labeling Methods
Labels for supervised learning:
- **Fixed horizon labeling**: Classify price movements over fixed timeframes
- **Triple barrier method**: Path-dependent labeling with profit/stop-loss barriers
- **Trend-scanning**: Statistical tests for trend detection

### Cross-Validation
Time-series aware splitting:
- **PurgedKFold**: Removes test data leakage via information embargo
- **CombinatorialPurgedKFold**: Generates multiple train/test paths
- Uses embargo periods and purging windows

### Portfolio Optimization & Bet Sizing
Asset allocation strategies:
- **Mean-variance optimization**: Modern portfolio theory implementations
- **Hierarchical Risk Parity (HRP)**: Machine learning-based allocation
- **Kelly criterion**: Optimal bet sizing
- **EF3M**: Event-driven bet sizing with mixture models

### Feature Engineering
- **Microstructural features**: VPIN, Kyle lambda, Amihud lambda, Hasbrouck lambda
- **Fractional differentiation**: Stationarity while preserving memory  
- **Structural breaks**: CUSUM, explosion tests, Chow-type tests

## Design Patterns

MLFinLab uses clean architectural patterns:

- **Strategy Pattern** (23 instances): Labeling methods, weighting schemes, bar types
- **Factory Pattern** (20 instances): Data structure generation, feature creation
- **Template Method**: Base classes with customizable implementations
- **Adapter**: Scikit-learn compatible interfaces

See `references/patterns/detected_patterns.json` for complete pattern analysis.

## Usage Examples

High-quality usage examples are available in `references/test_examples/test_examples.md` (614 examples). These show:

- Instantiation patterns with real parameters
- Method calls with expected outputs
- Common workflows and configurations
- Test data setup patterns

**Example**: Creating dollar bars
```python
from mlfinlab.data_structures import get_dollar_bars

# From CSV file
dollar_bars = get_dollar_bars(
    file_path='tick_data.csv',
    threshold=100000,  # $100k threshold
    batch_size=10000
)

# From DataFrame
dollar_bars = get_dollar_bars(
    file_path_or_df=tick_df,
    threshold=100000
)
```

## Reference Materials

Access these resources as needed for detailed information:

- **API Reference** (`references/api_reference/`): 147 API documentation files covering all modules
- **Test Examples** (`references/test_examples/test_examples.md`): 614 real usage examples from test suite
- **Design Patterns** (`references/patterns/detected_patterns.json`): Comprehensive pattern analysis with locations and evidence

## Quick Tips

1. **Data structures**: Start with dollar bars for financial data - they're information-time sampled
2. **Labeling**: Use triple barrier method for realistic ML labels with profit targets and stop losses
3. **Cross-validation**: Always use PurgedKFold to prevent lookahead bias in time series
4. **Sample weights**: Apply time decay and uniqueness weights for better model training
5. **Architecture**: MLFinLab follows scikit-learn conventions - use `.fit()`, `.predict()` patterns
