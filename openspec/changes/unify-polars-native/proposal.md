# Proposal: Unify Polars Native

## Why

The AFML project has a fundamental design inconsistency: the codebase was migrated to Polars for performance, but the test suite still uses pandas fixtures and expects pandas return types. This causes 70% of tests to fail with AttributeError and type mismatches. The project cannot claim to be "Polars Native" while the test infrastructure assumes pandas.

## What Changes

1. **Update all test fixtures to return Polars DataFrames** - Convert pandas fixtures in `tests/test_*.py` to return Polars native types
2. **Fix attribute mismatches** - Add missing attributes that tests expect (`ffd_check_stationarity`, `selected_features_`, `concurrency_`, etc.)
3. **Fix return type mismatches** - Ensure methods return correct column names (`sample_weight` vs `weight`)
4. **Fix type conversion bug in labeling.py** - Add pandas-to-Polars conversion in `fit()` methods
5. **Fix unused imports** - Clean up lint warnings across modules

## Capabilities

### New Capabilities

- **polars-native-tests**: Convert test infrastructure to use Polars native types throughout
- **auto-conversion-layer**: Add automatic pandas-to-Polars conversion in processor fit() methods

### Modified Capabilities

- None - this is a test infrastructure change, not a behavioral change

## Impact

### Files Modified

- `tests/test_features.py` - Convert fixtures to return Polars, add expected attributes to FeatureEngineer
- `tests/test_bet_sizing.py` - Convert fixtures, add step_size attribute to BetSizer
- `tests/test_sample_weights.py` - Convert fixtures, add concurrency_/uniqueness_ attributes
- `tests/test_triple_barrier.py` - Add pandas-to-Polars conversion in labeler
- `tests/test_meta_labeling.py` - Add n_splits attribute to pipeline
- `src/afml/labeling.py` - Fix pandas Series handling in fit()
- `src/afml/bet_sizing.py` - Remove unused variables, add step_size attribute
- `src/afml/sample_weights.py` - Rename columns to match test expectations
- `src/afml/features.py` - Add missing attributes
- `src/afml/dollar_bars.py` - Remove unused numpy import
- `src/afml/metrics.py` - Remove unused Optional import
- `src/afml/stationarity.py` - Remove unused Optional import

### Dependencies

- No new dependencies required
- Maintains existing sklearn compatibility
