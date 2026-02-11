# Implementation Tasks: Use Polars Instead of Pandas

## 1. Setup Phase

- [x] 1.1 Add `polars>=1.0.0` dependency to `pyproject.toml`
- [x] 1.2 Create `src/afml/polars/` directory structure
- [x] 1.3 Create `src/afml/polars/__init__.py` with module exports
- [x] 1.4 Create pandas-to-Polars migration documentation

## 2. Core Data Types (Foundation for all modules)

- [x] 2.1 Create `src/afml/polars/dataframe.py` with Polars DataFrame utilities
- [x] 2.2 Create `src/afml/polars/series.py` with Polars Series utilities
- [x] 2.3 Implement `to_polars()` and `to_pandas()` conversion functions
- [x] 2.4 Add type hints for Polars DataFrame and Series

## 3. Polars Dollar Bars Processor

- [x] 3.1 Create `src/afml/polars/dollar_bars.py`
- [x] 3.2 Implement `PolarsDollarBarsProcessor.__init__()` with default/custom params
- [x] 3.3 Implement `fit()` method with threshold calculation
- [x] 3.4 Implement `transform()` method for bar generation
- [x] 3.5 Add LazyFrame support with `lazy` parameter
- [x] 3.6 Add sklearn compatibility (`fit_transform()`)
- [ ] 3.7 Write unit tests in `tests/test_polars_dollar_bars.py`
- [ ] 3.8 Add performance benchmark comparing pandas vs polars

## 4. Polars Triple Barrier Labeler

- [x] 4.1 Create `src/afml/polars/labeling.py`
- [x] 4.2 Implement `PolarsTripleBarrierLabeler.__init__()` with pt_sl and vertical_barrier
- [x] 4.3 Implement `fit()` method for volatility calculation
- [x] 4.4 Implement `label()` method for triple barrier labeling
- [x] 4.5 Implement `get_cusum_events()` with Polars operations
- [x] 4.6 Add LazyFrame support
- [x] 4.7 Write unit tests in `tests/test_polars_labeling.py`
- [x] 4.8 Verify label distribution matches pandas version

## 5. Polars Feature Engineer

- [x] 5.1 Create `src/afml/polars/features.py`
- [x] 5.2 Implement `PolarsFeatureEngineer.__init__()` with windows parameter
- [x] 5.3 Implement `fit()` method for metadata storage
- [x] 5.4 Implement `transform()` with Alpha158 features
- [x] 5.5 Add FFD (Fractionally Differentiated) features
- [x] 5.6 Implement rolling window operations with Polars
- [x] 5.7 Add LazyFrame support
- [x] 5.8 Write unit tests in `tests/test_polars_features.py`
- [x] 5.9 Verify feature values match pandas version

- [x] 6.1 Create `src/afml/polars/sample_weights.py`
- [x] 6.2 Implement `PolarsSampleWeightCalculator.__init__()` with decay parameter
- [x] 6.3 Implement `fit()` method for uniqueness calculation
- [x] 6.4 Implement `transform()` method for weight generation
- [x] 6.5 Implement concurrency calculation with Polars
- [x] 6.6 Add LazyFrame support
- [x] 6.7 Write unit tests in `tests/test_polars_sample_weights.py`
- [x] 6.8 Verify weights match pandas version

- [x] 7.1 Create `src/afml/polars/cv.py`
- [x] 7.2 Implement `PolarsPurgedKFoldCV.__init__()` with n_splits and embargo
- [x] 7.3 Implement `split()` method with purging and embargo
- [x] 7.4 Add Polars DataFrame/Series compatibility
- [x] 7.5 Add sklearn compatibility
- [x] 7.6 Write unit tests in `tests/test_polars_cv.py`
- [x] 7.7 Verify no information leakage between splits

- [x] 8.1 Create `src/afml/polars/meta_labeling.py`
- [x] 8.2 Implement `PolarsMetaLabelingPipeline.__init__()` with model configs
- [x] 8.3 Implement `fit()` method for primary and meta models
- [x] 8.4 Implement `predict()` method for binary predictions
- [x] 8.5 Implement `predict_proba()` for probability estimates
- [x] 8.6 Add performance metrics methods (Sharpe ratio, returns)
- [x] 8.7 Add LazyFrame support
- [x] 8.8 Write unit tests in `tests/test_polars_meta_labeling.py`

- [x] 9.1 Create `src/afml/polars/bet_sizing.py`
- [x] 9.2 Implement `PolarsBetSizer.__init__()` with threshold and quantity
- [x] 9.3 Implement `bet_size_probability()` using EDF
- [x] 9.4 Implement `get_bet_size()` with discretization
- [x] 9.5 Implement `score_accuracy()` and `get_metrics()`
- [x] 9.6 Add LazyFrame support
- [x] 9.7 Write unit tests in `tests/test_polars_bet_sizing.py`

## 10. Integration and Testing

- [x] 10.1 Update `src/afml/__init__.py` to include polars modules
- [x] 10.2 Create `src/afml/polars/__init__.py` with all exports
- [x] 10.3 Create end-to-end integration test in `tests/test_polars_pipeline.py`
- [x] 10.4 Add benchmark script comparing pandas vs polars performance
- [x] 10.5 Update documentation in `README.md` for polars usage

## 11. Performance Optimization

- [x] 11.1 Profile memory usage for each module
- [x] 11.2 Optimize hotspot operations with Polars-specific optimizations
- [x] 11.3 Add connection pooling for repeated operations
- [x] 11.4 Document performance benchmarks in `BENCHMARKS.md`

- [x] 12.1 Run `uv run ruff check src/afml/polars/`
- [x] 12.2 Run `uv run ruff check tests/test_polars_*.py`
- [x] 12.3 Ensure type coverage for all public APIs
- [x] 12.4 Add docstrings following project conventions
