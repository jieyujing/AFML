# OOP Refactoring Implementation Tasks

## 1. Project Structure Setup

- [ ] 1.1 Create `src/afml/` directory structure
- [ ] 1.2 Create `src/afml/__init__.py` with module exports
- [ ] 1.3 Create `src/afml/base.py` with abstract base class
- [ ] 1.4 Create `config/processor_defaults.yaml` configuration file

## 2. DollarBarsProcessor Implementation

- [ ] 2.1 Implement `DollarBarsProcessor` class in `src/afml/dollar_bars.py`
- [ ] 2.2 Add `__init__` with configurable parameters
- [ ] 2.3 Implement `fit()` method for threshold calculation
- [ ] 2.4 Implement `transform()` for fixed dollar bars
- [ ] 2.5 Implement `fit_transform()` sklearn-compatible method
- [ ] 2.6 Write unit tests for DollarBarsProcessor

## 3. TripleBarrierLabeler Implementation

- [ ] 3.1 Implement `TripleBarrierLabeler` class in `src/afml/labeling.py`
- [ ] 3.2 Add `__init__` with barrier configuration
- [ ] 3.3 Implement `fit()` for volatility calculation
- [ ] 3.4 Implement `label()` for triple barrier application
- [ ] 3.5 Implement CUSUM filter integration
- [ ] 3.6 Write unit tests for TripleBarrierLabeler

## 4. FeatureEngineer Implementation

- [ ] 4.1 Implement `FeatureEngineer` class in `src/afml/features.py`
- [ ] 4.2 Add `__init__` with window parameters
- [ ] 4.3 Implement Alpha158 feature generation
- [ ] 4.4 Implement FFD feature generation
- [ ] 4.5 Implement market regime features
- [ ] 4.6 Implement feature selection based on importance
- [ ] 4.7 Write unit tests for FeatureEngineer

## 5. SampleWeightCalculator Implementation

- [ ] 5.1 Implement `SampleWeightCalculator` class in `src/afml/sample_weights.py`
- [ ] 5.2 Add `__init__` with decay parameter
- [ ] 5.3 Implement `fit()` for concurrency calculation
- [ ] 5.4 Implement `transform()` for weight generation
- [ ] 5.5 Write unit tests for SampleWeightCalculator

## 6. BetSizer Implementation

- [ ] 6.1 Implement `BetSizer` class in `src/afml/bet_sizing.py`
- [ ] 6.2 Add `__init__` with step size parameter
- [ ] 6.3 Implement `calculate()` for bet sizing
- [ ] 6.4 Implement active signal averaging
- [ ] 6.5 Write unit tests for BetSizer

## 7. MetaLabelingPipeline Implementation

- [ ] 7.1 Implement `MetaLabelingPipeline` class in `src/afml/meta_labeling.py`
- [ ] 7.2 Integrate with PurgedKFoldCV
- [ ] 7.3 Implement primary model training with PurgedKFold
- [ ] 7.4 Implement meta-label generation
- [ ] 7.5 Implement secondary model training
- [ ] 7.6 Implement prediction filtering
- [ ] 7.7 Write unit tests for MetaLabelingPipeline

## 8. CLI Integration and Migration

- [ ] 8.1 Update `src/process_bars.py` to use DollarBarsProcessor
- [ ] 8.2 Update `src/labeling.py` to use TripleBarrierLabeler
- [ ] 8.3 Update `src/features.py` to use FeatureEngineer
- [ ] 8.4 Update `src/sample_weights.py` to use SampleWeightCalculator
- [ ] 8.5 Update `src/bet_sizing.py` to use BetSizer
- [ ] 8.6 Update `src/meta_labeling.py` to use MetaLabelingPipeline

## 9. Testing and Verification

- [ ] 9.1 Verify output matches original implementation
- [ ] 9.2 Run full pipeline with sample data
- [ ] 9.3 Validate all unit tests pass
- [ ] 9.4 Run ruff linter check
- [ ] 9.5 Update AGENTS.md with new module documentation
