# Tasks: Refactor Pipeline for Modular Step Execution

## 1. CLI Enhancement

- [x] 1.1 Add `PipelineStep` enum with all step names
- [x] 1.2 Add `--step` argument to argparse with choices from enum
- [x] 1.3 Add `--streaming` flag for memory-efficient mode
- [x] 1.4 Add `--output` argument to override default output paths

## 2. Step Runner Functions

- [x] 2.1 Create `run_step_load()` function with visualization
- [x] 2.2 Create `run_step_bars()` function with visualization
- [x] 2.3 Create `run_step_labels()` function with visualization
- [x] 2.4 Create `run_step_features()` function with visualization
- [x] 2.5 Create `run_step_weights()` function with visualization
- [x] 2.6 Create `run_step_cv()` function with visualization
- [x] 2.7 Create `run_step_meta()` function with visualization
- [x] 2.8 Create `run_step_bet()` function with visualization
- [x] 2.9 Create `run_step_verify()` function with visualization

## 3. Main Dispatcher

- [x] 3.1 Refactor `main()` to dispatch to step runner based on `--step`
- [x] 3.2 Add auto-detection of input files based on step
- [x] 3.3 Handle missing dependencies gracefully (e.g., no labels file for features step)

## 4. Memory Optimization

- [x] 4.1 Implement streaming mode for large parquet directories
- [x] 4.2 Add row count estimation before full collection

## 5. Testing

- [x] 5.1 Test `--step bars` runs correctly
- [x] 5.2 Test `--step labels` runs correctly
- [x] 5.3 Test `--step features` runs correctly
- [x] 5.4 Verify visualization files are created
