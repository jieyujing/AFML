# Proposal: Refactor afml_polars_pipeline.py for Modular Step Execution

## Why

The current `afml_polars_pipeline.py` runs all AFML pipeline steps sequentially in a single `run_pipeline()` call. This makes it difficult to:
- Run individual steps (e.g., just generate dollar bars for debugging)
- Experiment with parameters for specific stages without rerunning everything
- Verify visualization outputs at each step independently
- Handle large datasets that may cause memory issues during full pipeline runs

## What Changes

Refactor the pipeline to support:
1. **Modular step execution**: Add `--step` parameter to run individual pipeline stages
2. **Standalone step scripts**: Each step can be executed independently with its own parameters
3. **Enhanced visualization**: Generate verification plots after each step completion
4. **Memory optimization**: Leverage Polars lazy evaluation and streaming for large datasets

## Capabilities

### New Capabilities
- **Step-wise execution**: Run `--step bars`, `--step labels`, `--step features`, etc.
- **Step verification**: Each step outputs visualization to confirm correctness
- **Memory-efficient mode**: Use Polars streaming for large parquet directories

### Modified Capabilities
- **CLI interface**: Add `--step` and `--output` arguments
- **Pipeline functions**: Accept optional `prev_output` for chaining steps

## Impact

- `afml_polars_pipeline.py`: Add `--step` CLI option, refactor main()
- New optional: `src/run_step.py`: Standalone step runner (if needed)
- Each step function gains visualization output capability
