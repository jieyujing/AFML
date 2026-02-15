# Design: Refactor afml_polars_pipeline.py for Modular Step Execution

## Context

The current `afml_polars_pipeline.py` implements a monolithic `run_pipeline()` function that:
1. Loads raw data
2. Generates dollar bars
3. Applies labels
4. Generates features
5. Calculates weights
6. Runs cross-validation
7. Runs meta-labeling
8. Calculates bet sizes
9. Verifies strategy

All steps run sequentially in one call. There's no way to run individual steps for debugging or experimentation.

## Goals / Non-Goals

**Goals:**
- Add `--step` CLI parameter to run individual pipeline stages
- Each step generates its own visualization output for verification
- Support memory-efficient streaming mode for large datasets
- Maintain backward compatibility (default runs all steps)

**Non-Goals:**
- Create separate script files for each step (keep in one file)
- Change the underlying AFML module APIs
- Add new ML models or algorithms

## Decisions

### Decision 1: Step Enum vs String

**Choice:** Use Python `Enum` for step names with validation

**Rationale:** Provides type safety and IDE autocomplete, prevents typos, allows easy iteration for help text.

```python
class PipelineStep(str, Enum):
    LOAD = "load"
    BARS = "bars"
    LABELS = "labels"
    FEATURES = "features"
    WEIGHTS = "weights"
    CV = "cv"
    META = "meta"
    BET = "bet"
    VERIFY = "verify"
    ALL = "all"
```

### Decision 2: Step Execution Pattern

**Choice:** Each step function returns a dict with results and intermediate DataFrames

**Rationale:** Allows later steps to use previous outputs without reloading from disk.

```python
def run_step_bars(df, args) -> StepResult:
    dollar_bars = generate_dollar_bars(...)
    # Generate visualization
    viz.plot_bar_stats(dollar_bars)
    return StepResult(data={"dollar_bars": dollar_bars})
```

### Decision 3: Streaming Mode

**Choice:** Add `--streaming` flag to enable Polars streaming

**Rationale:** Large parquet directories (100M+ rows) can cause OOM. Streaming processes in chunks.

```python
if args.streaming:
    df = pl.scan_parquet(path).sink_parquet(temp_stream)
    # Process from temp stream
```

### Decision 4: Input Auto-Detection

**Choice:** If `--step` is specified but no `--input`, infer from step

**Rationale:** User can run `--step labels` and it auto-loads `data/dollar_bars_polars.csv`

**Files checked for each step:**
- `load`: uses `--input` or default data dir
- `bars`: loads from `--input` or default
- `labels`: loads `data/dollar_bars_polars.csv`
- `features`: loads `data/labeled_polars.csv`
- etc.

---

## Implementation Plan

1. Add `PipelineStep` enum
2. Refactor `main()` to parse `--step` parameter
3. Create step runner function that dispatches to appropriate handler
4. Each handler: load data → process → visualize → return results
5. Add `--streaming` flag implementation
6. Add `--output` flag to override default output paths
