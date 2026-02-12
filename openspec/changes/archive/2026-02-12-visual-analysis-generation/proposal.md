## Why

Implementing a rigorous AFML pipeline is essential, but understanding and verifying each step visually is critical for research and debugging. Currently, the pipeline provides numeric outputs and logs, but lacks intuitive visual validation (e.g., bar distributions, label balances, feature stationarity, cross-validation splits).

## What Changes

A new visualization module will be added to the project. This module will provide functions to generate plots for each step of the AFML pipeline. The `src/afml_polars_pipeline.py` will be updated to optionally (or by default) save these visualizations to a designated `visual_analysis` directory.

## Capabilities

### New Capabilities
- `visual-analysis`: Provides a suite of visualization tools for bars, labeling, features, weights, and strategy verification.

### Modified Capabilities
- `afml-pipeline`: Integrated support for visual analysis generation during pipeline execution.

## Impact

- `src/afml/visualization.py`: New module for all plotting logic.
- `src/afml_polars_pipeline.py`: Updated to call visualization functions.
- `visual_analysis/`: New directory for storing generated plots.
