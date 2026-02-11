## Context

The current `afml_polars_pipeline.py` script provides a basic implementation of the AFML workflow using Polars. However, it lacks strict stationarity enforcement (automatic fractional differentiation) and rigorous statistical verification using Deflated Sharpe Ratio (DSR), which are core tenets of the AFML methodology as outlined in the `afml-quant-factory` skill.

## Goals / Non-Goals

**Goals:**
1.  Update the pipeline to enforce stationarity by automatically finding the minimum fractional differentiation `d` such that the p-value < 0.05.
2.  Implement Deflated Sharpe Ratio (DSR) calculation for final strategy verification.
3.  Expose all key parameters via command-line arguments for flexible experimentation.
4.  Ensure the pipeline is modular and reusable.
5.  Generate a comprehensive report including stationarity metrics, label distribution, model performance, and DSR.

**Non-Goals:**
1.  Support for non-CSV data formats.
2.  Real-time trading execution.

## Decisions

1.  **Use Polars**: Leverage `polars` for high-performance data processing.
2.  **Enhance Existing Script**: We will update `src/afml_polars_pipeline.py` rather than creating a new one, as it already has the structure we need. We will add the missing logic.
3.  **DSR Implementation**: Create a new module `src/afml/metrics.py` (or similar) to house the DSR calculation logic if not present.
4.  **Stationarity Logic**: Integrate stationarity checks directly into the feature generation step or as a pre-processing step.

## Risks / Trade-offs

1.  **Performance overhead**: Automatic stationarity checks (iterating `d`) can be computationally expensive. We will mitigate this by using a binary search or efficient grid search for `d`.
2.  **Data Quality**: The pipeline assumes clean tick/bar data. Garbage in, garbage out.
