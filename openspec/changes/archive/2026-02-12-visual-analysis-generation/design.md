## Context

The current `src/afml_polars_pipeline.py` executes a full AFML workflow but only provides numeric feedback and CSV artifacts. In quantitative research, visual confirmation of intermediate steps (like how triple barriers are triggered or how CV splits are purged) is essential for validating the logic and identifying potential data issues.

## Goals / Non-Goals

**Goals:**
- Create a reusable `afml.visualization` module.
- Generate high-quality plots for every major step of the pipeline.
- Save plots automatically to `visual_analysis/` in PNG format.
- Ensure visualizations work seamlessly with Polars DataFrames.

**Non-Goals:**
- Building an interactive web dashboard (focus on static artifacts for reporting).
- Real-time plotting during execution (plots should be saved at the end of each step).
- Heavyweight dependencies beyond `matplotlib` and `seaborn`.

## Decisions

- **Library Selection**: Use `matplotlib` and `seaborn` for robust, publication-quality static plots.
- **Directory Structure**: All plots saved to `visual_analysis/` with names corresponding to steps (e.g., `01_dollar_bars.png`, `02_label_distribution.png`).
- **Encapsulation**: Plotting logic will be encapsulated in classes/functions within `src/afml/visualization.py` to keep the main pipeline script clean.
- **Data Handling**: Plots will handle Polars DataFrames directly, converting to NumPy/Pandas only when strictly necessary for specific plotting functions.

## Risks / Trade-offs

- **Memory Usage**: Plotting large datasets can be memory-intensive. I will implement sampling logic for time-series plots (e.g., plotting only the first/last N events).
- **Dependency Management**: Adding `matplotlib` and `seaborn` as dependencies. These are already common in quant environments.
