## 1. Setup and Infrastructure

- [x] 1.1 Create `src/afml/visualization.py` module.
- [x] 1.2 Update project dependencies with `matplotlib`, `seaborn`, and `scikit-learn` (for metrics plotting).

## 2. Core Visualization Logic

- [x] 2.1 Implement `plot_bar_stats(bars_df, output_path)`: Bar count/size distribution.
- [x] 2.2 Implement `plot_triple_barrier_sample(price_df, labeled_df, output_path)`: Visualizes TP/SL barriers on raw data points.
- [x] 2.3 Implement `plot_label_distribution(labeled_df, output_path)`: Bar chart of label counts.
- [x] 2.4 Implement `plot_feature_heatmap(features_df, output_path)`: Correlation matrix of technical features.
- [x] 2.5 Implement `plot_stationarity_search(d_values, p_values, output_path)`: ADF p-value curve vs FracDiff d.
- [x] 2.6 Implement `plot_cv_timeline(splits, n_samples, output_path)`: Timeline showing training/test intervals with purge/embargo regions.
- [x] 2.7 Implement `plot_meta_performance(y_actual, y_predicted, output_path)`: Confusion matrix and ROC/PR curves.
- [x] 2.8 Implement `plot_equity_curve(strategy_returns, output_path)`: Cumulative returns and drawdown.

## 3. Pipeline Integration

- [x] 3.1 Update `src/afml_polars_pipeline.py` to instantiate `AFMLVisualizer`.
- [x] 3.2 Add visualization hooks to each step of the pipeline.
- [x] 3.3 Implement `--visualize` CLI flag and `VISUAL_ANALYSIS_DIR` configuration.

## 4. Verification and Documentation

- [x] 4.1 Run end-to-end pipeline with `T9999.CCFX-2020-1-1-To-2026-02-11-1m.csv`.
- [x] 4.2 Validate that plots in `visual_analysis/` are correctly formatted and saved.
- [x] 4.3 Update `README.md` to mention the new visualization capabilities.
