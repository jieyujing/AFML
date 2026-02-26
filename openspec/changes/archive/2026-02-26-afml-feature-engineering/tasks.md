## 1. Core Feature Extensions

- [x] 1.1 Implement an automatic stationarity test routine (using Augmented Dickey-Fuller) to iterate and find the optimal differencing order $d \in [0, 1]$.
- [x] 1.2 Verify `afmlkit.feature.core.frac_diff` FFD implementation correctly enforces memory truncation bounds to prevent heavy computational loads.

## 2. Feature Generation Pipeline Setup

- [x] 2.1 Create `scripts/feature_engineering.py` and load the raw continuous Dollar Bars DataFrame (`outputs/dollar_bars/dollar_bars_freq20.csv`).
- [x] 2.2 Compute log returns and use `afmlkit.feature.kit` or discrete utilities to generate multi-window exponentially weighted volatility features (e.g., spans: 10, 50, 100).
- [x] 2.3 Compute momentum and structural metric features (e.g., log price distances to short/long EMAs).
- [x] 2.4 Apply Fractional Differentiation (FFD) with the automatically discovered optimal $d$ to the log price series.

## 3. Causal Alignment & Cleansing

- [x] 3.1 Load the previously generated CUSUM sampled labels and weights DataFrame (`outputs/dollar_bars/cusum_sampled_bars.csv`).
- [x] 3.2 Extract the `DatetimeIndex` from the labels DataFrame and use it to strictly slice matching rows from the continuous features DataFrame.
- [x] 3.3 Join the discrete features matrix with the labels (`bin`, `t1`) and weights (`avg_uniqueness`, `return_attribution`).
- [x] 3.4 Purge (drop) any rows containing `NaN` feature values caused by windowed indicator initializations, ensuring their corresponding labels are safely dropped as well.

## 4. Verification & Output

- [x] 4.1 Log the final optimal $d$ used and print a summary of the fully merged feature/label dataset.
- [x] 4.2 Save the pristine, aligned data matrix to `outputs/dollar_bars/feature_matrix.csv`.
