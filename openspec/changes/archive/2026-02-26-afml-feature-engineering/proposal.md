## Why

In the AFML (Advances in Financial Machine Learning) framework, after generating Dollar Bars, sampling events via the CUSUM filter, and assigning labels with sample weights via the Triple Barrier Method (TBM), the next critical step is Feature Engineering. This step is necessary to transform raw market data (prices, volumes, bar structures) into meaningful, predictive variables (features) that machine learning models can learn from. The core challenge this solves is resolving the dilemma between "stationarity" (required by ML algorithms) and "memory" (required for predictive power) using techniques like Fractional Differentiation.

## What Changes

- Implement core feature extraction pipelines utilizing the `afmlkit.feature.kit.FeatureKit`.
- Add Fractional Differentiation (frac-diff) processing for price series to achieve stationarity while retaining long-term memory.
- Add multi-window rolling volatility features.
- Add basic momentum and structural features (e.g., Log Returns, moving average crossovers, RSI) adapted for Dollar Bars.
- Create an alignment script/pipeline to precisely match the computed features with the TBM label timestamps (`cusum_sampled_bars.csv`) without introducing lookahead bias.

## Capabilities

### New Capabilities
- `fractional-differentiation`: Transformation of non-stationary time series into stationary series while preserving memory, including routines to find the optimal differencing order $d$.
- `volatility-features`: Calculation of exponentially weighted and rolling volatility metrics across various observation windows.
- `momentum-features`: Standard technical and structural momentum indicators computed on event-driven bars.
- `feature-alignment`: A pipeline component responsible for joining the primary feature matrix $X$ with the label/weight matrix $y$ accurately.

### Modified Capabilities
- 

## Impact

- **Workflows/Scripts**: Introduces `scripts/feature_engineering.py` as the next stage in the AFML pipeline.
- **Data Output**: Will produce a final modeled dataset (e.g., `outputs/feature_matrix.csv`) containing $X$ (features) aligned with $y$ (labels) and $w$ (weights).
- **Core Kit Utilization**: Heavily leverages `afmlkit.feature.kit` and related core modules, bringing them into the active trading pipeline.
