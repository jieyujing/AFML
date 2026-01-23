# Quantitative R&D Progress Tracking

## Project Status
- **Current Phase**: Phase 5 - Evaluation & Refinement
- **Last Updated**: 2026-01-23
- **Dataset**: `IF9999.CCFX-2020-1-1-To-2026-01-22-1m.csv`
- **Model Performance**:
  - Meta-Model Purged CV AUC: **0.5729** (Previous Best)
  - Walk-Forward Sharpe Ratio: **0.655** (Confirmed)
  - Win Rate: **21.80%** (Low)
  - Probabilistic Sharpe Ratio (PSR): **1.000** (Significant)
  - Deflated Sharpe Ratio (DSR): **0.000** (Not Significant)

## Recent Achievements
- [x] **Primary Strategy**: Trained directional model (RF Momentum) -> Generated `predicted_side.csv`.
- [x] **Meta-Labeling**: Applied Triple Barrier (1:2 Risk/Reward) on top of Primary signals.
- [x] **Feature Engineering V2**: Generated 230 features (Microstructure, Signal, FFD) for Meta-Model.
- [x] **Meta-Modeling**: Validated Random Forest Meta-Model with Purged K-Fold CV (Meta Mode Enabled).
- [x] **Position Sizing**: Implemented Probabilistic Bet Sizing (size ~ 2*prob - 1) with Concurrency Adjustment.
- [x] **Backtesting**: Completed Walk-Forward Validation (Expanding Window). Result: Sharpe 0.655.

## Current Task: Strategy Refinement
- **Objective**: Improve Win Rate (>30%) and Sharpe Ratio (>1.0).
- **Status**: Strategy is profitable (SR > 0) but suffers from low win rate.
- **Diagnosis**: High positive skewness (2.53) indicates option-like payoff, but frequent small losses reduce consistency.

## Next Steps
1. **Threshold Optimization**: Increase Meta-Labeling probability threshold (e.g., 0.5 -> 0.6) to filter low-confidence bets.
2. **Feature Selection**: Investigate top features from MDI to remove noise.
3. **Stop Loss/Profit Taking**: Optimize Triple Barrier width (currently 1.0 std).