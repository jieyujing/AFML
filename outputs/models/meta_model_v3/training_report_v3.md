# Meta-Model v3 Training Report

## 1. Data Overview

| Metric | Value |
|--------|-------|
| Samples | 4699 |
| Features | 15 |
| Label Distribution | {0: 2432, 1: 2267} |
| Avg Uniqueness | 0.5000 |

## 2. Feature List (15 features)

1. `log_return`
2. `vol_atr_14`
3. `vol_rel_20`
4. `vol_ewm_100`
5. `vol_parkinson`
6. `ema_diff`
7. `trend_variance_ratio_20`
8. `corr_pv_10`
9. `liq_amihud`
10. `vol_garman_klass`
11. `vol_yang_zhang`
12. `mom_macd`
13. `vol_kyle_lambda`
14. `entropy_shannon`
15. `prob` (Primary Model prediction)

## 3. CV Performance

| Metric | Value | Evaluation |
|--------|-------|------------|
| ROC AUC | 0.5127 | ✗ Need Improvement |
| Accuracy | 0.5165 | ✗ Need Improvement |
| Precision | 0.4974 | - |
| Recall | 0.2073 | - |
| F1 Score | 0.2927 | - |
| Brier Score | 0.2510 | ✗ Need Improvement |

## 4. Full Sample Performance

| Metric | Value |
|--------|-------|
| ROC AUC | 0.5786 |
| Accuracy | 0.5542 |
| Brier Score | 0.2448 |

## 5. Configuration

| Parameter | Value |
|-----------|-------|
| N Estimators | 1000 |
| CV Splits | 5 |
| Embargo | 1.0% |
| Calibration | Isotonic |

## 6. Comparison with v2

| Metric | v2 (old) | v3 (new) | Change |
|--------|----------|----------|--------|
| ROC AUC | 0.5168 | 0.5127 | -0.0041 |
| Brier Score | 0.2524 | 0.2510 | +0.0014 |

## 7. Next Steps

- [ ] Run Bet Size analysis
- [ ] Backtest strategy performance
- [ ] Compare with v2 workflow

