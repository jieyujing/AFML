---
name: afml
description: Implement machine learning algorithms for quantitative finance based on "Advances in Financial Machine Learning"(afml) by Marcos López de Prado. Use when working with: (1) Financial data structures (tick/volume/dollar/imbalance bars), (2) Triple-barrier labeling or meta-labeling, (3) Sample weights for overlapping labels, (4) Purged/embargoed cross-validation, (5) Feature importance (MDI/MDA/SFI), (6) Backtest methodology and overfitting detection (PBO), (7) Hierarchical Risk Parity (HRP) portfolios, (8) Market microstructure analysis (VPIN, order flow), (9) Structural break detection (CUSUM, SADF), (10) Fractionally differentiated features, (11) High-performance computing for finance.
---

# Advances in Financial Machine Learning

Implement quantitative finance ML techniques from Marcos López de Prado's book.

## Core Concepts

### Bar Types (Chapter 2)
Use information-driven bars instead of time bars:
- **Tick bars**: Sample every N ticks
- **Volume bars**: Sample every N volume units  
- **Dollar bars**: Sample every $N traded
- **Imbalance bars**: Sample on signed volume imbalance

See `references/part_1.md` for data structures.

### Triple-Barrier Labeling (Chapter 3)
Label based on first barrier touched:
```python
# Upper barrier: Take-profit (pt * target)
# Lower barrier: Stop-loss (-sl * target)
# Vertical barrier: Max holding period
def get_events(close, t_events, pt_sl, target, min_ret):
    pass  # Returns first barrier touch time and label
```
See `references/3_form_labels_using_the_triple_barrier_method_with_symmetric.md`.

### Sample Weights (Chapter 4)
Account for overlapping labels with uniqueness weighting:
```python
def get_avg_uniqueness(t_in, num_co_events):
    # Uniqueness = 1 / concurrent labels at each time
    wght = pd.Series(index=t_in.index)
    for t_in_i, t_out_i in t_in.items():
        wght.loc[t_in_i] = (1. / num_co_events.loc[t_in_i:t_out_i]).mean()
    return wght
```

### Purged Cross-Validation (Chapter 7)
Prevent leakage from overlapping labels:
- **Purge**: Remove train samples overlapping test period
- **Embargo**: Gap after test period

See `references/5_the_cv_must_be_purged_and_embargoed_for_the_reasons_explained_in.md`.

### Feature Importance (Chapter 8)
| Method | Source | Use Case |
|--------|--------|----------|
| MDI | In-sample impurity | Fast, biased toward high-cardinality |
| MDA | OOS permutation | Robust, detects generalization |
| SFI | Single-feature CV | Detects substitution effects |

### Hierarchical Risk Parity (Chapter 16)
Portfolio construction without matrix inversion:
1. Hierarchical clustering on correlation
2. Quasi-diagonalization (reorder by cluster)
3. Recursive bisection (inverse-variance weights)

### Backtest Overfitting (Chapters 11-14)
- **Deflated Sharpe**: Account for multiple testing
- **PBO (Probability of Backtest Overfitting)**: Estimate IS-optimal underperforming OOS

See `references/1_survivorship_bias_using_as_investment_universe_the_current_one_hence.md`.

## Best Practices

### Data Preparation
1. Never use time bars—use volume/dollar bars
2. Apply CUSUM filter for sampling
3. Fractionally differentiate to preserve memory

### Modeling
1. Use triple-barrier labeling for path dependency
2. Apply meta-labeling to separate signal from sizing
3. Always purge and embargo cross-validation

### Backtesting
1. Deflate Sharpe ratio for multiple testing
2. Compute PBO to estimate overfitting
3. Validate with synthetic data

## Reference Files

### By Topic

**Data Structures & Labeling**:
- `references/part_1.md` - Data analysis fundamentals
- `references/3_form_labels_using_the_triple_barrier_method_with_symmetric.md` - Triple barrier implementation
- `references/1_sample_bars_using_the_cusum_filter_where_y_t_are_absolute_returns_and_h_.md` - CUSUM filtering

**Cross-Validation & Feature Importance**:
- `references/5_the_cv_must_be_purged_and_embargoed_for_the_reasons_explained_in.md` - Purged CV
- `references/1_masking_effects_take_place_when_some_features_are_systematically_ignored.md` - Feature masking
- `references/4_fit_classifiers_on_the.md` - Classifier fitting

**Backtesting**:
- `references/part_3.md` - Backtesting methodology
- `references/part_3_backtesting.md` - Backtesting deep dive
- `references/1_survivorship_bias_using_as_investment_universe_the_current_one_hence.md` - Bias detection
- `references/chapter_15_understanding_strategy_risk.md` - Risk metrics

**High-Performance Computing**:
- `references/1_chapter_20_multiprocessing_and_vectorization.md` - Parallel processing
- `references/4_using_the_optimized_code_what_is_the_problem_dimensionality_that.md` - Optimization

**Bibliography**:
- `references/index.md` - Complete structure
- `references/9_bibliography.md` - Academic references

## Related Libraries

- **mlfinlab**: Python implementation of book techniques
- **sklearn**: ML algorithms
- **pandas/numpy**: Data structures

---

**Source**: "Advances in Financial Machine Learning" by Marcos López de Prado (Wiley, 2018)
