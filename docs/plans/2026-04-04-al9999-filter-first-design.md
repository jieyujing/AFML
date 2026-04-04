# AL9999 Filter-First Optimized Version Design

**Date:** 2026-04-04  
**Scope:** `strategies/AL9999`  
**Status:** Approved

---

## 1. Background and Goal

Current `AL9999` has usable alpha but does not pass AFML acceptance on the key metric `Combined(OOS) DSR > 0.95`.  
This design defines a medium-scope optimization version with one primary objective:

- Raise `Combined(OOS) DSR` toward `0.95+` under controlled trade shrinkage.

Confirmed constraints:

- Priority: AFML significance first (DSR).
- Allowed trade shrinkage: medium (`15%` to `30%`).
- Avoid full research-stack rebuild in this iteration.

---

## 2. Strategy Direction

Chosen direction: **Filter-First**.

Keep the existing research pipeline structure:

- Dollar Bars -> Feature Engineering -> MA Primary -> TBM -> Meta Filter -> Rolling Backtest

Only optimize decision and execution layers:

1. `Meta Filter` layer (feature scheme + threshold policy)
2. `Execution Guard` layer (reverse/cooldown controls)
3. `Side Governance` layer (short-side fallback switch)

This minimizes blast radius while directly targeting the observed bottlenecks.

---

## 3. Main Design

### 3.1 Meta Scheme Selector

Candidate schemes are restricted to:

- `full`
- `mda_positive` (from `07b` positive MDA clusters)

Selection priority:

1. `Combined(OOS) DSR`
2. `Combined(OOS) Sharpe`
3. `holdout_f1`

Rationale: selection metric must align with strategy objective (OOS robustness), not only classification quality.

### 3.2 Threshold Sweep Evaluator

Replace fixed threshold with constrained sweep (e.g. `0.50` to `0.56`).

For each threshold compute:

- `Combined(OOS) DSR`
- `Combined(OOS) Sharpe`
- `Combined(OOS) n_trades`
- trade shrinkage vs baseline

Hard constraints:

- shrinkage in `[15%, 30%]`
- `Combined(OOS) Sharpe > 0`
- `Combined(OOS) Sharpe >= Primary(Full) Sharpe`

Ranking among feasible thresholds:

1. OOS DSR
2. OOS Sharpe
3. Higher feasible OOS trade count

### 3.3 Execution Guard

Add configurable protections in single-position backtest/execution:

- `min_hold_bars`: block premature flip
- `reverse_confirmation_prob`: stronger condition for reverse than normal entry
- `cooldown_bars`: block immediate post-exit reversal

Objective: reduce `reverse_signal` loss concentration without breaking TBM-triggered exits.

### 3.4 Side Governance (Fallback)

Provide mode switch:

- `both` (default)
- `both_with_short_penalty`
- `long_only` (fallback only)

`long_only` is not default. It is a stage-2 fallback if DSR target cannot be reached and short-side drag remains material.

---

## 4. Validation and Acceptance

### 4.1 Primary Acceptance

- `Combined(OOS) DSR >= 0.95`

### 4.2 Secondary Acceptance

- trade shrinkage within `15%` to `30%`
- `PBO <= 0.10` non-degrading
- `Combined(OOS) Sharpe > 0`
- `Combined(OOS) Sharpe >= Primary(Full) Sharpe`

### 4.3 Diagnostics

- `reverse_signal` ratio decreases
- `reverse_signal` aggregate loss decreases significantly
- monthly OOS contribution is not dominated by only a few months

---

## 5. Risks and Controls

1. Threshold overfitting risk  
Control: select stable threshold band, avoid single-point peak chasing.

2. Excessive trade shrinkage risk  
Control: enforce `[15%, 30%]` shrinkage window.

3. Over-strong reversal blocking risk  
Control: incremental activation of guards with component-level comparison.

4. Premature short-side disable risk  
Control: keep short governance as fallback, not default.

---

## 6. Out of Scope (This Iteration)

- Rebuilding Dollar Bar construction logic
- Rebuilding CUSUM/event generation
- Large-scale new feature expansion
- Structural rewrite of vn.py runtime engine

This iteration is a targeted optimization release, not a full research architecture reset.

---

## 7. Planned Deliverables

1. Design-approved optimized decision layer
2. Threshold sweep and selection report
3. Execution-guarded backtest comparison report
4. AFML acceptance summary for optimized version

