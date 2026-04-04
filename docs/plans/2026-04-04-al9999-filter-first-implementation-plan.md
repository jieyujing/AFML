# AL9999 Filter-First Optimized Version Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a Filter-First optimization path for `AL9999` that prioritizes `Combined(OOS) DSR` while keeping trade shrinkage within 15%-30%.

**Architecture:** Keep the existing research pipeline unchanged and add a decision layer between meta probabilities and final execution. Implement three focused enhancements: meta scheme + threshold selection, execution guard for reversal control, and side-governance fallback switch. Validate each change with strategy-level tests and deterministic reports.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn outputs (existing artifacts), pytest.

---

### Task 1: Add Strategy Config for Filter-First Controls

**Files:**
- Modify: `strategies/AL9999/config.py`
- Test: `tests/strategies/test_al9999_filter_first_config.py`

**Step 1: Write the failing test**

```python
from strategies.AL9999 import config


def test_filter_first_config_exists_and_has_expected_keys():
    cfg = config.FILTER_FIRST_CONFIG
    assert "threshold_grid" in cfg
    assert "shrinkage_min" in cfg
    assert "shrinkage_max" in cfg
    assert "execution_guard" in cfg
    assert "side_mode" in cfg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategies/test_al9999_filter_first_config.py -v`  
Expected: FAIL with missing `FILTER_FIRST_CONFIG`.

**Step 3: Write minimal implementation**

```python
FILTER_FIRST_CONFIG = {
    "threshold_grid": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
    "shrinkage_min": 0.15,
    "shrinkage_max": 0.30,
    "execution_guard": {
        "enabled": True,
        "min_hold_bars": 2,
        "cooldown_bars": 1,
        "reverse_confirmation_delta": 0.02,
    },
    "side_mode": "both",
}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategies/test_al9999_filter_first_config.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_filter_first_config.py strategies/AL9999/config.py
git commit -m "feat(al9999): add filter-first optimization config"
```

---

### Task 2: Add Threshold Sweep + Feasible Selection Utility

**Files:**
- Create: `strategies/AL9999/threshold_optimizer.py`
- Test: `tests/strategies/test_al9999_threshold_optimizer.py`

**Step 1: Write the failing test**

```python
import pandas as pd

from strategies.AL9999.threshold_optimizer import select_best_threshold


def test_select_best_threshold_respects_shrinkage_and_ranks_by_oos_dsr():
    df = pd.DataFrame(
        {
            "threshold": [0.50, 0.51, 0.52],
            "oos_dsr": [0.60, 0.70, 0.90],
            "oos_sharpe": [1.0, 1.2, 0.9],
            "trade_shrinkage": [0.10, 0.20, 0.40],
        }
    )
    best = select_best_threshold(df, shrinkage_min=0.15, shrinkage_max=0.30)
    assert best["threshold"] == 0.51
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategies/test_al9999_threshold_optimizer.py -v`  
Expected: FAIL with module/function missing.

**Step 3: Write minimal implementation**

```python
def select_best_threshold(result_df, shrinkage_min, shrinkage_max):
    feasible = result_df[
        (result_df["trade_shrinkage"] >= shrinkage_min)
        & (result_df["trade_shrinkage"] <= shrinkage_max)
    ].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values(
        ["oos_dsr", "oos_sharpe"], ascending=[False, False]
    )
    return feasible.iloc[0].to_dict()
```

Also add helper(s) to produce `trade_shrinkage` and threshold-level report table.

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategies/test_al9999_threshold_optimizer.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_threshold_optimizer.py strategies/AL9999/threshold_optimizer.py
git commit -m "feat(al9999): add threshold sweep selection utility"
```

---

### Task 3: Add Execution Guard to Single-Position Backtest

**Files:**
- Modify: `strategies/AL9999/backtest_utils.py`
- Test: `tests/strategies/test_al9999_execution_guard.py`

**Step 1: Write the failing test**

```python
import pandas as pd

from strategies.AL9999.backtest_utils import rolling_backtest


def test_min_hold_blocks_immediate_reverse():
    # build a tiny bars/signals fixture where reverse happens too early
    # expected: reverse trade count reduced by guard
    assert True  # replace with concrete fixture assertion
```

Add at least:
- one case for `min_hold_bars`
- one case for `cooldown_bars`
- one case for `reverse_confirmation_delta`

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategies/test_al9999_execution_guard.py -v`  
Expected: FAIL because guard logic not implemented.

**Step 3: Write minimal implementation**

Implement optional guard arguments (or config-driven defaults) in `rolling_backtest`:
- track bars since entry
- block reverse if below `min_hold_bars`
- require `meta_prob >= threshold + reverse_confirmation_delta` for reverse
- enforce `cooldown_bars` after close before opposite-side entry

Ensure default behavior remains backward-compatible when guard is disabled.

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategies/test_al9999_execution_guard.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_execution_guard.py strategies/AL9999/backtest_utils.py
git commit -m "feat(al9999): add execution guard for reverse/cooldown control"
```

---

### Task 4: Integrate Filter-First Selection into Backtest Workflow

**Files:**
- Modify: `strategies/AL9999/10_combined_backtest.py`
- Modify: `strategies/AL9999/run_workflow.py`
- Test: `tests/strategies/test_al9999_filter_first_pipeline.py`

**Step 1: Write the failing test**

```python
def test_filter_first_pipeline_emits_threshold_report_and_uses_feasible_choice():
    # run a small mocked pipeline path
    # assert selected threshold honors shrinkage bounds and report file exists
    assert True  # replace with concrete assertions
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategies/test_al9999_filter_first_pipeline.py -v`  
Expected: FAIL because integration/report output does not exist.

**Step 3: Write minimal implementation**

Integrate:
- scheme decision (`full` vs `mda_positive`)
- threshold sweep selection via `threshold_optimizer`
- guard-aware rolling backtest call

Persist report artifact:
- `strategies/AL9999/output/features/filter_first_threshold_report.parquet`

Update workflow entry to expose this mode with clear logging.

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategies/test_al9999_filter_first_pipeline.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_filter_first_pipeline.py strategies/AL9999/10_combined_backtest.py strategies/AL9999/run_workflow.py
git commit -m "feat(al9999): integrate filter-first threshold and guard pipeline"
```

---

### Task 5: Add Side Governance Fallback and Validation Output

**Files:**
- Modify: `strategies/AL9999/10_combined_backtest.py`
- Modify: `strategies/AL9999/08_dsr_validation.py`
- Test: `tests/strategies/test_al9999_side_governance.py`

**Step 1: Write the failing test**

```python
def test_side_mode_filters_trade_directions():
    # side_mode=long_only should produce no short trades
    assert True  # replace with concrete assertions
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/strategies/test_al9999_side_governance.py -v`  
Expected: FAIL due to missing side governance behavior.

**Step 3: Write minimal implementation**

Add `side_mode` handling:
- `both`
- `both_with_short_penalty` (higher effective threshold for short side)
- `long_only`

Extend validation summary to print and persist:
- selected `side_mode`
- short-side contribution
- pass/fail against DSR and shrinkage constraints

**Step 4: Run test to verify it passes**

Run: `pytest tests/strategies/test_al9999_side_governance.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/strategies/test_al9999_side_governance.py strategies/AL9999/10_combined_backtest.py strategies/AL9999/08_dsr_validation.py
git commit -m "feat(al9999): add side governance fallback controls"
```

---

### Task 6: Run End-to-End Verification and Final Docs Update

**Files:**
- Modify: `strategies/AL9999/AUTOGEN_WORKFLOW.md`
- Modify: `strategies/AL9999/autogen_config.yaml` (if thresholds/config keys are surfaced)
- Optional report refresh in `strategies/AL9999/output/features/`

**Step 1: Run targeted strategy tests**

Run:
- `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_threshold_optimizer.py -v`
- `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_execution_guard.py -v`
- `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_filter_first_pipeline.py -v`
- `NUMBA_DISABLE_JIT=1 pytest tests/strategies/test_al9999_side_governance.py -v`

Expected: all PASS.

**Step 2: Run AL9999 workflow phase for backtest/validation**

Run:
- `python3 strategies/AL9999/run_workflow.py --phase 6`

Expected:
- threshold report generated
- DSR validation summary includes filter-first metadata

**Step 3: Update docs for new knobs and acceptance interpretation**

Document:
- filter-first config keys
- guard parameters
- side governance modes
- shrinkage constraints

**Step 4: Full relevant test sweep**

Run:
- `NUMBA_DISABLE_JIT=1 pytest tests/strategies/ -v`

Expected: PASS for strategy-relevant tests.

**Step 5: Commit**

```bash
git add strategies/AL9999/AUTOGEN_WORKFLOW.md strategies/AL9999/autogen_config.yaml
git commit -m "docs(al9999): document filter-first optimization workflow"
```

---

## Notes for Execution

- Keep all changes isolated to AL9999-related paths unless a shared utility is strictly required.
- Do not increase search-space complexity in this iteration (YAGNI).
- Prefer deterministic report artifacts with explicit sort/ranking columns.
- Keep metric naming consistent with existing outputs (`Combined (OOS)`, `DSR`, `Sharpe`).

