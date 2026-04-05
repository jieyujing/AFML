# DMA + TBM Exit & Trend Scanning Meta-Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simple time-barrier exit with full TBM triple-barrier in DMA backtest; fix TREND_WINDOWS config; replace ad-hoc sample weights with z-score normalization.

**Architecture:** Two independent changes — (1) TBM exit logic injected into `10_combined_backtest.py` using existing `afmlkit.label.tbm.triple_barrier()`; (2) sample weight formula refactored in `06_meta_labels.py`. `backtest_utils.py` interface is unchanged.

**Tech Stack:** Python, NumPy, Pandas, `afmlkit.label.tbm`, `afmlkit.feature.core.trend_scan`

---

## File Map

| File | Role |
|------|------|
| `strategies/AL9999/config.py` | `TREND_WINDOWS` update (line 79); delete `DMA_EXIT_CONFIG` |
| `strategies/AL9999/10_combined_backtest.py` | Add TBM touch_idx computation; remove `DMA_EXIT_CONFIG` import |
| `strategies/AL9999/06_meta_labels.py` | Refactor `compute_sample_weights()` to z-score normalization |

---

## Task 1: Update TREND_WINDOWS and remove DMA_EXIT_CONFIG from config.py

**Files:**
- Modify: `strategies/AL9999/config.py:79`
- Modify: `strategies/AL9999/config.py:225-232` (delete `DMA_EXIT_CONFIG` block)

- [ ] **Step 1: Update TREND_WINDOWS**

Read `strategies/AL9999/config.py` line 79:

```python
# before
TREND_WINDOWS = [5, 10, 15]  # 趋势窗口: 5/10/15 bars ≈ 0.3/0.7/1.0 个交易日

# after
TREND_WINDOWS = [5, 10, 15, 20]  # 趋势窗口: 5/10/15/20 bars ≈ 0.3/0.7/1.0/1.3 个交易日
```

- [ ] **Step 2: Delete DMA_EXIT_CONFIG block**

Read lines 222-232 of `config.py`. Delete the entire `DMA_EXIT_CONFIG` block:

```python
# ============================================================
# Phase 5: DMA Exit Strategy 参数（替代 TBM）
# ============================================================

DMA_EXIT_CONFIG = {
    # 基于波动率的动态止损/止盈（TBM 作为风控）
    'vol_sl_mult': 2.0,       # 止损乘数（×波动率）
    'vol_tp_mult': 2.0,       # 止盈乘数（×波动率）
    'max_hold_bars': 80,      # 最大持仓 bar 数（时间耗尽）
    'vol_col': 'feat_ewm_vol_20',  # 波动率特征列
}
```

- [ ] **Step 3: Verify TBM_CONFIG is still intact**

Read the `TBM_CONFIG` block (around lines 214-220). Confirm it contains:
```python
TBM_CONFIG = {
    'target_ret_col': 'feat_ewm_vol_20',
    'profit_loss_barriers': (2.0, 2.0),
    'vertical_barrier_bars': 80,
    'min_ret': 0.002,
    'min_close_time_sec': 60,
}
```

- [ ] **Step 4: Commit**

```bash
git add strategies/AL9999/config.py
git commit -m "feat(AL9999): unify TREND_WINDOWS to [5,10,15,20] and remove DMA_EXIT_CONFIG"
```

---

## Task 2: Add TBM touch_idx computation to 10_combined_backtest.py

**Files:**
- Modify: `strategies/AL9999/10_combined_backtest.py`
  - Add import: `from afmlkit.label.tbm import triple_barrier`
  - Remove import: `DMA_EXIT_CONFIG` from the config import block
  - Insert TBM computation block after signal merge (after line ~374, before threshold scan)
  - Remove `max_hold = int(DMA_EXIT_CONFIG.get(...))` line

- [ ] **Step 1: Read current imports and find exact lines to change**

Read `strategies/AL9999/10_combined_backtest.py` lines 31-46 (import block). Confirm current imports include `DMA_EXIT_CONFIG`:

```python
from strategies.AL9999.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    BARS_DIR,
    META_MODEL_CONFIG,
    FILTER_FIRST_CONFIG,
    DMA_EXIT_CONFIG,    # ← remove this line
    COMMISSION_RATE,
    SLIPPAGE_POINTS,
    TARGET_DAILY_BARS,
)
```

Remove `DMA_EXIT_CONFIG` from the import block.

- [ ] **Step 2: Read combined signal merge section**

Read lines 360-380. The current code at line 369-374 is:
```python
# 计算时间耗尽索引 (touch_idx = event_idx + max_hold_bars)
max_hold = int(DMA_EXIT_CONFIG.get('max_hold_bars', 80))
combined['touch_idx'] = [
    bars.index.get_loc(ts) + max_hold if ts in bars.index else None
    for ts in combined.index
]
```

Replace this entire block (lines ~369-374) with the TBM computation below. **Insert immediately after the combined signal merge block and before the logger.info for "可用无前视样本"**.

```python
# Step 2.5: TBM 出场计算（替代纯时间屏障）
logger.info("[Step 2.5] 计算 TBM 三重屏障 touch_idx...")

# 2.5.1 加载波动率目标
events_features = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))
if 'timestamp' in events_features.columns:
    events_features = events_features.set_index('timestamp')

# 2.5.2 对齐波动率到 combined 信号
common_idx_for_tbm = combined.index.intersection(events_features.index)
combined_tbm = combined.loc[common_idx_for_tbm].copy()
vol_targets = events_features.loc[common_idx_for_tbm, TBM_CONFIG['target_ret_col']].values.astype(np.float64)

# 2.5.3 准备 triple_barrier 输入
timestamps = bars.index.values.astype(np.int64)
close = bars['close'].values.astype(np.float64)
event_idxs = np.array([
    bars.index.get_loc(ts) for ts in common_idx_for_tbm
], dtype=np.int64)
sides = combined_tbm['side'].values.astype(np.int8)

# 2.5.4 计算垂直屏障（秒）
vb_bars = int(TBM_CONFIG['vertical_barrier_bars'])
bar_duration_sec = 4 * 3600  # 假设 6 bars/day, 1 bar ≈ 4 hours
vertical_barrier_sec = vb_bars * bar_duration_sec

# 2.5.5 过滤末尾事件（无法完整评估）
max_end_idx = len(bars) - 1
valid_mask = event_idxs + vb_bars < max_end_idx
n_before = len(combined_tbm)
combined_tbm = combined_tbm[valid_mask].copy()
event_idxs = event_idxs[valid_mask]
vol_targets = vol_targets[valid_mask]
sides = sides[valid_mask]
logger.info(f"  过滤末尾事件: {n_before} → {len(combined_tbm)}")

# 2.5.6 调用 triple_barrier
labels, touch_idxs, rets, max_rb_ratios = triple_barrier(
    timestamps=timestamps,
    close=close,
    event_idxs=event_idxs,
    targets=vol_targets,
    profit_loss_barriers=TBM_CONFIG['profit_loss_barriers'],
    vertical_barrier=vertical_barrier_sec,
    min_close_time_sec=TBM_CONFIG['min_close_time_sec'],
    side=sides,
    min_ret=TBM_CONFIG['min_ret'],
)

# 2.5.7 写入 combined DataFrame（对齐到原始 combined 索引）
combined.loc[combined_tbm.index, 'touch_idx'] = touch_idxs
combined.loc[combined_tbm.index, 'tbm_ret'] = rets
combined.loc[combined_tbm.index, 'max_rb_ratio'] = max_rb_ratios
logger.info(f"  TBM touch_idx 计算完成: {len(combined_tbm)} 个有效信号")
```

- [ ] **Step 3: Verify backtest_utils.rolling_backtest call is unchanged**

Read the call to `rolling_backtest` (search for `rolling_backtest` in the file). The `touch_idx` column is already read from `combined` — no interface change needed.

- [ ] **Step 4: Run the file to check for import/syntax errors**

```bash
cd /Users/link/Documents/AFMLKIT
python -c "from strategies.AL9999.config import TBM_CONFIG, TREND_WINDOWS; print('TREND_WINDOWS:', TREND_WINDOWS); print('TBM_CONFIG:', TBM_CONFIG)"
```
Expected: No ImportError, TREND_WINDOWS = [5, 10, 15, 20], TBM_CONFIG printed.

```bash
python -c "import ast; ast.parse(open('strategies/AL9999/10_combined_backtest.py').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 5: Commit**

```bash
git add strategies/AL9999/10_combined_backtest.py
git commit -m "feat(AL9999): replace time-barrier exit with TBM triple-barrier

- Add triple_barrier() computation for touch_idx in combined backtest
- Remove DMA_EXIT_CONFIG import (unified into TBM_CONFIG)
- touch_idx now reflects actual SL/TP/time barrier touch position
- backtest_utils.rolling_backtest() interface unchanged

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Refactor compute_sample_weights() in 06_meta_labels.py

**Files:**
- Modify: `strategies/AL9999/06_meta_labels.py:99-112`

- [ ] **Step 1: Read current function**

Read `strategies/AL9999/06_meta_labels.py` lines 99-112:

```python
def compute_sample_weights(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算样本权重 (基于 |t_value| 幅度)。

    |t_value| 越大，趋势越明确，样本权重越高。
    """
    t_abs = labels_df['t_value'].abs()
    t_norm = t_abs / (t_abs.max() + 1e-8)
    labels_df['sample_weight'] = np.clip(np.exp(t_norm * 2), 0.1, 1.0)
    print(f"  样本权重范围: [{labels_df['sample_weight'].min():.4f}, {labels_df['sample_weight'].max():.4f}]")
    return labels_df
```

- [ ] **Step 2: Replace with z-score normalized version**

Replace the entire function with:

```python
def compute_sample_weights(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算样本权重 (基于 |t_value| 的 z-score 归一化)。

    高于平均 |t| 的样本权重 > 1，低于平均的权重 < 1。
    相比 min-max 归一化，z-score 保留相对分布信息。
    """
    t_abs = labels_df['t_value'].abs()
    t_mean = t_abs.mean()
    t_std = t_abs.std()

    if t_std > 0:
        t_z = (t_abs - t_mean) / t_std
    else:
        t_z = t_abs - t_mean

    labels_df['sample_weight'] = np.clip(t_z + 1.0, 0.1, 1.0)
    print(f"  样本权重范围: [{labels_df['sample_weight'].min():.4f}, {labels_df['sample_weight'].max():.4f}]")
    print(f"  |t| 均值: {t_mean:.4f}, 标准差: {t_std:.4f}")
    return labels_df
```

- [ ] **Step 3: Verify syntax**

```bash
python -c "import ast; ast.parse(open('strategies/AL9999/06_meta_labels.py').read()); print('Syntax OK')"
```

- [ ] **Step 4: Commit**

```bash
git add strategies/AL9999/06_meta_labels.py
git commit -m "feat(AL9999): use z-score normalization for sample weights

Replace ad-hoc exp(t_norm*2) with z-score based weighting:
weight = clip((|t| - mean) / std + 1.0, 0.1, 1.0)
This preserves relative distribution information vs min-max.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Verification

**Files:**
- Run: `strategies/AL9999/03_trend_scanning.py`
- Run: `strategies/AL9999/06_meta_labels.py`
- Run: `strategies/AL9999/10_combined_backtest.py`

- [ ] **Step 1: Re-run trend scanning with new TREND_WINDOWS**

```bash
cd /Users/link/Documents/AFMLKIT
python strategies/AL9999/03_trend_scanning.py
```
Expected: "窗口范围: [5, 10, 15, 20]" in output. Output file `FEATURES_DIR/trend_labels.parquet` regenerated.

- [ ] **Step 2: Re-run meta labels to verify new weight formula**

```bash
python strategies/AL9999/06_meta_labels.py
```
Expected: "样本权重范围: [0.1xxx, 1.0xxx]" and "|t| 均值: X.XXXX, 标准差: X.XXXX" printed.

- [ ] **Step 3: Re-run combined backtest to verify TBM exit**

```bash
python strategies/AL9999/10_combined_backtest.py
```
Expected: "[Step 2.5] 计算 TBM 三重屏障 touch_idx..." in output. No `DMA_EXIT_CONFIG` ImportError.

- [ ] **Step 4: Commit verification artifacts (optional)**

```bash
git add strategies/AL9999/output/features/trend_labels.parquet
git commit -m "chore(AL9999): regenerate trend_labels with TREND_WINDOWS=[5,10,15,20]

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Spec Coverage Check

| Spec Requirement | Task |
|-----------------|------|
| TREND_WINDOWS = [5,10,15,20] | Task 1, Step 1 |
| Delete DMA_EXIT_CONFIG | Task 1, Step 2 |
| TBM touch_idx in 10_combined_backtest.py | Task 2 |
| Z-score sample weights | Task 3 |
| Verification runs | Task 4 |

No gaps found.

## Placeholder Scan

- No "TBD" or "TODO" in plan steps
- All code blocks show exact replacement strings
- All file paths are absolute from repo root
- `DMA_EXIT_CONFIG` removal confirmed (was referenced at line ~37 import and line ~370 usage)
