# DMA + TBM 出场 & Trend Scanning 元标签改进设计

## 状态

- **日期**: 2026-04-06
- **状态**: 已批准，待实现
- **版本**: v2（修订版，整合 TBM 出场 + Trend Scanning 元标签改进）

---

## Part I: TBM 出场（替代纯时间屏障）

### 背景

当前 `10_combined_backtest.py` 中 DMA 出场机制：
```python
combined['touch_idx'] = [bars.index.get_loc(ts) + max_hold for ts in combined.index]
```
仅实现**时间屏障**，没有动态止损（SL）/止盈（TP）。

### 目标

用 `afmlkit.label.tbm.triple_barrier()` 替代简单时间屏障，计算真实的屏障触碰位置作为 `touch_idx`。

### 设计决策

1. **计算位置**: `10_combined_backtest.py`（方案 A）——信号合并之后独立计算
2. **配置整合**: `DMA_EXIT_CONFIG` 合并至 `TBM_CONFIG`，删除 `DMA_EXIT_CONFIG`
3. **接口不变**: `backtest_utils.rolling_backtest()` 只接收 `touch_idx`，无需改动

### 数据流

```
dma_signals.parquet          events_features.parquet
        │                              │
        └─── merge ───────────────────►│ feat_ewm_vol_20
                              combined (has: side + vol)
                                      │
                              triple_barrier() ◄── TBM_CONFIG
                                      │
                              touch_idx (真实屏障触碰位置)
                                      │
                              backtest_utils.rolling_backtest()
```

---

## Part II: Trend Scanning 元标签改进

### 当前问题

1. **配置不一致**: `TREND_WINDOWS = [5, 10, 15]` vs `vnpy_strategy["trend_scan"]["windows"] = [5, 10, 15, 20]`
2. **样本权重公式 ad-hoc**: `np.exp(t_norm * 2)` 缺乏统计意义

### 改进 1: 统一 TREND_WINDOWS

**改动**: `config.py` 第79行
```python
# 修改前
TREND_WINDOWS = [5, 10, 15]

# 修改后
TREND_WINDOWS = [5, 10, 15, 20]
```

同时确保 `vnpy_strategy["trend_scan"]["windows"]` 保持一致（已为 `[5, 10, 15, 20]`）。

> 窗口含义：5/10/15/20 bars ≈ 0.3/0.7/1.0/1.3 个交易日，更宽的窗口能捕捉更长期的趋势。

### 改进 2: 统计归一化样本权重

**改动**: `06_meta_labels.py` 中 `compute_sample_weights()` 函数

```python
# 修改前（ad-hoc 指数权重）
t_abs = labels_df['t_value'].abs()
t_norm = t_abs / (t_abs.max() + 1e-8)
labels_df['sample_weight'] = np.clip(np.exp(t_norm * 2), 0.1, 1.0)

# 修改后（z-score 归一化权重）
t_abs = labels_df['t_value'].abs()
t_mean = t_abs.mean()
t_std = t_abs.std()
if t_std > 0:
    t_z = (t_abs - t_mean) / t_std
else:
    t_z = t_abs - t_mean
labels_df['sample_weight'] = np.clip(t_z + 1.0, 0.1, 1.0)
```

**公式含义**:
- 高于平均 t 值 → 权重 > 1（高置信度样本）
- 低于平均 t 值 → 权重 < 1（低置信度样本）
- clip 到 [0.1, 1.0] 防止极端权重

> **为什么不用 min-max**: min-max 归一化使最大权重始终为 1.0，与实际 t-value 绝对值脱钩。z-score 保留相对分布信息。

### 改进 3: 保持现有过滤策略

`|t_value| < MIN_T_VALUE` 的事件继续过滤，仅保留有效趋势事件。

---

## 实现步骤总览

### Step 1: 统一 TREND_WINDOWS（`config.py`）

```python
# Line 79
TREND_WINDOWS = [5, 10, 15, 20]  # 新增 20 bar 窗口
```

### Step 2: TBM touch_idx 计算（`10_combined_backtest.py`）

紧接信号合并之后（第374行附近）、阈值扫描之前插入：

```python
from afmlkit.label.tbm import triple_barrier
from strategies.AL9999.config import TBM_CONFIG

# Step 2.5: TBM 出场计算
logger.info("[Step 2.5] 计算 TBM 三重屏障 touch_idx...")

# 2.5.1 加载波动率目标
events_features = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))
if 'timestamp' in events_features.columns:
    events_features = events_features.set_index('timestamp')

# 2.5.2 对齐波动率到 combined 信号
common_idx = combined.index.intersection(events_features.index)
combined = combined.loc[common_idx].copy()
vol_targets = events_features.loc[common_idx, TBM_CONFIG['target_ret_col']].values.astype(np.float64)

# 2.5.3 准备 triple_barrier 输入
timestamps = bars.index.values.astype(np.int64)
close = bars['close'].values.astype(np.float64)
event_idxs = np.array([
    bars.index.get_loc(ts) for ts in common_idx
], dtype=np.int64)
sides = combined['side'].values.astype(np.int8)

# 2.5.4 计算垂直屏障（秒）
vb_bars = int(TBM_CONFIG['vertical_barrier_bars'])
bar_duration_sec = 4 * 3600
vertical_barrier_sec = vb_bars * bar_duration_sec

# 2.5.5 过滤末尾事件
max_end_idx = len(bars) - 1
valid_mask = event_idxs + vb_bars < max_end_idx
n_before = len(combined)
combined = combined[valid_mask].copy()
event_idxs = event_idxs[valid_mask]
vol_targets = vol_targets[valid_mask]
sides = sides[valid_mask]
logger.info(f"  过滤末尾事件: {n_before} → {len(combined)}")

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

# 2.5.7 写入 touch_idx
combined['touch_idx'] = touch_idxs
combined['tbm_ret'] = rets
combined['max_rb_ratio'] = max_rb_ratios
logger.info(f"  TBM touch_idx 计算完成: {len(combined)} 个有效信号")
```

### Step 3: 更新 `compute_sample_weights()`（`06_meta_labels.py`）

```python
def compute_sample_weights(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算样本权重 (基于 |t_value| 的 z-score 归一化)。

    高于平均 |t| 的样本权重 > 1，低于平均的权重 < 1。
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

### Step 4: 删除 `DMA_EXIT_CONFIG`（`config.py` + `10_combined_backtest.py`）

- 从 `config.py` 删除 `DMA_EXIT_CONFIG` 定义
- 从 `10_combined_backtest.py` 导入语句中删除 `DMA_EXIT_CONFIG`

---

## 屏障触发逻辑

| 持仓方向 | 触发条件 |
|----------|----------|
| Long (+1) | ret ≥ +2σ → TP 触发；ret ≤ -2σ → SL 触发 |
| Short (-1) | ret ≤ -2σ → TP 触发；ret ≥ +2σ → SL 触发 |
| 两者 | 时间屏障到期（80 bars）→ 时间触发 |

---

## 验证清单

- [ ] 对比修改前后 `exit_reason` 分布：`TBM_trigger` 应包含 SL/TP/时间三种触发原因
- [ ] 对比 PnL 分布：TBM 版本应比纯时间屏障版本有更好的风险控制
- [ ] 运行 `06_meta_labels.py` 验证新样本权重分布
- [ ] 确认 `TREND_WINDOWS` 更新后 `03_trend_scanning.py` 重新运行生成新 trend_labels

---

## 注意事项

1. **bar_duration_sec**: 当前硬编码为 `4 * 3600`（4小时），基于 6 bars/day 的假设。
2. **`feat_ewm_vol_20` 缺失**: 如果某些事件没有波动率特征，需要做 fillna 或 skip 处理。
3. **TBM 用于出场 vs Trend Scanning 用于元标签**: 两者职责分离——TBM 做风控执行，Trend Scanning 定义趋势真相。
4. **`TREND_WINDOWS` 更新后需重跑**: `03_trend_scanning.py` 和 `06_meta_labels.py` 需要重新运行以生成新窗口的 trend_labels。

---

## 依赖

- `afmlkit.label.tbm.triple_barrier`
- `events_features.parquet`（含 `feat_ewm_vol_20` 列）
- `trend_labels.parquet`（重新生成，含 20-bar 窗口）
