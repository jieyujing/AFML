# DMA + TBM 完整三重屏障出场设计

## 状态

- **日期**: 2026-04-05
- **状态**: 已批准，待实现
- **问题**: DMA 出场使用纯时间屏障（`event_idx + 80`），未使用 TBM 动态 SL/TP

## 背景

当前 `10_combined_backtest.py` 中 DMA 出场机制：
```python
combined['touch_idx'] = [
    bars.index.get_loc(ts) + max_hold
    for ts in combined.index
]
```

这仅实现了**时间屏障**，没有动态止损（SL）和止盈（TP）。

TBM（Triple-Barrier Method）已在 `afmlkit.label.tbm.triple_barrier()` 中实现，但仅被 MA/RF Primary Model 使用（`04_ma_primary_model.py`），DMA 信号未接入。

## 目标

将完整的 TBM 三重屏障接入 DMA 回测，用 `afmlkit.label.tbm.triple_barrier()` 替代简单的时间屏障。

## 设计决策

1. **计算位置**: `10_combined_backtest.py`（方案 A）——紧接信号合并之后独立计算
2. **配置整合**: `DMA_EXIT_CONFIG` 合并至 `TBM_CONFIG`，删除 `DMA_EXIT_CONFIG`
3. **接口不变**: `backtest_utils.rolling_backtest()` 只接收 `touch_idx`，无需改动

## 数据流

```
dma_signals.parquet          events_features.parquet
        │                              │
        └─── merge ───────────────────►│ feat_ewm_vol_20
                                      │
                              combined (has: side + vol)
                                      │
                              triple_barrier() ◄── TBM_CONFIG
                                      │
                              touch_idx (真实屏障触碰位置)
                                      │
                              backtest_utils.rolling_backtest()
```

## 实现步骤

### Step 1: 统一配置（`config.py`）

将 `DMA_EXIT_CONFIG` 的参数合并至 `TBM_CONFIG`：

```python
# Phase 4b: TBM (Triple Barrier Method) 参数
TBM_CONFIG = {
    'target_ret_col': 'feat_ewm_vol_20',
    'profit_loss_barriers': (2.0, 2.0),   # (tp_mult, sl_mult)
    'vertical_barrier_bars': 80,
    'min_ret': 0.002,
    'min_close_time_sec': 60,
    # 以下为 DMA 出场专用（与 MA/RF 共用同一配置）
    'max_hold_bars': 80,        # 等于 vertical_barrier_bars，供回测使用
}
```

删除 `DMA_EXIT_CONFIG` 及其所有引用。

### Step 2: TBM touch_idx 计算（`10_combined_backtest.py`）

在合并信号之后、阈值扫描之前，插入 TBM 计算：

```python
from afmlkit.label.tbm import triple_barrier
from strategies.AL9999.config import TBM_CONFIG

# Step 2（原有）之后新增：
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
bar_duration_sec = 4 * 3600  # 假设 1 bar ≈ 4 hours
vertical_barrier_sec = vb_bars * bar_duration_sec

# 2.5.5 过滤末尾事件（无法完整评估）
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

# 2.5.7 写入 touch_idx（替换原有的简单时间屏障）
combined['touch_idx'] = touch_idxs
combined['tbm_ret'] = rets  # 可选：保留 TBM 收益用于分析
combined['max_rb_ratio'] = max_rb_ratios  # 可选：保留屏障接近度用于权重

logger.info(f"  TBM touch_idx 计算完成: {len(combined)} 个有效信号")
```

### Step 3: 删除 DMA_EXIT_CONFIG 引用（`config.py`）

从 `config.py` 删除 `DMA_EXIT_CONFIG` 定义。
从 `10_combined_backtest.py` 导入语句中删除 `DMA_EXIT_CONFIG`。

### Step 4: 验证

- 对比修改前后 `exit_reason` 分布：`TBM_trigger` 应包含 SL/TP/时间三种触发原因
- 对比 PnL 分布：TBM 版本应比纯时间屏障版本有更好的风险控制

## 屏障触发逻辑（来自 `afmlkit.label.tbm`）

| 持仓方向 | 触发条件 |
|----------|----------|
| Long (+1) | ret ≥ +2σ → TP 触发；ret ≤ -2σ → SL 触发 |
| Short (-1) | ret ≤ -2σ → TP 触发；ret ≥ +2σ → SL 触发 |
| 两者 | 时间屏障到期（80 bars）→ 时间触发 |

## 注意事项

1. **bar_duration_sec**: 当前硬编码为 `4 * 3600`（4小时），基于 6 bars/day 的假设。如果实际数据采样率不同，需要从 bars 数据中动态计算。
2. **`feat_ewm_vol_20` 缺失**: 如果某些事件没有波动率特征，需要做 fillna 或 skip 处理。
3. **与 MA/RF Primary Model 的 TBM 的区别**: MA/RF 的 TBM 用于生成**标签**（meta-labeling），而 DMA 的 TBM 仅用于**出场时序**，不涉及标签生成逻辑。

## 依赖

- `afmlkit.label.tbm.triple_barrier`
- `events_features.parquet`（需包含 `feat_ewm_vol_20` 列）
