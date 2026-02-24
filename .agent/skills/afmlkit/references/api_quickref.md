# AFMLKit API 速查表

## 模块索引

| 模块 | 核心类/函数 | 功能 |
|------|------------|------|
| `bar.data_model` | `TradesData` | 交易数据加载、预处理、HDF5 存储 |
| `bar.io` | `H5Inspector`, `AddTimeBarH5`, `TimeBarReader` | HDF5 工具类 |
| `bar.kit` | `TimeBarKit`, `VolumeBarKit`, `DollarBarKit` | K 线构建 |
| `feature.base` | `SISOTransform`, `MISOTransform`, `CoreTransform` | 自定义 Transform 基类 |
| `feature.kit` | `Feature`, `FeatureKit` | 特征管道 |
| `feature.core.ma` | `ewma`, `sma` | 移动平均（Numba 加速）|
| `feature.core.volatility` | `ewms`, `ewmst`, `atr`, `realized_vol`, `bollinger_percent_b`, `variance_ratio_1_4_core` | 波动率系列 |
| `sampling.filters` | `cusum_filter`, `z_score_peak_filter` | 事件检测过滤器 |
| `label.kit` | `TBMLabel`, `SampleWeights` | 三重屏障标签（高层 API）|
| `label.tbm` | `triple_barrier` | 三重屏障法 Numba 核心 |
| `label.weights` | `average_uniqueness`, `return_attribution`, `time_decay`, `class_balance_weights` | 样本权重 Numba 函数 |

---

## TradesData — 关键签名

```python
# 创建
TradesData(ts, px, qty, id=None, is_buyer_maker=None, side=None,
           timestamp_unit=None, preprocess=False, proc_res=None, name=None)

# 加载 HDF5
TradesData.load_trades_h5(filepath, key=None, start_time=None, end_time=None,
                           enable_multiprocessing=True, n_workers=None)

# 保存 HDF5
trades.save_h5(filepath, complib='blosc:lz4', complevel=1, mode='a')

# 时间视图（不复制数据）
trades.set_view_range(start, end)
trades.data        # pd.DataFrame，含 timestamp/price/amount/side
```

---

## K 线构建 — 关键签名

```python
from afmlkit.bar.kit import TimeBarKit, VolumeBarKit, DollarBarKit, TickBarKit

TimeBarKit(trades, bar_timedelta)          # pd.Timedelta
VolumeBarKit(trades, volume_ths)
DollarBarKit(trades, dollar_thrs)
TickBarKit(trades, tick_ths)

kit.build_ohlcv() -> pd.DataFrame          # open/high/low/close/volume/...
```

---

## TBMLabel — 关键签名

```python
TBMLabel(
    features,              # pd.DataFrame，含 target_ret_col 列
    target_ret_col,        # log-return 空间的波动率列
    min_ret,               # 最小收益阈值（低于此值的事件被丢弃）
    horizontal_barriers,   # (stop_loss_mult, take_profit_mult)，inf 禁用
    vertical_barrier,      # pd.Timedelta，时间屏障
    min_close_time=pd.Timedelta(seconds=1),
    is_meta=False          # True 时 features 需含 'side' 列
)

tbm.compute_labels(trades)   -> (features_df, labels_df)
tbm.compute_weights(trades)  -> weights_df  # avg_uniqueness, return_attribution, vertical_touch_weights

SampleWeights.compute_final_weights(
    avg_uniqueness, time_decay_intercept=1.0,
    return_attribution=None, vertical_touch_weights=None, labels=None
) -> pd.DataFrame   # 含 combined_weights 列
```

---

## CUSUM 过滤器

```python
from afmlkit.sampling.filters import cusum_filter

event_positions = cusum_filter(
    raw_time_series,   # NDArray[float64] 价格序列
    threshold          # NDArray，单元素=固定阈值，多元素=动态阈值（建议用波动率）
)  # 返回 NDArray[int64]：positions（不是时间戳！）
```

---

## FeatureKit — Transform 命名约定

- `SISOTransform(input_col, output_col)` → 输出列名 = `{input_col}_{output_col}`
  - 例：`input_col='close'`, `output_col='sma_20'` → `close_sma_20`
- `MISOTransform(input_cols, output_col)` → 输出列名 = `output_col`（直接使用）

```python
kit = FeatureKit(features_list, retain=['close', 'volume'])
df_out = kit.build(ohlcv_df, backend='nb')   # 'nb'=Numba, 'pd'=Pandas
```

---

## 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|----------|
| Numba JIT 首次慢 | 首次调用编译，几秒到几分钟 | 生产环境提前 warmup 或用 `NUMBA_DISABLE_JIT=1` 测试 |
| 时间戳格式混淆 | `triple_barrier` / `average_uniqueness` 需纳秒 `int64` | 用 `timestamps.view('int64')` 转换 |
| `event_idx` 列 | TBMLabel 若 features 无此列则按时间戳自动推断 | 显式传入 CUSUM 事件的纳秒时间戳更精确 |
| 元标签 side 为 0 | `is_meta=True` 时 side 只能为 -1 或 1 | 确保主模型输出转换为 -1/1，过滤掉 0 |
| `min_ret` 过滤 | 低于 `min_ret * max(horizontal_barriers)` 的事件被丢弃 | 调小 `min_ret` 或提高波动率估计精度 |
| 竖直屏障权重 | 竖直屏障触碰标签噪声更大 | 传入 `vertical_touch_weights` 到 `compute_final_weights` |
| 测试时禁用 JIT | Numba JIT 使 pytest 非常慢 | `$env:NUMBA_DISABLE_JIT=1; uv run pytest` |

---

## 底层 Numba 函数（直接调用）

```python
from afmlkit.label.tbm import triple_barrier
from afmlkit.label.weights import average_uniqueness, return_attribution, time_decay, class_balance_weights

# triple_barrier 返回 4 元组
labels, touch_idxs, returns, barrier_ratios = triple_barrier(
    timestamps,           # NDArray[int64]，纳秒
    close,                # NDArray[float64]
    event_idxs,           # NDArray[int64]，events 纳秒时间戳（timestamps 的子集）
    targets,              # NDArray[float64]，log-return 目标
    horizontal_barriers,  # (bottom_mult, top_mult)，np.inf 禁用
    vertical_barrier,     # float，秒数，np.inf 禁用
    min_close_time_sec,   # float
    side,                 # None=方向预测；NDArray[int8]=元标签
    min_ret               # float
)
# barrier_ratios ∈ [0,1]：越接近 1 表示越靠近水平屏障（nan=屏障禁用）

# 样本权重流水线
avg_uniq, concurrency = average_uniqueness(timestamps, event_idxs, touch_idxs)
ret_attr = return_attribution(event_idxs, touch_idxs, close, concurrency, normalize=True)
decayed_w = time_decay(avg_uniq, last_weight=0.5)           # last_weight ∈ [-1, 1]
classes, class_w, class_cnt, final_w = class_balance_weights(labels, decayed_w)
```
