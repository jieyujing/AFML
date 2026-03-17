# CUSUM 双层联动可视化设计

## 概述

将现有的 CUSUM 单层可视化升级为**双层联动图表**，直观展示 CUSUM 采样的"因果机制"：

- **上层（表象层）**：价格曲线 + 事件触发点散点标记 — 展示"结果"
- **下层（机制层）**：累积和曲线（S⁺/S⁻）+ 静态阈值线（±h）— 展示"原因"
- **联动**：时间轴同步，鼠标悬停时两层高亮相同时间点

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 状态数据返回 | 扩展现有函数，新增 `return_state` 参数 | 向后兼容，按需获取 |
| 图表交互 | 时间轴同步联动 | 直观理解价格变化与累积和的对应关系 |
| 重置点展示 | 静态曲线自然呈现 | 简洁清晰，曲线垂直下落已足够表达 |
| 阈值线 | 静态水平线（取平均值） | 简化展示，适合快速理解 |
| 布局比例 | 黄金分割 (60:40) | 价格是主视图，机制层为辅助 |

## 第一部分：数据层改造

### 改动文件

`scripts/cusum_filtering.py`

### 改动内容

扩展 `compute_dynamic_cusum_filter` 函数：

```python
def compute_dynamic_cusum_filter(
    df: pd.DataFrame,
    price_col: str = 'close',
    vol_span: int = 50,
    threshold_multiplier: float = 2.0,
    use_frac_diff: bool = True,
    return_state: bool = False,  # 新增参数
) -> Union[
    Tuple[pd.DataFrame, pd.DatetimeIndex],
    Tuple[pd.DataFrame, pd.DatetimeIndex, dict]
]:
    """
    通过 CUSUM 对时间序列的微观波动去噪。

    :param return_state: 如果为 True，返回 CUSUM 状态历史用于可视化。
    :returns:
        - return_state=False: (filtered_df, t_events)
        - return_state=True: (filtered_df, t_events, cusum_state)
    """
```

### 状态数据结构

当 `return_state=True` 时，第三个返回值 `cusum_state` 结构如下：

```python
cusum_state = {
    's_pos': np.ndarray,           # 正累积和历史 (float64)
    's_neg': np.ndarray,           # 负累积和历史 (float64)
    'threshold': float,            # 静态阈值（阈值数组的平均值）
    'time_index': pd.DatetimeIndex # 对应的时间索引
}
```

### 实现要点

1. 调用 `cusum_filter_with_state` 替代 `cusum_filter` 获取状态历史
2. 对齐时间索引：状态数组长度可能与输入 DataFrame 不同（因波动率计算和分数差分会丢弃前置数据）
3. 阈值取平均值：`threshold = float(np.mean(threshold_array))`
4. **两个分支均需使用 `cusum_filter_with_state`**（`use_frac_diff=True` 和 `False`）

### 完整实现示例

在 `compute_dynamic_cusum_filter` 函数末尾，两个分支（`use_frac_diff=True/False`）的 CUSUM 调用需要改为：

```python
# ===== 非 FracDiff 分支 =====
if not use_frac_diff:
    prices_for_cusum = prices
    dynamic_threshold = clean_volatility * threshold_multiplier

    diff_series = np.zeros_like(prices_for_cusum)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_series[1:] = np.log(prices_for_cusum[1:] / prices_for_cusum[:-1])
    diff_series = np.nan_to_num(diff_series, nan=0.0)

    if return_state:
        event_indices, s_pos_history, s_neg_history, threshold_arr = cusum_filter_with_state(
            diff_series, dynamic_threshold
        )
        # 初始化 s_pos/s_neg[0]
        s_pos_history[0] = 0.0
        s_neg_history[0] = 0.0
        cusum_state = {
            's_pos': s_pos_history,
            's_neg': s_neg_history,
            'threshold': float(np.mean(threshold_arr)),
            'time_index': df.index  # valid_mask 过滤后的 df
        }
    else:
        event_indices = cusum_filter(diff_series, dynamic_threshold)
        cusum_state = None

# ===== FracDiff 分支 =====
else:
    # ... FracDiff 处理逻辑（df 经过 common_indices 二次过滤）...

    if return_state:
        event_indices, s_pos_history, s_neg_history, threshold_arr = cusum_filter_with_state(
            diff_series, dynamic_threshold
        )
        s_pos_history[0] = 0.0
        s_neg_history[0] = 0.0
        cusum_state = {
            's_pos': s_pos_history,
            's_neg': s_neg_history,
            'threshold': float(np.mean(threshold_arr)),
            'time_index': df.index  # 二次过滤后的 df
        }
    else:
        event_indices = cusum_filter(diff_series, dynamic_threshold)
        cusum_state = None

# ===== 返回值 =====
if return_state:
    return filtered_df, t_events, cusum_state
else:
    return filtered_df, t_events
```

### 向后兼容

- `return_state=False`（默认）时，保持原有返回值和行为
- 现有调用无需修改

---

## 第二部分：可视化组件

### 改动文件

`webapp/components/cusum_viz.py`

### 新增函数：`plot_cusum_dual_layer`

```python
def plot_cusum_dual_layer(
    price_df: pd.DataFrame,
    event_indices: np.ndarray,
    cusum_state: dict,
    price_col: str = 'close',
    title: str = "CUSUM 采样：价格与累积和联动"
) -> go.Figure:
    """
    绘制双层联动的 CUSUM 可视化图表。

    上层：价格曲线 + 事件散点
    下层：累积和曲线 (S⁺/S⁻) + 静态阈值线 (±h)

    :param price_df: 原始价格 DataFrame，含 DatetimeIndex
    :param event_indices: 事件点的整数索引数组
    :param cusum_state: CUSUM 状态字典，含 s_pos, s_neg, threshold, time_index
    :param price_col: 价格列名
    :param title: 图表标题
    :returns: Plotly Figure 对象
    """
```

### 布局结构

```
┌─────────────────────────────────────────────┐
│  价格曲线 (close)                            │  60%
│  + 事件散点 (红色三角)                        │
├─────────────────────────────────────────────┤
│  S⁺ 正累积和 (绿色)                          │
│  S⁻ 负累积和 (红色)                          │  40%
│  +h 阈值线 (橙色虚线)                        │
│  -h 阈值线 (橙色虚线)                        │
└─────────────────────────────────────────────┘
        ↑
    共享时间轴 (X轴同步)
```

### Plotly 实现

```python
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    subplot_titles=('价格序列', 'CUSUM 累积和')
)
```

### 颜色方案

| 元素 | 颜色 | 样式 |
|------|------|------|
| 价格曲线 | `#4e79a7` | 实线，width=1 |
| 事件散点 | `#e15759` | 三角形，size=8 |
| S⁺ 正累积和 | `#2ca02c` | 实线，width=1.5 |
| S⁻ 负累积和 | `#d62728` | 实线，width=1.5 |
| 阈值线 ±h | `#ff7f0e` | 虚线，width=1 |

### 交互特性

- `hovermode='x unified'`：鼠标悬停时两层同步显示相同时间点数据
- Tooltip 内容：
  - 上层：时间、价格
  - 下层：S⁺、S⁻、阈值

### 废弃函数

以下函数将被新函数替代，标记为废弃：

- `plot_cusum_cumulative_sum` — 功能合并到 `plot_cusum_dual_layer`
- `plot_volatility_and_threshold` — 不再需要

### 保留函数

- `plot_price_with_events` — 保留，供其他页面或简化场景使用
- `render_sampling_rate_panel` — 保留，不变

---

## 第三部分：页面集成

### 改动文件

`webapp/pages/03_cusum_sampling.py`

### 改动点

#### 1. 执行采样时获取状态数据

位置：`compute_dynamic_cusum_filter` 调用处（按钮点击回调内）

```python
# 现有代码
sampled_df, t_events = compute_dynamic_cusum_filter(
    selected_df,
    price_col='close',
    vol_span=vol_span,
    threshold_multiplier=threshold_multiplier,
    use_frac_diff=use_frac_diff
)

# 改为
sampled_df, t_events, cusum_state = compute_dynamic_cusum_filter(
    selected_df,
    price_col='close',
    vol_span=vol_span,
    threshold_multiplier=threshold_multiplier,
    use_frac_diff=use_frac_diff,
    return_state=True  # 新增
)

# 新增：保存状态到 Session
SessionManager.update('cusum_state', cusum_state)
```

#### 2. 渲染可视化区块

位置：采样结果展示区块（`cusum_sampled_data` 不为 None 时）

```python
# 现有代码
if cusum_events is not None and len(cusum_events) > 0:
    event_indices = np.array([...])
    if len(event_indices) > 0:
        fig_price = plot_price_with_events(original_df, event_indices)
        st.plotly_chart(fig_price, use_container_width=True)

# 改为
cusum_state = SessionManager.get('cusum_state')
if cusum_events is not None and len(cusum_events) > 0:
    event_indices = np.array([...])
    if len(event_indices) > 0:
        if cusum_state is not None:
            fig = plot_cusum_dual_layer(
                price_df=original_df,
                event_indices=event_indices,
                cusum_state=cusum_state
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback：无状态数据时使用旧版单层图
            fig = plot_price_with_events(original_df, event_indices)
            st.plotly_chart(fig, use_container_width=True)
```

### Session 存储结构

新增 `cusum_state` 键：

```python
SessionManager.update('cusum_state', cusum_state)
```

---

## 实现优先级

1. **P0 - 数据层**：扩展 `compute_dynamic_cusum_filter` 支持返回状态
2. **P0 - 可视化组件**：实现 `plot_cusum_dual_layer` 函数
3. **P1 - 页面集成**：修改 CUSUM 采样页面使用新组件

## 测试要点

### 数据层测试

1. **返回值类型测试**
   - `return_state=False` 时返回 `(DataFrame, DatetimeIndex)`
   - `return_state=True` 时返回 `(DataFrame, DatetimeIndex, dict)`

2. **状态数据有效性**
   - `s_pos` 和 `s_neg` 长度与 `time_index` 一致
   - `s_pos[0] == 0.0` 和 `s_neg[0] == 0.0`（已初始化）
   - `threshold` 为正数

3. **FracDiff 分支专项**
   - FracDiff 后 `time_index` 与二次过滤后的 `df.index` 一致
   - 状态数组长度正确

4. **边界条件**
   - 最小有效数据长度：`vol_span` 个数据点（波动率计算需求）
   - 空数据或长度不足时正确报错

### 可视化组件测试

1. **渲染正确性**
   - 双层图表正确渲染（rows=2）
   - 颜色和样式符合设计

2. **交互特性**
   - 时间轴联动生效（`shared_xaxes=True`）
   - `hovermode='x unified'` 正常工作

3. **边缘情况**
   - 空事件数组：仅渲染曲线，无散点
   - 缺少必需键：抛出 `ValueError`
   - 数组长度不一致：抛出 `ValueError`

### 页面集成测试

1. **数据流**
   - 执行采样后状态数据正确保存到 Session
   - 页面刷新后状态丢失（Session 行为正常）

2. **Fallback 逻辑**
   - `cusum_state is None` 时使用旧版单层图
   - 图表正确展示

3. **端到端测试**
   - 从 Dollar Bar → CUSUM 采样 → 图表展示完整流程

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 状态数据量大时内存占用 | 仅在 `return_state=True` 时计算和返回 |
| 时间索引对齐复杂 | 在数据层严格对齐，可视化层直接使用 |
| 现有功能回归 | 保留原有函数，新增函数独立实现 |

---

## 补充：数据层实现细节

### Import 变更

`scripts/cusum_filtering.py` 需要新增 import：

```python
# 现有
from afmlkit.sampling.filters import cusum_filter

# 改为
from afmlkit.sampling.filters import cusum_filter, cusum_filter_with_state
```

### 时间索引构建

根据 `use_frac_diff` 参数，时间索引的构建方式不同：

#### 非 FracDiff 分支 (`use_frac_diff=False`)

```python
# df 经过 valid_mask 过滤（波动率计算丢弃前置数据）
time_index = df.index
```

#### FracDiff 分支 (`use_frac_diff=True`)

```python
# df 经过 valid_mask 过滤 + common_indices 二次过滤
# 二次过滤发生在 scripts/cusum_filtering.py 第 76-91 行
common_indices = df.index.intersection(stationary_prices.index)
df = df.loc[common_indices].copy()

# 时间索引与二次过滤后的 df 对齐
time_index = df.index
```

#### 构建状态字典

```python
cusum_state = {
    's_pos': s_pos_history,
    's_neg': s_neg_history,
    'threshold': float(np.mean(threshold)),
    'time_index': time_index  # 对应当前分支过滤后的 df
}
```

### s_pos_history[0] 初始化

`cusum_filter_with_state` 返回的数组 `s_pos_history[0]` 和 `s_neg_history[0]` 未初始化（垃圾值）。

**处理方案**：在数据层初始化为 0

```python
# 在调用 cusum_filter_with_state 后
s_pos_history[0] = 0.0
s_neg_history[0] = 0.0
```

---

## 补充：SessionManager 更新

### 改动文件

`webapp/session.py`

### 改动内容

在 `SessionManager.KEYS` 列表中添加 `'cusum_state'`：

```python
KEYS = [
    'raw_data', 'bar_data', 'dollar_bars', 'features', 'labels', 'sample_weights',
    # ... 现有键 ...
    'cusum_config', 'cusum_sampled_data', 'cusum_events', 'cusum_state'  # 新增
]
```

---

## 补充：废弃策略

### 废弃函数处理

以下函数标记为废弃，但保留 API 兼容。

**实现方式**：在函数开头添加 `warnings.warn` 调用，原函数体保持不变（完整复制原有逻辑）。

```python
import warnings

def plot_cusum_cumulative_sum(s_pos, s_neg, threshold, title="CUSUM 累积和曲线"):
    """
    [已废弃] 请使用 plot_cusum_dual_layer 替代。
    此函数将在 v2.0.0 移除。
    """
    warnings.warn(
        "plot_cusum_cumulative_sum is deprecated, use plot_cusum_dual_layer instead",
        DeprecationWarning,
        stacklevel=2
    )
    # ===== 原有实现保持不变 =====
    n = len(s_pos)
    x_axis = np.arange(n)
    fig = make_subplots(rows=1, cols=1)
    # ... 原有绘图代码 ...

def plot_volatility_and_threshold(volatility, dynamic_threshold, title="波动率与动态阈值"):
    """
    [已废弃] 此函数不再需要，将在 v2.0.0 移除。
    """
    warnings.warn(
        "plot_volatility_and_threshold is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=2
    )
    # ===== 原有实现保持不变 =====
    n = len(volatility)
    x_axis = np.arange(n)
    fig = make_subplots(rows=1, cols=1)
    # ... 原有绘图代码 ...
```

### 移除计划

- 在下一个 major version（v2.0.0）移除废弃函数

---

## 补充：边缘情况处理

### 可视化组件参数验证

在 `plot_cusum_dual_layer` 开头添加完整验证：

```python
def plot_cusum_dual_layer(
    price_df: pd.DataFrame,
    event_indices: np.ndarray,
    cusum_state: dict,
    price_col: str = 'close',
    title: str = "CUSUM 采样：价格与累积和联动"
) -> go.Figure:
    # ===== 参数验证 =====

    # 1. 必需键检查
    required_keys = ['s_pos', 's_neg', 'threshold', 'time_index']
    missing_keys = [k for k in required_keys if k not in cusum_state]
    if missing_keys:
        raise ValueError(f"cusum_state missing required keys: {missing_keys}")

    # 2. 价格列检查
    if price_col not in price_df.columns:
        raise ValueError(f"Column '{price_col}' not found in price_df")

    # 3. 数组长度一致性
    if len(cusum_state['s_pos']) != len(cusum_state['time_index']):
        raise ValueError("s_pos length mismatch with time_index")
    if len(cusum_state['s_neg']) != len(cusum_state['time_index']):
        raise ValueError("s_neg length mismatch with time_index")

    # 4. 状态数组不能超过价格数据长度
    if len(cusum_state['s_pos']) > len(price_df):
        raise ValueError("State array exceeds price data length")

    # 5. 数据有效性检查（非全 NaN）
    if np.all(np.isnan(cusum_state['s_pos'])):
        raise ValueError("s_pos contains all NaN values")
    if np.all(np.isnan(cusum_state['s_neg'])):
        raise ValueError("s_neg contains all NaN values")

    # ... 后续绘图逻辑 ...
```

### 空事件数组处理

```python
# 如果没有事件，仍渲染双层图表，但上层不添加散点标记
if len(event_indices) == 0:
    # 仅渲染价格曲线 + 累积和曲线，无事件标记
    pass
```

### 数组对齐

状态数组的 `time_index` 是 `price_df` 的子集（因波动率计算丢弃前置数据）。绘图时使用 `time_index` 作为 X 轴：

```python
# 上层：使用完整的 price_df.index
fig.add_trace(go.Scatter(x=price_df.index, y=price_df[price_col], ...))

# 下层：使用 cusum_state['time_index']
fig.add_trace(go.Scatter(
    x=cusum_state['time_index'],
    y=cusum_state['s_pos'],
    ...
))
```

### 时间范围视觉对齐（预期行为）

由于 `cusum_state['time_index']` 是 `price_df.index` 的子集（波动率计算丢弃了前置数据），会导致：

- **下层图表左侧可能有空白区域**：累积和曲线从有效数据点开始，而价格曲线从原始数据起点开始
- **两个子图的时间轴起点不一致**：上层可能比下层多展示一段时间

**这是预期行为**，原因：
1. 保持价格数据的完整性，让用户看到完整的上下文
2. 累积和曲线从有效起点开始，避免误导（前置数据无有效波动率）
3. 用户可以通过时间轴缩放自行调整查看范围

如果需要两层时间范围完全对齐，可在实现中添加配置选项。