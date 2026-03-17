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

位置：第 187-199 行

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

位置：第 252-261 行

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

1. **数据层**：
   - `return_state=False` 时返回值与原有行为一致
   - `return_state=True` 时返回的状态数组长度与时间索引对齐
   - 阈值计算正确（取平均值）

2. **可视化组件**：
   - 双层图表正确渲染
   - 时间轴联动生效
   - 颜色和样式符合设计

3. **页面集成**：
   - 执行采样后状态数据正确保存到 Session
   - 图表正确展示
   - Fallback 逻辑正常工作

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 状态数据量大时内存占用 | 仅在 `return_state=True` 时计算和返回 |
| 时间索引对齐复杂 | 在数据层严格对齐，可视化层直接使用 |
| 现有功能回归 | 保留原有函数，新增函数独立实现 |