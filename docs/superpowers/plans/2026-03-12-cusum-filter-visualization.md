# CUSUM Filter 可视化实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 WebUI 数据导入页面的 CUSUM filter 功能添加可视化图表和采样率统计面板。

**Architecture:**
- 在 `webapp/pages/01_data_import.py` 中扩展 CUSUM K 线构建逻辑
- 创建新的可视化组件文件 `webapp/components/cusum_viz.py` 包含 4 个图表函数
- 修改 `afmlkit/sampling/filters.py` 返回 CUSUM 中间状态（s_pos, s_neg, volatility）
- 使用 Streamlit expander 实现简单/高级模式切换

**Tech Stack:**
- Streamlit (WebUI 框架)
- Plotly (图表可视化)
- NumPy/Pandas (数据处理)
- `afmlkit.sampling.filters.cusum_filter` (核心算法)

---

## 文件结构

**Create:**
- `webapp/components/cusum_viz.py` - CUSUM 专用可视化组件

**Modify:**
- `webapp/pages/01_data_import.py:309-332` - CUSUM K 线构建逻辑扩展
- `afmlkit/sampling/filters.py:6-64` - cusum_filter 返回中间状态

**No test files needed** - WebUI 组件，手动测试

---

### Task 1: 创建 CUSUM 可视化组件

**Files:**
- Create: `webapp/components/cusum_viz.py`
- Test: 手动在 Streamlit 中验证

- [ ] **Step 1: 创建 cusum_viz.py 文件骨架**

创建文件并导入依赖：
```python
"""CUSUM Filter 可视化组件"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def plot_price_with_events(
    df: pd.DataFrame,
    event_indices: np.ndarray,
    price_col: str = 'close',
    title: str = "价格序列 + CUSUM 事件"
) -> go.Figure:
    """绘制价格序列并用标记标出 CUSUM 事件位置"""
    pass


def plot_cusum_cumulative_sum(
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    threshold: np.ndarray,
    title: str = "CUSUM 累积和曲线"
) -> go.Figure:
    """绘制正负累积和曲线及阈值线"""
    pass


def plot_volatility_and_threshold(
    volatility: np.ndarray,
    dynamic_threshold: np.ndarray,
    title: str = "波动率与动态阈值"
) -> go.Figure:
    """绘制滚动波动率和动态阈值"""
    pass


def render_sampling_rate_panel(
    original_rows: int,
    sampled_rows: int,
    time_range_days: float,
) -> Tuple[str, go.Figure]:
    """
    渲染采样率面板
    :returns: (指标文本，进度条 Figure)
    """
    pass
```

- [ ] **Step 2: 实现 plot_price_with_events 函数**

```python
def plot_price_with_events(
    df: pd.DataFrame,
    event_indices: np.ndarray,
    price_col: str = 'close',
    title: str = "价格序列 + CUSUM 事件"
) -> go.Figure:
    """绘制价格序列并用标记标出 CUSUM 事件位置"""
    fig = make_subplots(rows=1, cols=1, figsize=(14, 6))

    # 价格折线
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        line=dict(color='#4e79a7', width=1),
        name='价格'
    ))

    # 事件标记
    if len(event_indices) > 0:
        event_prices = df.iloc[event_indices][price_col]
        event_times = df.index[event_indices]
        fig.add_trace(go.Scatter(
            x=event_times,
            y=event_prices,
            mode='markers',
            marker=dict(
                color='#e15759',
                size=8,
                symbol='triangle-down'
            ),
            name='CUSUM 事件'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    return fig
```

- [ ] **Step 3: 实现 plot_cusum_cumulative_sum 函数**

```python
def plot_cusum_cumulative_sum(
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    threshold: np.ndarray,
    title: str = "CUSUM 累积和曲线"
) -> go.Figure:
    """绘制正负累积和曲线及阈值线"""
    n = len(s_pos)
    x_axis = np.arange(n)

    fig = make_subplots(rows=1, cols=1)

    # 正累积和
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=s_pos,
        mode='lines',
        line=dict(color='#2ca02c', width=1.5),
        name='S+ (正累积和)'
    ))

    # 负累积和
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=s_neg,
        mode='lines',
        line=dict(color='#d62728', width=1.5),
        name='S- (负累积和)'
    ))

    # 上阈值线
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=threshold,
        mode='lines',
        line=dict(color='#ff7f0e', width=1, dash='dash'),
        name='阈值'
    ))

    # 下阈值线
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=-threshold,
        mode='lines',
        line=dict(color='#ff7f0e', width=1, dash='dash'),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        xaxis_title="样本索引",
        yaxis_title="累积和",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    return fig
```

- [ ] **Step 4: 实现 plot_volatility_and_threshold 函数**

```python
def plot_volatility_and_threshold(
    volatility: np.ndarray,
    dynamic_threshold: np.ndarray,
    title: str = "波动率与动态阈值"
) -> go.Figure:
    """绘制滚动波动率和动态阈值"""
    n = len(volatility)
    x_axis = np.arange(n)

    fig = make_subplots(rows=1, cols=1)

    # 波动率
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=volatility,
        mode='lines',
        line=dict(color='#1f77b4', width=1.5),
        name='波动率 (EWMS)'
    ))

    # 动态阈值
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=dynamic_threshold,
        mode='lines',
        line=dict(color='#ff7f0e', width=1.5),
        name='动态阈值'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="样本索引",
        yaxis_title="值",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )

    return fig
```

- [ ] **Step 5: 实现 render_sampling_rate_panel 函数**

```python
def render_sampling_rate_panel(
    original_rows: int,
    sampled_rows: int,
    time_range_days: float,
) -> Tuple[str, go.Figure]:
    """
    渲染采样率面板
    :returns: (指标文本 MD, 进度条 Figure)
    """
    sampling_rate = sampled_rows / original_rows if original_rows > 0 else 0
    compression_ratio = original_rows / sampled_rows if sampled_rows > 0 else float('inf')
    events_per_day = sampled_rows / time_range_days if time_range_days > 0 else 0
    events_per_hour = events_per_day / 24

    # 指标文本
    metrics_md = f"""
    - **采样率**: {sampling_rate:.2%}
    - **原始行数**: {original_rows:,}
    - **采样后行数**: {sampled_rows:,}
    - **压缩比**: {compression_ratio:.2f}x
    - **事件频率**: {events_per_day:.1f} 次/天 ({events_per_hour:.2f} 次/小时)
    """

    # 进度条 Figure
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sampling_rate,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "采样率"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#4e79a7"},
            'steps': [
                {'range': [0, 0.1], 'color': "#fef9e7"},
                {'range': [0.1, 0.3], 'color': "#fdebd0"},
                {'range': [0.3, 1], 'color': "#d5dbdb"}
            ],
        }
    ))

    fig.update_layout(height=200)

    return metrics_md, fig
```

---

### Task 2: 扩展 cusum_filter 返回中间状态

**Files:**
- Modify: `afmlkit/sampling/filters.py:6-64`

- [ ] **Step 1: 修改 cusum_filter 函数签名和返回值**

当前函数只返回 `event_indices`，需要增加返回中间状态用于可视化：

```python
@njit(nogil=True)
def cusum_filter(
    diff_time_series: NDArray[np.float64],
    threshold: NDArray,
    return_state: bool = False  # 新增参数
) -> Tuple[NDArray[np.int64], Optional[Dict]]:  # 修改返回类型
    """...docstring..."""
    # ... 现有代码保持不变 ...

    # 在循环中记录状态（新增）
    if return_state:
        s_pos_history = np.empty(n, dtype=np.float64)
        s_neg_history = np.empty(n, dtype=np.float64)

    # ... 在循环内记录 ...
    for i in range(1, n):
        ret = diff_time_series[i]
        thrs = threshold[i]

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if return_state:
            s_pos_history[i] = s_pos
            s_neg_history[i] = s_neg

        if s_neg < -thrs:
            s_neg = 0.0
            event_indices[num_events] = i
            num_events += 1
        elif s_pos > thrs:
            s_pos = 0.0
            event_indices[num_events] = i
            num_events += 1

    result = event_indices[:num_events]

    if return_state:
        state = {
            's_pos': s_pos_history[:num_events + 1],
            's_neg': s_neg_history[:num_events + 1],
        }
        return result, state

    return result
```

- [ ] **Step 2: 运行测试验证修改**

```bash
cd /Users/link/Documents/AFMLKIT
pytest tests/sampling/test_cusum_filter.py -v
```

预期：现有测试通过（return_state 默认为 False）

- [ ] **Step 3: 提交**

```bash
git add afmlkit/sampling/filters.py
git commit -m "feat: add return_state option to cusum_filter for visualization"
```

---

### Task 3: 修改数据导入页面添加可视化

**Files:**
- Modify: `webapp/pages/01_data_import.py:309-332`
- Create: `webapp/components/cusum_viz.py` (已在 Task 1 完成)

- [ ] **Step 1: 在 CUSUM K 线构建逻辑后添加可视化显示**

修改 `elif bar_type == 'cusum':` 块（约 309-332 行）：

```python
elif bar_type == 'cusum':
    # === 简单/高级模式切换 ===
    advanced_mode = st.checkbox("高级模式", value=False)

    with st.expander("参数配置", expanded=advanced_mode):
        if advanced_mode:
            col1, col2, col3 = st.columns(3)
            with col1:
                vol_span = st.number_input(
                    "波动率窗口 (vol_span)",
                    min_value=10, max_value=200, value=50, step=10
                )
            with col2:
                threshold_multiplier = st.number_input(
                    "阈值乘数",
                    min_value=0.5, max_value=5.0, value=2.0, step=0.1
                )
            with col3:
                use_frac_diff = st.checkbox("启用分数阶差分", value=True)

            # 固定阈值模式（不推荐，但保留）
            threshold = st.number_input(
                "CUSUM 阈值 (仅当使用固定阈值模式)",
                min_value=0.01, max_value=1.0, value=0.05, step=0.01
            )
        else:
            # 简单模式：默认参数
            vol_span = 50
            threshold_multiplier = 2.0
            use_frac_diff = True
            threshold = 0.05

    if st.button("构建 CUSUM K 线"):
        try:
            from scripts.cusum_filtering import compute_dynamic_cusum_filter

            # 执行 CUSUM filter
            with st.spinner("正在执行 CUSUM 过滤..."):
                sampled_df, t_events = compute_dynamic_cusum_filter(
                    prepared_data,
                    price_col='close',
                    vol_span=vol_span,
                    threshold_multiplier=threshold_multiplier,
                    use_frac_diff=use_frac_diff
                )

            # 保存到 Session
            SessionManager.update('bar_data', sampled_df)
            SessionManager.update('bar_config', {
                'type': bar_type,
                'vol_span': vol_span,
                'threshold_multiplier': threshold_multiplier,
                'use_frac_diff': use_frac_diff
            })
            SessionManager.update('filter_events', t_events)

            st.success(f"检测到 {len(t_events)} 个 CUSUM 事件")

            # === 新增：显示可视化 ===
            st.markdown("### 📊 CUSUM 采样可视化")

            from webapp.components.cusum_viz import (
                plot_price_with_events,
                render_sampling_rate_panel
            )

            # 计算统计指标
            original_rows = len(prepared_data)
            sampled_rows = len(sampled_df)
            time_range_days = (
                prepared_data.index[-1] - prepared_data.index[0]
            ).total_seconds() / 86400 if len(prepared_data) > 0 else 1

            # 采样率面板
            col1, col2 = st.columns([2, 1])
            with col1:
                metrics_md, gauge_fig = render_sampling_rate_panel(
                    original_rows, sampled_rows, time_range_days
                )
                st.markdown(metrics_md)
            with col2:
                st.plotly_chart(gauge_fig, use_container_width=True)

            # 价格序列 + 事件标记
            fig_price = plot_price_with_events(
                prepared_data,
                np.array([prepared_data.index.get_loc(t) for t in t_events if t in prepared_data.index])
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # 显示采样数据预览
            with st.expander("采样数据预览"):
                st.dataframe(sampled_df.head(20))

        except ImportError as e:
            st.warning(f"CUSUM Filter 导入失败：{e}")
        except Exception as e:
            st.error(f"执行失败：{str(e)}")
            import traceback
            st.code(traceback.format_exc())
```

- [ ] **Step 2: 启动 Streamlit 验证**

```bash
cd /Users/link/Documents/AFMLKIT/webapp
streamlit run app.py
```

手动测试流程：
1. 导航到 "数据导入" 页面
2. 选择/加载数据
3. 在 "K 线构建" 步骤选择 "CUSUM 过滤 K 线"
4. 点击"构建 CUSUM K 线"
5. 验证 4 个图表和采样率面板正确显示

- [ ] **Step 3: 提交**

```bash
git add webapp/pages/01_data_import.py webapp/components/cusum_viz.py
git commit -m "feat: add CUSUM filter visualization in data import page"
```

---

### Task 4: 可选增强 - CUSUM 中间状态可视化

**Files:**
- Modify: `webapp/components/cusum_viz.py` (添加 2 个图表函数)
- Modify: `webapp/pages/01_data_import.py` (添加额外图表显示)

此 Task 为可选增强，如果 Task 3 已完成基础可视化，可根据时间决定是否实现：

- [ ] **可选 Step: 添加 CUSUM 曲线和波动率图表**

在 `cusum_viz.py` 中实现 `plot_cusum_cumulative_sum` 和 `plot_volatility_and_threshold`，并在页面中以 expander 形式展示：

```python
with st.expander("CUSUM 诊断图表"):
    st.plotly_chart(cusum_sum_fig, use_container_width=True)
    st.plotly_chart(vol_threshold_fig, use_container_width=True)
```

---

## 完成标准

- [ ] Task 1: CUSUM 可视化组件创建完成
- [ ] Task 2: cusum_filter 支持返回中间状态
- [ ] Task 3: 数据导入页面显示可视化图表
- [ ] 所有 git 提交完成，无未提交更改

---

## 测试检查清单

在 Streamlit 中手动验证：
- [ ] 简单模式下点击按钮可执行 CUSUM filter
- [ ] 高级模式可展开并显示完整参数
- [ ] 采样率指标正确计算并显示
- [ ] 价格序列图正确标记事件点
- [ ] 参数调整后重新执行可获得不同结果
