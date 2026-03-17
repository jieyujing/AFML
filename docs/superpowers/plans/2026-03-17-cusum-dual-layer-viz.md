# CUSUM 双层联动可视化实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 CUSUM 单层可视化升级为双层联动图表，直观展示采样的因果机制。

**Architecture:** 扩展数据层返回状态历史，新增 Plotly 双层图表组件，集成到现有 Streamlit 页面。数据层使用 `return_state` 参数保持向后兼容。

**Tech Stack:** Python, NumPy, Pandas, Plotly, Streamlit

**Spec:** [docs/superpowers/specs/2026-03-17-cusum-dual-layer-viz-design.md](docs/superpowers/specs/2026-03-17-cusum-dual-layer-viz-design.md)

---

## File Structure

```
webapp/
├── session.py                    # [MODIFY] 添加 'cusum_state' 到 KEYS
├── components/
│   └── cusum_viz.py              # [MODIFY] 新增 plot_cusum_dual_layer，废弃旧函数
└── pages/
    └── 03_cusum_sampling.py      # [MODIFY] 使用新可视化组件

scripts/
└── cusum_filtering.py            # [MODIFY] 扩展 return_state 参数

tests/
└── test_cusum_dual_viz.py        # [CREATE] 新增测试文件
```

---

## Task 1: 更新 SessionManager KEYS

**Files:**
- Modify: `webapp/session.py:58`

- [ ] **Step 1: 添加 'cusum_state' 到 KEYS 列表**

```python
# webapp/session.py 第 58 行
# 现有代码:
'cusum_config', 'cusum_sampled_data', 'cusum_events'

# 改为:
'cusum_config', 'cusum_sampled_data', 'cusum_events', 'cusum_state'
```

- [ ] **Step 2: 验证修改**

```bash
grep -n "cusum_state" webapp/session.py
```

Expected: 找到 `'cusum_state'` 在 KEYS 列表中

- [ ] **Step 3: 提交**

```bash
git add webapp/session.py
git commit -m "feat: add cusum_state to SessionManager KEYS"
```

---

## Task 2: 扩展数据层函数

**Files:**
- Modify: `scripts/cusum_filtering.py`
- Test: `tests/test_cusum_dual_viz.py`

### 2.1 更新 import 语句

- [ ] **Step 1: 添加 cusum_filter_with_state 导入**

```python
# scripts/cusum_filtering.py 第 9 行
# 现有代码:
from afmlkit.sampling.filters import cusum_filter

# 改为:
from afmlkit.sampling.filters import cusum_filter, cusum_filter_with_state
```

### 2.2 修改函数签名

- [ ] **Step 2: 添加 return_state 参数**

```python
# scripts/cusum_filtering.py 第 13-19 行
# 修改函数签名:

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

需要添加 Union 导入:
```python
from typing import Union, Tuple
```

### 2.3 修改非 FracDiff 分支

- [ ] **Step 3: 修改非 FracDiff 分支逻辑**

找到非 FracDiff 分支（约第 106-122 行），修改 CUSUM 调用：

```python
# 现有代码 (约第 106-122 行):
else:
    prices_for_cusum = prices
    dynamic_threshold = clean_volatility * threshold_multiplier

    diff_series = np.zeros_like(prices_for_cusum)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_series[1:] = np.log(prices_for_cusum[1:] / prices_for_cusum[:-1])

    diff_series = np.nan_to_num(diff_series, nan=0.0)

    print("Applying Standard CUSUM filter...")
    start_time = time.time()
    event_indices = cusum_filter(diff_series, dynamic_threshold)
    elapsed = time.time() - start_time
    print(f"CUSUM filter completed in {elapsed:.4f} seconds.")

# 改为:
else:
    prices_for_cusum = prices
    dynamic_threshold = clean_volatility * threshold_multiplier

    diff_series = np.zeros_like(prices_for_cusum)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_series[1:] = np.log(prices_for_cusum[1:] / prices_for_cusum[:-1])

    diff_series = np.nan_to_num(diff_series, nan=0.0)

    print("Applying Standard CUSUM filter...")
    start_time = time.time()

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
            'time_index': df.index
        }
    else:
        event_indices = cusum_filter(diff_series, dynamic_threshold)
        cusum_state = None

    elapsed = time.time() - start_time
    print(f"CUSUM filter completed in {elapsed:.4f} seconds.")
```

### 2.4 修改 FracDiff 分支

- [ ] **Step 4: 修改 FracDiff 分支逻辑**

找到 FracDiff 分支（约第 65-105 行），修改 CUSUM 调用：

```python
# 现有代码 (约第 95-105 行):
print("Applying FracDiff Custom CUSUM filter...")
start_time = time.time()

diff_series = np.zeros_like(prices_for_cusum)
diff_series[1:] = prices_for_cusum[1:] - prices_for_cusum[:-1]

event_indices = cusum_filter(diff_series, dynamic_threshold)

elapsed = time.time() - start_time
print(f"FracDiff CUSUM filter completed in {elapsed:.4f} seconds.")

# 改为:
print("Applying FracDiff Custom CUSUM filter...")
start_time = time.time()

diff_series = np.zeros_like(prices_for_cusum)
diff_series[1:] = prices_for_cusum[1:] - prices_for_cusum[:-1]

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

elapsed = time.time() - start_time
print(f"FracDiff CUSUM filter completed in {elapsed:.4f} seconds.")
```

### 2.5 修改返回值

- [ ] **Step 5: 修改函数返回值**

```python
# 找到函数末尾的返回语句 (约第 154 行):
return filtered_df, t_events

# 改为:
if return_state:
    return filtered_df, t_events, cusum_state
else:
    return filtered_df, t_events
```

### 2.6 编写测试

- [ ] **Step 6: 创建测试文件**

```python
# tests/test_cusum_dual_viz.py
"""CUSUM 双层可视化测试"""
import pytest
import numpy as np
import pandas as pd
from scripts.cusum_filtering import compute_dynamic_cusum_filter


@pytest.fixture
def sample_df():
    """创建测试数据"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)


class TestReturnState:
    """测试 return_state 参数"""

    def test_return_state_false_returns_two_values(self, sample_df):
        """return_state=False 返回两个值"""
        result = compute_dynamic_cusum_filter(
            sample_df, return_state=False, use_frac_diff=False
        )
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DatetimeIndex)

    def test_return_state_true_returns_three_values(self, sample_df):
        """return_state=True 返回三个值"""
        result = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=False
        )
        assert len(result) == 3
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DatetimeIndex)
        assert isinstance(result[2], dict)

    def test_cusum_state_structure(self, sample_df):
        """测试 cusum_state 结构"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=False
        )
        required_keys = ['s_pos', 's_neg', 'threshold', 'time_index']
        for key in required_keys:
            assert key in cusum_state, f"Missing key: {key}"

    def test_s_pos_initialized(self, sample_df):
        """测试 s_pos[0] 已初始化为 0"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=False
        )
        assert cusum_state['s_pos'][0] == 0.0
        assert cusum_state['s_neg'][0] == 0.0

    def test_threshold_is_positive(self, sample_df):
        """测试阈值为正数"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=False
        )
        assert cusum_state['threshold'] > 0

    def test_time_index_alignment(self, sample_df):
        """测试时间索引与状态数组对齐"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=False
        )
        assert len(cusum_state['s_pos']) == len(cusum_state['time_index'])
        assert len(cusum_state['s_neg']) == len(cusum_state['time_index'])

    def test_frac_diff_branch(self, sample_df):
        """测试 FracDiff 分支"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True, use_frac_diff=True
        )
        assert cusum_state is not None
        assert len(cusum_state['s_pos']) == len(cusum_state['time_index'])
```

- [ ] **Step 7: 运行测试验证通过**

```bash
NUMBA_DISABLE_JIT=1 pytest tests/test_cusum_dual_viz.py -v
```

Expected: 所有测试通过

- [ ] **Step 8: 提交**

```bash
git add scripts/cusum_filtering.py tests/test_cusum_dual_viz.py
git commit -m "feat: add return_state parameter to compute_dynamic_cusum_filter"
```

---

## Task 3: 实现双层可视化组件

**Files:**
- Modify: `webapp/components/cusum_viz.py`

### 3.1 新增 plot_cusum_dual_layer 函数

- [ ] **Step 1: 添加 plot_cusum_dual_layer 函数**

在 `webapp/components/cusum_viz.py` 文件末尾添加：

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
    :param event_indices: 事件点的整数索引数组（相对于 cusum_state['time_index']）
    :param cusum_state: CUSUM 状态字典，含 s_pos, s_neg, threshold, time_index
    :param price_col: 价格列名
    :param title: 图表标题
    :returns: Plotly Figure 对象
    :raises ValueError: 参数验证失败
    """
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

    # ===== 创建双层图表 =====
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=('价格序列', 'CUSUM 累积和')
    )

    # ===== 上层：价格曲线 =====
    fig.add_trace(
        go.Scatter(
            x=price_df.index,
            y=price_df[price_col],
            mode='lines',
            line=dict(color='#4e79a7', width=1),
            name='价格'
        ),
        row=1, col=1
    )

    # ===== 上层：事件散点 =====
    if len(event_indices) > 0:
        # event_indices 是相对于 cusum_state['time_index'] 的索引
        # 需要映射到 price_df 的索引
        event_times = cusum_state['time_index'][event_indices]
        event_prices = price_df.loc[event_times, price_col]

        fig.add_trace(
            go.Scatter(
                x=event_times,
                y=event_prices,
                mode='markers',
                marker=dict(
                    color='#e15759',
                    size=8,
                    symbol='triangle-down'
                ),
                name='CUSUM 事件'
            ),
            row=1, col=1
        )

    # ===== 下层：S⁺ 正累积和 =====
    fig.add_trace(
        go.Scatter(
            x=cusum_state['time_index'],
            y=cusum_state['s_pos'],
            mode='lines',
            line=dict(color='#2ca02c', width=1.5),
            name='S⁺ (正累积和)'
        ),
        row=2, col=1
    )

    # ===== 下层：S⁻ 负累积和 =====
    fig.add_trace(
        go.Scatter(
            x=cusum_state['time_index'],
            y=cusum_state['s_neg'],
            mode='lines',
            line=dict(color='#d62728', width=1.5),
            name='S⁻ (负累积和)'
        ),
        row=2, col=1
    )

    # ===== 下层：阈值线 =====
    threshold = cusum_state['threshold']

    # +h 阈值线
    fig.add_trace(
        go.Scatter(
            x=[cusum_state['time_index'][0], cusum_state['time_index'][-1]],
            y=[threshold, threshold],
            mode='lines',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            name=f'+h ({threshold:.4f})',
            showlegend=True
        ),
        row=2, col=1
    )

    # -h 阈值线
    fig.add_trace(
        go.Scatter(
            x=[cusum_state['time_index'][0], cusum_state['time_index'][-1]],
            y=[-threshold, -threshold],
            mode='lines',
            line=dict(color='#ff7f0e', width=1, dash='dash'),
            name=f'-h ({-threshold:.4f})',
            showlegend=False
        ),
        row=2, col=1
    )

    # ===== 布局设置 =====
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(title_text="时间", row=2, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="累积和", row=2, col=1)

    return fig
```

### 3.2 添加废弃警告

- [ ] **Step 2: 更新 plot_cusum_cumulative_sum 函数**

```python
# 在文件开头添加 import
import warnings

# 修改 plot_cusum_cumulative_sum 函数
def plot_cusum_cumulative_sum(
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    threshold: np.ndarray,
    title: str = "CUSUM 累积和曲线"
) -> go.Figure:
    """
    [已废弃] 请使用 plot_cusum_dual_layer 替代。
    此函数将在 v2.0.0 移除。

    绘制正负累积和曲线及阈值线。
    """
    warnings.warn(
        "plot_cusum_cumulative_sum is deprecated, use plot_cusum_dual_layer instead",
        DeprecationWarning,
        stacklevel=2
    )

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

- [ ] **Step 3: 更新 plot_volatility_and_threshold 函数**

```python
def plot_volatility_and_threshold(
    volatility: np.ndarray,
    dynamic_threshold: np.ndarray,
    title: str = "波动率与动态阈值"
) -> go.Figure:
    """
    [已废弃] 此函数不再需要，将在 v2.0.0 移除。

    绘制滚动波动率和动态阈值。
    """
    warnings.warn(
        "plot_volatility_and_threshold is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=2
    )

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

- [ ] **Step 4: 验证语法正确**

```bash
python -m py_compile webapp/components/cusum_viz.py
```

Expected: 无输出（编译成功）

- [ ] **Step 5: 提交**

```bash
git add webapp/components/cusum_viz.py
git commit -m "feat: add plot_cusum_dual_layer and deprecate old functions"
```

---

## Task 4: 更新页面集成

**Files:**
- Modify: `webapp/pages/03_cusum_sampling.py`

### 4.1 更新 import

- [ ] **Step 1: 更新 import 语句**

```python
# webapp/pages/03_cusum_sampling.py 第 16 行
# 现有代码:
from components.cusum_viz import plot_price_with_events, render_sampling_rate_panel

# 改为:
from components.cusum_viz import (
    plot_price_with_events,
    render_sampling_rate_panel,
    plot_cusum_dual_layer
)
```

### 4.2 修改采样执行逻辑

- [ ] **Step 2: 修改 compute_dynamic_cusum_filter 调用**

找到执行 CUSUM 采样的代码块（`if st.button("🔬 执行 CUSUM 采样", type="primary"):` 内）：

```python
# 现有代码:
sampled_df, t_events = compute_dynamic_cusum_filter(
    selected_df,
    price_col='close',
    vol_span=vol_span,
    threshold_multiplier=threshold_multiplier,
    use_frac_diff=use_frac_diff
)

# 改为:
sampled_df, t_events, cusum_state = compute_dynamic_cusum_filter(
    selected_df,
    price_col='close',
    vol_span=vol_span,
    threshold_multiplier=threshold_multiplier,
    use_frac_diff=use_frac_diff,
    return_state=True
)

# 新增：保存状态到 Session
SessionManager.update('cusum_state', cusum_state)
```

### 4.3 修改可视化渲染逻辑

- [ ] **Step 3: 修改图表渲染代码**

找到采样结果展示区块（`if cusum_sampled_data is not None:` 内的价格图渲染部分）：

```python
# 现有代码:
cusum_events = SessionManager.get('cusum_events')
if cusum_events is not None and len(cusum_events) > 0:
    event_indices = np.array([
        original_df.index.get_loc(t)
        for t in cusum_events
        if t in original_df.index
    ])

    if len(event_indices) > 0:
        fig_price = plot_price_with_events(original_df, event_indices)
        st.plotly_chart(fig_price, use_container_width=True)

# 改为:
cusum_events = SessionManager.get('cusum_events')
cusum_state = SessionManager.get('cusum_state')

if cusum_events is not None and len(cusum_events) > 0:
    # 获取事件在 cusum_state['time_index'] 中的位置
    cusum_time_index = cusum_state['time_index'] if cusum_state else original_df.index
    event_indices = np.array([
        cusum_time_index.get_loc(t)
        for t in cusum_events
        if t in cusum_time_index
    ])

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
            fig_price = plot_price_with_events(original_df, event_indices)
            st.plotly_chart(fig_price, use_container_width=True)
```

- [ ] **Step 4: 验证语法正确**

```bash
python -m py_compile webapp/pages/03_cusum_sampling.py
```

Expected: 无输出（编译成功）

- [ ] **Step 5: 提交**

```bash
git add webapp/pages/03_cusum_sampling.py
git commit -m "feat: integrate plot_cusum_dual_layer into CUSUM sampling page"
```

---

## Task 5: 集成测试

- [ ] **Step 1: 启动 Web 应用**

```bash
streamlit run webapp/app.py
```

- [ ] **Step 2: 手动测试流程**

1. 导入数据或使用已有 Dollar Bars
2. 进入 CUSUM 采样页面
3. 执行 CUSUM 采样
4. 验证双层图表正确渲染：
   - 上层显示价格曲线和事件散点
   - 下层显示 S⁺/S⁻ 曲线和阈值线
   - 时间轴联动正常

- [ ] **Step 3: 运行全部测试**

```bash
NUMBA_DISABLE_JIT=1 pytest tests/test_cusum_dual_viz.py -v
```

Expected: 所有测试通过

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: complete CUSUM dual-layer visualization implementation"
```

---

## Summary

| Task | Files | Status |
|------|-------|--------|
| 1. 更新 SessionManager KEYS | `webapp/session.py` | [ ] |
| 2. 扩展数据层函数 | `scripts/cusum_filtering.py`, `tests/test_cusum_dual_viz.py` | [ ] |
| 3. 实现双层可视化组件 | `webapp/components/cusum_viz.py` | [ ] |
| 4. 更新页面集成 | `webapp/pages/03_cusum_sampling.py` | [ ] |
| 5. 集成测试 | - | [ ] |