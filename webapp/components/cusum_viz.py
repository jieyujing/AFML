"""CUSUM Filter 可视化组件"""
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Tuple


def plot_price_with_events(
    df: pd.DataFrame,
    event_indices: np.ndarray,
    price_col: str = 'close',
    title: str = "价格序列 + CUSUM 事件"
) -> go.Figure:
    """绘制价格序列并用标记标出 CUSUM 事件位置"""
    fig = make_subplots(rows=1, cols=1)

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
