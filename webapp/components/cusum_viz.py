"""CUSUM Filter 可视化组件"""
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
