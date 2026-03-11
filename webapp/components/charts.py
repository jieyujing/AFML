"""通用图表组件"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List


def render_candlestick_chart(
    df: pd.DataFrame,
    title: str = "K 线图",
    limit: int = 100
) -> go.Figure:
    """渲染 K 线图

    Args:
        df: 包含 OHLC 数据的 DataFrame
        title: 图表标题
        limit: 显示的数据点数量

    Returns:
        Plotly Figure 对象
    """
    df_view = df.tail(limit).copy()

    fig = go.Figure(data=[go.Candlestick(
        x=df_view.index,
        open=df_view['open'],
        high=df_view['high'],
        low=df_view['low'],
        close=df_view['close'],
        name='OHLC'
    )])

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="价格",
        template="plotly_dark",
        height=600
    )

    return fig


def render_line_chart(
    df: pd.DataFrame,
    y_columns: List[str],
    title: str = "",
    x_label: str = "时间",
    y_label: str = "值"
) -> go.Figure:
    """渲染折线图

    Args:
        df: DataFrame
        y_columns: Y 轴列名列表
        title: 图表标题
        x_label: X 轴标签
        y_label: Y 轴标签

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    for col in y_columns:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )

    return fig


def render_bar_chart(
    values: pd.Series,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    orientation: str = 'v'
) -> go.Figure:
    """渲染柱状图

    Args:
        values: 值 Series
        title: 图表标题
        x_label: X 轴标签
        y_label: Y 轴标签
        orientation: 方向 ('v' 或 'h')

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=values.index if orientation == 'v' else values.values,
        y=values.values if orientation == 'v' else values.index,
        orientation=orientation
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=500
    )

    return fig


def render_heatmap(
    data: pd.DataFrame,
    title: str = "",
    colorscale: str = "RdBu"
) -> go.Figure:
    """渲染热力图

    Args:
        data: 数据 DataFrame
        title: 图表标题
        colorscale: 颜色刻度

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.index,
        colorscale=colorscale
    ))

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        template="plotly_white",
        height=600
    )

    return fig


def render_distribution_chart(
    values: np.ndarray,
    title: str = "",
    bins: int = 50
) -> go.Figure:
    """渲染分布图（直方图 + KDE）

    Args:
        values: 值数组
        title: 图表标题
        bins: 直方图 bin 数量

    Returns:
        Plotly Figure 对象
    """
    from scipy import stats

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

    # 直方图
    fig.add_trace(
        go.Histogram(x=values, nbinsx=bins, name='直方图'),
        row=1, col=1
    )

    # KDE
    kde_x = np.linspace(values.min(), values.max(), 100)
    kde_y = stats.gaussian_kde(values)(kde_x)

    fig.add_trace(
        go.Scatter(x=kde_x, y=kde_y, name='KDE', line=dict(color='red')),
        row=2, col=1
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500,
        showlegend=False
    )

    return fig


def render_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    size_col: Optional[str] = None
) -> go.Figure:
    """渲染散点图

    Args:
        df: DataFrame
        x_col: X 轴列
        y_col: Y 轴列
        color_col: 颜色列
        title: 标题
        size_col: 大小列

    Returns:
        Plotly Figure 对象
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        template="plotly_white"
    )

    fig.update_layout(height=500)

    return fig


def render_equity_curve(
    equity: pd.Series,
    title: str = "权益曲线",
    benchmark: Optional[pd.Series] = None
) -> go.Figure:
    """渲染权益曲线

    Args:
        equity: 权益序列
        title: 标题
        benchmark: 基准序列（可选）

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode='lines',
        name='策略权益'
    ))

    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='基准'
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="权益",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )

    return fig


def render_drawdown_chart(
    drawdown: pd.Series,
    title: str = "回撤图"
) -> go.Figure:
    """渲染回撤图

    Args:
        drawdown: 回撤序列
        title: 标题

    Returns:
        Plotly Figure 对象
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=-drawdown.values * 100,  # 转换为正数百分比
        mode='lines',
        fill='tozeroy',
        name='回撤'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="回撤 (%)",
        template="plotly_white",
        height=400
    )

    return fig


def display_metrics(metrics: Dict[str, Any]):
    """在列中显示指标

    Args:
        metrics: 指标字典
    """
    cols = st.columns(len(metrics))

    for i, (name, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                st.metric(name, f"{value:.4f}")
            elif isinstance(value, int):
                st.metric(name, f"{value:,}")
            else:
                st.metric(name, str(value))
