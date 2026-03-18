"""Primary Model 可视化组件"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional


def plot_optimization_results(result) -> go.Figure:
    """参数分布热力图

    :param result: OptimizationResult 实例
    :returns: Plotly Figure
    """
    df = result.all_results

    # 检查是否有 short_window 和 long_window 列
    if 'short_window' not in df.columns or 'long_window' not in df.columns:
        # 如果没有，返回一个简单的柱状图
        fig = px.bar(
            df,
            x='fold',
            y='test_recall',
            title='各 Fold 测试集 Recall',
            labels={'fold': 'Fold', 'test_recall': 'Recall'}
        )
        return fig

    # 统计每个参数组合的出现频率
    param_counts = df.groupby(['short_window', 'long_window']).size().reset_index(name='count')

    fig = px.density_heatmap(
        param_counts,
        x='short_window',
        y='long_window',
        z='count',
        title='最优参数分布',
        labels={
            'short_window': '短期均线',
            'long_window': '长期均线',
            'count': '频次'
        }
    )

    return fig


def plot_fold_performance(result) -> go.Figure:
    """各 Fold Recall 趋势

    :param result: OptimizationResult 实例
    :returns: Plotly Figure
    """
    df = result.all_results

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['fold'],
        y=df['train_recall'],
        mode='lines+markers',
        name='训练集 Recall',
        line=dict(color='#1f77b4')
    ))

    fig.add_trace(go.Scatter(
        x=df['fold'],
        y=df['test_recall'],
        mode='lines+markers',
        name='测试集 Recall',
        line=dict(color='#2ca02c')
    ))

    fig.add_hline(
        y=result.best_score,
        line_dash='dash',
        line_color='red',
        annotation_text=f'平均: {result.best_score:.2%}'
    )

    fig.update_layout(
        title='Walk-Forward Recall 趋势',
        xaxis_title='Fold',
        yaxis_title='Recall',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def plot_signals_overview(
    data: pd.DataFrame,
    signals: pd.Series,
    labels: Optional[pd.DataFrame] = None
) -> go.Figure:
    """信号概览图

    :param data: 价格数据，必须包含 'price' 列
    :param signals: 信号序列 (±1/0)
    :param labels: 可选的 TBM 标签 DataFrame
    :returns: Plotly Figure
    """
    fig = go.Figure()

    # 价格线
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['price'],
        mode='lines',
        name='价格',
        line=dict(color='gray', width=1)
    ))

    # 做多信号
    long_idx = signals[signals == 1].index
    if len(long_idx) > 0:
        fig.add_trace(go.Scatter(
            x=long_idx,
            y=data.loc[long_idx, 'price'],
            mode='markers',
            name='做多信号',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))

    # 做空信号
    short_idx = signals[signals == -1].index
    if len(short_idx) > 0:
        fig.add_trace(go.Scatter(
            x=short_idx,
            y=data.loc[short_idx, 'price'],
            mode='markers',
            name='做空信号',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

    fig.update_layout(
        title='交易信号概览',
        xaxis_title='时间',
        yaxis_title='价格',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig