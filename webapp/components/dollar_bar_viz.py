"""Dollar Bar 可视化组件 - Plotly 交互式图表"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats


def plot_iid_comparison(
    results: Dict[int, Dict],
    best_freq: int,
    title: str = "IID 评估指标对比"
) -> go.Figure:
    """
    绘制 IID 指标对比图（JB 统计量和自相关）

    :param results: 评估结果字典
    :param best_freq: 最优频率
    :param title: 图表标题
    :return: Plotly Figure
    """
    freqs = sorted(results.keys())

    # 提取指标
    jb_stats = [results[f].get('jb_stat', np.nan) for f in freqs]
    autocorrs = [results[f].get('autocorr_1', np.nan) for f in freqs]

    # 颜色设置
    best_color = "#e15759"
    default_color = "#4e79a7"
    bar_colors = [best_color if f == best_freq else default_color for f in freqs]

    # 创建子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Jarque-Bera 统计量", "一阶自相关系数")
    )

    # JB 统计量
    fig.add_trace(
        go.Bar(
            x=[str(f) for f in freqs],
            y=jb_stats,
            marker_color=bar_colors,
            name="JB 统计量",
            text=[f"{v:.1f}" if not np.isnan(v) else "N/A" for v in jb_stats],
            textposition="auto"
        ),
        row=1, col=1
    )

    # 自相关
    fig.add_trace(
        go.Bar(
            x=[str(f) for f in freqs],
            y=autocorrs,
            marker_color=bar_colors,
            name="自相关",
            text=[f"{v:.4f}" if not np.isnan(v) else "N/A" for v in autocorrs],
            textposition="auto"
        ),
        row=1, col=2
    )

    # 更新布局
    fig.update_layout(
        title=title,
        showlegend=False,
        height=500,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="每日目标 Bar 数", row=1, col=1)
    fig.update_xaxes(title_text="每日目标 Bar 数", row=1, col=2)
    fig.update_yaxes(title_text="JB 统计量 (越小越好)", row=1, col=1)
    fig.update_yaxes(title_text="自相关系数 (接近 0 越好)", row=1, col=2)

    # JB 统计量使用对数刻度
    fig.update_yaxes(type="log", row=1, col=1)

    return fig


def plot_return_distribution(
    returns: pd.Series,
    best_freq: int,
    title: str = "收益率分布"
) -> go.Figure:
    """
    绘制收益率分布图（直方图 + 正态拟合）

    :param returns: 收益率 Series
    :param best_freq: 最优频率
    :param title: 图表标题
    :return: Plotly Figure
    """
    # 创建图形
    fig = go.Figure()

    # 直方图
    n_bins = min(200, max(50, len(returns) // 100))

    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=n_bins,
        name="收益率",
        marker_color="#4e79a7",
        opacity=0.7,
        histnorm="probability density"
    ))

    # 正态分布拟合
    mu, sigma = returns.mean(), returns.std()
    x_range = np.linspace(returns.min(), returns.max(), 300)
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)

    fig.add_trace(go.Scatter(
        x=x_range,
        y=normal_pdf,
        name=f"正态拟合 (μ={mu:.6f}, σ={sigma:.6f})",
        line=dict(color="#e15759", width=3)
    ))

    # 更新布局
    fig.update_layout(
        title=f"{title} - 最优频率：{best_freq} bars/day",
        xaxis_title="对数收益率",
        yaxis_title="概率密度",
        template="plotly_white",
        height=500,
        showlegend=True
    )

    fig.update_xaxes(range=[x_range.min(), x_range.max()])

    return fig


def plot_bars_count_comparison(
    bars_dict: Dict[int, pd.DataFrame],
    title: str = "各频率 Bar 数量对比"
) -> go.Figure:
    """
    绘制各频率 Bar 数量对比图

    :param bars_dict: bars 字典
    :param title: 图表标题
    :return: Plotly Figure
    """
    freqs = sorted(bars_dict.keys())
    counts = [len(bars_dict[f]) for f in freqs]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[str(f) for f in freqs],
        y=counts,
        marker_color="#4e79a7",
        text=[f"{c:,}" for c in counts],
        textposition="auto"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="每日目标 Bar 数",
        yaxis_title="生成的 Bar 数量",
        template="plotly_white",
        height=400
    )

    return fig


def plot_time_series(
    bars_dict: Dict[int, pd.DataFrame],
    best_freq: int,
    title: str = "收盘价时间序列",
    max_points: int = 5000
) -> go.Figure:
    """
    绘制最优频率的收盘价时间序列

    :param bars_dict: bars 字典
    :param best_freq: 最优频率
    :param title: 图表标题
    :param max_points: 最大显示点数
    :return: Plotly Figure
    """
    if best_freq not in bars_dict or len(bars_dict[best_freq]) == 0:
        return go.Figure().add_annotation(text="无数据", showarrow=False)

    df = bars_dict[best_freq]

    # 降采样（如果数据量太大）
    if len(df) > max_points:
        step = len(df) // max_points
        plot_data = df.iloc[::step]
    else:
        plot_data = df

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_data.index,
        y=plot_data['close'],
        mode='lines',
        line=dict(color="#4e79a7", width=1),
        name=f"{best_freq} bars/day"
    ))

    fig.update_layout(
        title=f"{title} - {best_freq} bars/day ({len(df):,} bars)",
        xaxis_title="时间",
        yaxis_title="价格",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )

    # Y 轴格式化为千分位
    fig.update_yaxes(tickformat=",d")

    return fig


def plot_cumulative_returns(
    bars_dict: Dict[int, pd.DataFrame],
    title: str = "累计收益对比",
    max_points: int = 5000
) -> go.Figure:
    """
    绘制各频率累计收益对比图

    :param bars_dict: bars 字典
    :param title: 图表标题
    :param max_points: 最大显示点数
    :return: Plotly Figure
    """
    fig = go.Figure()

    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    for i, freq in enumerate(sorted(bars_dict.keys())):
        df = bars_dict[freq]
        if len(df) == 0:
            continue

        # 降采样
        if len(df) > max_points:
            step = len(df) // max_points
            plot_data = df.iloc[::step]
        else:
            plot_data = df

        # 计算累计收益
        log_ret = np.log(plot_data['close'] / plot_data['close'].shift(1))
        cum_ret = (1 + log_ret).cumprod()

        fig.add_trace(go.Scatter(
            x=plot_data.index,
            y=cum_ret,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1.5),
            name=f"{freq} bars/day"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="时间",
        yaxis_title="累计收益",
        template="plotly_white",
        height=500,
        hovermode='x unified'
    )

    return fig


def create_evaluation_dashboard(
    results: Dict[int, Dict],
    bars_dict: Dict[int, pd.DataFrame],
    best_freq: int
) -> go.Figure:
    """
    创建 4 面板评估仪表板

    :param results: 评估结果字典
    :param bars_dict: bars 字典
    :param best_freq: 最优频率
    :return: Plotly Figure
    """
    # 颜色设置
    best_color = "#e15759"

    # 创建 2x2 子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "IID 指标对比",
            "收益率分布",
            "Bar 数量对比",
            "收盘价时间序列"
        ),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )

    freqs = sorted(results.keys())

    # 1. IID 指标对比 (JB 统计量)
    jb_stats = [results[f].get('jb_stat', np.nan) for f in freqs]
    fig.add_trace(
        go.Bar(
            x=[str(f) for f in freqs],
            y=jb_stats,
            marker_color=[best_color if f == best_freq else "#4e79a7" for f in freqs],
            name="JB 统计量",
            showlegend=False
        ),
        row=1, col=1
    )

    # 2. 收益率分布
    if best_freq in bars_dict and len(bars_dict[best_freq]) > 0:
        returns = np.log(bars_dict[best_freq]['close'] / bars_dict[best_freq]['close'].shift(1)).dropna()
        n_bins = min(100, max(30, len(returns) // 100))

        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=n_bins,
                name="收益率",
                marker_color="#4e79a7",
                opacity=0.7,
                histnorm="probability density",
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Bar 数量对比
    counts = [len(bars_dict[f]) for f in freqs]
    fig.add_trace(
        go.Bar(
            x=[str(f) for f in freqs],
            y=counts,
            marker_color="#4e79a7",
            name="Bar 数量",
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. 收盘价时间序列
    if best_freq in bars_dict and len(bars_dict[best_freq]) > 0:
        df = bars_dict[best_freq]
        if len(df) > 5000:
            step = len(df) // 5000
            plot_data = df.iloc[::step]
        else:
            plot_data = df

        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data['close'],
                mode='lines',
                line=dict(color="#4e79a7", width=1),
                name="收盘价",
                showlegend=False
            ),
            row=2, col=2
        )

    # 更新布局
    fig.update_layout(
        title=f"Dollar Bar 评估仪表板 - 最优频率：{best_freq} bars/day",
        height=800,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="每日目标 Bar 数", row=1, col=1)
    fig.update_xaxes(title_text="JB 统计量", row=1, col=1)
    fig.update_xaxes(title_text="收益率", row=1, col=2)
    fig.update_xaxes(title_text="每日目标 Bar 数", row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=2)

    fig.update_yaxes(title_text="JB 统计量", type="log", row=1, col=1)
    fig.update_yaxes(title_text="概率密度", row=1, col=2)
    fig.update_yaxes(title_text="Bar 数量", row=2, col=1)
    fig.update_yaxes(title_text="价格", row=2, col=2)

    return fig


def plot_iid_summary_table(results: Dict[int, Dict]) -> go.Figure:
    """
    创建 IID 评估结果汇总表

    :param results: 评估结果字典
    :return: Plotly Figure (表格)
    """
    # 构建表格数据
    data = []
    for freq in sorted(results.keys()):
        r = results[freq]
        if "error" in r:
            data.append([
                freq, r.get('n_bars', 0), "ERROR", "-", "-", "-", "-", "-"
            ])
        else:
            data.append([
                freq,
                r['n_bars'],
                f"{r['jb_stat']:.2f}",
                f"{r['jb_pvalue']:.6f}",
                f"{r['autocorr_1']:.4f}",
                f"{r['skew']:.4f}",
                f"{r['kurtosis']:.4f}",
                f"{r['mean_ret']:.6f}"
            ])

    # 创建表格
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["频率", "Bar 数", "JB 统计量", "JB p-value", "自相关 (1)", "偏度", "峰度", "均值"],
            fill_color='#4e79a7',
            align='center',
            font=dict(color='white', size=12)
        ),
        cells=dict(
            values=list(zip(*data)),
            fill_color='#fafafa',
            align='center',
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="IID 评估结果汇总",
        height=400
    )

    return fig
