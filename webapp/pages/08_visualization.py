"""可视化中心页面"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="可视化中心", page_icon="🎨", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import (
    render_candlestick_chart, render_line_chart, render_bar_chart,
    render_heatmap, render_equity_curve, render_drawdown_chart
)

# 导入 AFML 可视化函数
from scripts.afml_visual_guides import (
    plot_cusum_filter_events,
    plot_tbm_bounds_single_case,
    plot_all_tbm_labels,
    plot_sample_uniqueness_and_concurrency
)

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("🎨 可视化中心")

st.markdown("""
本页面提供 AFML 核心概念的可视化：
- **CUSUM 过滤**: 采样事件在价格曲线上的分布
- **TBM 标签**: 三重屏障法的标签结果
- **样本权重**: 并发性和唯一性可视化
- **特征分析**: 相关性、重要性图表
- **回测结果**: 权益曲线、回撤图
""")

# 可视化类别
viz_categories = {
    "数据可视化": ["K 线图", "CUSUM 过滤事件"],
    "标签可视化": ["TBM 单样本", "TBM 全量标签", "样本权重与并发性"],
    "特征可视化": ["特征相关性", "特征重要性"],
    "回测可视化": ["权益曲线", "回撤图", "收益分布"]
}

# 侧边栏选择
st.sidebar.markdown("### 选择可视化类型")

selected_category = st.sidebar.selectbox(
    "类别",
    list(viz_categories.keys())
)

selected_viz = st.sidebar.selectbox(
    "图表",
    viz_categories[selected_category]
)

# 数据可视化
if selected_category == "数据可视化":
    if selected_viz == "K 线图":
        st.markdown("### 📊 K 线图")

        bar_data = SessionManager.get('bar_data')

        if bar_data is None:
            st.warning("请先导入数据并构建 K 线")
        else:
            limit = st.slider("显示 K 线数量", 50, 500, 100)

            if 'open' in bar_data.columns and 'high' in bar_data.columns:
                fig = render_candlestick_chart(bar_data, limit=limit)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("数据缺少 OHLC 列")

    elif selected_viz == "CUSUM 过滤事件":
        st.markdown("### 📍 CUSUM 过滤事件")

        filter_events = SessionManager.get('filter_events')
        bar_data = SessionManager.get('bar_data')

        if filter_events is None:
            st.warning("请先执行 CUSUM 过滤")

            # 提供执行 CUSUM 过滤的选项
            if bar_data is not None:
                if st.button("执行 CUSUM 过滤"):
                    try:
                        from afmlkit.sampling import CUSUMFilter

                        threshold = st.number_input("阈值", 0.01, 0.5, 0.05, 0.01)
                        cusum_filter = CUSUMFilter(threshold=threshold)
                        events = cusum_filter.filter(bar_data['close'])

                        SessionManager.update('filter_events', events)
                        st.success(f"检测到 {len(events)} 个事件")

                    except Exception as e:
                        st.error(f"CUSUM 过滤失败：{str(e)}")
        else:
            # 使用 matplotlib 绘制
            if bar_data is not None and 'close' in bar_data.columns:
                fig_mpl, ax = plt.subplots(figsize=(14, 7))

                ax.plot(bar_data.index, bar_data['close'], color='lightgray', linewidth=1.5, label='价格')

                valid_events = filter_events[filter_events.isin(bar_data.index)]
                event_prices = bar_data.loc[valid_events, 'close']

                ax.scatter(event_prices.index, event_prices.values, color='red', marker='v', s=50, label='CUSUM 事件', zorder=5)

                ax.set_title("CUSUM 过滤采样事件")
                ax.set_xlabel('时间')
                ax.set_ylabel('价格')
                ax.legend()

                st.pyplot(fig_mpl)
                plt.close()

# 标签可视化
elif selected_category == "标签可视化":
    if selected_viz == "TBM 单样本":
        st.markdown("### 🎯 TBM 单样本可视化")

        label_data = SessionManager.get('label_data')

        if label_data is None:
            st.warning("请先生成 TBM 标签")
        else:
            st.info("展示单个交易样本的三重屏障方法")

            # 选择样本
            sample_idx = st.slider("选择样本索引", 0, len(label_data) - 1, 0)

            sample = label_data.iloc[sample_idx]

            if 't0' in sample.index and 't1' in sample.index:
                bar_data = SessionManager.get('bar_data')

                if bar_data is not None:
                    t0 = sample['t0'] if hasattr(sample['t0'], 'timestamp') else pd.Timestamp(sample['t0'])
                    t1 = sample['t1'] if hasattr(sample['t1'], 'timestamp') else pd.Timestamp(sample['t1'])

                    # 获取价格
                    price_series = bar_data['close'] if 'close' in bar_data.columns else None

                    if price_series is not None:
                        fig_mpl, ax = plt.subplots(figsize=(12, 7))

                        ax.plot(price_series.index, price_series.values, color='black', linewidth=2, label='价格路径', alpha=0.6)

                        # 画屏障
                        if 'profit_barrier' in SessionManager.get('label_config', {}):
                            config = SessionManager.get('label_config')
                            upper = config.get('profit_barrier', 0.02)
                            lower = -config.get('loss_barrier', 0.02)

                            entry_price = price_series.loc[price_series.index >= t0].iloc[0] if len(price_series.loc[price_series.index >= t0]) > 0 else price_series.iloc[0]

                            ax.hlines(y=entry_price * (1 + upper), xmin=t0, xmax=t1, color='green', linestyle='-', linewidth=2, label='止盈屏障')
                            ax.hlines(y=entry_price * (1 + lower), xmin=t0, xmax=t1, color='red', linestyle='-', linewidth=2, label='止损屏障')
                            ax.vlines(x=t1, ymin=entry_price * (1 + lower), ymax=entry_price * (1 + upper), color='gray', linestyle='--', linewidth=2, label='时间期限')

                        ax.set_title(f"TBM 单样本 (索引={sample_idx}, 标签={sample.get('bin', 'N/A')})")
                        ax.set_xlabel('时间')
                        ax.set_ylabel('价格')
                        ax.legend()

                        st.pyplot(fig_mpl)
                        plt.close()

    elif selected_viz == "TBM 全量标签":
        st.markdown("### 🏷️ TBM 全量标签分布")

        label_data = SessionManager.get('label_data')
        bar_data = SessionManager.get('bar_data')

        if label_data is None:
            st.warning("请先生成 TBM 标签")
        else:
            if 'bin' in label_data.columns:
                label_dist = label_data['bin'].value_counts().sort_index()

                cols = st.columns(3)
                colors = ['red', 'gray', 'green']
                labels = ['止损 (-1)', '持有 (0)', '止盈 (+1)']

                for i, (label, count) in enumerate(label_dist.items()):
                    with cols[i + 1]:
                        color = colors[int(label) + 1] if label in [-1, 0, 1] else 'gray'
                        st.metric(labels[int(label) + 1] if label in [-1, 0, 1] else f"标签{label}", int(count))

                # 柱状图
                fig = render_bar_chart(label_dist, title="TBM 标签分布")
                st.plotly_chart(fig, use_container_width=True)

    elif selected_viz == "样本权重与并发性":
        st.markdown("### ⚖️ 样本权重与并发性")

        label_data = SessionManager.get('label_data')

        if label_data is None:
            st.warning("请先生成标签和权重")
        else:
            st.info("展示样本的并发数量和唯一性权重")

            # 计算并发性
            if 't1' in label_data.columns:
                t0_series = pd.Series(label_data.index)
                t1_series = label_data['t1']

                # 构建事件流
                starts = pd.Series(1, index=pd.to_datetime(t0_series))
                ends = pd.Series(-1, index=pd.to_datetime(t1_series) + pd.Timedelta(milliseconds=1))

                event_flow = pd.concat([starts, ends]).sort_index()
                c_t = event_flow.cumsum()
                c_t = c_t[~c_t.index.duplicated(keep='last')]

                # 绘制
                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

                if not c_t.empty:
                    ax1.plot(c_t.index, c_t.values, color='tab:blue', drawstyle='steps-post', label='并发标签数')
                ax1.set_ylabel('并发数量')
                ax1.set_title('样本并发性与唯一性')
                ax1.legend()

                # 唯一性权重
                if 'avg_uniqueness' in label_data.columns:
                    ax2.scatter(label_data.index, label_data['avg_uniqueness'], color='tab:orange', alpha=0.6, label='唯一性权重')
                ax2.set_ylabel('唯一性权重')
                ax2.set_xlabel('事件时间')
                ax2.legend()

                st.pyplot(fig)
                plt.close()

# 特征可视化
elif selected_category == "特征可视化":
    if selected_viz == "特征相关性":
        st.markdown("### 🔗 特征相关性热力图")

        corr_matrix = SessionManager.get('corr_matrix')

        if corr_matrix is None:
            st.warning("请先执行特征聚类分析")
        else:
            fig = render_heatmap(corr_matrix, title="特征相关性矩阵", colorscale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

    elif selected_viz == "特征重要性":
        st.markdown("### 📊 特征重要性")

        feature_importance = SessionManager.get('feature_importance')

        if feature_importance is None:
            st.warning("请先计算特征重要性")
        else:
            top_n = st.slider("显示 Top N 特征", 5, 50, 20)

            if 'mda_score' in feature_importance.columns:
                fig = render_bar_chart(
                    feature_importance.nlargest(top_n, 'mda_score')['mda_score'],
                    title=f"Top {top_n} 特征重要性 (MDA)",
                    orientation='h'
                )
            else:
                fig = render_bar_chart(
                    feature_importance.nlargest(top_n, 'importance')['importance'],
                    title=f"Top {top_n} 特征重要性",
                    orientation='h'
                )

            st.plotly_chart(fig, use_container_width=True)

# 回测可视化
elif selected_category == "回测可视化":
    if selected_viz == "权益曲线":
        st.markdown("### 📈 权益曲线")

        backtest_results = SessionManager.get('backtest_results')

        if backtest_results is None:
            st.warning("请先运行回测")
        else:
            equity = backtest_results['equity']
            fig = render_equity_curve(equity, title="策略权益曲线")
            st.plotly_chart(fig, use_container_width=True)

    elif selected_viz == "回撤图":
        st.markdown("### 📉 回撤图")

        backtest_results = SessionManager.get('backtest_results')

        if backtest_results is None:
            st.warning("请先运行回测")
        else:
            drawdown = backtest_results['drawdown']
            fig = render_drawdown_chart(drawdown, title="策略回撤走势")
            st.plotly_chart(fig, use_container_width=True)

    elif selected_viz == "收益分布":
        st.markdown("### 📊 收益分布")

        backtest_results = SessionManager.get('backtest_results')

        if backtest_results is None:
            st.warning("请先运行回测")
        else:
            from components.charts import render_distribution_chart

            returns = backtest_results['returns']
            fig = render_distribution_chart(returns.dropna().values, title="策略收益分布")
            st.plotly_chart(fig, use_container_width=True)

# 导出图表
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 导出")

if st.sidebar.button("导出当前图表"):
    st.info("图表导出功能尚未实现")
