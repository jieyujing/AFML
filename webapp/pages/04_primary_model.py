"""Primary Model 页面 - 双均线策略信号生成与优化"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.session import SessionManager
from webapp.utils.primary_model import DualMAStrategy, WalkForwardOptimizer
from webapp.components.primary_model_viz import (
    plot_optimization_results,
    plot_fold_performance,
    plot_signals_overview
)


def main():
    st.title("📈 Primary Model - 双均线策略")
    st.markdown("""
    基于 **Meta-Labeling** 框架的 Primary Model。
    目标：**最大化 Recall**，捕获尽可能多的盈利机会。
    """)

    sm = SessionManager()

    # ========== Step 1: 数据源确认 ==========
    st.header("Step 1: 数据源确认")

    cusum_data = sm.get('cusum_sampled_data')
    if cusum_data is None:
        st.warning("⚠️ 请先在「CUSUM 采样」页面生成采样数据")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("事件数", len(cusum_data))
    with col2:
        if isinstance(cusum_data.index, pd.DatetimeIndex):
            st.metric("时间范围",
                      f"{cusum_data.index[0].date()} ~ {cusum_data.index[-1].date()}")
        else:
            st.metric("数据点数", len(cusum_data))
    with col3:
        cusum_config = sm.get('cusum_config', {})
        sample_rate = cusum_config.get('sample_rate', 'N/A')
        st.metric("采样率", f"{sample_rate}" if sample_rate else "N/A")

    with st.expander("📋 数据预览"):
        st.dataframe(cusum_data.head(10))

    # ========== Step 2: TBM 参数配置 ==========
    st.header("Step 2: 三重屏障参数")

    col1, col2, col3 = st.columns(3)
    with col1:
        tp_ratio = st.number_input(
            "止盈倍数",
            min_value=0.5, max_value=5.0,
            value=2.0, step=0.1,
            help="止盈阈值 = 波动率 × 此倍数"
        )
    with col2:
        sl_ratio = st.number_input(
            "止损倍数",
            min_value=0.5, max_value=5.0,
            value=1.0, step=0.1,
            help="止损阈值 = 波动率 × 此倍数"
        )
    with col3:
        time_barrier = st.number_input(
            "时间屏障（事件数）",
            min_value=0, max_value=100,
            value=0, step=5,
            help="0 表示不使用时间屏障"
        )

    st.info(f"📌 止盈/止损比 = {tp_ratio}:{sl_ratio}")

    # ========== Step 3: 策略参数范围 ==========
    st.header("Step 3: 双均线参数范围")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("短期均线")
        short_min = st.number_input("最小周期", 2, 20, 3, key="short_min")
        short_max = st.number_input("最大周期", 5, 50, 10, key="short_max")
        short_step = st.number_input("步长", 1, 10, 2, key="short_step")

    with col2:
        st.subheader("长期均线")
        long_min = st.number_input("最小周期", 10, 100, 15, key="long_min")
        long_max = st.number_input("最大周期", 20, 200, 50, key="long_max")
        long_step = st.number_input("步长", 1, 20, 5, key="long_step")

    # 参数组合预览
    short_vals = list(range(short_min, short_max + 1, short_step))
    long_vals = list(range(long_min, long_max + 1, long_step))
    n_combinations = len([1 for s in short_vals for l in long_vals if l > s])

    st.caption(f"参数组合数: {n_combinations}")

    # ========== Step 4: Walk-Forward 配置 ==========
    st.header("Step 4: Walk-Forward 配置")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_size = st.number_input(
            "训练窗口（事件数）",
            min_value=20, max_value=500, value=100
        )
    with col2:
        test_size = st.number_input(
            "测试窗口（事件数）",
            min_value=10, max_value=100, value=30
        )
    with col3:
        embargo = st.number_input(
            "隔离期（事件数）",
            min_value=0, max_value=20, value=5
        )

    # 预估 Fold 数
    required = train_size + embargo + test_size
    n_folds = max(0, (len(cusum_data) - required) // test_size) if test_size > 0 else 0
    st.caption(f"需要至少 {required} 个事件，当前 {len(cusum_data)} 个，预计 Fold 数: ~{n_folds}")

    # ========== 执行优化 ==========
    st.header("Step 5: 执行优化")

    if st.button("🚀 开始 Walk-Forward 优化", type="primary"):
        if n_folds <= 0:
            st.error(f"❌ 数据量不足！需要至少 {required} 个事件，当前仅 {len(cusum_data)} 个")
        else:
            with st.spinner("优化中..."):
                # 创建策略实例
                strategy = DualMAStrategy(
                    short_range=(short_min, short_max),
                    long_range=(long_min, long_max),
                    step=max(short_step, long_step),
                    tp_ratio=tp_ratio,
                    sl_ratio=sl_ratio,
                    time_barrier=time_barrier if time_barrier > 0 else None,
                    price_col='close'  # CUSUM 数据来自 Dollar Bars
                )

                # 创建优化器
                optimizer = WalkForwardOptimizer(
                    train_size=train_size,
                    test_size=test_size,
                    embargo=embargo
                )

                # 执行优化
                result = optimizer.optimize(cusum_data, strategy, metric='recall')

                # 保存到 session
                sm.update('primary_model_result', result)
                sm.update('primary_model_config', {
                    'tp_ratio': tp_ratio,
                    'sl_ratio': sl_ratio,
                    'time_barrier': time_barrier,
                    'short_range': (short_min, short_max),
                    'long_range': (long_min, long_max),
                    'train_size': train_size,
                    'test_size': test_size,
                    'embargo': embargo
                })

                st.success("✅ 优化完成!")
                st.rerun()

    # ========== 结果展示 ==========
    result = sm.get('primary_model_result')
    if result:
        st.header("Step 6: 结果展示")

        # 汇总指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "最优短期均线",
                result.best_params.get('short_window', 'N/A')
            )
        with col2:
            st.metric(
                "最优长期均线",
                result.best_params.get('long_window', 'N/A')
            )
        with col3:
            st.metric(
                "平均测试 Recall",
                f"{result.best_score:.2%}"
            )

        # Fold 结果表
        st.subheader("📋 各 Fold 详情")
        st.dataframe(
            result.all_results.style.format({
                'train_recall': '{:.2%}',
                'test_recall': '{:.2%}'
            }),
            use_container_width=True
        )

        # 可视化
        st.subheader("📊 可视化")
        tab1, tab2 = st.tabs(["参数分布", "Recall 趋势"])

        with tab1:
            fig = plot_optimization_results(result)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = plot_fold_performance(result)
            st.plotly_chart(fig, use_container_width=True)

        # 导出
        st.subheader("💾 导出信号")
        if st.button("生成最终信号并导出"):
            # 用最优参数生成全量信号
            config = sm.get('primary_model_config', {})
            strategy = DualMAStrategy(
                tp_ratio=config.get('tp_ratio', 2.0),
                sl_ratio=config.get('sl_ratio', 1.0),
                time_barrier=config.get('time_barrier') if config.get('time_barrier', 0) > 0 else None,
                price_col='close'  # CUSUM 数据来自 Dollar Bars
            )
            final_result = strategy.generate_signals(
                cusum_data,
                **result.best_params
            )

            # 保存
            sm.update('primary_model_signals', final_result.signals)
            sm.update('primary_model_labels', final_result.events_with_labels)

            st.success("✅ 信号已生成并保存到会话状态")

            # 导出 CSV
            csv = final_result.events_with_labels.to_csv()
            st.download_button(
                "📥 下载信号 CSV",
                csv,
                "primary_model_signals.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()