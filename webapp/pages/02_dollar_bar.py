"""Dollar Bar 生成页面"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Dollar Bar 生成", page_icon="💵", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from utils.csv_loader import CSVDataloader
from utils.dollar_bar_generator import DollarBarGenerator
from utils.iid_evaluator import IIDEvaluator, compare_frequencies
from components.dollar_bar_viz import (
    plot_iid_comparison,
    plot_return_distribution,
    plot_bars_count_comparison,
    plot_time_series,
    plot_cumulative_returns,
    create_evaluation_dashboard
)

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("💵 Dollar Bar 生成与评估")

st.markdown("""
基于 Marcos López de Prado 的动态 Dollar Bar 理论，根据每日目标 bar 数量自适应生成 bars。

**功能特性:**
- 支持从 `data/csv` 目录加载 1 分钟 OHLCV 数据
- 动态阈值 Dollar Bar 生成（多频率并行）
- IID 评估（Jarque-Bera 检验、自相关分析）
- 最优频率推荐
- 交互式可视化
""")

# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. 数据选择", "2. 参数配置", "3. 生成 Dollar Bars", "4. 评估结果", "5. 导出"],
    horizontal=True
)

if step == "1. 数据选择":
    st.markdown("### 1️⃣ 数据选择")

    # 初始化加载器
    loader = CSVDataloader("data/csv")

    # 获取可用文件列表
    available_files = loader.list_available_files()

    if not available_files:
        st.warning("data/csv 目录下没有找到 CSV 文件")

        # 显示上传选项
        uploaded_file = st.file_uploader("上传 CSV 文件", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['datetime'], index_col='datetime')
                SessionManager.update('csv_data', df)
                st.success(f"成功加载 {len(df)} 行数据")
            except Exception as e:
                st.error(f"加载失败：{str(e)}")
    else:
        # 文件选择
        st.markdown("#### 从 data/csv 目录选择")

        file_options = {f.name: f for f in available_files}
        selected_file = st.selectbox("选择文件", list(file_options.keys()))

        if selected_file:
            filepath = file_options[selected_file]

            # 显示文件信息
            file_info = loader.get_file_info(filepath)
            st.json({
                "文件名": file_info.get('filename', ''),
                "大小": f"{file_info.get('size_mb', 0)} MB",
                "估计行数": f"{file_info.get('estimated_rows', 0):,}",
                "列名": file_info.get('columns', [])
            })

            # 加载数据
            if st.button("加载文件"):
                try:
                    with st.spinner("正在加载数据..."):
                        df, warnings = loader.load_and_validate(filepath)

                    SessionManager.update('csv_data', df)
                    SessionManager.update('csv_warnings', warnings)

                    st.success(f"成功加载 {len(df)} 行数据")

                    if warnings:
                        st.warning("⚠️ 数据警告：")
                        for w in warnings:
                            st.write(f"- {w}")

                    # 显示数据预览
                    st.markdown("#### 数据预览")
                    st.dataframe(df.head(10))

                    st.markdown("#### 数据统计")
                    st.dataframe(df.describe())

                except Exception as e:
                    st.error(f"加载失败：{str(e)}")

    # 检查已加载的数据
    csv_data = SessionManager.get('csv_data')
    if csv_data is not None:
        st.info(f"✅ 当前已加载：{len(csv_data)} 行数据")

elif step == "2. 参数配置":
    st.markdown("### 2️⃣ 参数配置")

    csv_data = SessionManager.get('csv_data')

    if csv_data is None:
        st.warning("请先选择并加载数据")
    else:
        st.markdown("#### Dollar Bar 参数")

        col1, col2 = st.columns(2)

        with col1:
            # 每日目标 bar 数量
            target_options = [2, 4, 6, 10, 15, 20, 30, 50, 75, 100]
            selected_targets = st.multiselect(
                "每日目标 Bar 数",
                options=target_options,
                default=[4, 6, 10, 20, 50]
            )

        with col2:
            # EWMA Span
            ewma_span = st.slider("EWMA 平滑窗口", min_value=5, max_value=60, value=20)

        if not selected_targets:
            st.error("请至少选择一个目标频率")
        else:
            st.markdown("#### 当前配置")
            st.json({
                "每日目标 Bars": selected_targets,
                "EWMA Span": ewma_span
            })

            # 保存配置
            if st.button("保存配置"):
                SessionManager.update('dollar_bar_config', {
                    'target_daily_bars': selected_targets,
                    'ewma_span': ewma_span
                })
                st.success("配置已保存")

elif step == "3. 生成 Dollar Bars":
    st.markdown("### 3️⃣ 生成 Dollar Bars")

    csv_data = SessionManager.get('csv_data')
    config = SessionManager.get('dollar_bar_config', {})

    if csv_data is None:
        st.warning("请先加载数据")
    elif not config:
        st.warning("请先配置参数")
    else:
        st.markdown("#### 配置确认")
        st.json({
            "数据行数": f"{len(csv_data):,}",
            "每日目标 Bars": config.get('target_daily_bars', []),
            "EWMA Span": config.get('ewma_span', 20)
        })

        # 进度显示
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(value):
            progress_bar.progress(value)
            status_text.text(f"处理进度：{int(value * 100)}%")

        if st.button("开始生成"):
            try:
                start_time = time.time()

                # 创建生成器
                generator = DollarBarGenerator(
                    target_daily_bars=config['target_daily_bars'],
                    ewma_span=config['ewma_span']
                )

                # 运行生成
                status_text.text("阶段 1: 计算日度 Dollar Volume...")
                bars_dict = generator.run(csv_data, progress_callback=update_progress)

                elapsed = time.time() - start_time

                # 保存结果
                SessionManager.update('dollar_bars', bars_dict)
                SessionManager.update('generation_time', elapsed)

                status_text.text("✅ 生成完成!")

                # 显示摘要
                st.success(f"生成完成！耗时：{elapsed:.2f}秒")

                st.markdown("#### 生成结果摘要")

                for freq in sorted(bars_dict.keys()):
                    df = bars_dict[freq]
                    if len(df) > 0:
                        st.info(f"频率 {freq} bars/day: {len(df):,} bars")

            except Exception as e:
                st.error(f"生成失败：{str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif step == "4. 评估结果":
    st.markdown("### 4️⃣ 评估结果")

    bars_dict = SessionManager.get('dollar_bars')

    if bars_dict is None:
        st.warning("请先生成 Dollar Bars")
    else:
        # 执行 IID 评估
        evaluator = IIDEvaluator()

        results = {}
        for freq, bars_df in bars_dict.items():
            if len(bars_df) > 0:
                results[freq] = evaluator.evaluate(bars_df)

        if not results:
            st.warning("没有可用的评估结果")
        else:
            # 计算最优频率
            best_freq, score_df, report = compare_frequencies(bars_dict)

            SessionManager.update('iid_results', results)
            SessionManager.update('best_freq', best_freq)
            SessionManager.update('iid_score_df', score_df)

            # 自动保存最佳结果到 outputs/dollar_bars
            output_dir = Path("outputs/dollar_bars")
            output_dir.mkdir(parents=True, exist_ok=True)
            best_bars = bars_dict[best_freq]
            if len(best_bars) > 0:
                output_path = output_dir / f"dollar_bars_best_{best_freq}.csv"
                best_bars.to_csv(output_path)
                st.success(f"✅ 最佳结果已自动保存：{output_path.absolute()}")

            # 显示最优频率
            st.markdown("#### 🏆 评估结论")
            st.success(f"**最优频率：{best_freq} bars/day**")

            # 显示评分详情
            st.markdown("#### 评分详情")
            st.dataframe(score_df)

            # 文本报告
            st.markdown("#### 文本报告")
            st.code(report)

            # 可视化
            st.markdown("#### 可视化分析")

            viz_tabs = st.tabs([
                "IID 指标对比",
                "收益率分布",
                "Bar 数量对比",
                "时间序列",
                "累计收益",
                "仪表板"
            ])

            with viz_tabs[0]:
                fig = plot_iid_comparison(results, best_freq)
                st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[1]:
                if best_freq in bars_dict:
                    returns = np.log(bars_dict[best_freq]['close'] / bars_dict[best_freq]['close'].shift(1)).dropna()
                    fig = plot_return_distribution(returns, best_freq)
                    st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[2]:
                fig = plot_bars_count_comparison(bars_dict)
                st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[3]:
                fig = plot_time_series(bars_dict, best_freq)
                st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[4]:
                fig = plot_cumulative_returns(bars_dict)
                st.plotly_chart(fig, use_container_width=True)

            with viz_tabs[5]:
                fig = create_evaluation_dashboard(results, bars_dict, best_freq)
                st.plotly_chart(fig, use_container_width=True)

elif step == "5. 导出":
    st.markdown("### 5️⃣ 导出结果")

    bars_dict = SessionManager.get('dollar_bars')
    best_freq = SessionManager.get('best_freq')

    if bars_dict is None:
        st.warning("没有可导出的数据")
    else:
        st.markdown("#### CSV 导出")

        # 为每个频率创建下载按钮
        for freq in sorted(bars_dict.keys()):
            df = bars_dict[freq]
            if len(df) == 0:
                continue

            csv_bytes = df.to_csv().encode('utf-8')

            st.download_button(
                label=f"下载 {freq} bars/day CSV ({len(df):,} rows)",
                data=csv_bytes,
                file_name=f"dollar_bars_{freq}.csv",
                mime="text/csv"
            )

        # 导出评估结果
        results = SessionManager.get('iid_results')
        if results:
            st.markdown("#### 评估结果导出")

            score_df = SessionManager.get('iid_score_df')
            if score_df is not None:
                csv_bytes = score_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="下载评估结果汇总 CSV",
                    data=csv_bytes,
                    file_name="iid_evaluation_summary.csv",
                    mime="text/csv"
                )

        # 保存路径信息
        st.markdown("#### 保存位置")
        output_dir = Path("outputs/dollar_bars")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 显示已保存的文件
        if best_freq is not None:
            saved_file = output_dir / f"dollar_bars_best_{best_freq}.csv"
            if saved_file.exists():
                st.success(f"✅ 最佳结果已保存：{saved_file.absolute()}")
                st.info(f"文件名：dollar_bars_best_{best_freq}.csv")

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回数据导入", use_container_width=True):
        navigate_to("1️⃣ 数据导入")
        st.rerun()

with col3:
    if SessionManager.get('dollar_bars') is not None:
        if st.button("前往特征工程 ➡️", use_container_width=True):
            navigate_to("2️⃣ 特征工程")
            st.rerun()
