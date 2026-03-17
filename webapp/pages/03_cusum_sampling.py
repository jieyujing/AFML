"""CUSUM 采样页面"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="CUSUM 采样", page_icon="🔬", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.cusum_viz import plot_price_with_events, render_sampling_rate_panel

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("🔬 CUSUM 采样")
st.markdown("基于 CUSUM Filter 的事件采样，用于提取显著的行情变化点。")

# ==================== 数据源区块 ====================
st.markdown("### 📊 数据源")

data_source_mode = st.radio(
    "选择数据来源",
    ["从 Dollar Bars 采样", "加载已保存结果"],
    horizontal=True
)

selected_df = None
original_df = None
selected_freq = None

if data_source_mode == "从 Dollar Bars 采样":
    dollar_bars = SessionManager.get('dollar_bars')

    if dollar_bars is None or all(len(df) == 0 for df in dollar_bars.values()):
        st.warning("⚠️ 请先生成 Dollar Bars 数据")
        if st.button("前往 Dollar Bar 生成"):
            navigate_to("💵 Dollar Bar")
            st.rerun()
        st.stop()

    # 频率选择
    available_freqs = sorted([f for f, df in dollar_bars.items() if len(df) > 0])
    selected_freq = st.selectbox(
        "选择 Dollar Bar 频率",
        options=available_freqs,
        format_func=lambda x: f"{x} bars/day ({len(dollar_bars[x]):,} rows)"
    )

    selected_df = dollar_bars[selected_freq]
    st.info(f"已选择：{selected_freq} bars/day, 共 {len(selected_df)} 行")

    # 数据验证
    if 'close' not in selected_df.columns:
        st.error("数据缺少 'close' 列，无法执行 CUSUM 采样")
        st.stop()

    if len(selected_df) < 100:
        st.warning(f"数据行数较少 ({len(selected_df)})，采样结果可能不稳定")

else:
    # 加载已保存结果
    cusum_dir = Path("outputs/cusum_sampling")
    if not cusum_dir.exists():
        st.warning("未找到 outputs/cusum_sampling 目录")
        st.stop()

    saved_files = list(cusum_dir.glob("cusum_sampled_*.csv")) + list(cusum_dir.glob("cusum_sampled_*.parquet"))

    if not saved_files:
        st.warning("未找到已保存的 CUSUM 采样结果")
        st.stop()

    file_options = {f.name: f for f in saved_files}
    selected_file_name = st.selectbox("选择文件", list(file_options.keys()))

    if selected_file_name:
        selected_file = file_options[selected_file_name]
        try:
            if selected_file.suffix == '.csv':
                selected_df = pd.read_csv(selected_file, parse_dates=['timestamp'], index_col='timestamp')
            else:
                selected_df = pd.read_parquet(selected_file)

            SessionManager.update('cusum_sampled_data', selected_df)
            st.success(f"✅ 已加载：{selected_file_name} ({len(selected_df)} 行)")

            with st.expander("数据预览"):
                st.dataframe(selected_df.head(10))

            # 跳过后续步骤直接显示结果
            st.stop()

        except Exception as e:
            st.error(f"加载失败：{e}")
            st.stop()

# ==================== 参数配置区块 ====================
st.markdown("### ⚙️ 参数配置")

advanced_mode = st.checkbox("高级模式", value=False, key="cusum_advanced_mode")

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
else:
    vol_span = 50
    threshold_multiplier = 2.0
    use_frac_diff = True

# 保存配置
cusum_config = {
    'vol_span': vol_span,
    'threshold_multiplier': threshold_multiplier,
    'use_frac_diff': use_frac_diff,
    'freq': selected_freq
}

# ==================== 执行采样区块 ====================
st.markdown("### ▶️ 执行采样")

if st.button("🔬 执行 CUSUM 采样", type="primary"):
    try:
        from scripts.cusum_filtering import compute_dynamic_cusum_filter

        SessionManager.set_processing(True)
        progress_bar = st.progress(0, text="正在执行 CUSUM 过滤...")

        # 执行 CUSUM filter
        sampled_df, t_events = compute_dynamic_cusum_filter(
            selected_df,
            price_col='close',
            vol_span=vol_span,
            threshold_multiplier=threshold_multiplier,
            use_frac_diff=use_frac_diff
        )

        progress_bar.progress(100, text="采样完成!")

        # 保存到 Session
        SessionManager.update('cusum_sampled_data', sampled_df)
        SessionManager.update('cusum_events', t_events)
        SessionManager.update('cusum_config', cusum_config)
        SessionManager.set_processing(False)

        st.success(f"✅ 检测到 {len(t_events)} 个 CUSUM 事件")

        # 更新变量用于显示结果
        original_df = selected_df.copy()

    except ImportError as e:
        SessionManager.set_processing(False)
        st.error(f"CUSUM Filter 导入失败：{e}")
    except Exception as e:
        SessionManager.set_processing(False)
        st.error(f"执行失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())

# ==================== 采样结果区块 ====================
cusum_sampled_data = SessionManager.get('cusum_sampled_data')

if cusum_sampled_data is not None:
    st.markdown("---")
    st.markdown("### 📈 采样结果")

    # 获取原始数据用于统计
    cusum_config = SessionManager.get('cusum_config', {})
    dollar_bars = SessionManager.get('dollar_bars', {})
    freq = cusum_config.get('freq')

    if freq and freq in dollar_bars:
        original_df = dollar_bars[freq]

    if original_df is not None:
        original_rows = len(original_df)
        sampled_rows = len(cusum_sampled_data)
        time_range_days = (
            (original_df.index[-1] - original_df.index[0]).total_seconds() / 86400
            if len(original_df) > 1 else 1
        )

        # 采样率面板
        col1, col2 = st.columns([2, 1])
        with col1:
            metrics_md, gauge_fig = render_sampling_rate_panel(
                original_rows, sampled_rows, time_range_days
            )
            st.markdown(metrics_md)
        with col2:
            st.plotly_chart(gauge_fig, use_container_width=True)

        # 价格序列 + 事件标记图
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

    # 数据预览
    with st.expander("📋 采样数据预览"):
        st.dataframe(cusum_sampled_data.head(20))

# ==================== 导出区块 ====================
if cusum_sampled_data is not None:
    st.markdown("---")
    st.markdown("### 💾 导出")

    col1, col2 = st.columns([1, 1])

    with col1:
        export_format = st.selectbox("导出格式", ["CSV", "Parquet"])
        default_filename = f"cusum_sampled_{datetime.now().strftime('%Y%m%d')}"
        export_filename = st.text_input("文件名", value=default_filename)

    with col2:
        st.markdown(" ")  # 占位
        st.markdown(" ")

    if st.button("💾 导出结果"):
        output_dir = Path("outputs/cusum_sampling")
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_format == "CSV":
            filepath = output_dir / f"{export_filename}.csv"
            if filepath.exists():
                if not st.checkbox(f"文件已存在，确认覆盖 {filepath.name}?"):
                    st.stop()
            cusum_sampled_data.to_csv(filepath)
        else:
            filepath = output_dir / f"{export_filename}.parquet"
            if filepath.exists():
                if not st.checkbox(f"文件已存在，确认覆盖 {filepath.name}?"):
                    st.stop()
            cusum_sampled_data.to_parquet(filepath)

        st.success(f"✅ 已导出到：{filepath}")

# ==================== 导航按钮 ====================
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回 Dollar Bar", use_container_width=True):
        navigate_to("💵 Dollar Bar")
        st.rerun()

with col3:
    if SessionManager.get('cusum_sampled_data') is not None:
        if st.button("前往特征工程 ➡️", use_container_width=True):
            navigate_to("2️⃣ 特征工程")
            st.rerun()