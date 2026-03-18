"""特征工程页面"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import project modules
from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import render_line_chart
from webapp.utils.feature_calculator import compute_all_features

st.set_page_config(page_title="特征工程", page_icon="🔧", layout="wide")

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get("current_page", "首页"):
    navigate_to(selected_page)
    st.rerun()

st.title("🔧 特征工程")

# 检查是否有 K 线数据或 Dollar Bars 数据
bar_data = SessionManager.get("bar_data")
dollar_bars = SessionManager.get("dollar_bars")

# 如果 Session 中没有数据，尝试从文件加载
if bar_data is None and dollar_bars is None:
    # 尝试从 outputs/dollar_bars 加载已生成的 Dollar Bars
    output_dir = Path("outputs/dollar_bars")
    if output_dir.exists():
        dollar_bar_files = list(output_dir.glob("dollar_bars_*.csv"))
        if dollar_bar_files:
            st.info(
                f"📁 从 `{output_dir}` 发现 {len(dollar_bar_files)} 个 Dollar Bars 文件"
            )

            # 让用户选择文件
            file_options = {f.name: f for f in dollar_bar_files}
            selected_file_name = st.selectbox(
                "选择 Dollar Bars 文件", options=list(file_options.keys())
            )

            if selected_file_name:
                selected_file = file_options[selected_file_name]
                try:
                    df = pd.read_csv(selected_file, parse_dates=["timestamp"])
                    df = df.set_index("timestamp")

                    # 提取频率（从文件名）
                    freq = selected_file.stem.replace("dollar_bars_", "")

                    # 保存到 Session
                    SessionManager.update("dollar_bars", {freq: df})
                    dollar_bars = SessionManager.get("dollar_bars")
                    st.success(f"✅ 已加载：{selected_file_name}")
                except Exception as e:
                    st.error(f"加载失败：{e}")

# 数据源选择逻辑
if bar_data is None and (
    dollar_bars is None or all(len(df) == 0 for df in dollar_bars.values())
):
    # 既没有 bar_data 也没有 dollar_bars
    st.warning("⚠️ 请先导入数据并构建 K 线")
    if st.button("前往数据导入"):
        navigate_to("1️⃣ 数据导入")
        st.rerun()
    st.stop()

elif dollar_bars is not None:
    # 存在 Dollar Bars 数据，让用户选择频率
    st.markdown("### 📊 数据源选择")

    # 获取可用的频率选项
    available_freqs = sorted([f for f, df in dollar_bars.items() if len(df) > 0])

    if not available_freqs:
        st.warning("没有找到有效的 Dollar Bars 数据")
        if st.button("前往 Dollar Bar 生成"):
            navigate_to("💵 Dollar Bar")
            st.rerun()
        st.stop()

    # 频率选择器
    selected_freq = st.selectbox(
        "选择 Dollar Bar 频率",
        options=available_freqs,
        format_func=lambda x: f"{x} bars/day ({len(dollar_bars[x]):,} rows)",
    )

    # 使用选中的频率数据作为 bar_data
    bar_data = dollar_bars[selected_freq]

    st.info(f"当前使用：{selected_freq} bars/day, 共 {len(bar_data)} 行")

    # 显示数据统计
    with st.expander("📊 数据统计摘要"):
        st.dataframe(bar_data.describe())

elif bar_data is not None:
    # 只有传统 bar_data（非 Dollar Bar 流程）
    st.info(f"当前 K 线数据：{len(bar_data)} 行")

# 特征类别
FEATURE_CATEGORIES = {
    "波动率": ["volatility"],
    "动量": ["momentum"],
    "趋势": ["trend"],
    "成交量": ["volume"],
    "相关性": ["correlation"],
    "结构断点": ["structural_break"],
}

# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. 特征配置", "2. 特征计算", "3. 特征预览", "4. 特征导出"],
    horizontal=True,
)

if step == "1. 特征配置":
    st.markdown("### 1️⃣ 特征配置")

    # 特征类别选择
    selected_categories = st.multiselect(
        "选择要计算的特征类别",
        options=list(FEATURE_CATEGORIES.keys()),
        default=["波动率", "动量"],
    )

    # 参数配置
    st.markdown("#### 参数配置")

    # 波动率参数
    with st.expander("波动率参数", expanded=True):
        vol_spans_input = st.text_input(
            "波动率窗口 (逗号分隔)",
            value="10,50,100",
            help="EWM 波动率计算的时间窗口，多个值用逗号分隔",
        )
        vol_spans = [
            int(x.strip()) for x in vol_spans_input.split(",") if x.strip().isdigit()
        ]

    # 动量参数
    with st.expander("动量参数", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            rsi_window = st.number_input(
                "RSI 窗口", min_value=5, max_value=50, value=14
            )
        with col2:
            roc_period = st.number_input(
                "ROC 周期", min_value=5, max_value=50, value=10
            )

    # 高级参数
    with st.expander("高级参数 (FFD)", expanded=False):
        frac_diff_enabled = st.checkbox("启用分数阶差分", value=True)
        col1, col2 = st.columns(2)
        with col1:
            frac_diff_thres = st.number_input(
                "FFD 阈值",
                min_value=0.00001,
                max_value=0.001,
                value=0.0001,
                format="%.5f",
                step=0.00001,
            )
        with col2:
            frac_diff_d_step = st.number_input(
                "d 步长", min_value=0.01, max_value=0.2, value=0.05, step=0.01
            )

    # Alpha158 配置 (FFD 改造版)
    with st.expander("Alpha158 特征 (FFD 改造版)", expanded=False):
        alpha158_enabled = st.checkbox(
            "启用 Alpha158 特征",
            value=False,
            help="启用后将计算 FFD 变换的 Alpha158 风格特征，与现有特征并存",
        )

        if alpha158_enabled:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**波动率窗口**")
                alpha158_vol_spans = st.text_input(
                    "波动率 spans (逗号分隔)", value="5,10,20"
                )

            with col2:
                st.markdown("**均线窗口**")
                alpha158_ma_windows = st.text_input(
                    "MA windows (逗号分隔)", value="5,10,20"
                )

            alpha158_rank_enabled = st.checkbox("启用时序 Rank 特征", value=True)

            alpha158_rank_window = st.number_input(
                "Rank 窗口", min_value=5, max_value=60, value=20
            )

    # CUSUM 对齐选项
    with st.expander("CUSUM 事件对齐", expanded=False):
        # 优先从 Session 获取 CUSUM 数据
        cusum_sampled_data = SessionManager.get("cusum_sampled_data")

        if cusum_sampled_data is not None:
            st.success(f"✅ 已从 Session 加载 CUSUM 数据：{len(cusum_sampled_data)} 行")
            align_to_cusum = st.checkbox(
                "对齐到 CUSUM 事件时间戳",
                value=True,
                help="如果启用，特征将与 CUSUM 采样的事件时间戳对齐",
            )
            cusum_path = None  # 不需要路径，使用 Session 数据
        else:
            st.info("Session 中没有 CUSUM 数据，可以从文件加载")
            align_to_cusum = st.checkbox(
                "对齐到 CUSUM 事件时间戳",
                value=False,
                help="如果启用，特征将与 CUSUM 采样的事件时间戳对齐",
            )
            cusum_path = st.text_input(
                "CUSUM 文件路径", value="outputs/cusum_sampling/cusum_sampled.csv"
            )

    # 保存配置
    if st.button("保存特征配置"):
        feature_config = {
            "categories": selected_categories,
            "volatility": {"spans": vol_spans},
            "momentum": {"rsi_window": rsi_window, "roc_period": roc_period},
            "fractional_diff": {
                "enabled": frac_diff_enabled,
                "threshold": frac_diff_thres,
                "d_step": frac_diff_d_step,
            },
            "cusum": {"align_enabled": align_to_cusum, "path": cusum_path},
        }

        # Add Alpha158 config if enabled
        if alpha158_enabled:
            feature_config["alpha158"] = {
                "enabled": True,
                "volatility": {
                    "spans": [
                        int(x.strip())
                        for x in alpha158_vol_spans.split(",")
                        if x.strip().isdigit()
                    ]
                },
                "ma": {
                    "windows": [
                        int(x.strip())
                        for x in alpha158_ma_windows.split(",")
                        if x.strip().isdigit()
                    ]
                },
                "rank": {
                    "enabled": alpha158_rank_enabled,
                    "window": alpha158_rank_window,
                },
            }

        SessionManager.update("feature_config", feature_config)
        st.success("特征配置已保存")
        st.json(feature_config)

elif step == "2. 特征计算":
    st.markdown("### 2️⃣ 特征计算")

    feature_config = SessionManager.get("feature_config")

    if not feature_config:
        st.warning("请先配置特征参数")
    else:
        st.markdown("#### 当前配置")
        st.json(feature_config)

        if not st.button("开始计算特征"):
            st.stop()

        SessionManager.set_processing(True)
        progress_bar = st.progress(0)
        status_text = st.empty()
        info_container = st.empty()

        try:
            df = bar_data.copy()
            status_text.text("准备数据...")
            progress_bar.progress(10)

            # 获取 CUSUM 数据：优先 Session，fallback 到文件路径
            cusum_sampled_data = SessionManager.get("cusum_sampled_data")
            cusum_path = feature_config.get("cusum", {}).get("path", "")
            align_enabled = feature_config.get("cusum", {}).get("align_enabled", False)

            # 验证 CUSUM 数据可用性
            if align_enabled:
                if cusum_sampled_data is None and cusum_path:
                    if not Path(cusum_path).exists():
                        st.warning(f"CUSUM 文件不存在：{cusum_path}，将使用连续特征模式")
                        align_enabled = False
                elif cusum_sampled_data is None and not cusum_path:
                    st.warning("Session 中没有 CUSUM 数据且未指定文件路径，将使用连续特征模式")
                    align_enabled = False

            status_text.text("计算特征 (可能需要几分钟)...")
            progress_bar.progress(30)

            features_df, metadata = compute_all_features(
                df=df,
                config=feature_config,
                cusum_data=cusum_sampled_data if align_enabled else None,
                cusum_path=cusum_path if align_enabled and cusum_sampled_data is None else None,
                align_to_cusum=align_enabled,
            )

            progress_bar.progress(80)
            status_text.text("计算完成!")

            with st.expander("📊 计算摘要", expanded=True):
                st.metric("最优 FFD 参数 d", f"{metadata.get('optimal_d', 'N/A'):.4f}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("清理前行数", f"{metadata.get('rows_before_clean', 0):,}")
                with col2:
                    st.metric("清理后行数", f"{metadata.get('rows_after_clean', 0):,}")
                with col3:
                    st.metric("丢弃行数", f"{metadata.get('rows_dropped', 0):,}")

                # Alpha158 metadata
                if metadata.get("alpha158_enabled"):
                    st.markdown("**Alpha158 特征:**")
                    st.write(
                        f"  - 最优 d*: {metadata.get('alpha158_optimal_d', 'N/A'):.4f}"
                    )
                    st.write(
                        f"  - 特征数量：{len(metadata.get('alpha158_columns', []))}"
                    )

                if metadata.get("label_distribution"):
                    st.markdown("**标签分布:**")
                    label_dist = metadata["label_distribution"]
                    for label, count in sorted(label_dist.items()):
                        total = sum(label_dist.values())
                        pct = count / total * 100
                        st.write(f"  bin={int(label):+d}: {count:,} ({pct:.1f}%)")

            progress_bar.progress(90)

            SessionManager.update("features", features_df)
            SessionManager.update("feature_metadata", metadata)
            SessionManager.set_processing(False)
            progress_bar.progress(100)

            st.success(
                f"✅ 特征计算完成！共 {len(features_df)} 行，{len(features_df.columns)} 个特征"
            )

            st.markdown("#### 特征列表")
            feature_cols = [
                col
                for col in features_df.columns
                if col not in ["open", "high", "low", "close", "volume", "timestamp"]
            ]
            st.write(feature_cols)

        except Exception as e:
            SessionManager.set_processing(False)
            st.error(f"特征计算失败：{str(e)}")
            import traceback

            st.code(traceback.format_exc())

elif step == "3. 特征预览":
    st.markdown("### 3️⃣ 特征预览")

    features = SessionManager.get("features")
    metadata = SessionManager.get("feature_metadata")

    if features is None:
        st.warning("请先计算特征")
    else:
        if metadata:
            with st.expander("📊 计算元数据", expanded=False):
                st.json(metadata)

        st.markdown("#### 数据统计")
        st.dataframe(features.describe())

        st.markdown("#### 特征相关性")

        feature_cols = [
            col
            for col in features.columns
            if col
            not in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "timestamp",
                "bin",
                "t1",
                "avg_uniqueness",
                "return_attribution",
            ]
        ]

        if len(feature_cols) > 1:
            corr_matrix = features[feature_cols].corr()
            st.dataframe(corr_matrix)

            from components.charts import render_heatmap

            fig = render_heatmap(corr_matrix, title="特征相关性矩阵")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 特征分布")

        selected_features = st.multiselect(
            "选择特征查看分布",
            feature_cols,
            default=feature_cols[:3] if feature_cols else [],
        )

        if selected_features:
            for feat in selected_features:
                if feat in features.columns:
                    fig = render_line_chart(features, [feat], title=feat)
                    st.plotly_chart(fig, use_container_width=True)

elif step == "4. 特征导出":
    st.markdown("### 4️⃣ 特征导出")

    features = SessionManager.get("features")

    if features is None:
        st.warning("没有可导出的特征数据")
    else:
        st.markdown(f"当前特征数据：{len(features)} 行 × {len(features.columns)} 列")

        # 导出格式选择
        fmt = st.selectbox("导出格式", ["CSV", "Parquet", "HDF5"])

        col1, col2 = st.columns(2)

        with col1:
            filename = st.text_input("文件名", "feature_matrix")

        with col2:
            if st.button("导出文件"):
                output_dir = Path(__file__).parent.parent / "outputs"
                output_dir.mkdir(exist_ok=True)

                if fmt == "CSV":
                    filepath = output_dir / f"{filename}.csv"
                    features.to_csv(filepath)
                elif fmt == "Parquet":
                    filepath = output_dir / f"{filename}.parquet"
                    features.to_parquet(filepath)
                elif fmt == "HDF5":
                    import h5py

                    filepath = output_dir / f"{filename}.h5"
                    with h5py.File(filepath, "w") as f:
                        for col in features.columns:
                            f.create_dataset(col, data=features[col].values)

                st.success(f"已导出到 {filepath}")

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回数据导入", use_container_width=True):
        navigate_to("1️⃣ 数据导入")
        st.rerun()

with col3:
    if SessionManager.get("features") is not None:
        if st.button("前往标签生成 ➡️", use_container_width=True):
            navigate_to("3️⃣ 标签生成")
            st.rerun()
