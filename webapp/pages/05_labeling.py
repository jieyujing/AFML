"""标签生成页面 - TBM 三重屏障法"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="标签生成", page_icon="🏷️", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from components.charts import render_line_chart, render_bar_chart, display_metrics

# 初始化会话
SessionManager.init_session()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("🏷️ 标签生成 - 三重屏障法 (TBM)")

# 检查是否有特征数据
features = SessionManager.get('features')
bar_data = SessionManager.get('bar_data')

if bar_data is None:
    st.warning("⚠️ 请先导入数据并构建 K 线")
    st.stop()

st.info(f"当前数据：{len(bar_data)} 根 K 线")

# 步骤选择
step = st.radio(
    "选择步骤",
    ["1. TBM 参数配置", "2. 生成标签", "3. 样本权重", "4. 标签分析"],
    horizontal=True
)

if step == "1. TBM 参数配置":
    st.markdown("### 1️⃣ TBM 参数配置")

    st.markdown("""
    **三重屏障法 (Triple Barrier Method)** 是 López de Prado 提出的标签生成方法：
    - **上屏障**: 止盈线
    - **下屏障**: 止损线
    - **垂直屏障**: 时间期限
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        profit_barrier = st.number_input(
            "止盈屏障 (上屏障)",
            min_value=0.001,
            max_value=1.0,
            value=0.02,
            step=0.001,
            help="价格上漲超过此值时标记为 +1"
        )

    with col2:
        loss_barrier = st.number_input(
            "止损屏障 (下屏障)",
            min_value=0.001,
            max_value=1.0,
            value=0.02,
            step=0.001,
            help="价格下跌超过此值时标记为 -1"
        )

    with col3:
        time_horizon = st.number_input(
            "时间期限 (小时)",
            min_value=1,
            max_value=168,
            value=24,
            step=1,
            help="超过此时间后强制平仓"
        )

    st.markdown("#### 高级参数")

    col1, col2 = st.columns(2)

    with col1:
        min_return = st.number_input(
            "最小收益率",
            min_value=0.0,
            max_value=0.1,
            value=0.0,
            step=0.001,
            help="低于此收益率的样本被过滤"
        )

    with col2:
        is_meta = st.checkbox(
            "Meta 标签模式",
            value=False,
            help="启用后生成 Meta 标签用于模型选择"
        )

    # 保存配置
    if st.button("保存 TBM 配置"):
        label_config = {
            'profit_barrier': profit_barrier,
            'loss_barrier': loss_barrier,
            'time_horizon': time_horizon,
            'min_return': min_return,
            'is_meta': is_meta
        }
        SessionManager.update('label_config', label_config)
        st.success("TBM 配置已保存")
        st.json(label_config)

elif step == "2. 生成标签":
    st.markdown("### 2️⃣ 生成标签")

    label_config = SessionManager.get('label_config', {})

    if not label_config:
        st.warning("请先配置 TBM 参数")
    else:
        st.markdown("#### 当前配置")
        st.json(label_config)

        if st.button("开始生成 TBM 标签"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("准备数据...")
                progress_bar.progress(10)

                # 准备数据
                df = bar_data.copy()

                # 计算波动率
                status_text.text("计算波动率...")
                from afmlkit.feature.core.volatility import ewms

                log_ret = np.log(df['close'] / df['close'].shift(1)).values.astype(np.float64)
                volatility = ewms(log_ret, span=50)
                df['volatility'] = volatility
                progress_bar.progress(30)

                # 创建 TradesData
                status_text.text("创建 TradesData 对象...")
                from afmlkit.bar.data_model import TradesData

                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')

                ts = df.index.astype(np.int64).values
                px = df['close'].values.astype(np.float64)
                qty = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones(len(px))

                trades = TradesData(
                    ts=ts,
                    px=px,
                    qty=qty,
                    id=np.arange(len(df)),
                    timestamp_unit='ns',
                    preprocess=False
                )
                progress_bar.progress(50)

                # 创建 TBMLabel
                status_text.text("实例化 TBMLabel...")
                from afmlkit.label.kit import TBMLabel

                tbm = TBMLabel(
                    features=df,
                    target_ret_col='volatility',
                    min_ret=label_config.get('min_return', 0.0),
                    horizontal_barriers=(
                        label_config.get('profit_barrier', 1.0),
                        label_config.get('loss_barrier', 1.0)
                    ),
                    vertical_barrier=pd.Timedelta(hours=label_config.get('time_horizon', 24)),
                    is_meta=label_config.get('is_meta', False),
                    min_close_time=pd.Timedelta(seconds=1)
                )
                progress_bar.progress(70)

                # 计算标签
                status_text.text("计算 TBM 标签...")
                start_time = time.time()

                feats, out = tbm.compute_labels(trades)

                elapsed = time.time() - start_time
                st.info(f"计算耗时：{elapsed:.2f}秒")
                progress_bar.progress(90)

                # 显示标签分布
                st.markdown("#### 标签分布")
                if 'bin' in out.columns:
                    label_dist = out['bin'].value_counts().sort_index()
                    st.write(label_dist)

                    # 可视化
                    fig = render_bar_chart(label_dist, title="标签分布")
                    st.plotly_chart(fig, use_container_width=True)

                # 保存结果
                SessionManager.update('labels', out.get('bin'))
                SessionManager.update('label_data', out)
                SessionManager.update('tbm', tbm)

                SessionManager.set_processing(False)
                status_text.text("标签生成完成!")
                progress_bar.progress(100)

                st.success(f"✅ 生成 {len(out)} 个标签样本")

            except Exception as e:
                SessionManager.set_processing(False)
                st.error(f"标签生成失败：{str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif step == "3. 样本权重":
    st.markdown("### 3️⃣ 样本权重")

    tbm = SessionManager.get('tbm')
    label_data = SessionManager.get('label_data')

    if tbm is None or label_data is None:
        st.warning("请先生成 TBM 标签")
    else:
        st.markdown("#### 样本权重计算方法")

        weight_method = st.selectbox(
            "选择权重方法",
            ["concurrency", "return_attribution", "time_decay", "combined"],
            format_func=lambda x: {
                'concurrency': '并发权重 (Concurrency)',
                'return_attribution': '收益归因权重',
                'time_decay': '时间衰减权重',
                'combined': '组合权重'
            }.get(x, x)
        )

        if weight_method == 'time_decay':
            decay_factor = st.slider("衰减因子", 0.0, 1.0, 0.5, 0.1)

        if st.button("计算样本权重"):
            try:
                from afmlkit.bar.data_model import TradesData

                # 重新创建 TradesData (如果需要)
                label_data_copy = label_data.copy()

                status_text = st.empty()
                status_text.text("计算权重...")

                # 获取必要的列
                if 't1' in label_data.columns:
                    # 计算权重
                    weights = tbm.compute_weights(None)  # trades 可能不需要

                    if isinstance(weights, pd.DataFrame):
                        if 'weight' in weights.columns:
                            label_data['weight'] = weights['weight']
                        elif 'sample_weight' in weights.columns:
                            label_data['weight'] = weights['sample_weight']
                    else:
                        label_data['weight'] = weights

                    st.success(f"✅ 计算 {len(label_data)} 个样本权重")

                    # 显示权重分布
                    st.markdown("#### 权重分布")
                    st.dataframe(label_data[['weight']].describe())

                    # 权重直方图
                    fig = render_bar_chart(
                        pd.Series(label_data['weight'].values),
                        title="样本权重分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 保存
                    SessionManager.update('sample_weights', label_data['weight'])
                    SessionManager.update('label_data', label_data)

                else:
                    st.warning("缺少 t1 列，无法计算并发权重")

            except Exception as e:
                st.error(f"权重计算失败：{str(e)}")
                import traceback
                st.code(traceback.format_exc())

elif step == "4. 标签分析":
    st.markdown("### 4️⃣ 标签分析")

    label_data = SessionManager.get('label_data')

    if label_data is None:
        st.warning("请先生成标签")
    else:
        st.markdown("#### 标签统计")

        # 基本统计
        if 'bin' in label_data.columns:
            col1, col2, col3 = st.columns(3)

            with col1:
                total = len(label_data)
                st.metric("总样本数", total)

            with col2:
                pos_count = (label_data['bin'] == 1).sum()
                st.metric("正样本数", pos_count)

            with col3:
                neg_count = (label_data['bin'] == -1).sum()
                st.metric("负样本数", neg_count)

            # 正负样本比例
            if neg_count > 0:
                ratio = pos_count / neg_count
                st.info(f"正负样本比例：{ratio:.2f}")
            else:
                st.info("没有负样本")

        # 标签随时间分布
        st.markdown("#### 标签时间序列")

        label_data_copy = label_data.copy()
        label_data_copy['label_abs'] = label_data_copy['bin'].abs()

        # 按日期聚合
        if hasattr(label_data_copy.index, 'to_period'):
            label_data_copy['date'] = label_data_copy.index.date
            daily_labels = label_data_copy.groupby('date')['bin'].mean()

            fig = render_line_chart(
                pd.DataFrame({'daily_label_mean': daily_labels}),
                ['daily_label_mean'],
                title="日均标签值"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 收益分布
        if 'return' in label_data.columns or 'ret' in label_data.columns:
            st.markdown("#### 收益分布")

            ret_col = 'return' if 'return' in label_data.columns else 'ret'
            from components.charts import render_distribution_chart

            fig = render_distribution_chart(
                label_data[ret_col].dropna().values,
                title="收益分布"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 权重分析
        if 'weight' in label_data.columns:
            st.markdown("#### 权重分析")

            weight_stats = label_data['weight'].describe()
            st.dataframe(weight_stats)

# 导航按钮
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("⬅️ 返回特征工程", use_container_width=True):
        navigate_to("2️⃣ 特征工程")
        st.rerun()

with col3:
    if SessionManager.get('labels') is not None:
        if st.button("前往特征分析 ➡️", use_container_width=True):
            navigate_to("4️⃣ 特征分析")
            st.rerun()
