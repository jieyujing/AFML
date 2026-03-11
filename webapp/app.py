"""AFMLKit Web UI - Streamlit 主入口"""
import streamlit as st
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="AFMLKit - 金融机器学习工具包",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
from session import SessionManager
SessionManager.init_session()

# 导入组件
from components.sidebar import render_sidebar, navigate_to, PAGES
from components.data_loader import render_data_loader, render_data_preview
from components.charts import display_metrics

# 设置页面标题
st.title("📈 AFMLKit Web UI")
st.markdown("基于 Marcos López de Prado《Advances in Financial Machine Learning》的金融机器学习工具包")

# 渲染侧边栏并获取选择
selected_page = render_sidebar()

# 处理页面导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

# 显示当前页面描述
if st.session_state.get('current_page', '首页') == '首页':
    st.markdown("---")

    # 项目概述
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 功能特性")
        st.markdown("""
        - **数据导入**: 支持 CSV、Parquet、HDF5 格式
        - **K 线构建**: 时间/Tick/成交量/金额/CUSUM/不平衡 K 线
        - **特征工程**: 丰富的特征变换和构建器
        - **标签生成**: 三重屏障法（TBM）和样本权重
        - **特征分析**: 聚类分析和特征重要性
        - **模型训练**: Purged CV 和随机森林/梯度提升
        - **回测评估**: Sharpe、PSR、DSR、最大回撤
        """)

    with col2:
        st.markdown("### 🚀 快速开始")
        st.markdown("""
        1. 从左侧导航栏选择"数据导入"
        2. 上传或选择您的交易数据
        3. 配置 K 线参数并构建
        4. 进行特征工程和标签生成
        5. 训练模型并回测评估
        """)

    st.markdown("---")

    # 显示当前数据状态
    st.markdown("### 📊 当前数据状态")

    data_status = {
        "原始数据": st.session_state.get('raw_data'),
        "K 线数据": st.session_state.get('bar_data'),
        "特征": st.session_state.get('features'),
        "标签": st.session_state.get('labels'),
        "模型": st.session_state.get('model'),
        "回测结果": st.session_state.get('backtest_results')
    }

    status_cols = st.columns(6)
    for i, (name, data) in enumerate(data_status.items()):
        with status_cols[i]:
            if data is not None:
                if isinstance(data, dict) and not data:
                    st.info(f"⬜ {name}")
                else:
                    st.success(f"✅ {name}")
            else:
                st.info(f"⬜ {name}")

    # 最近实验
    from config import ConfigManager
    config_mgr = ConfigManager()
    experiments = config_mgr.list_experiments()

    if experiments:
        st.markdown("---")
        st.markdown("### 📁 最近实验")

        exp_cols = st.columns(min(3, len(experiments)))
        for i, exp in enumerate(experiments[:3]):
            with exp_cols[i]:
                st.markdown(f"**{exp.name}**")
                st.caption(f"修改时间：{exp.stat().st_mtime}")

    # 页脚
    st.markdown("---")
    st.markdown("### ℹ️ 关于")
    st.markdown(
        "AFMLKit 是一个基于 Advances in Financial Machine Learning 理论构建的金融机器学习工具包。\n"
        "Web UI 版本提供了交互式的界面，让研究和实验变得更加简单。"
    )

# 注意：实际页面渲染由各个页面文件处理
# 这里是首页内容
