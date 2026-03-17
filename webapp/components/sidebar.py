"""侧边栏导航组件"""
import streamlit as st
from pathlib import Path
from typing import Dict, List


# 页面配置
PAGES: Dict[str, dict] = {
    "首页": {
        "icon": "🏠",
        "file": "app.py",
        "description": "AFMLKit Web UI 首页"
    },
    "1️⃣ 数据导入": {
        "icon": "📥",
        "file": "pages/01_data_import.py",
        "description": "导入交易数据并构建 K 线"
    },
    "💵 Dollar Bar": {
        "icon": "💵",
        "file": "pages/02_dollar_bar.py",
        "description": "生成和评估 Dollar Bars"
    },
    "🔬 CUSUM 采样": {
        "icon": "🔬",
        "file": "pages/03_cusum_sampling.py",
        "description": "CUSUM 事件采样与可视化"
    },
    "2️⃣ 特征工程": {
        "icon": "🔧",
        "file": "pages/04_feature_engineering.py",
        "description": "构建和变换特征"
    },
    "3️⃣ 标签生成": {
        "icon": "🏷️",
        "file": "pages/05_labeling.py",
        "description": "生成 TBM 标签和样本权重"
    },
    "4️⃣ 特征分析": {
        "icon": "📊",
        "file": "pages/06_feature_analysis.py",
        "description": "特征重要性和聚类分析"
    },
    "5️⃣ 模型训练": {
        "icon": "🤖",
        "file": "pages/07_model_training.py",
        "description": "训练和评估模型"
    },
    "6️⃣ 回测评估": {
        "icon": "📈",
        "file": "pages/08_backtest.py",
        "description": "回测策略和绩效评估"
    },
    "🎨 可视化中心": {
        "icon": "🎨",
        "file": "pages/09_visualization.py",
        "description": "查看所有可视化结果"
    }
}


def render_sidebar() -> str:
    """渲染侧边栏导航

    Returns:
        选中的页面名称
    """
    with st.sidebar:
        st.title("AFMLKit")
        st.markdown("---")

        # 项目信息
        st.markdown("### 📦 金融机器学习工具包")
        st.markdown(
            "基于 Marcos López de Prado 的 "
            "《Advances in Financial Machine Learning》"
        )
        st.markdown("---")

        # 导航菜单
        st.markdown("### 🧭 导航")

        # 获取当前页面
        current_page = st.session_state.get('current_page', '首页')

        # 创建页面选择
        page_options = list(PAGES.keys())
        selected_index = page_options.index(current_page) if current_page in page_options else 0

        selected_label = st.radio(
            "选择页面",
            options=page_options,
            index=selected_index,
            format_func=lambda x: f"{PAGES[x]['icon']} {x}",
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 当前步骤指示器
        st.markdown("### 📍 当前步骤")
        current_step = st.session_state.get('current_step', 0)
        steps = [
            "数据导入",
            "Dollar Bar",
            "CUSUM 采样",
            "特征工程",
            "标签生成",
            "特征分析",
            "模型训练",
            "回测评估"
        ]
        for i, step in enumerate(steps):
            if i < current_step:
                st.success(f"✅ {step}")
            elif i == current_step:
                st.info(f"🔄 {step}")
            else:
                st.markdown(f"⬜ {step}")

        st.markdown("---")

        # 数据状态
        st.markdown("### 📊 数据状态")
        has_raw = st.session_state.get('raw_data') is not None
        has_bar = st.session_state.get('bar_data') is not None
        has_dollar_bars = st.session_state.get('dollar_bars') is not None
        has_features = st.session_state.get('features') is not None
        has_labels = st.session_state.get('labels') is not None

        status_icon = "✅" if has_raw else "❌"
        st.markdown(f"{status_icon} 原始数据")

        status_icon = "✅" if (has_bar or has_dollar_bars) else "❌"
        st.markdown(f"{status_icon} K 线/Dollar Bars")

        has_cusum = st.session_state.get('cusum_sampled_data') is not None
        status_icon = "✅" if has_cusum else "❌"
        st.markdown(f"{status_icon} CUSUM 采样")

        status_icon = "✅" if has_features else "❌"
        st.markdown(f"{status_icon} 特征")

        status_icon = "✅" if has_labels else "❌"
        st.markdown(f"{status_icon} 标签")

        st.markdown("---")

        # 快速操作
        st.markdown("### ⚡ 快速操作")
        if st.button("🔄 重置所有数据", use_container_width=True):
            from session import SessionManager
            SessionManager.reset_all()
            st.rerun()

        if st.button("💾 保存实验快照", use_container_width=True):
            from session import SessionManager
            name = st.text_input("快照名称", key="snapshot_name_input")
            if name:
                SessionManager.save_snapshot(name)
                st.success(f"已保存快照：{name}")

        # 页脚
        st.markdown("---")
        st.markdown("### ℹ️ 关于")
        st.markdown("AFMLKit Web UI v0.1.0")
        st.markdown("Built with Streamlit")

        return selected_label


def navigate_to(page: str):
    """导航到指定页面

    Args:
        page: 页面名称
    """
    st.session_state.current_page = page

    # 根据页面更新当前步骤
    step_mapping = {
        '首页': 0,
        '1️⃣ 数据导入': 0,
        '💵 Dollar Bar': 1,
        '🔬 CUSUM 采样': 2,
        '2️⃣ 特征工程': 3,
        '3️⃣ 标签生成': 4,
        '4️⃣ 特征分析': 5,
        '5️⃣ 模型训练': 6,
        '6️⃣ 回测评估': 7,
        '🎨 可视化中心': 8
    }
    st.session_state.current_step = step_mapping.get(page, 0)

    # 获取页面文件路径并切换
    page_file = PAGES.get(page, {}).get('file')
    if page_file:
        # 构建页面文件的绝对路径
        pages_dir = Path(__file__).parent.parent
        page_path = pages_dir / page_file

        # 使用 st.switch_page 切换到目标页面
        # 注意：st.switch_page 会立即中断当前脚本执行
        # 因此调用 navigate_to 后的 st.rerun() 不会被执行
        if page_path.exists():
            st.switch_page(str(page_path))


def get_page_description(page: str) -> str:
    """获取页面描述

    Args:
        page: 页面名称

    Returns:
        页面描述
    """
    return PAGES.get(page, {}).get('description', '')
