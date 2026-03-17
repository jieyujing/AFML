"""实验管理页面"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="实验管理", page_icon="📁", layout="wide")

from session import SessionManager
from components.sidebar import render_sidebar, navigate_to
from config import ConfigManager

# 初始化会话
SessionManager.init_session()
config_mgr = ConfigManager()

# 渲染侧边栏
selected_page = render_sidebar()

# 处理导航
if selected_page != st.session_state.get('current_page', '首页'):
    navigate_to(selected_page)
    st.rerun()

st.title("📁 实验管理")

st.markdown("""
实验管理功能允许您：
- 保存当前研究配置和结果
- 加载历史实验进行比较
- 删除不再需要的实验
""")

# 功能选择
feature = st.radio(
    "选择功能",
    ["保存实验", "加载实验", "实验列表", "删除实验"],
    horizontal=True
)

if feature == "保存实验":
    st.markdown("### 💾 保存实验")

    # 实验名称
    exp_name = st.text_input("实验名称", value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 实验备注
    exp_notes = st.text_area("实验备注", placeholder="记录实验目的、参数调整等信息...")

    # 选择保存内容
    st.markdown("#### 保存内容")

    save_data = st.checkbox("保存数据", value=False)
    save_config = st.checkbox("保存配置", value=True)
    save_model = st.checkbox("保存模型", value=False)
    save_results = st.checkbox("保存回测结果", value=False)

    if st.button("保存实验"):
        try:
            # 收集要保存的数据
            experiment_data = {
                'name': exp_name,
                'timestamp': datetime.now().isoformat(),
                'notes': exp_notes,
            }

            if save_config:
                experiment_data['bar_config'] = SessionManager.get('bar_config')
                experiment_data['feature_config'] = SessionManager.get('feature_config')
                experiment_data['label_config'] = SessionManager.get('label_config')
                experiment_data['model_config'] = SessionManager.get('model_config')
                experiment_data['backtest_config'] = SessionManager.get('backtest_config')

            if save_model:
                model = SessionManager.get('model')
                if model is not None:
                    st.warning("模型保存功能尚未实现，仅保存模型配置")
                    experiment_data['model_config'] = SessionManager.get('model_config')

            if save_results:
                experiment_data['backtest_results'] = SessionManager.get('backtest_results')
                experiment_data['cv_results'] = SessionManager.get('cv_results')

            # 保存为 YAML
            filepath = config_mgr.save_config(experiment_data, exp_name)

            st.success(f"✅ 实验已保存到：{filepath}")

        except Exception as e:
            st.error(f"保存失败：{str(e)}")

elif feature == "加载实验":
    st.markdown("### 📂 加载实验")

    # 获取实验列表
    experiments = config_mgr.list_experiments()

    if not experiments:
        st.warning("没有找到已保存的实验")
    else:
        # 实验选择
        exp_options = {str(exp.name): exp for exp in experiments}
        selected_exp = st.selectbox("选择实验", list(exp_options.keys()))

        if selected_exp:
            exp_path = exp_options[selected_exp]

            # 显示实验信息
            st.markdown("#### 实验信息")

            try:
                exp_data = config_mgr.load_config(exp_path)

                st.json({
                    '名称': exp_data.get('name', 'N/A'),
                    '时间': exp_data.get('timestamp', 'N/A'),
                    '备注': exp_data.get('notes', 'N/A')
                })

                # 选择加载内容
                st.markdown("#### 加载内容")

                load_config = st.checkbox("加载配置", value=True)
                load_results = st.checkbox("加载结果", value=False)

                if st.button("加载实验"):
                    try:
                        if load_config:
                            if 'bar_config' in exp_data:
                                SessionManager.update('bar_config', exp_data['bar_config'])
                            if 'feature_config' in exp_data:
                                SessionManager.update('feature_config', exp_data['feature_config'])
                            if 'label_config' in exp_data:
                                SessionManager.update('label_config', exp_data['label_config'])
                            if 'model_config' in exp_data:
                                SessionManager.update('model_config', exp_data['model_config'])
                            if 'backtest_config' in exp_data:
                                SessionManager.update('backtest_config', exp_data['backtest_config'])

                        if load_results:
                            if 'backtest_results' in exp_data:
                                SessionManager.update('backtest_results', exp_data['backtest_results'])
                            if 'cv_results' in exp_data:
                                SessionManager.update('cv_results', exp_data['cv_results'])

                        st.success("✅ 实验已加载")

                        if st.button("前往回测页面"):
                            navigate_to("6️⃣ 回测评估")
                            st.rerun()

                    except Exception as e:
                        st.error(f"加载失败：{str(e)}")

            except Exception as e:
                st.error(f"读取实验失败：{str(e)}")

elif feature == "实验列表":
    st.markdown("### 📋 实验列表")

    experiments = config_mgr.list_experiments()

    if not experiments:
        st.info("暂无已保存的实验")
    else:
        # 构建实验信息表格
        exp_data_list = []

        for exp in experiments:
            try:
                data = config_mgr.load_config(exp)
                exp_data_list.append({
                    '名称': data.get('name', exp.name),
                    '时间': data.get('timestamp', 'N/A'),
                    '备注': data.get('notes', '-')[:50],
                    '文件大小': f"{exp.stat().st_size / 1024:.1f} KB"
                })
            except:
                exp_data_list.append({
                    '名称': exp.name,
                    '时间': 'N/A',
                    '备注': '-',
                    '文件大小': f"{exp.stat().st_size / 1024:.1f} KB"
                })

        exp_df = pd.DataFrame(exp_data_list)
        st.dataframe(exp_df)

        st.markdown(f"共 {len(experiments)} 个实验")

elif feature == "删除实验":
    st.markdown("### 🗑️ 删除实验")

    experiments = config_mgr.list_experiments()

    if not experiments:
        st.info("暂无已保存的实验")
    else:
        # 实验选择
        exp_options = {str(exp.name): exp for exp in experiments}
        selected_exp = st.selectbox("选择要删除的实验", list(exp_options.keys()))

        if selected_exp:
            exp_path = exp_options[selected_exp]

            # 确认删除
            st.warning(f"确定要删除实验 **{selected_exp}** 吗？此操作不可恢复!")

            if st.button("确认删除"):
                try:
                    config_mgr.delete_experiment(exp_path.name)
                    st.success(f"✅ 已删除实验：{selected_exp}")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败：{str(e)}")

# 侧边栏快捷操作
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ 快捷操作")

if st.sidebar.button("保存当前快照"):
    exp_name = st.sidebar.text_input("快照名称", key="sidebar_exp_name")
    if exp_name:
        try:
            filepath = config_mgr.save_config({
                'name': exp_name,
                'timestamp': datetime.now().isoformat(),
                'snapshot': True
            }, exp_name)
            st.sidebar.success(f"已保存：{filepath}")
        except Exception as e:
            st.sidebar.error(f"保存失败：{str(e)}")
