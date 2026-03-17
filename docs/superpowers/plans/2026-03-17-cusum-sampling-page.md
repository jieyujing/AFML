# CUSUM 采样页面实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 CUSUM Filter 功能从数据导入页面拆分为独立页面，实现完整的数据源选择、参数配置、执行采样、结果可视化和导出功能。

**Architecture:** 单页分组布局，5 个功能区块；复用现有 cusum_viz.py 可视化组件和 scripts/cusum_filtering.py 计算逻辑；通过 SessionManager 传递数据到特征工程页面。

**Tech Stack:** Streamlit, Plotly, Pandas, NumPy

---

## 文件结构

### 新增文件

| 文件 | 职责 |
|------|------|
| `webapp/pages/03_cusum_sampling.py` | CUSUM 采样页面主文件 |
| `outputs/cusum_sampling/.gitkeep` | 导出目录占位 |

### 重命名文件

| 原路径 | 新路径 |
|--------|--------|
| `webapp/pages/02_dollar_bar_generator.py` | `webapp/pages/02_dollar_bar.py` |
| `webapp/pages/03_feature_engineering.py` | `webapp/pages/04_feature_engineering.py` |
| `webapp/pages/04_labeling.py` | `webapp/pages/05_labeling.py` |
| `webapp/pages/05_feature_analysis.py` | `webapp/pages/06_feature_analysis.py` |
| `webapp/pages/06_model_training.py` | `webapp/pages/07_model_training.py` |
| `webapp/pages/07_backtest.py` | `webapp/pages/08_backtest.py` |
| `webapp/pages/08_visualization.py` | `webapp/pages/09_visualization.py` |
| `webapp/pages/09_experiment.py` | `webapp/pages/10_experiment.py` |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `webapp/session.py` | 添加 CUSUM 状态键 |
| `webapp/components/sidebar.py` | 更新 PAGES、steps、step_mapping、数据状态面板 |
| `webapp/pages/01_data_import.py` | 移除 cusum K 线选项代码 |
| `webapp/pages/04_feature_engineering.py` | 从 Session 读取 cusum_sampled_data |

---

## Task 1: 重命名现有页面文件

**Files:**
- Rename: `webapp/pages/02_dollar_bar_generator.py` → `webapp/pages/02_dollar_bar.py`
- Rename: `webapp/pages/03_feature_engineering.py` → `webapp/pages/04_feature_engineering.py`
- Rename: `webapp/pages/04_labeling.py` → `webapp/pages/05_labeling.py`
- Rename: `webapp/pages/05_feature_analysis.py` → `webapp/pages/06_feature_analysis.py`
- Rename: `webapp/pages/06_model_training.py` → `webapp/pages/07_model_training.py`
- Rename: `webapp/pages/07_backtest.py` → `webapp/pages/08_backtest.py`
- Rename: `webapp/pages/08_visualization.py` → `webapp/pages/09_visualization.py`
- Rename: `webapp/pages/09_experiment.py` → `webapp/pages/10_experiment.py`

- [ ] **Step 1: 执行文件重命名**

```bash
cd /Users/link/Documents/AFMLKIT/webapp/pages
mv 02_dollar_bar_generator.py 02_dollar_bar.py
mv 03_feature_engineering.py 04_feature_engineering.py
mv 04_labeling.py 05_labeling.py
mv 05_feature_analysis.py 06_feature_analysis.py
mv 06_model_training.py 07_model_training.py
mv 07_backtest.py 08_backtest.py
mv 08_visualization.py 09_visualization.py
mv 09_experiment.py 10_experiment.py
```

- [ ] **Step 2: 验证文件重命名成功**

Run: `ls -la /Users/link/Documents/AFMLKIT/webapp/pages/`
Expected: 显示新编号的文件列表

- [ ] **Step 3: 提交重命名更改**

```bash
git add webapp/pages/
git commit -m "refactor: renumber page files for CUSUM sampling insertion"
```

---

## Task 2: 更新 SessionManager 状态键

**Files:**
- Modify: `webapp/session.py:50-58`

- [ ] **Step 1: 在 KEYS 列表添加 CUSUM 状态键**

在 `webapp/session.py` 第 57 行后添加：

```python
KEYS = [
    'raw_data', 'bar_data', 'dollar_bars', 'features', 'labels', 'sample_weights',
    'bar_config', 'feature_config', 'label_config', 'model_config', 'backtest_config',
    'model', 'model_results', 'feature_importance', 'feature_metadata',
    'backtest_results', 'plots',
    'current_step', 'is_processing', 'last_updated',
    'experiment_name', 'experiment_notes',
    'iid_results', 'iid_score_df', 'best_freq', 'generation_time',
    'cusum_config', 'cusum_sampled_data', 'cusum_events'  # 新增
]
```

- [ ] **Step 2: 验证语法正确**

Run: `python -c "from webapp.session import SessionManager; print(SessionManager.KEYS)"`
Expected: 输出包含 `cusum_config`, `cusum_sampled_data`, `cusum_events`

- [ ] **Step 3: 提交更改**

```bash
git add webapp/session.py
git commit -m "feat: add CUSUM state keys to SessionManager"
```

---

## Task 3: 创建 CUSUM 采样页面

**Files:**
- Create: `webapp/pages/03_cusum_sampling.py`

- [ ] **Step 1: 创建页面框架和导入**

```python
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
```

- [ ] **Step 2: 实现数据源区块**

在页面框架后添加：

```python
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
```

- [ ] **Step 3: 实现参数配置区块**

在数据源区块后添加：

```python
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
```

- [ ] **Step 4: 实现执行采样区块**

在参数配置后添加：

```python
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
```

- [ ] **Step 5: 实现采样结果区块**

在执行区块后添加：

```python
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
```

- [ ] **Step 6: 实现导出区块**

在结果区块后添加：

```python
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
```

- [ ] **Step 7: 添加导航按钮**

在文件末尾添加：

```python
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
```

- [ ] **Step 8: 验证页面语法**

Run: `python -m py_compile webapp/pages/03_cusum_sampling.py`
Expected: 无错误输出

- [ ] **Step 9: 提交新页面**

```bash
git add webapp/pages/03_cusum_sampling.py
git commit -m "feat: add CUSUM sampling page"
```

---

## Task 4: 更新侧边栏导航

**Files:**
- Modify: `webapp/components/sidebar.py:8-54` (PAGES 字典)
- Modify: `webapp/components/sidebar.py:98-105` (steps 列表)
- Modify: `webapp/components/sidebar.py:170-179` (step_mapping)
- Modify: `webapp/components/sidebar.py:116-134` (数据状态面板)

- [ ] **Step 1: 更新 PAGES 字典**

将 `webapp/components/sidebar.py` 第 8-54 行替换为：

```python
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
```

- [ ] **Step 2: 更新 steps 列表**

将第 98-105 行替换为：

```python
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
```

- [ ] **Step 3: 更新 step_mapping**

将第 170-179 行替换为：

```python
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
```

- [ ] **Step 4: 更新数据状态面板**

在第 134 行 `st.markdown(f"{status_icon} 标签")` 后添加：

```python
        has_cusum = st.session_state.get('cusum_sampled_data') is not None
        status_icon = "✅" if has_cusum else "❌"
        st.markdown(f"{status_icon} CUSUM 采样")
```

- [ ] **Step 5: 验证语法正确**

Run: `python -c "from webapp.components.sidebar import PAGES; print(list(PAGES.keys()))"`
Expected: 输出包含 `🔬 CUSUM 采样`

- [ ] **Step 6: 提交更改**

```bash
git add webapp/components/sidebar.py
git commit -m "feat: update sidebar navigation for CUSUM sampling page"
```

---

## Task 5: 清理数据导入页面的 CUSUM 代码

**Files:**
- Modify: `webapp/pages/01_data_import.py:197-207` (移除 cusum 选项)
- Modify: `webapp/pages/01_data_import.py:310-418` (删除 cusum 分支)

- [ ] **Step 1: 从 K 线类型选择器移除 cusum 选项**

找到第 197-207 行的 `bar_type` selectbox，修改为：

```python
        bar_type = st.selectbox(
            "选择 K 线类型",
            ["time", "tick", "volume", "dollar"],  # 移除 "cusum"
            format_func=lambda x: {
                'time': '时间 K 线',
                'tick': 'Tick K 线',
                'volume': '成交量 K 线',
                'dollar': '金额 K 线'
            }.get(x, x)
        )
```

- [ ] **Step 2: 删除 cusum 分支代码**

删除第 310-418 行（整个 `elif bar_type == 'cusum':` 分支）。

- [ ] **Step 3: 验证语法正确**

Run: `python -m py_compile webapp/pages/01_data_import.py`
Expected: 无错误输出

- [ ] **Step 4: 提交清理更改**

```bash
git add webapp/pages/01_data_import.py
git commit -m "refactor: remove CUSUM option from data import page"
```

---

## Task 6: 更新特征工程页面兼容性

**Files:**
- Modify: `webapp/pages/04_feature_engineering.py:214-223` (CUSUM 配置区块)

- [ ] **Step 1: 移除 cusum_path 输入框，改为从 Session 自动读取**

找到第 214-223 行的 "CUSUM 事件对齐" expander，替换为：

```python
    # CUSUM 对齐选项
    with st.expander("CUSUM 事件对齐", expanded=False):
        # 优先从 Session 读取
        cusum_data = SessionManager.get('cusum_sampled_data')

        if cusum_data is not None:
            st.success(f"✅ 已从 Session 加载 CUSUM 采样数据 ({len(cusum_data)} 行)")
            align_to_cusum = st.checkbox(
                "对齐到 CUSUM 事件时间戳",
                value=True,
                help="特征将与 CUSUM 采样的事件时间戳对齐"
            )
            cusum_path = None  # 使用 Session 数据
        else:
            st.info("未找到 CUSUM 采样数据，可从文件加载")
            align_to_cusum = st.checkbox(
                "对齐到 CUSUM 事件时间戳",
                value=False,
                help="特征将与 CUSUM 采样的事件时间戳对齐"
            )
            cusum_path = st.text_input(
                "CUSUM 文件路径",
                value="outputs/dollar_bars/cusum_sampled_bars.csv"
            ) if align_to_cusum else None
```

- [ ] **Step 2: 更新 compute_all_features 调用**

找到第 291-307 行的 `compute_all_features` 调用，修改为：

```python
            cusum_path = feature_config.get("cusum", {}).get("path", "")
            align_enabled = feature_config.get("cusum", {}).get("align_enabled", False)

            # 优先使用 Session 数据
            cusum_data = SessionManager.get('cusum_sampled_data')
            if align_enabled and cusum_data is not None:
                cusum_path = None  # 使用 cusum_data

            status_text.text("计算特征 (可能需要几分钟)...")
            progress_bar.progress(30)

            features_df, metadata = compute_all_features(
                df=df,
                config=feature_config,
                cusum_path=cusum_path if not cusum_data else None,
                align_to_cusum=align_enabled,
            )
```

- [ ] **Step 3: 验证语法正确**

Run: `python -m py_compile webapp/pages/04_feature_engineering.py`
Expected: 无错误输出

- [ ] **Step 4: 提交更改**

```bash
git add webapp/pages/04_feature_engineering.py
git commit -m "feat: update feature engineering to read CUSUM from Session"
```

---

## Task 7: 端到端测试

- [ ] **Step 1: 启动 Web UI**

Run: `streamlit run webapp/app.py`
Expected: 浏览器打开，显示 AFMLKit 首页

- [ ] **Step 2: 验证侧边栏导航**

检查侧边栏是否显示：
- `🔬 CUSUM 采样` 菜单项
- 更新后的步骤指示器
- `CUSUM 采样` 数据状态项

- [ ] **Step 3: 验证 CUSUM 采样页面**

1. 点击侧边栏 `🔬 CUSUM 采样`
2. 验证页面标题和 5 个区块显示正确
3. 验证无 Dollar Bars 时显示警告提示

- [ ] **Step 4: 验证数据导入页面**

1. 进入数据导入页面
2. 验证 K 线类型选择器不包含 "CUSUM 过滤 K 线" 选项

- [ ] **Step 5: 验证特征工程页面**

1. 进入特征工程页面
2. 验证 CUSUM 对齐区块显示正确
3. 验证 Session 有 CUSUM 数据时自动加载

- [ ] **Step 6: 完成验收并提交**

```bash
git add -A
git commit -m "feat: complete CUSUM sampling page implementation"
```

---

## 验收检查清单

- [ ] 新页面 `03_cusum_sampling.py` 可独立访问
- [ ] 能从 Dollar Bars 执行 CUSUM 采样并显示结果
- [ ] 能加载已保存的 CUSUM 结果文件
- [ ] 采样结果可导出为 CSV/Parquet
- [ ] 数据导入页面不再显示 CUSUM K 线选项
- [ ] 侧边栏导航正确显示新页面
- [ ] 采样结果可通过 Session 传递到特征工程页面
- [ ] 无 Dollar Bars 数据时显示正确引导提示
- [ ] 文件已存在时提示覆盖确认
- [ ] 特征工程页面可从 Session 读取 CUSUM 数据