# Clustered MDA WebUI 集成实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** 在 WebUI 特征分析页面集成 Clustered MDA 可视化功能，支持特征聚类、Purged CV、误差线可视化和"毒药"簇警告。

**Architecture:** 复用现有 `afmlkit/importance` 模块，扩展 `webapp/components/charts.py` 添加可视化函数，修改 `webapp/pages/05_feature_analysis.py` 添加 UI 集成。

**Tech Stack:** Python, pandas, numpy, scikit-learn, scipy, Streamlit, plotly, afmlkit

---

### Task 1: 实现 `render_clustered_mda_chart` 函数

**Files:**
- Modify: `webapp/components/charts.py:1-50`
- Test: `tests/webapp/test_charts.py`

**Step 1: 查看现有 charts.py 结构**

Read `webapp/components/charts.py` 了解现有函数格式和导入。

**Step 2: 编写失败的测试**

```python
# tests/webapp/test_charts.py
import pandas as pd
import pytest
from webapp.components.charts import render_clustered_mda_chart


def test_render_clustered_mda_chart_basic():
    """测试基本图表渲染"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [0.05, 0.03, 0.01],
        'std_importance': [0.01, 0.005, 0.002]
    })

    fig = render_clustered_mda_chart(mda_df)

    assert fig is not None
    assert len(fig.data) >= 2  # bar + error bars


def test_render_clustered_mda_chart_poison_highlight():
    """测试毒药簇高亮（重要性 <= 0）"""
    mda_df = pd.DataFrame({
        'cluster_id': [1, 2, 3],
        'features': [['feat_a'], ['feat_b'], ['feat_c']],
        'mean_importance': [0.05, -0.02, 0.01],  # Cluster 2 is poison
        'std_importance': [0.01, 0.005, 0.002]
    })

    fig = render_clustered_mda_chart(mda_df, highlight_poison=True)

    # 验证毒药簇使用红色
    assert fig is not None


def test_render_clustered_mda_chart_empty():
    """测试空 DataFrame 处理"""
    mda_df = pd.DataFrame()

    with pytest.raises(ValueError, match="MDA DataFrame cannot be empty"):
        render_clustered_mda_chart(mda_df)
```

**Step 3: 运行测试验证失败**

Run: `pytest tests/webapp/test_charts.py::test_render_clustered_mda_chart_basic -v`
Expected: FAIL with "cannot import name 'render_clustered_mda_chart'"

**Step 4: 实现最小功能代码**

```python
# webapp/components/charts.py - 添加到文件末尾

import plotly.graph_objects as go
from typing import Optional


def render_clustered_mda_chart(
    mda_df: pd.DataFrame,
    title: str = "Clustered MDA 特征重要性",
    highlight_poison: bool = True,
) -> go.Figure:
    """
    绘制 Clustered MDA 水平条形图（带误差线）

    :param mda_df: DataFrame with columns [cluster_id, features, mean_importance, std_importance]
    :param title: 图表标题
    :param highlight_poison: 是否高亮毒药簇（重要性 ≤ 0）
    :returns: Plotly Figure
    """
    if mda_df.empty:
        raise ValueError("MDA DataFrame cannot be empty")

    # 按重要性降序排序
    mda_df = mda_df.sort_values('mean_importance', ascending=False).reset_index(drop=True)

    # 准备数据
    y_labels = [f"Cluster {cid}: {feats}" for cid, feats in zip(
        mda_df['cluster_id'], mda_df['features']
    )]
    importance = mda_df['mean_importance'].values
    std = mda_df['std_importance'].values

    # 确定条形颜色（毒药簇用红色）
    colors = []
    for imp in importance:
        if highlight_poison and imp <= 0:
            colors.append('#d62728')  # 红色
        else:
            colors.append('#1f77b4')  # 蓝色

    # 创建图形
    fig = go.Figure()

    # 添加条形图
    fig.add_trace(go.Bar(
        y=y_labels,
        x=importance,
        orientation='h',
        marker_color=colors,
        error_x=dict(
            type='data',
            array=std,
            width=6,
            color='red'
        ),
        name='特征重要性'
    ))

    # 添加 0 轴参考线
    fig.add_shape(
        type='line',
        x0=0, y0=-0.5,
        x1=0, y1=len(y_labels) - 0.5,
        line=dict(color='gray', dash='dash'),
    )

    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title='特征重要性 (MDA 分数)',
        yaxis_title='特征簇',
        template='plotly_white',
        height=max(400, len(y_labels) * 60),
        showlegend=False
    )

    return fig
```

**Step 5: 运行测试验证通过**

Run: `pytest tests/webapp/test_charts.py -v`
Expected: All 3 tests PASS

**Step 6: 提交**

```bash
git add webapp/components/charts.py tests/webapp/test_charts.py
git commit -m "feat: add render_clustered_mda_chart for Clustered MDA visualization"
```

---

### Task 2: 修改特征分析页面添加 Clustered MDA 模式

**Files:**
- Modify: `webapp/pages/05_feature_analysis.py:194-310`

**Step 1: 查看当前"3. 特征重要性"步骤代码**

Read `webapp/pages/05_feature_analysis.py:194-310` 了解当前结构。

**Step 2: 添加模式选择和 Clustered MDA 逻辑**

替换现有的 "3. 特征重要性" 步骤代码：

```python
elif step == "3. 特征重要性":
    st.markdown("### 3️⃣ 特征重要性分析")

    features_data = SessionManager.get('features')
    labels_data = SessionManager.get('labels')

    if features_data is None or labels_data is None:
        st.warning("请确保已有特征和标签数据")
    else:
        # 模式选择
        mode = st.radio(
            "分析模式",
            ["快速 MDA (单个特征)", "Clustered MDA (推荐)"],
            horizontal=True
        )

        if mode == "Clustered MDA (推荐)":
            _render_clustered_mda_section(features_data, labels_data)
        else:
            _render_simple_mda_section(features_data, labels_data)
```

**Step 3: 实现 Clustered MDA 渲染函数**

在文件顶部添加辅助函数：

```python
def _render_clustered_mda_section(
    features_data: pd.DataFrame,
    labels_data: pd.DataFrame,
) -> None:
    """渲染 Clustered MDA 分析部分"""
    from components.charts import render_clustered_mda_chart
    from afmlkit.importance.clustering import cluster_features
    from afmlkit.importance.mda import clustered_mda

    st.markdown("#### Clustered MDA (聚类 MDA) 特征重要性")

    st.markdown("""
    **Clustered MDA** 通过打乱整个特征簇来测量重要性，消除替代效应：
    - 使用 ONC 算法自动确定最优聚类数
    - 同时打乱同一簇内的所有特征
    - 使用 Purged CV 和 Log-loss 计算样本外重要性
    """)

    # 参数配置
    with st.expander("Clustered MDA 参数", expanded=False):
        n_repeats = st.number_input("重复次数", min_value=1, max_value=50, value=10)
        n_splits = st.number_input("CV 折数", min_value=2, max_value=10, value=5)
        embargo_pct = st.number_input("Embargo 比例", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
        random_state = st.number_input("随机种子", value=42)

    if not st.button("计算 Clustered MDA"):
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 准备数据
        status_text.text("准备数据...")
        numeric_cols = features_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['open', 'high', 'low', 'close', 'volume']]

        X = features_data[feature_cols].dropna()

        # 获取 Triple Barrier 标签
        if isinstance(labels_data, pd.Series):
            y = labels_data
            t1 = y.index + pd.Timedelta(minutes=30)  # 默认 30 分钟
        else:
            y = labels_data.get('bin', labels_data.iloc[:, 0])
            t1 = labels_data.get('t1', y.index + pd.Timedelta(minutes=30))

        # 对齐索引
        common_idx = X.index.intersection(y.index).intersection(t1.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        t1 = t1.loc[common_idx]

        progress_bar.progress(20)

        # Step 1: 特征聚类
        status_text.text("执行特征聚类...")
        clusters = cluster_features(X)
        progress_bar.progress(40)

        # 显示聚类结果
        with st.expander("📊 聚类结果预览", expanded=True):
            st.write(f"**最优聚类数：** {len(clusters)} (通过 Silhouette Score 自动选择)")
            for cid, feats in sorted(clusters.items()):
                st.write(f"- **Cluster {cid}**: {', '.join(feats)}")

        # Step 2: 计算 Clustered MDA
        status_text.text("计算 Clustered MDA (可能需要几分钟)...")
        mda_df = clustered_mda(
            X=X,
            y=y,
            clusters=clusters,
            t1=t1,
            n_splits=n_splits,
            embargo_pct=embargo_pct,
            n_repeats=n_repeats,
            random_state=random_state
        )
        progress_bar.progress(80)

        # Step 3: 可视化
        status_text.text("渲染图表...")
        fig = render_clustered_mda_chart(mda_df, highlight_poison=True)
        st.plotly_chart(fig, use_container_width=True)

        # 毒药簇警告
        poison_clusters = mda_df[mda_df['mean_importance'] <= 0]
        if len(poison_clusters) > 0:
            st.warning(f"""
            ⚠️ **毒药簇警告：**

            以下簇在打乱后模型表现反而变好（重要性 ≤ 0），说明这些特征严重误导模型：
            """)
            for _, row in poison_clusters.iterrows():
                st.write(f"- **Cluster {row['cluster_id']}** (重要性={row['mean_importance']:.4f}): {', '.join(row['features'])}")
            st.write("""
            **建议：** 在实盘代码中直接 drop 掉这些特征！
            """)

        # 保存结果
        SessionManager.update('mda_results', mda_df)
        SessionManager.update('feature_clusters', clusters)

        status_text.text("分析完成!")
        progress_bar.progress(100)
        st.success("✅ Clustered MDA 分析完成")

    except Exception as e:
        st.error(f"Clustered MDA 分析失败：{str(e)}")
        import traceback
        st.code(traceback.format_exc())
```

**Step 4: 保留简单 MDA 作为备选**

```python
def _render_simple_mda_section(
    features_data: pd.DataFrame,
    labels_data: pd.DataFrame,
) -> None:
    """渲染简单 MDA 分析部分（保留原有逻辑）"""
    # ... 保留原有的简单 MDA 代码 ...
```

**Step 5: 运行 Streamlit 验证**

Run: `streamlit run webapp/app.py`
Expected: 页面正常加载，Clustered MDA 模式可用

**Step 6: 提交**

```bash
git add webapp/pages/05_feature_analysis.py
git commit -m "feat: add Clustered MDA mode to feature analysis page"
```

---

### Task 3: 编写集成测试

**Files:**
- Create: `tests/webapp/test_clustered_mda_integration.py`

**Step 1: 编写完整流程测试**

```python
# tests/webapp/test_clustered_mda_integration.py
"""Clustered MDA 集成测试"""
import numpy as np
import pandas as pd
import pytest
from webapp.components.charts import render_clustered_mda_chart
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda


@pytest.fixture
def sample_features_and_labels():
    """生成模拟特征和 Triple Barrier 标签"""
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2023-01-01', periods=n, freq='5min')

    # 生成特征
    data = {
        f'feat_{i}': np.random.randn(n) * 0.01
        for i in range(10)
    }
    # 添加一些相关特征
    data['feat_correlated'] = data['feat_0'] * 0.8 + np.random.randn(n) * 0.002

    X = pd.DataFrame(data, index=dates)

    # 生成 Triple Barrier 标签
    y = pd.Series(
        np.random.choice([-1, 0, 1], n),
        index=dates,
        name='bin'
    )
    t1 = dates + pd.Timedelta(minutes=30)

    return X, y, t1


def test_clustered_mda_full_pipeline(sample_features_and_labels):
    """测试完整 Clustered MDA 流程"""
    X, y, t1 = sample_features_and_labels

    # Step 1: 特征聚类
    clusters = cluster_features(X)
    assert len(clusters) >= 1
    assert len(clusters) <= len(X.columns)

    # Step 2: Clustered MDA
    mda_df = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        n_splits=3,  # 减少折数加快测试
        n_repeats=2,
        random_state=42
    )

    # 验证结果
    assert len(mda_df) == len(clusters)
    assert 'cluster_id' in mda_df.columns
    assert 'mean_importance' in mda_df.columns
    assert 'std_importance' in mda_df.columns

    # Step 3: 可视化
    fig = render_clustered_mda_chart(mda_df)
    assert fig is not None
    assert len(fig.data) >= 1


def test_clustered_mda_with_poison_clusters(sample_features_and_labels):
    """测试毒药簇检测"""
    X, y, t1 = sample_features_and_labels

    clusters = cluster_features(X)
    mda_df = clustered_mda(
        X=X, y=y, clusters=clusters, t1=t1,
        n_splits=2, n_repeats=1, random_state=42
    )

    # 验证图表能正确处理毒药簇
    fig = render_clustered_mda_chart(mda_df, highlight_poison=True)
    assert fig is not None


def test_clustered_mda_missing_labels():
    """测试缺失标签时的降级处理"""
    X = pd.DataFrame({
        'feat_1': np.random.randn(100),
        'feat_2': np.random.randn(100)
    })

    # 没有提供 y 和 t1，应该抛出有意义的错误
    with pytest.raises(ValueError, match="labels|y"):
        clustered_mda(X=X, y=None, clusters={1: ['feat_1']}, t1=None)
```

**Step 2: 运行测试**

Run: `pytest tests/webapp/test_clustered_mda_integration.py -v`
Expected: All tests PASS

**Step 3: 提交**

```bash
git add tests/webapp/test_clustered_mda_integration.py
git commit -m "test: add integration tests for Clustered MDA"
```

---

### Task 4: 更新文档

**Files:**
- Modify: `webapp/FEATURE_ENGINEERING_GUIDE.md`

**Step 1: 添加 Clustered MDA 章节**

在特征分析指南中添加：

```markdown
## Clustered MDA 特征重要性

### 为什么使用 Clustered MDA？

传统 MDA（单个特征打乱）存在**替代效应**问题：当两个特征高度相关时，打乱其中一个，另一个可以替代它，导致重要性被低估。

Clustered MDA 的解决方案：
1. 使用 ONC 算法自动识别特征簇
2. 同时打乱整个簇的所有特征
3. 使用 Purged CV 计算样本外 Log-loss 变化

### 解读结果

**条形图：**
- 蓝色条形：重要性 > 0，特征有价值
- 红色条形：重要性 ≤ 0，"毒药"簇

**误差线：**
- 红色误差线表示标准差
- 误差线越短，重要性估计越稳定

**毒药簇警告：**
如果发现红色簇（重要性 ≤ 0），说明这些特征在打乱后模型表现反而变好。这通常意味着：
- 特征包含严重噪音
- 特征导致模型过拟合
- 建议：直接从特征池中移除

### 使用示例

```python
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda

# Step 1: 特征聚类
clusters = cluster_features(X_features)

# Step 2: 计算 Clustered MDA
mda_df = clustered_mda(
    X=X_features,
    y=y_labels,
    clusters=clusters,
    t1=t1_events,
    n_splits=5,
    embargo_pct=0.01,
    n_repeats=10
)

# Step 3: 识别毒药簇
poison = mda_df[mda_df['mean_importance'] <= 0]
print(f"需要移除的簇：{poison['cluster_id'].tolist()}")
```
```

**Step 2: 提交**

```bash
git add webapp/FEATURE_ENGINEERING_GUIDE.md
git commit -m "docs: add Clustered MDA usage guide"
```

---

### Task 5: 最终验证

**Step 1: 运行完整测试套件**

Run: `pytest tests/webapp/ -v`
Expected: All tests PASS

**Step 2: 手动验证 Streamlit 应用**

Run: `streamlit run webapp/app.py`
Expected:
- 特征分析页面正常加载
- Clustered MDA 模式可用
- 图表正常渲染
- 毒药簇警告正确显示

**Step 3: 代码风格检查**

Run: `ruff check webapp/`
Expected: No errors

---

## 验收标准

1. ✅ `render_clustered_mda_chart` 函数正确实现
2. ✅ Clustered MDA 模式在 UI 中可用
3. ✅ 特征聚类结果正确显示
4. ✅ 毒药簇（重要性 ≤ 0）用红色高亮
5. ✅ 毒药簇警告正确显示
6. ✅ 所有单元测试通过
7. ✅ 所有集成测试通过
8. ✅ 文档更新完成
9. ✅ 代码风格检查通过
