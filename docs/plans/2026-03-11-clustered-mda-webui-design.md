# Clustered MDA WebUI 集成设计

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 WebUI 特征分析页面的"3. 特征重要性"步骤中，集成 Clustered MDA 可视化功能，支持特征聚类、Purged CV、误差线可视化和"毒药"簇警告。

**Architecture:** 复用现有 `afmlkit/importance/clustering.py` 和 `afmlkit/importance/mda.py` 模块，在 `webapp/pages/05_feature_analysis.py` 中添加 Streamlit UI 集成层和 Plotly 可视化组件。

**Tech Stack:** Python, pandas, numpy, scikit-learn, scipy, Streamlit, plotly, afmlkit

---

## 1. 系统架构

### 1.1 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                    webapp/pages/05_feature_analysis.py          │
│                                                                 │
│  SessionManager.get('features')  →  X (特征矩阵)                │
│  SessionManager.get('labels')    →  y (Triple Barrier 标签)     │
│                                  →  t1 (事件结束时间)           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 1: cluster_features(X)                            │   │
│  │           → clusters: dict[int, list[str]]              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 2: clustered_mda(X, y, clusters, t1)              │   │
│  │           → mda_df: DataFrame                           │   │
│  │              [cluster_id, features, mean_importance,    │   │
│  │               std_importance]                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Step 3: render_clustered_mda_chart(mda_df)             │   │
│  │           → Plotly Figure (水平条形图 + 误差线)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖

| 模块 | 用途 | 状态 |
|------|------|------|
| `afmlkit/importance/clustering.py` | ONC 算法、特征聚类 | 已有 |
| `afmlkit/importance/mda.py` | Clustered MDA 计算 | 已有 |
| `webapp/components/charts.py` | 可视化组件 | 需扩展 |
| `webapp/pages/05_feature_analysis.py` | UI 集成 | 需修改 |

---

## 2. UI 设计

### 2.1 特征重要性步骤改造

**当前状态：** 使用简单 MDA（单个特征打乱 + 准确率评分）

**改造后：** 双模式设计

```python
mode = st.radio(
    "分析模式",
    ["快速 MDA (单个特征)", "Clustered MDA (推荐)"],
    horizontal=True
)
```

### 2.2 Clustered MDA 配置项

```python
with st.expander("Clustered MDA 参数", expanded=False):
    n_repeats = st.number_input("重复次数", min_value=1, max_value=50, value=10)
    n_splits = st.number_input("CV 折数", min_value=2, max_value=10, value=5)
    embargo_pct = st.number_input("Embargo 比例", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
    random_state = st.number_input("随机种子", value=42)
```

### 2.3 聚类结果预览

```python
st.markdown("#### 特征聚类结果")
st.write(f"最优聚类数：{len(clusters)} (通过 Silhouette Score 自动选择)")

# 显示每个簇的特征
for cid, feats in sorted(clusters.items()):
    st.write(f"**Cluster {cid}:** {', '.join(feats)}")
```

### 2.4 Clustered MDA 可视化

**图表规格：**
- **类型：** 水平条形图
- **X 轴：** 特征重要性（MDA 分数）
- **Y 轴：** 簇 ID（按重要性降序排列）
- **条形颜色：**
  - 重要性 > 0：蓝色 (`#1f77b4`)
  - 重要性 ≤ 0：红色 (`#d62728`)，标记为"毒药"
- **误差线：** 红色，双向，表示标准差

**毒药簇警告框：**
```python
poison_clusters = mda_df[mda_df['mean_importance'] <= 0]
if len(poison_clusters) > 0:
    st.warning("""
    ⚠️ **毒药簇警告：**
    以下簇在打乱后模型表现反而变好（重要性 ≤ 0），说明这些特征严重误导模型：
    - Cluster X: [特征列表]
    建议：在实盘代码中直接 drop 掉这些特征！
    """)
```

---

## 3. 组件设计

### 3.1 `render_clustered_mda_chart` 函数

**位置：** `webapp/components/charts.py`

**函数签名：**
```python
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
```

**实现逻辑：**
1. 按 `mean_importance` 降序排序
2. 确定条形颜色（毒药簇用红色）
3. 使用 `go.Figure` 和 `go.Bar` 创建水平条形图
4. 使用 `go.Scatter` 添加误差线（红色，双向）
5. 添加 0 轴参考线（虚线）

### 3.2 误差线数据格式

```python
# Plotly 误差线格式
error_y = dict(
    type='data',           # 使用实际数据值
    array=std_values,      # 标准差数组
    width=6,               # 误差线横线宽度
    color='red'            # 红色
)
```

---

## 4. 错误处理

| 场景 | 检测条件 | 处理方式 |
|------|----------|----------|
| 没有 Triple Barrier 标签 | `labels is None or 'bin' not in labels` | 降级为简单收益率三分位，显示警告 |
| 特征数量 < 2 | `len(feature_cols) < 2` | 显示错误："特征数量不足，至少需要 2 个特征" |
| 聚类失败 | `cluster_features` 抛出异常 | 降级为简单 MDA，显示警告 |
| CV 折数 > 样本数 | `n_splits > n_samples // 5` | 自动调整为 `max(2, n_samples // 10)` |
| 没有毒药簇 | `len(poison_clusters) == 0` | 不显示警告框 |

---

## 5. 测试策略

### 5.1 单元测试

**文件：** `tests/webapp/test_clustered_mda_ui.py`

**测试用例：**
1. `test_render_clustered_mda_chart_basic` - 基本图表渲染
2. `test_render_clustered_mda_chart_poison_highlight` - 毒药簇高亮
3. `test_error_bar_data_format` - 误差线数据格式验证

### 5.2 集成测试

**文件：** `tests/webapp/test_clustered_mda_integration.py`

**测试用例：**
1. `test_full_clustered_mda_pipeline` - 完整流程测试
2. `test_clustered_mda_with_triple_barrier_labels` - Triple Barrier 标签集成

---

## 6. 实现任务分解

### Task 1: 扩展 `webapp/components/charts.py`
- 添加 `render_clustered_mda_chart` 函数
- 支持水平条形图 + 误差线
- 支持毒药簇高亮

### Task 2: 修改 `webapp/pages/05_feature_analysis.py`
- 在"3. 特征重要性"步骤添加模式选择
- 实现 Clustered MDA 计算逻辑
- 集成聚类结果预览
- 添加毒药簇警告

### Task 3: 编写单元测试
- 测试 `render_clustered_mda_chart`
- 测试错误处理逻辑

### Task 4: 编写集成测试
- 完整流程测试
- Triple Barrier 标签集成测试

### Task 5: 验证和文档
- 运行完整测试套件
- 更新 `webapp/FEATURE_ENGINEERING_GUIDE.md`

---

## 7. 验收标准

1. ✅ 能够在 UI 中选择"Clustered MDA"模式
2. ✅ 能够自动计算最优聚类数并显示
3. ✅ 能够绘制带误差线的水平条形图
4. ✅ 毒药簇（重要性 ≤ 0）用红色高亮
5. ✅ 显示毒药簇警告和移除建议
6. ✅ 所有测试通过（单元 + 集成）
7. ✅ 与现有简单 MDA 并存，可切换

---

## 8. 性能考虑

| 优化项 | 目标 |
|--------|------|
| 聚类计算 | < 5 秒（100 特征） |
| Clustered MDA | < 30 秒（5 折 CV，10 次重复） |
| 图表渲染 | < 2 秒 |

**优化策略：**
- 使用 `n_jobs=-1` 并行计算
- 缓存聚类结果（SessionManager）
- 使用 Plotly 的 WebGL 渲染（`render_mode='webgl'`）
