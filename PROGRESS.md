# AFML 项目进度报告

## ✅ 已完成工作

### 1. Dollar Bars 生成 ✓
**文件**: `src/process_bars.py`

已成功实现两种类型的 dollar bars:
- **Fixed Dollar Bars**: 使用固定阈值
- **Dynamic Dollar Bars**: 使用 EMA 动态阈值

**输出文件**:
- `dynamic_dollar_bars.csv` - 3,927 个 dollar bars
- 时间范围: 2022-01-04 至 2026-01-20

**关键发现**:
- Dollar bars 成功降低了 Jarque-Bera 统计量
- 收益分布更接近正态分布
- 动态阈值表现优于固定阈值

---

### 2. Triple Barrier Labeling ✓
**文件**: `src/labeling.py`

实现了 AFML 核心的三重障碍标签法:
- **上障碍**: 止盈 (1x 波动率)
- **下障碍**: 止损 (1x 波动率)
- **垂直障碍**: 时间限制 (1天)

**输出文件**:
- `labeled_events.csv` - 3,707 个标记事件
- `dollar_bars_labeled.csv` - 带标签的完整数据集

**标签分布**:
- Loss (-1): 53.33% (1,977 events)
- Profit (1): 46.67% (1,730 events)

**收益统计**:
- 平均收益: -0.014%
- 标准差: 0.907%
- 范围: -3.947% 至 8.880%

**持仓时间**:
- 平均: 22.51 小时
- 中位数: 20.33 小时
- 范围: 0.12 至 263.88 小时

---

### 3. 标签可视化分析 ✓
**文件**: `src/visualize_labels.py`

生成了三组综合可视化:

1. **label_distribution.png**
   - 标签计数和百分比分布
   - 按标签的收益分布
   - 收益箱线图

2. **temporal_analysis.png**
   - 时间序列标签分布
   - 滚动标签平衡 (100-bar window)
   - 月度标签分布

3. **barrier_analysis.png**
   - 持仓时间分布
   - 按标签的持仓时间
   - 障碍宽度(波动率)分布
   - 收益 vs 障碍宽度散点图

---

### 4. 特征工程 (Alpha158 + FFD Momentum) ✓
**文件**: `src/features.py`

实现了整合了传统金融特征和高级 ML 特征的工程管线。特别针对**记忆保留 (Memory Preservation)** 进行了增强：

**A. Alpha158 特征集 (基准对照)**:
- 包含经典的 **ROC/Returns** 特征（相当于 $d=1$ 整数差分）
- 这些特征虽然平稳，但完全丢失了价格的长记忆信息
- 用于作为模型性能的 Baseline

**B. 分数阶差分 (Fractional Differentiation) 增强版**:
- **核心理念**: 用 FFD 替代 ROC 实现更优秀的趋势跟踪
- **FFD Level**: 自动寻找最小 $d$（如 $d=0.2$）的平稳价格序列，保留了 80% 的记忆
- **FFD Momentum (Trend)**: 在平稳的 FFD 序列上计算移动平均 (`FFD_CLOSE_MAw`)
- **FFD Volatility (Risk)**: 在平稳的 FFD 序列上计算波动率 (`FFD_CLOSE_STDw`)
- **FFD Slope**: 在平稳序列上计算局部斜率

**输出文件**:
- `features_labeled.csv` - 包含 Alpha158 + FFD 特征 + 标签的完整数据集
- 特征总数: ~180 个 (大幅增强了特征维度)

---

### 5. 样本权重 (Sample Weights) ✓
**文件**: `src/sample_weights.py`

实现了金融数据的样本权重计算 (Chapter 4)，解决了样本重叠 (Concurrency) 导致的非独立同分布问题：

**核心指标**:
- **Concurrency (并发性)**: 计算每个时间点上重叠的标签数量
- **Average Uniqueness (平均唯一性)**: 标签在其生命周期内的平均独立程度 (Mean: ~0.90)
- **Sample Weights**: 结合了收益率归因和唯一性的最终权重

**关键发现**:
- 平均唯一性高达 0.90，说明在当前 Barrier 设置下，大部分样本是相对独立的
- 权重分布呈现长尾特征，部分高收益且独立的样本获得了极高的权重 (Max > 6.0)

**输出文件**:
- `sample_weights.csv` - 包含唯一性分数和最终权重
- `features_labeled.csv` - 已更新，新增 `sample_weight` 和 `avg_uniqueness` 列
- `visual_analysis/weights_distribution.png` - 权重分布图
- `visual_analysis/weights_correlations.png` - 权重与其他变量的相关性

---

### 6. 交叉验证策略 (Cross-Validation) ✓
**文件**: `src/cv_setup.py`

成功实现了 AFML Chapter 7 的 **Purged K-Fold Cross-Validation**，确保模型评估的严谨性。

**核心机制**:
1.  **Purging (清洗)**: 在训练集中剔除所有与测试集在时间上有重叠的样本。由于我们的标签具有持续期 (Holding Period)，如果不进行 Purging，训练集中的样本可能会包含测试集的信息（Look-ahead Bias）。
2.  **Embargo (禁运)**: 在测试集之后设立了 **1%** 的禁运期，进一步切断长记忆特征可能带来的信息泄露。

**验证结果**:
- 进行了 5-Fold 分割
- 自动化测试确认: **无泄漏 (No Leakage Detected)**。所有的 Train/Test 边界都已正确处理。
- 可视化验证: 生成了 `visual_analysis/cv_splits.png`，直观展示了训练集与测试集之间的“隔离带”。

---

### 7. 初步模型训练 (Baseline Model) ✓
**文件**: `src/train_model.py`

完成了基于 **Random Forest** 的基准模型训练，并在 **Purged K-Fold Cross-Validation** 下进行了评估。

**模型配置**:
- **算法**: Random Forest Classifier
- **参数**: n_estimators=1000, max_depth=5, class_weight='balanced_subsample', criterion='entropy'
- **验证**: 5-Fold Purged CV (1% Embargo)

**性能指标 (Baseline)**:
- **Accuracy**: 48.84% (+/- 3.42%)
- **F1 Score**: 0.4884 (+/- 0.0341)
- **ROC AUC**: 0.5082 (+/- 0.0248)
- **Log Loss**: 0.8001 (+/- 0.0272)

**结果分析**:
- 模型的预测能力目前接近随机猜测 (AUC ~0.50)，这是金融数据的常见起点。
- 考虑到使用了较为严格的 Purged CV，这反映了真实的泛化能力，没有过拟合。
- **特征重要性**: 已提取并保存至 `feature_importance.csv`，并生成了 `visual_analysis/feature_importance.png`。

---

### 8. 特征重要性分析 (Feature Importance) ✓
**文件**: `src/feature_importance.py`

成功实现了 AFML Chapter 8 的特征重要性分析,使用了两种方法:

**A. MDI (Mean Decrease Impurity)**:
- 快速的In-sample方法
- 基于树的分裂不纯度降低
- 优点: 计算快速,特征重要性总和为1
- 缺点: 容易过拟合,对高基数特征有偏向

**B. MDA (Mean Decrease Accuracy) - 金标准 ⭐**:
- 稳健的Out-of-sample排列重要性方法
- 使用 Purged 5-Fold CV 防止信息泄露
- 通过排列特征来衡量性能下降
- 这是金融ML中最可靠的特征重要性方法

**特征筛选结果**:
- **原始特征数**: 181个 (Alpha158 + FFD)
- **正向MDA重要性特征**: 125个 (69%)
- **负向MDA重要性特征**: 56个 (31%) - 这些特征实际上降低了模型性能!

**Top 5 Most Important Features (MDA)**:
1. **KSFT**: 0.0092 - K线形态(Shift)
2. **KSFT2**: 0.0069 - K线形态标准化
3. **KMID2**: 0.0062 - K线中点标准化
4. **KMID**: 0.0057 - K线中点
5. **RESI5**: 0.0042 - 5期线性回归残差

**关键发现**:
- K线形态特征(KSFT系列)表现最优,说明价格形态对预测至关重要
- 许多FFD特征的重要性为负值,说明过度的特征工程可能引入噪音
- 短期技术指标(5-10期)比长期指标(50期)更重要

**输出文件**:
- `feature_importance_mdi.csv` - MDI重要性分数
- `feature_importance_mda.csv` - MDA重要性分数(推荐使用)
- `selected_features.csv` - 筛选后的125个有效特征
- `visual_analysis/feature_importance_comparison.png` - MDI vs MDA 对比图
- `visual_analysis/feature_clustering.png` - 特征聚类分析

---

## 📊 数据质量评估

### 优点 ✅
1. **记忆性增强**: 新增的 FFD-Momentum 特征理论上比 ROC 具有更强的预测能力
2. **标签平衡良好**: Loss/Profit 比例接近 53/47
3. **收益分离清晰**: Loss 平均 -0.714%, Profit 平均 +0.787%
4. **特征集完善**: 提供了 "Integer Diff (ROC)" vs "Fractional Diff (FFD)" 的完整对比组
5. **验证严谨**: Purged CV 成功消除了 Look-ahead Bias。

### 需要注意 ⚠️
1. **模型欠拟合**: 深度限制 (max_depth=5) 可能导致模型未能捕捉复杂模式。
2. **维度增加**: 特征数量增加，建议后续增加特征筛选步骤 (Feature Selection)。
3. **Warm-up**: FFD 需要较长的历史数据来预热，导致早期样本被丢弃。

---

## 🎯 下一步建议

现在我们已经识别了有效特征,接下来应该用这些筛选后的特征重新训练模型。

### 选项 1: 使用筛选后的特征重新训练 (推荐) ⭐
**目标**: 仅使用125个正向MDA重要性特征重新训练Random Forest
**预期效果**: 
- 去除噪音特征,提升Out-of-Sample性能
- 减少过拟合风险
- 加快训练速度
**实施**: 修改 `train_model.py` 加载 `selected_features.csv`

### 选项 2: 超参数优化
**目标**: 调整 Random Forest 参数 (如 max_depth, min_samples_leaf)
**理由**: 当前 max_depth=5 可能太保守,可以尝试更深的树。

### 选项 3: 尝试其他模型
**目标**: 使用 XGBoost 或 LightGBM
**理由**: 梯度提升树通常在表格数据上表现更好。

---

## 📁 当前项目结构

```
AFML/
├── src/
│   ├── process_bars.py          # Dollar bars 生成
│   ├── labeling.py               # Triple barrier 标签
│   ├── features.py               # Alpha158 + FFD 增强特征工程
│   ├── sample_weights.py         # 样本权重计算
│   └── cv_setup.py               # 交叉验证配置 (New!)
├── features_labeled.csv          # 完整训练数据 (Features + Labels + Weights)
└── visual_analysis/
    ├── weights_distribution.png  # 权重分布
    └── cv_splits.png             # CV 分割示意图 (New!)
```

## ❓ 您的下一步指示?

请告诉我您想从以下哪个选项开始:
1.  **模型训练** - 训练 Random Forest 并评估性能
2.  **特征重要性** - 分析哪些因子最有效

我已经准备好帮您训练第一个 AFML 模型! 🚀
