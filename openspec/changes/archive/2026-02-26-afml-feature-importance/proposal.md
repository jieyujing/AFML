## Why

当前工作流存在“修改特征 -> 跑回测看曲线”的盲目循环，这种方式容易导致严重的过拟合和错误发现（False Discoveries）。为了建立科学的特征筛选机制，必须在最终模型训练和回测之前，引入基于 AFML（Advances in Financial Machine Learning）标准的特征重要性分析流程。

## What Changes

- **新增特征聚类 (Feature Clustering)**：在评估特征重要性之前，先对特征进行聚类，以解决金融数据中常见的严重多重共线性（Multicollinearity）问题。
- **新增聚类 MDA/MDI 评估 (Clustered MDA/MDI)**：基于聚类结果，计算每个特征簇（Cluster）的 Mean Decrease Accuracy (MDA) 和 Mean Decrease Impurity (MDI)。
- **引入 Purged K-Fold CV**：在交叉验证过程中严格实施 Purged CV（切断时间穿越和序列相关性带来的信息泄露）和 Embargo 机制。
- **采用 Log-loss 作为评估指标**：评估特征重要性时，默认采用 Log-loss（对数损失）等考虑概率分布的指标，以更准确地衡量分类器输出概率的真实有效性（结合样本权重）。

## Capabilities

### New Capabilities
- `feature-clustering`: 实现基于特征相关性或信息论指标的层次聚类（Hierarchical Clustering）等机制。
- `purged-cv`: 实现带有 Purge 和 Embargo 机制的交叉验证逻辑，针对金融时间序列进行特化。
- `clustered-mda`: 实现基于 Purged CV 和 Log-loss 评估指标的聚类 MDA（Mean Decrease Accuracy/Log-loss）特征重要性计算。

### Modified Capabilities


## Impact

- `scripts/`: 将新增用于特征重要性评估和筛选的标准化脚本，作为特征工程（Feature Engineering）后的必经环节。
- `afmlkit/`: 将在 `afmlkit` 工具包内（如 `afmlkit.feature_importance` 或类似模块）添加特征聚类、Purged CV 和 Clustered MDA 计算的核心库函数实现。
