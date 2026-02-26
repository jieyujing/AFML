## Context

在金融机器学习（AFML）框架下，特征工程的结果不能直接用于训练模型和跑全量回测。金融时序特征具有两个显著特性：
1. **高度多重共线性（Multicollinearity）**：许多技术指标或衍生特征之间存在极强的线性或非线性相关性。在树模型（如 Random Forest, XGBoost）中，这会导致“替代效应（Substitution Effect）”，使得本应重要的特征由于被其他相关特征替代，而在单纯的 MDI/MDA 评估中得分极低。
2. **序列相关性（Serial Correlation）**：样本数据在时间轴上有重叠或相关性。如果使用标准的 K-Fold 交叉验证，测试集的信息会通过时间上的邻近性“泄露”（Leakage）到训练集中，导致模型在 CV 期间表现极好，实盘却迅速崩溃。

为了解决这两个问题，我们必须引入 AFML 范式中的 **特征聚类** 与 **Purged K-Fold CV**，并最终通过 **聚类 MDA（Clustered Mean Decrease Accuracy/Log-loss）** 来评估特征重要性。

## Goals / Non-Goals

**Goals:**
- 在 `afmlkit` 中提供独立、可复用的模块来支持特征科学验证。
- 实现 `PurgedKFold`，切断交叉验证过程中的时间穿越（Purge）并添加隔离期（Embargo）。
- 实现基于相关性距离的特征分层聚类（Hierarchical Feature Clustering），自动确定最佳簇数（如使用 Silhouette 轮廓系数）。
- 实现 `ClusteredMDA` 算法，在特征簇的级别打乱数据，计算 Log-loss 的衰减量。
- 提供对应的顶层脚本 `scripts/feature_importance_analysis.py`，串接事件驱动样本和标签数据，执行特征重要性分析。

**Non-Goals:**
- 本次变更**不**直接向任何交易模型或回测引擎传入重要性数据，只负责特征评估分析本身。
- **不**引入超大规模的深度学习特征自动提取，保持重心在人工定义与统计验证的特征上。

## Decisions

**1. 目录和模块架构调整**
- **Decision:** 在 `afmlkit` 工具包中新增 `afmlkit/validation` 和 `afmlkit/importance` 目录。
- **Rationale:** 严格遵循单一职责原则。特征的“生成”归属 `feature`，标签“合成”归属 `label`，交叉验证防穿越归属 `validation`，而特征选择“评估”则归属 `importance`。这样也利于与 `sklearn` API 的松耦合。

**2. 交叉验证类的设计 (`PurgedKFold`)**
- **Decision:** `PurgedKFold` 将实现 `sklearn.model_selection._split.BaseCrossValidator` 接口。需要传入事件时间（`t1`：样本结束时间）来进行区间交叉比对，清洗训练集。
- **Rationale:** 兼容标准的 Sklearn 生态，使其可以直接被诸如 `GridSearchCV` 等原生工具使用。

**3. 距离度量与聚类方式**
- **Decision:** 使用基于 Pearson 相关系数的距离矩阵：$D = \sqrt{0.5 \times (1 - \rho)}$，结合 `scipy.cluster.hierarchy` 进行特征的凝聚层次聚类（Agglomerative Hierarchical Clustering）。最佳簇数量通过最大的平均轮廓系数（Silhouette Score）决定。
- **Rationale:** 该距离变换满足度量空间（Metric Space）特性，相比于直接利用原始相关系数具有更好的聚类数值稳定性。

**4. 重要性评估指标 (Log-loss vs Accuracy)**
- **Decision:** `ClusteredMDA` 将默认并被推荐使用带有样本权重的 `Log-loss`（对数损失）或类似的概率相关评分进行评估，而不是仅仅看胜率（Accuracy）。
- **Rationale:** 胜率掩盖了置信度。在 AFML 的 Meta-labeling 中，我们需要输出精准的概率进行头寸下注（Bet Sizing）。只有基于概率优劣来反向衡量特征重要性，才能找出能提供明确“信号强度”的特征。

## Risks / Trade-offs

- **[Risk] Purged CV 导致训练集大幅萎缩**
  - **Trade-off/Mitigation:** 带有极宽 Embargo 窗口的 Purged CV 会导致可用的训练样本量断崖式下跌，增加模型欠拟合风险。 mitigation：将 `n_splits` 适当调整，或动态计算并打印出每个 Fold 实际可用的训练样本比例，若比例过低触发警告，提醒用户检查采样频率。
- **[Risk] 聚类过程中的维度灾难及噪音分组**
  - **Trade-off/Mitigation:** 极端噪音特征可能会在寻找最佳轮廓系数（Silhouette）时干扰算法。 mitigation：作为回退策略，代码应当支持用户不使用自适应聚类，而是强制手动设置 `n_clusters`；同时在重要性脚本中先做一步最基础的 `MDI` 低优特征物理剔除。
- **[Risk] 样本权重（Sample Weights）在 Log-loss 中的使用**
  - **Trade-off/Mitigation:** 某些基础模型或实现不支持自带 `sample_weight`，导致 Log-loss 难以反映财务现实上的“收益权重”。 mitigation：使用支持样本加权的分类器（如 `RandomForestClassifier(class_weight='balanced')` 和自带 `fit(X, y, sample_weight=w)` 的实现）。
