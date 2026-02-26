## ADDED Requirements

### Requirement: Feature Distance Matrix Calculation
系统 MUST 提供基于特征相关性计算距离矩阵的方法（例如 $D = \sqrt{0.5 \times (1 - \rho)}$）。

#### Scenario: Distance Calculation with Pearson Correlation
- **WHEN** 给定一个特征数据集 $X$ 时
- **THEN** 系统能够计算出特征间的距离矩阵，并且该距离矩阵是对角线为 0、元素且非负的对称矩阵。

### Requirement: Hierarchical Feature Clustering
系统 MUST 提供基于上述距离矩阵的凝聚层次聚类（Agglomerative Hierarchical Clustering）功能。

#### Scenario: Clustering with Automatically Determined K
- **WHEN** 聚类算法执行时且未指定具体聚类数 `k`
- **THEN** 系统根据轮廓系数（Silhouette Score）或 ONC（Optimal Number of Clusters）算法自动寻找并返回最理想的聚类数目及分组方案。

#### Scenario: Clustering with Predefined K
- **WHEN** 聚类算法执行时且通过参数指定了明确的 `k` 值
- **THEN** 系统直接输出包含 `k` 个簇的分组方案。
