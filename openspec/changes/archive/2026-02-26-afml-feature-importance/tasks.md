## 1. Setup and Module Architecture

- [x] 1.1 Create `afmlkit/validation` and `afmlkit/importance` directories
- [x] 1.2 Add empty `__init__.py` files to make them proper Python packages

## 2. Feature Clustering Calculation (`afmlkit.importance.clustering`)

- [x] 2.1 Implement `get_feature_distance_matrix` to calculate the distance matrix: $D = \sqrt{0.5 \times (1 - \rho)}$
- [x] 2.2 Implement `hierarchical_clustering` function using `scipy.cluster.hierarchy`
- [x] 2.3 Implement logic to find the optimal number of clusters (k) automatically using Silhouette Score if `k` is not predefined

## 3. Purged K-Fold CV (`afmlkit.validation.purged_cv`)

- [x] 3.1 Create `PurgedKFold` class inheriting from `sklearn.model_selection._split.BaseCrossValidator`
- [x] 3.2 Implement `Purge` logic: 剔除训练集中与验证集时间区间存在交集的样本
- [x] 3.3 Implement `Embargo` logic: 在测试集结束的时间点向后延伸一定宽度的保护期，跳过这期间的训练集样本
- [x] 3.4 动态增加打印当前 Fold 训练集有效样本比例的检查机制（防止样本暴降）

## 4. Clustered MDA Scoring (`afmlkit.importance.mda`)

- [x] 4.1 Implement `clustered_mda` algorithm integrating the custom cluster sets and `PurgedKFold` CV methodology
- [x] 4.2 Ensure perturbation (特征乱序) happens strictly at the cluster level
- [x] 4.3 Provide evaluation wrapper using `log_loss` supporting `sample_weight` argument instead of base classification accuracy

## 5. Integration Scripts (`scripts/`)

- [x] 5.1 Create `scripts/feature_importance_analysis.py`
- [x] 5.2 Implement main pipeline to read `feature_matrix.csv` (and related tags/weights if needed)
- [x] 5.3 Execute clustering and clustered MDA using `PurgedKFold`, outputting importance values logically and ideally visualize output results
