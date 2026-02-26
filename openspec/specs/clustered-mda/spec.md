## ADDED Requirements

### Requirement: Mean Decrease Accuracy on Clusters
系统 MUST 实现基于特征分簇进行的 Mean Decrease Accuracy（MDA）变体，确保一次性打乱整个特征群，来计算信息下降量。

#### Scenario: Dropping accuracy on clustered permutations
- **WHEN** 给定一个训练模型和按照聚类组合的特定特征子集，并且在评分过程中打乱该特征子集的值
- **THEN** 会重新计算在验证集上的损失（或正确率）分值，比较打乱前后的降低幅度作为该簇的重要度。

### Requirement: Out-of-Sample Log-loss Evaluation
系统 MUST 支持以 Out-of-Sample（OOS，结合 Purged K-Fold CV）与基于概率及样本权重的 Log-loss 计算机制作为默认。

#### Scenario: Evaluate MDA with Log-loss
- **WHEN** 计算得分变量并在基础分类器输出了预测概率（`predict_proba`）
- **THEN** 系统采用带有 `sample_weight` 限制的 Log-Loss 指标来计算真实置信度损失，而非单纯依赖胜率变化。
