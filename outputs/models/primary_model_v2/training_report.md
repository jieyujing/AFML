# Primary Model v1 训练报告

## 1. 模型目标

| 目标 | 目标值 | 实际值 |
|------|--------|--------|
| Recall | > 0.7 | 0.4757 |
| Precision | > 0.4 | 0.4885 |
| ROC AUC | > 0.55 | 0.5207 |

## 2. 数据概览

| 指标 | 值 |
|------|-----|
| 样本数量 | 4699 |
| 特征数量 | 35 |
| 标签分布 (0/1) | {0: 2429, 1: 2270} |

## 3. CV 性能指标

| 指标 | 值 | 评价 |
|------|-----|------|
| ROC AUC | 0.5207 | ⚠ 一般 |
| Accuracy | 0.5032 | - |
| Precision | 0.4885 | - |
| **Recall** | **0.4757** | ✗ 需改进 |
| F1 Score | 0.4520 | - |

## 4. 全样本性能指标

| 指标 | 值 |
|------|-----|
| ROC AUC | 0.9816 |
| Recall | 0.9445 |
| Precision | 0.9008 |

## 5. Side 信号分布

| Side | 数量 | 占比 |
|------|------|------|
| +1 (看多) | 3913 | 83.3% |
| -1 (看空) | 786 | 16.7% |
| 0 (中性) | 0 | 0.0% |

## 6. 配置参数

| 参数 | 值 |
|------|-----|
| N Estimators | 200 |
| Max Depth | 10 |
| Min Samples Split | 5 |
| Min Samples Leaf | 2 |
| Class Weight | balanced_subsample |
| CV Splits | 5 |
| Positive Threshold | 0.35 |
| Negative Threshold | 0.65 |

## 7. 下一步

1. 使用生成的 side 信号运行 Triple Barrier 生成 bin 标签
2. 用新标签重新训练 Meta Model
3. 验证完整工作流表现

