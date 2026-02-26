## ADDED Requirements

### Requirement: Purged K-Fold Iteration
系统 MUST 实现符合 Sklearn 标准交叉验证 API的 `PurgedKFold`。

#### Scenario: Dropping Overlapped Samples (Purge)
- **WHEN** 生成某个 Fold 的验证集事件后，发现训练集部分事件的时间跨度与验证集事件范围有交集 $[t0, t1]$
- **THEN** 验证生成器切断穿越（Purge），自动剔除这些会引发信息泄露的训练集样本。

### Requirement: Embargo Support
系统 MUST 允许用户在一个特定宽度 `embargo` 下指定观察缓冲期。

#### Scenario: Applying Embargo After Test Set Ends
- **WHEN** 测试集结束时存在序列相关性
- **THEN** 系统强行跳过（Embargo）紧接在测试集结尾后宽度占比为 `h` 的训练数据，以消除自相关导致的信息延后泄露。
