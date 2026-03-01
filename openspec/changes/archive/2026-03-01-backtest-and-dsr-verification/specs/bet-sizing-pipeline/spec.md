# Bet Sizing Pipeline (Delta Package)

## Modified Requirements

### Requirement: Concurrent Signal Tracking
增加对极端数值和边界情形的鲁棒性处理。

#### Scenario: Computing Target Average (Updated)
- **WHEN** 计算活跃信号的均值 $S_t$
- **THEN** 它必须检测结果中的 `inf` 或 `-inf`
- **THEN** 若检测到无穷大值或非数值，必须将其替换为 $0.0$ 以确保交易连续性
- **THEN** 它必须确保即使在信号快速切换的边界点，也不会输出非数值结果。
