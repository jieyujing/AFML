# Statistical Verification (DSR) Specification

## Purpose
根据 AFML 方法论，对策略的夏普比率进行折现处理，以校正多次试验（Multiple Testing）导致的虚假发现风险。

## Requirements

### Requirement: Probabilistic Sharpe Ratio (PSR)
根据给定的基准夏普比率（$SR^*$），计算当前策略超过基准的概率。

#### Scenario: PSR Computation
- **WHEN** 输入策略的日度收益序列和基准 $SR^*$（通常为 0）
- **THEN** 它必须基于序列的均值、标准差、偏度（Skewness）和峰度（Kurtosis）计算 PSR 统计量。
- **THEN** 它必须输出一个概率值 $P(SR > SR^*)$。

### Requirement: Deflated Sharpe Ratio (DSR)
在 PSR 的基础上，考虑实验的总次数（Number of Trials），计算折现后的显著性概率。

#### Scenario: DSR Verification
- **WHEN** 输入策略结果及试验次数 $N$
- **THEN** 它必须计算预期的最大夏普比率 $\mathbb{E}[\max(\text{Trials})]$。
- **THEN** 它必须使用该预期值作为 PSR 的新基准进行计算。
- **THEN** 它必须判断最终 DSR 概率是否大于 0.95。

### Requirement: Data Reporting
提供 DSR 分析的详细结果报告。

#### Scenario: DSR Summary
- **WHEN** 验证完成
- **THEN** 它必须报告：
  - **原始夏普 (Original SR)**
  - **试验次数 (Number of Trials)**
  - **DSR 概率 (DSR Probability)**
  - **建议 (Status: ACCEPT/REJECT)**
