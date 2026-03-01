# Strategy Backtesting Specification

## Purpose
实现基于离散化头寸和资产价格序列的收益核算逻辑，提供策略性能的初步评估。

## Requirements

### Requirement: PnL Calculation
系统必须利用离散化的头寸序列（Sizes）和 Dollar Bar 的价格序列（Close）计算资产收益。

#### Scenario: Vectorized Returns
- **WHEN** 输入包含 `discretized_size` 的 DataFrame 和原始 `price` 序列
- **THEN** 它必须通过 `size.shift(1) * price.pct_change()` 计算每个观测点的对数/百分比收益（PnL）
- **THEN** 它必须正确处理数据对齐，确保没有前视偏差（即 $t$ 时刻的收益是基于 $t-1$ 时刻确定的头寸）

### Requirement: Performance Metrics
回测引擎必须计算并输出标准性能指标。

#### Scenario: Basic Metrics
- **WHEN** 回测运行结束
- **THEN** 它必须计算：
  - **累计收益 (Cumulative Return)**
  - **年化夏普比率 (Annualized Sharpe Ratio)**
  - **最大回撤 (Maximum Drawdown)**
  - **盈亏比 (Profit/Loss Ratio)**

### Requirement: Visualization
引擎必须生成关键的可视化图表，辅助分析。

#### Scenario: Performance Plots
- **WHEN** 调用绘图函数
- **THEN** 它必须生成包含累计收益曲线和回撤区域的图表。
