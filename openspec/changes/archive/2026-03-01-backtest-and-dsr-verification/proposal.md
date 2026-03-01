## Why

当前已经完成了从数据采样、特征工程到头寸管理（Bet Sizing）的研发流程。为了验证策略信号在模拟市场环境下的真实表现，并防止金融机器学习中常见的“多重测试陷阱”和过拟合问题，需要建立标准化的回测流水线及基于 Deflated Sharpe Ratio (DSR) 的统计验证机制。

## What Changes

- **修复数值稳定性**：解决 `afmlkit/label/bet_size.py` 中并发头寸累计可能产生的 `inf`/`-inf` 问题。
- **回测引擎实现**：开发 `scripts/backtest_performance.py`，根据离散化头寸计算累计收益（PnL）、最大回撤（MDD）及头寸分布。
- **DSR 验证机制**：实现 DSR 算法，根据试验次数、收益序列的偏度/峰度对夏普比率进行折现验证，确保策略具有统计显著性。
- **报告生成**：输出包含回测曲线和 DSR 结论的综合评估报告。

## Capabilities

### New Capabilities
- `strategy-backtesting`: 实现从头寸序列到收益序列的映射，提供 PnL 持久化与可视化功能。
- `statistical-verification-dsr`: 提供 DSR 计算逻辑，用于评估策略的过拟合风险。

### Modified Capabilities
- `bet-sizing-pipeline`: 改进 `concurrent_sizes` 计算逻辑，增强对空档期和极端数值的处理鲁棒性。

## Impact

- `afmlkit/label/bet_size.py`: 逻辑修正与鲁棒性增强。
- `outputs/backtest/`: 新增回测结果输出目录。
- `scripts/`: 新增回测与验证脚本。
