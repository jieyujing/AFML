## 1. 基础逻辑修复与增强

- [x] 1.1 修复 `afmlkit/label/bet_size.py` 中的 `get_concurrent_sizes` 逻辑，处理 `inf`/`-inf`。
- [x] 1.2 重新运行 `scripts/bet_sizing_pipeline.py` 并验证 `discretized_sizes.csv` 无异常。

## 2. 回测引擎实现 (strategy-backtesting)

- [x] 2.1 创建 `scripts/backtest_performance.py` 脚本，加载 Dollar Bars 和头寸数据。
- [x] 2.2 实现向量化收益计算逻辑（考虑 T+1 执行滞后）。
- [x] 2.3 实现性能指标计算：累计收益、年度夏普、最大回撤、盈亏比。
- [x] 2.4 实现回测曲线与回撤的可视化导出。

## 3. 统计验证实现 (statistical-verification-dsr)

- [x] 3.1 在 `afmlkit/utils/stats.py` 中（或脚本内）实现 PSR 计算函数。
- [x] 3.2 实现 DSR 折现逻辑，纳入 Meta-Model 调优阶段的试验次数 $N$。
- [x] 3.3 输出 DSR 验证报告，判断策略显著性。

## 4. 综合验收

- [x] 4.1 运行完整回测及验证链路，导出最终报告。
- [x] 4.2 验证回测结果与策略逻辑的逻辑一致性。
