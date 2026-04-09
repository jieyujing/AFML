# AL9999 CTA 策略最终报告
生成时间: 2026-04-08 19:33:14

## 1. Primary Rule
- 策略: DMA(3, 15)
- OOS Sharpe: 0.736
- OOS 交易数: 765
- 参数簇稳定性: ✅

## 2. Exit 策略
- 最优类型: reverse_signal
- OOS Sharpe: 0.736
- 均交易额: 138.88

## 3. Filter
- 启用: 否 (baseline)
- OOS Sharpe: 0.736

## 4. Risk
- 最大回撤: -175.00
- 仓位大小: 1.0

## 5. 稳健性测试
- Walk-Forward OOS Sharpe 中位数: 0.882
- WF 所有分片 Sharpe > 0: ✅
- 成本压力 Sharpe >= 0: ✅ (worst=0.736)
- 参数邻域稳定: ✅

## 6. Pass Criteria 检查
- 交易数: 765 >= 30 → ✅
- OOS 窗口数: 5 >= 3 → ✅
- OOS Sharpe: 0.736 >= 0.0 → ✅
- 稳健性测试 (WF + 成本 + 扰动): ✅

## 7. 最终决策
**GO** ✅ 策略满足最低验收标准，可作为实盘候选。

---
*本报告由 AL9999 CTA Workflow 自动生成*