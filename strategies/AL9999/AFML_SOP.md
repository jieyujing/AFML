# AL9999 CTA Trend Development Standard Operating Procedure

本 SOP 面向 AL9999 策略的 CTA 趋势跟踪开发，基于 `cta-trend-development` skill 的 8 步流程。目标是确保每个策略迭代都可验证、稳健、可复现。

## 总原则

1. **拒绝机器学习预测模型**：只使用规则化的技术指标（如 DMA、突破、动量）。
2. **一次只推进一层**：在复杂化之前，必须确保当前层通过验证。
3. **样本外（OOS）为准**：任何可交易性判断都必须基于成本后的 OOS 表现。
4. **参数簇而非单点**：选择参数组合的稳定簇，而非单一最优参数。
5. **成本压力为底线**：在 worst-case 成本下仍应有正期望。

## 8 步工作流

### Step 0: 数据边界定义

**输入**: 原始数据 CSV (1-minute 或其他高频)
**输出**: `output/bars/dollar_bars_target15.parquet`

- 构建 Dollar Bars，目标每日 bars 数 `TARGET_DAILY_BARS = 15`
- 验证数据质量：连续性、无异常缺口、交易量充足
- 确定时间范围（训练 + 测试），保留至少 2 年数据
- 统一交易日定义（夜盘归属下一交易日）

### Step 1: 基础趋势 Edge 验证

**输入**: `dollar_bars.parquet`
**输出**: `output/results/edge_verification.parquet`

- 使用最简单趋势定义：DMA(5, 20) 生成信号
- 固定出场：反向信号
- 固定仓位：1 手
- OOS 切分 70/30
- 检验条件：
  - OOS 净收益 > 0（成本后）
  - OOS Sharpe >= 0
  - Full & OOS 交易数均 >= 30
  - 参数邻域（fast=4-6, slow=18-22）全部 OOS Sharpe >= 0

**不通过则回退到 Step 0**（换品种、换 bar 阈值、换时间周期）。

### Step 2: Primary Rule 确认

**输入**: `dollar_bars.parquet`
**输出**: `output/results/primary_rule.parquet`

- 扫描参数网格：fast ∈ [3,7], slow ∈ [15,25] (fast < slow)
- 每个组合在 OOS 上回测
- 筛选满足 `OOS Sharpe > 0` 且 `trade_count >= 30` 的组合
- 从合格簇中选择：
  1. 参数簇最稳定者（相邻参数 Sharpe 不塌陷）
  2. 逻辑最简单（fast/slow 小值优先）
  3. Sharpe 最优者（若簇大小相近）
- 记录 `selected_fast`, `selected_slow`

**若无合格簇，回退到 Step 1** 重新验证 edge。

### Step 3: Exit 策略比较

**输入**: `primary_rule.parquet`
**输出**: `output/results/exit_comparison.parquet`

- 固定 entry 为 Step 2 确定的 DMA 信号
- 测试退出方案：
  1. `reverse_signal`（反向信号）
  2. `fixed_hold`（固定持有 N 根，默认 20）
  3. `trailing_stop`（ATR 追踪止损，2x ATR）
  4. `time_based`（最大持有时间，默认 60 根）
- 比较指标：
  - OOS Sharpe
  - 均交易收益（mean_trade）
  - 回撤（max_dd）
  - 交易数（trade count）
- 选择 OOS Sharpe > 0 且 mean_trade 最高的方案

**若所有方案 Sharpe <= 0，回退到 Step 2**。

### Step 4: Filter 机制测试

**输入**: `exit_comparison.parquet`
**输出**: `output/results/filter_test.parquet`

- 在 entry 之前应用过滤器（不影响 exit 逻辑）
- 测试：
  1. Baseline（无过滤）
  2. `volatility_regime`：过滤极高/极低波动（z-score > 1.5 或 < 0.5）
- 比较 filtered vs baseline：
  - OOS Sharpe 应提升或持平
  - 回撤应下降
  - 均交易收益不应显著下降
- 选择改善最显著的配置（或多保留一个组合）

**若过滤后恶化，禁用 filter，使用 baseline 继续**。

### Step 5: Position & Risk 集成

**输入**: `filter_test.parquet`
**输出**: `output/results/risk_position.parquet`

- 应用固定仓位大小 `position_size`（默认 1 手）
- 计算最终回测指标：
  - 最大回撤（PnL 序列最大回撤）
  - Sharpe
  - 交易数
- 检查是否违反上限（如 max_drawdown < 上限）
- 记录最终信号序列供后续使用

**若风险严重超标（回撤 > 50%），需调整仓位或增强 filter**。

### Step 6: 稳健性测试

**输入**: `risk_position.parquet`
**输出**: `output/results/stress_test.parquet` + 详细子文件

- **Walk-Forward Analysis**：
  - 分 5 折，每折训练 70% / 测试 30%
  - 所有分片 OOS Sharpe > 0
  - 中位数 Sharpe 作为稳健性指标

- **成本压力测试**：
  - Commission multipliers: [1.0, 1.5, 2.0]
  - Slippage multipliers: [1.0, 2.0, 3.0]
  - Worst-case Sharpe >= 0（至少 > -0.5 可特别标注）

- **参数扰动**：
  - fast±2, slow±5 范围内所有有效组合
  - 全部在 OOS 上 Sharpe >= 0

**任一项不通过，禁止进入 Step 7**。

### Step 7: 最终报告与决策

**输入**: 所有前置结果
**输出**: `final_report.md`, `go_no_go.txt`

- 汇总各步骤结果
- 对照 `PASS_CRITERIA` 逐项检查：
  - `trade_count >= 30`
  - `oos_windows >= 3`
  - `oos_sharpe >= 0`
  - 稳健性三项全部通过
- 输出决策：
  - **GO** ✅：策略满足所有标准，可作为实盘候选
  - **NO-GO** ❌：存在关键缺陷，需回退到相应步骤优化

## 运行方式

```bash
# 查看可用步骤
python run_workflow.py --list

# 运行完整工作流（必须按顺序）
python run_workflow.py --all

# 运行到某一步
python run_workflow.py --range 0-6
```

## 产物清单

| Step | 产物文件 | 说明 |
|------|----------|------|
| 0 | `output/bars/dollar_bars.parquet` | Dollar Bars 数据 |
| 1 | `output/results/edge_verification.parquet` | Edge 验证结果 |
| 2 | `output/results/primary_rule.parquet` | Primary 参数选择 |
| 3 | `output/results/exit_comparison.parquet` | Exit 策略对比 |
| 4 | `output/results/filter_test.parquet` | Filter 测试结果 |
| 5 | `output/results/risk_position.parquet` | 风险集成结果 |
| 6 | `output/results/stress_test.parquet` | 稳健性测试汇总 |
| 7 | `output/results/final_report.md` | 最终报告 |

## 重要提醒

- **严禁跳步**：Step N 依赖 Step N-1 的输出，必须按顺序执行。
- **样本外优先**：训练集表现不重要，OOS 表现才是关键。
- **成本为王**：所有回测必须包含佣金和滑点。
- **不追求单点冠军**：参数的稳定簇比最优参数更有意义。
- **失败回退**：任一阶段未通过，请按指引回退，不要强行推进。

---

*本 SOP 自动生成于 AL9999 CTA Workflow 改造完成时*
