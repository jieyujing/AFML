# AL9999 /afml 最佳实践 SOP

本 SOP 面向 `AL9999` 策略研发、验证和回测执行，目标是统一口径，避免“指标好看但不可复现”。

## 1. 总原则

- 先保证数据与标签质量，再优化模型复杂度。
- 同一轮评估必须使用同一份交易账本（trade ledger）。
- 先防泄露，再看收益；先看 OOS，再看 Full。
- 所有优化都要附带约束（交易收缩、成本口径、样本外显著性）。

## 2. 数据与采样

- 使用事件采样（Dollar Bars）作为默认输入，避免时间采样导致的信息密度不均。
- 采样参数变更后，必须复核独立性/同分布/正态性指标（AC1、Ljung-Box、VoV、JB）。
- 夜盘映射规则保持一致，禁止在不同脚本中使用不同交易日定义。

## 3. 标签与特征

- 标签使用 TBM + Meta Labeling，side 与是否入场分离。
- 特征只允许使用当前时点可得信息，未来信息仅用于分析脚本，禁止进入训练/回测主链路。
- 特征筛选优先使用可解释、可复现方案（如 `mda_positive`），并记录所选方案。

## 4. 训练与验证

- 训练阶段必须使用 Purged K-Fold + embargo，时间切分不可替换为随机切分。
- 保留 Holdout OOS 区间，所有“可实盘性”判断以 OOS 指标为准。
- DSR/PSR 为显著性主指标，必须显式记录参数试验次数（`n_trials`）。

## 5. 回测权威入口

- `10_combined_backtest.py` 是唯一权威回测入口。
- `10_rolling_backtest.py` 仅用于历史对照或调试，不作为验收依据。
- Phase 6 只跑：`08_dsr_validation.py` + `10_combined_backtest.py`。

## 6. 交易口径统一

- 成本参数统一从 `config.py` 读取，禁止脚本硬编码手续费/滑点。
- 收缩率（trade shrinkage）必须统一定义；阈值选择与验证阶段使用同一口径。
- 回测主链路必须落盘以下产物，供验证脚本读取：
  - `filter_first_primary_trades.parquet`
  - `filter_first_combined_trades.parquet`
  - `filter_first_selection.parquet`
  - `filter_first_threshold_report.parquet`

## 7. Filter-First 迭代规则

- 先在约束范围内选阈值，再比较 OOS DSR：
  - 交易收缩约束：`shrinkage_min <= shrinkage <= shrinkage_max`
  - 排序优先级：`OOS DSR` -> `OOS Sharpe` -> `OOS trades`
- side 治理默认允许 `both_with_short_penalty`，仅在证据充分时切到 `long_only`。
- execution guard（最短持仓、冷却期、反手确认）属于可控开关，必须记录参数。

## 8. 最低验收清单

- `Combined(OOS) DSR > 0.95`
- `Combined(OOS) Sharpe > 0`
- `Combined(OOS) Sharpe >= Primary(Full) Sharpe`
- `OOS trade shrinkage` 在配置约束范围内
- 验证脚本读取的是本轮 `filter_first_*` 交易产物（而非旧 rolling 产物）

## 9. 交付要求

- 每次策略迭代必须提供：
  - 关键参数快照（threshold、side_mode、guard 配置）
  - 统一口径的 OOS/Full 指标表
  - 对应交易账本产物路径
- 未满足验收清单时，必须标注“可研究，不可上线”。

