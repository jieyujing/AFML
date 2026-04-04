# AL9999 CPCV/PBO 审计报告

日期：2026-04-03  
任务：Task 8（仅文档审计）  
范围：`AL9999` 全链路，重点核查 `CPCV/PBO` 定义与实现一致性

## 审计结论摘要

- 本次审计按 5 层完成：特征与事件、TBM 与标签、模型训练与验证、回测执行、CPCV/PBO。
- 明确发现并确认以下历史问题：
  1. 旧实现 CPCV 仅切分不拼接路径。
  2. 旧实现 PBO 不是 rank 定义（而是单路径 Sharpe 概率口径）。
  3. 旧实现缺少多策略比较（无法形成 `strategy × path` 矩阵）。
- 已落地修复已可追溯：`09b` 新脚本、`afmlkit/validation` 层修复、旧脚本废弃标记、测试契约补齐。

## 逐步骤审计表

| 步骤 | 检查项 | 证据（文件/函数） | 结论 | 前视偏差等级 | 影响等级 | 修复优先级 |
|---|---|---|---|---|---|---|
| 1 | 分数差分与平稳化仅使用历史窗口 | `strategies/AL9999/02_feature_engineering.py::run_fracdiff_optimization`、`compute_dynamic_cusum_threshold` | 以滚动/差分计算为主，未见显式未来值读取 | 低 | 中 | P2 |
| 2 | CUSUM 事件采样是否只基于当前及过去信息 | `strategies/AL9999/02_feature_engineering.py::run_cusum_filter` | 事件由 `fracdiff.diff` 与阈值序列触发，未见未来点参与 | 低 | 中 | P2 |
| 3 | 事件特征是否在事件时点截面提取 | `strategies/AL9999/feature_compute.py::compute_event_features`、`02_feature_engineering.py` Step 8 | 使用 `event_timestamps` 做索引切片，口径正确 | 低 | 中 | P2 |
| 4 | TBM 标签计算与屏障触碰逻辑 | `strategies/AL9999/04_ma_primary_model.py::compute_tbm_labels`、`afmlkit.label.tbm.triple_barrier` | 标签来自事件后路径，作为监督目标使用，逻辑正确 | 中 | 高 | P1 |
| 5 | Meta 标签是否仅由 TBM 结果派生 | `strategies/AL9999/06_meta_labels.py::main` (`bin = ret > 0`) | 标签定义清晰，未见将未来收益反灌入特征 | 中 | 高 | P1 |
| 6 | 训练/样本外时间切分是否存在 | `strategies/AL9999/07_meta_model.py::split_train_holdout` | 保留末段 holdout，时间切分存在 | 低 | 高 | P1 |
| 7 | CV 是否执行 purge/embargo | `strategies/AL9999/07_meta_model.py` Step 4、`afmlkit/validation/purged_cv.py::_purge/_embargo` | PurgedKFold 路径完整，具备去泄漏机制 | 低 | 高 | P1 |
| 8 | 回测是否单仓位、是否避免重叠收益 | `strategies/AL9999/10_rolling_backtest.py::rolling_backtest` | 单仓位与反向信号平仓逻辑已实现 | 中 | 高 | P1 |
| 9 | 回测预测数据是否完全前瞻隔离 | `strategies/AL9999/10_rolling_backtest.py::get_meta_predictions`、`07_meta_model.py` | 使用训练后模型对全时段信号打分，研究口径可用，生产口径需额外时间隔离 | 中 | 中 | P2 |
| 10 | 旧 CPCV 是否真正拼接路径 | `strategies/AL9999/09_pbo_validation.py`（遍历 `cpcv.split` 直接累积 `sharpe_paths`） | **发现问题：仅“切分后计 Sharpe”，未做路径拼接** | 高 | 高 | P0（已修复） |
| 11 | 旧 PBO 是否为 AFML rank 定义 | `strategies/AL9999/09_pbo_validation.py::main`（1D `sharpe_array` 调用）+ `afmlkit/validation/pbo.py` legacy probability 分支 | **发现问题：旧口径是概率法代理，不是 rank-PBO 主定义** | 高 | 高 | P0（已修复） |
| 12 | 旧实现是否支持多策略比较 | `strategies/AL9999/09_pbo_validation.py`（仅 `rolling_combined_trades` 一条收益流） | **发现问题：缺少策略候选横向比较** | 高 | 高 | P0（已修复） |
| 13 | 新脚本是否实现多策略 + rank-PBO | `strategies/AL9999/09b_cpcv_pbo_validation.py::build_param_grid`、`_aggregate_cpcv_sharpes`、`compute_rank_pbo` | 已构建 `strategy × path` 的 IS/OOS Sharpe 矩阵并按 rank 计算 PBO | 低 | 高 | P0（已完成） |
| 14 | validation 层是否承接核心修复 | `afmlkit/validation/cpcv.py::generate_cpcv_paths/generate_paths`、`afmlkit/validation/pbo.py::calculate_pbo(method='rank')` | 核心路径分配、路径拼接、rank 统计均已下沉到通用层 | 低 | 高 | P0（已完成） |
| 15 | 旧脚本是否明确废弃，测试契约是否覆盖 | `strategies/AL9999/09_pbo_validation.py::emit_deprecation_warning`、`tests/validation/test_cpcv_path_contract.py`、`test_cpcv_generate_paths.py`、`test_pbo.py` | 废弃提醒与契约测试已就位 | 低 | 中 | P1（已完成） |

## 五层审计结论

### 1) 特征与事件层

- 结论：事件采样与事件特征提取流程基本符合“事件时点可得信息”原则。
- 关注点：特征函数库规模较大，建议后续补“逐特征时序可得性”自动化断言。

### 2) TBM 与标签层

- 结论：TBM 与 Meta Label 分层明确，未来路径用于监督标签而非直接特征输入。
- 关注点：应持续防止 `ret/label` 列在训练特征选择时被误纳入。

### 3) 模型训练与验证层

- 结论：时间 holdout + PurgedKFold + embargo 框架完整。
- 关注点：`t1` 的近似设定需与真实持仓周期一致化，以进一步降低残余泄漏风险。

### 4) 回测执行层

- 结论：单仓位滚动执行已修复“重叠收益重复计算”问题。
- 关注点：研究回测可接受；若用于生产评估，建议严格仅用 OOS/前滚模型输出。

### 5) CPCV/PBO 层

- 结论：历史三大问题已被新实现覆盖并修复；当前实现与测试契约一致。

## 本次明确发现（按要求）

1. 旧实现 CPCV 仅切分不拼接路径。  
2. 旧实现 PBO 不是 rank 定义。  
3. 旧实现缺少多策略比较。  

## 已落地修复项（核验）

- 新增 `strategies/AL9999/09b_cpcv_pbo_validation.py`，实现多策略矩阵、路径聚合与 rank-PBO。
- `afmlkit/validation/cpcv.py` 已提供路径分配与 `generate_paths()` 聚合能力。
- `afmlkit/validation/pbo.py` 已支持 `calculate_pbo(sharpe_is, sharpe_oos, method='rank')`。
- 旧 `strategies/AL9999/09_pbo_validation.py` 已带废弃提示（`DeprecationWarning`）。
- `tests/validation/` 已补契约测试，覆盖路径元数据、路径聚合与 rank-PBO 行为。

## 路径数口径差异说明

- 规格文档中出现了“`n_splits=6, n_test_splits=2 → 15 组合，10 路径`”与“15 组合→10 条完整路径”的表述。
- 当前代码与测试契约口径为：
  - `n_paths = C(n_splits - 1, n_test_splits - 1)`
  - 当 `n_splits=6, n_test_splits=2` 时，`n_paths = C(5,1)=5`
  - 见 `afmlkit/validation/cpcv.py::generate_cpcv_paths` 与 `tests/validation/test_cpcv_path_contract.py`
- 本次执行以**测试契约**为准；若未来要改为 10 路径口径，需同步更新规格、实现与测试。

## 后续建议

1. 增加 09b 端到端测试：校验 `cpcv_sharpe_is/oos` 维度与路径映射稳定性。  
2. 为回测新增严格前滚推断模式，避免训练期样本参与同周期打分。  
3. 将“路径数定义”沉淀为单一规范条目，避免规格与测试口径再次分叉。  
4. 在迁移窗口结束后移除旧 `09` 脚本，避免误用。  
