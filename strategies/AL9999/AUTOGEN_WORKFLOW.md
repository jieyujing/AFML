# AL9999 AutoGen 多 Agent 工作流

该工作流基于 `strategies/AL9999/run_workflow.py` 的既有 phase 脚本实现，不改动原有业务步骤，只增加多 agent 编排层。

## 1. 安装依赖

```bash
pip install -e .[dev]
pip install autogen-agentchat autogen-core autogen-ext openai
```

## 2. 设置环境变量

```bash
export OPENAI_API_KEY="your_api_key"
```

## 3. 配置文件

默认读取配置文件：

- `strategies/AL9999/autogen_config.yaml`

你可以在其中配置：

- `openai.api_key` / `openai.api_key_env`
- `openai.base_url`
- `openai.model`
- `openai.timeout` / `openai.max_retries`
- `workflow.phases` / `workflow.max_messages` 等
- `afml_acceptance.*`（AFML 合格标准阈值）

也支持通过 `--config` 指定其他配置文件路径。

## 4. 运行方式

```bash
# 默认：允许所有 phase（1-6）
python strategies/AL9999/autogen_multi_agent_workflow.py

# 指定 phase
python strategies/AL9999/autogen_multi_agent_workflow.py --phases 1,2,3

# phase 1 跳过参数优化
python strategies/AL9999/autogen_multi_agent_workflow.py --phases 1 --no-optimize

# 仅模拟执行（不真正运行 phase 脚本）
python strategies/AL9999/autogen_multi_agent_workflow.py --phases 1,2 --dry-run-execution

# 覆盖 Base URL（CLI 优先级最高）
python strategies/AL9999/autogen_multi_agent_workflow.py --base-url https://api.openai.com/v1
```

## 5. Agent 角色

- `planner`: 读取 phase 目录并输出执行计划
- `executor`: 调用 phase 执行工具运行脚本
- `validator`: 校验关键产物 + AFML 合格标准（Phase 6）
- `reporter`: 汇总执行结果、风险与后续动作

## 6. AFML 合格标准（默认）

- `PSR > 0.95`
- `DSR > 0.95`
- `PBO <= 0.10`（PBO 越低越好）
- `Combined(OOS) Sharpe > 0`
- `Combined(OOS) Sharpe - Primary(Full) Sharpe >= 0`

以上阈值可在 `autogen_config.yaml` 的 `afml_acceptance` 下调整。

## 7. 可调参数

- `--config`: 指定配置文件
- `--api-key`: 直接传 API Key
- `--api-key-env`: 指定 API Key 环境变量名
- `--base-url`: 指定 OpenAI 兼容网关地址
- `--model`: 覆盖模型名
- `--timeout`: 覆盖超时秒数
- `--max-retries`: 覆盖重试次数
- `--max-messages`: 覆盖群聊最大消息数

## 8. Filter-First 优化参数（新增）

在 `strategies/AL9999/config.py` 的 `FILTER_FIRST_CONFIG` 中可调：

- `threshold_grid`: 阈值扫描网格
- `shrinkage_min` / `shrinkage_max`: 交易收缩约束
- `execution_guard.*`: 反手确认、最短持仓、冷却期
- `short_penalty_delta`: 空头阈值惩罚（`both_with_short_penalty` 模式）
- `side_mode`: `both` / `both_with_short_penalty` / `long_only` / `short_only`

## 9. /afml 执行规范

统一执行请参考：

- `strategies/AL9999/AFML_SOP.md`

该 SOP 约束了权威回测入口、交易口径、收缩率定义和验收清单，建议作为研发与验收的默认标准。
