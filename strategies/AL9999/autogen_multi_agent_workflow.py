"""
autogen_multi_agent_workflow.py - AL9999 AutoGen 多 Agent 工作流入口。

使用示例:
    python strategies/AL9999/autogen_multi_agent_workflow.py
    python strategies/AL9999/autogen_multi_agent_workflow.py --phases 1,2,3
    python strategies/AL9999/autogen_multi_agent_workflow.py --phases 1 --no-optimize
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "autogen_config.yaml"
DEFAULT_TASK = "基于 AL9999 策略执行可复现实验工作流并输出验证结论"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_AFML_THRESHOLDS = {
    "psr_min": 0.95,
    "dsr_min": 0.95,
    "pbo_max": 0.10,
    "oos_sharpe_min": 0.0,
    "oos_vs_primary_sharpe_delta_min": 0.0,
}
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.AL9999.run_workflow import PHASES, run_phase


LOGGER = logging.getLogger("al9999.autogen.workflow")


@dataclass(frozen=True)
class WorkflowOptions:
    """
    多 Agent 运行参数。

    :param task: 业务任务描述
    :param phases: 允许执行的 phase 列表
    :param no_optimize: 是否禁用 phase 1 参数优化
    :param model: OpenAI 模型名
    :param api_key: OpenAI API Key
    :param client_kwargs: 传给 OpenAIChatCompletionClient 的额外参数
    :param afml_thresholds: AFML 合格阈值
    :param max_messages: 群聊最大消息数
    :param dry_run_execution: 若为 True，仅模拟执行 phase
    :param model_info: 自定义模型信息（非 OpenAI 模型必须提供）
    """

    task: str
    phases: list[int]
    no_optimize: bool
    model: str
    api_key: str
    client_kwargs: dict[str, Any]
    afml_thresholds: dict[str, float]
    max_messages: int
    dry_run_execution: bool
    model_info: dict[str, Any] | None = None


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    加载 YAML 配置文件。

    :param config_path: 配置文件路径
    :returns: 配置字典
    :raises ValueError: 配置格式非法
    """
    if not config_path.exists():
        LOGGER.info("配置文件不存在，使用参数与环境变量: %s", config_path)
        return {}

    raw_text = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件根节点必须是字典: {config_path}")
    return loaded


def filter_supported_client_kwargs(
    client_cls: Any,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    过滤当前 AutoGen 版本支持的模型客户端参数。

    :param client_cls: OpenAIChatCompletionClient 类
    :param kwargs: 候选参数
    :returns: 过滤后的参数
    """
    signature = inspect.signature(client_cls.__init__)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs

    supported = set(signature.parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if ignored:
        LOGGER.warning("以下模型客户端参数不受当前版本支持，已忽略: %s", ignored)
    return filtered


def resolve_options(args: argparse.Namespace) -> WorkflowOptions:
    """
    合并 CLI、配置文件与环境变量，生成工作流参数。

    优先级: CLI > YAML 配置 > 默认值。

    :param args: CLI 参数
    :returns: 工作流参数
    :raises ValueError: 配置不合法
    :raises RuntimeError: API Key 缺失
    """
    config_path = Path(args.config).expanduser()
    config = load_yaml_config(config_path)
    workflow_cfg = config.get("workflow", {})
    openai_cfg = config.get("openai", {})
    afml_cfg = config.get("afml_acceptance", {})

    if workflow_cfg and not isinstance(workflow_cfg, dict):
        raise ValueError("workflow 配置必须是字典。")
    if openai_cfg and not isinstance(openai_cfg, dict):
        raise ValueError("openai 配置必须是字典。")
    if afml_cfg and not isinstance(afml_cfg, dict):
        raise ValueError("afml_acceptance 配置必须是字典。")

    task = args.task if args.task is not None else workflow_cfg.get("task", DEFAULT_TASK)
    phase_expr = args.phases if args.phases is not None else workflow_cfg.get("phases", "all")
    phases = parse_phase_expression(phase_expr)

    no_optimize = bool(workflow_cfg.get("no_optimize", False)) or args.no_optimize
    dry_run_execution = bool(workflow_cfg.get("dry_run_execution", False)) or args.dry_run_execution

    model = args.model if args.model is not None else openai_cfg.get("model", "gpt-4o-mini")
    api_key_env = (
        args.api_key_env if args.api_key_env is not None else openai_cfg.get("api_key_env", "OPENAI_API_KEY")
    )
    api_key = args.api_key if args.api_key is not None else openai_cfg.get("api_key")
    if not api_key:
        api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"未找到 OpenAI API Key。可通过以下任一方式提供:\n"
            f"1) YAML 中 openai.api_key\n"
            f"2) 环境变量 {api_key_env}\n"
            f"3) CLI 参数 --api-key"
        )

    max_messages_raw = args.max_messages if args.max_messages is not None else workflow_cfg.get("max_messages", 20)
    try:
        max_messages = int(max_messages_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_messages 必须是整数。") from exc

    client_kwargs: dict[str, Any] = {}
    config_client_kwargs = openai_cfg.get("client_kwargs", {})
    if config_client_kwargs:
        if not isinstance(config_client_kwargs, dict):
            raise ValueError("openai.client_kwargs 必须是字典。")
        client_kwargs.update(config_client_kwargs)

    base_url = args.base_url if args.base_url is not None else openai_cfg.get("base_url")
    if base_url:
        client_kwargs["base_url"] = base_url

    organization = args.organization if args.organization is not None else openai_cfg.get("organization")
    if organization:
        client_kwargs["organization"] = organization

    timeout = args.timeout if args.timeout is not None else openai_cfg.get("timeout")
    if timeout is not None:
        try:
            client_kwargs["timeout"] = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout 必须是数字。") from exc

    max_retries = args.max_retries if args.max_retries is not None else openai_cfg.get("max_retries")
    if max_retries is not None:
        try:
            client_kwargs["max_retries"] = int(max_retries)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_retries 必须是整数。") from exc

    afml_thresholds = {**DEFAULT_AFML_THRESHOLDS}
    for key in DEFAULT_AFML_THRESHOLDS:
        if key in afml_cfg and afml_cfg[key] is not None:
            try:
                afml_thresholds[key] = float(afml_cfg[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"afml_acceptance.{key} 必须是数字。") from exc

    # 读取 model_info 配置（非 OpenAI 模型必须提供）
    model_info = openai_cfg.get("model_info")
    if model_info is not None and not isinstance(model_info, dict):
        raise ValueError("openai.model_info 必须是字典。")

    return WorkflowOptions(
        task=str(task),
        phases=phases,
        no_optimize=no_optimize,
        model=str(model),
        api_key=str(api_key),
        client_kwargs=client_kwargs,
        afml_thresholds=afml_thresholds,
        max_messages=max_messages,
        dry_run_execution=dry_run_execution,
        model_info=model_info,
    )


def parse_phase_expression(phase_expression: str | None) -> list[int]:
    """
    解析 phase 表达式。

    :param phase_expression: 例如 "1,2,3" 或 "all"
    :returns: phase 编号列表
    :raises ValueError: phase 表达式非法
    """
    if phase_expression is None or phase_expression.strip().lower() == "all":
        return sorted(PHASES.keys())

    normalized = phase_expression.replace("，", ",")
    values = [item.strip() for item in normalized.split(",") if item.strip()]
    if not values:
        raise ValueError("phase 不能为空。")

    phases: list[int] = []
    for value in values:
        if not value.isdigit():
            raise ValueError(f"phase '{value}' 不是数字。")
        phase_num = int(value)
        if phase_num not in PHASES:
            raise ValueError(f"phase '{phase_num}' 不存在，可用值: {sorted(PHASES.keys())}")
        if phase_num not in phases:
            phases.append(phase_num)
    return phases


def build_phase_catalog() -> dict[str, Any]:
    """
    构建 phase 元数据目录。

    :returns: phase 目录字典
    """
    catalog: dict[str, Any] = {}
    for phase_num, phase in PHASES.items():
        scripts: list[str]
        if "script" in phase:
            scripts = [phase["script"]]
        else:
            scripts = list(phase["scripts"])
        catalog[str(phase_num)] = {
            "name": phase["name"],
            "description": phase["description"],
            "scripts": scripts,
        }
    return catalog


def expected_artifact_patterns() -> dict[int, list[str]]:
    """
    Phase 级产物校验规则。

    :returns: phase -> glob pattern 列表
    """
    return {
        1: ["output/bars/dollar_bars_target*.parquet"],
        2: ["output/features/bars_features.parquet"],
        3: ["output/features/trend_labels.parquet"],
        4: ["output/features/ma_primary_signals.parquet", "output/features/tbm_results.parquet"],
        5: ["output/models/meta_model.pkl", "output/features/meta_labels.parquet"],
        6: ["output/features/backtest_stats.parquet"],
    }


def validate_artifacts_for_phases(phases: list[int]) -> dict[str, Any]:
    """
    校验 phase 对应核心产物是否存在。

    :param phases: phase 列表
    :returns: 校验结果
    """
    patterns = expected_artifact_patterns()
    results: dict[str, Any] = {}
    for phase_num in phases:
        phase_patterns = patterns.get(phase_num, [])
        matched_files: list[str] = []
        missing_patterns: list[str] = []
        for pattern in phase_patterns:
            matches = sorted(PROJECT_ROOT.glob(pattern))
            if matches:
                matched_files.extend(str(path.relative_to(PROJECT_ROOT)) for path in matches)
            else:
                missing_patterns.append(pattern)
        results[str(phase_num)] = {
            "phase_name": PHASES[phase_num]["name"],
            "ok": len(missing_patterns) == 0,
            "matched_files": matched_files,
            "missing_patterns": missing_patterns,
        }
    return results


def _to_float(value: Any) -> float:
    """
    将值转换为 float。

    :param value: 任意值
    :returns: float 值
    :raises ValueError: 无法转换
    """
    if value is None:
        raise ValueError("value is None")
    return float(value)


def evaluate_afml_acceptance(thresholds: dict[str, float]) -> dict[str, Any]:
    """
    根据 afml skill 定义评估策略是否合格。

    核心门槛：
    - PSR > 0.95
    - DSR > 0.95
    - PBO 越低越好（默认要求 <= 0.10）
    - OOS Sharpe > 0
    - OOS Sharpe 相对 Primary 不退化

    :param thresholds: 阈值配置
    :returns: 评估结果字典
    """
    features_dir = PROJECT_ROOT / "output" / "features"
    dsr_path = features_dir / "dsr_validation_results.parquet"
    pbo_path = features_dir / "cpcv_pbo_results.parquet"
    backtest_path = features_dir / "backtest_stats.parquet"

    required_files = {
        "dsr_validation_results": dsr_path,
        "cpcv_pbo_results": pbo_path,
        "backtest_stats": backtest_path,
    }
    missing_files = [
        key for key, path in required_files.items() if not path.exists()
    ]
    if missing_files:
        return {
            "ok": False,
            "reason": "missing_required_metrics_files",
            "missing_files": missing_files,
            "details": {},
        }

    dsr_df = pd.read_parquet(dsr_path)
    pbo_df = pd.read_parquet(pbo_path)
    backtest_df = pd.read_parquet(backtest_path)

    if dsr_df.empty or pbo_df.empty or backtest_df.empty:
        return {
            "ok": False,
            "reason": "metrics_files_empty",
            "missing_files": [],
            "details": {},
        }

    dsr_row = dsr_df.iloc[-1]
    pbo_row = pbo_df.iloc[-1]
    bt = backtest_df.set_index("Metric")

    psr_min = _to_float(thresholds.get("psr_min", DEFAULT_AFML_THRESHOLDS["psr_min"]))
    dsr_min = _to_float(thresholds.get("dsr_min", DEFAULT_AFML_THRESHOLDS["dsr_min"]))
    pbo_max = _to_float(thresholds.get("pbo_max", DEFAULT_AFML_THRESHOLDS["pbo_max"]))
    oos_sharpe_min = _to_float(
        thresholds.get("oos_sharpe_min", DEFAULT_AFML_THRESHOLDS["oos_sharpe_min"])
    )
    oos_vs_primary_sharpe_delta_min = _to_float(
        thresholds.get(
            "oos_vs_primary_sharpe_delta_min",
            DEFAULT_AFML_THRESHOLDS["oos_vs_primary_sharpe_delta_min"],
        )
    )

    oos_psr = _to_float(dsr_row.get("oos_psr"))
    oos_dsr = _to_float(dsr_row.get("oos_dsr"))
    pbo_value = _to_float(pbo_row.get("pbo"))

    if "年化夏普" not in bt.index:
        return {
            "ok": False,
            "reason": "backtest_metric_missing",
            "missing_files": [],
            "details": {"missing_metric": "年化夏普"},
        }

    primary_sharpe = _to_float(bt.loc["年化夏普", "Primary (Full)"])
    oos_sharpe = _to_float(bt.loc["年化夏普", "Combined (OOS)"])
    sharpe_delta = oos_sharpe - primary_sharpe

    checks = {
        "oos_psr_pass": oos_psr > psr_min,
        "oos_dsr_pass": oos_dsr > dsr_min,
        "pbo_pass": pbo_value <= pbo_max,
        "oos_sharpe_pass": oos_sharpe > oos_sharpe_min,
        "oos_vs_primary_sharpe_pass": sharpe_delta >= oos_vs_primary_sharpe_delta_min,
    }

    details = {
        "thresholds": {
            "psr_min": psr_min,
            "dsr_min": dsr_min,
            "pbo_max": pbo_max,
            "oos_sharpe_min": oos_sharpe_min,
            "oos_vs_primary_sharpe_delta_min": oos_vs_primary_sharpe_delta_min,
        },
        "metrics": {
            "oos_psr": oos_psr,
            "oos_dsr": oos_dsr,
            "pbo": pbo_value,
            "primary_sharpe": primary_sharpe,
            "oos_sharpe": oos_sharpe,
            "oos_vs_primary_sharpe_delta": sharpe_delta,
        },
        "checks": checks,
    }

    return {
        "ok": all(checks.values()),
        "reason": "afml_acceptance_evaluated",
        "missing_files": [],
        "details": details,
    }


def load_autogen_components() -> dict[str, Any]:
    """
    延迟加载 AutoGen 组件。

    :returns: 组件字典
    :raises RuntimeError: AutoGen 未安装
    """
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_agentchat.teams import SelectorGroupChat
        from autogen_core.tools import FunctionTool
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except ImportError as exc:
        raise RuntimeError(
            "缺少 AutoGen 依赖。请先安装: "
            "pip install autogen-agentchat autogen-core autogen-ext openai"
        ) from exc

    return {
        "AssistantAgent": AssistantAgent,
        "MaxMessageTermination": MaxMessageTermination,
        "SelectorGroupChat": SelectorGroupChat,
        "FunctionTool": FunctionTool,
        "OpenAIChatCompletionClient": OpenAIChatCompletionClient,
    }


async def run_workflow(options: WorkflowOptions) -> None:
    """
    运行 AL9999 AutoGen 多 Agent 工作流。

    :param options: 工作流参数
    :raises RuntimeError: 环境变量或依赖缺失
    """
    components = load_autogen_components()
    AssistantAgent = components["AssistantAgent"]
    MaxMessageTermination = components["MaxMessageTermination"]
    SelectorGroupChat = components["SelectorGroupChat"]
    FunctionTool = components["FunctionTool"]
    OpenAIChatCompletionClient = components["OpenAIChatCompletionClient"]

    client_init_kwargs = {
        "model": options.model,
        "api_key": options.api_key,
        **options.client_kwargs,
    }
    # 添加 model_info（非 OpenAI 模型必须提供）
    if options.model_info:
        client_init_kwargs["model_info"] = options.model_info
    client_init_kwargs = filter_supported_client_kwargs(
        OpenAIChatCompletionClient, client_init_kwargs
    )
    model_client = OpenAIChatCompletionClient(**client_init_kwargs)

    def tool_phase_catalog() -> str:
        """
        返回 phase 元数据目录（JSON 字符串）。

        :returns: JSON 字符串
        """
        LOGGER.info("tool_phase_catalog invoked")
        payload = {
            "allowed_phases": options.phases,
            "catalog": build_phase_catalog(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def tool_run_phase_batch(phases_csv: str) -> str:
        """
        执行多个 phase。

        :param phases_csv: 例如 "1,2,3"
        :returns: 执行结果 JSON 字符串
        """
        requested = parse_phase_expression(phases_csv)
        unauthorized = [phase for phase in requested if phase not in options.phases]
        if unauthorized:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"请求了未授权的 phase: {unauthorized}",
                    "allowed_phases": options.phases,
                },
                ensure_ascii=False,
                indent=2,
            )

        LOGGER.info("tool_run_phase_batch invoked: %s", requested)
        if options.dry_run_execution:
            return json.dumps(
                {
                    "ok": True,
                    "dry_run_execution": True,
                    "requested_phases": requested,
                    "no_optimize": options.no_optimize,
                },
                ensure_ascii=False,
                indent=2,
            )

        extra_args = ["--no-optimize"] if options.no_optimize else []
        phase_results: list[dict[str, Any]] = []
        all_ok = True

        for phase_num in requested:
            success = run_phase(phase_num, extra_args=extra_args)
            phase_results.append(
                {
                    "phase": phase_num,
                    "phase_name": PHASES[phase_num]["name"],
                    "success": success,
                }
            )
            if not success:
                all_ok = False
                break

        return json.dumps(
            {
                "ok": all_ok,
                "requested_phases": requested,
                "results": phase_results,
            },
            ensure_ascii=False,
            indent=2,
        )

    def tool_validate_artifacts(phases_csv: str) -> str:
        """
        校验 phase 产物。

        :param phases_csv: 例如 "1,2,3"
        :returns: 校验结果 JSON 字符串
        """
        requested = parse_phase_expression(phases_csv)
        unauthorized = [phase for phase in requested if phase not in options.phases]
        if unauthorized:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"请求了未授权的 phase: {unauthorized}",
                    "allowed_phases": options.phases,
                },
                ensure_ascii=False,
                indent=2,
            )

        LOGGER.info("tool_validate_artifacts invoked: %s", requested)
        checks = validate_artifacts_for_phases(requested)
        artifacts_ok = all(item["ok"] for item in checks.values())

        afml_eval = {
            "ok": True,
            "reason": "not_required_for_selected_phases",
            "missing_files": [],
            "details": {},
        }
        if 6 in requested:
            afml_eval = evaluate_afml_acceptance(options.afml_thresholds)

        ok = artifacts_ok and afml_eval["ok"]
        return json.dumps(
            {
                "ok": ok,
                "requested_phases": requested,
                "checks": checks,
                "artifacts_ok": artifacts_ok,
                "afml_acceptance": afml_eval,
            },
            ensure_ascii=False,
            indent=2,
        )

    phase_catalog_tool = FunctionTool(
        tool_phase_catalog,
        description="读取 AL9999 phase 目录与允许执行范围。",
    )
    run_phase_batch_tool = FunctionTool(
        tool_run_phase_batch,
        description="执行 phase 批任务，入参如 '1,2,3'。",
    )
    validate_artifacts_tool = FunctionTool(
        tool_validate_artifacts,
        description="校验 phase 产物与 AFML 合格标准（Phase 6 时启用），入参如 '1,2,3'。",
    )

    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        tools=[phase_catalog_tool],
        system_message=(
            "你是 AL9999 工作流规划代理。"
            "先调用 tool_phase_catalog，再给出最小可行执行计划。"
            "不要直接声称执行成功。"
        ),
    )

    executor = AssistantAgent(
        name="executor",
        model_client=model_client,
        tools=[run_phase_batch_tool],
        system_message=(
            "你是 AL9999 执行代理。"
            "仅通过 tool_run_phase_batch 执行脚本，并输出 JSON 结果要点。"
        ),
    )

    validator = AssistantAgent(
        name="validator",
        model_client=model_client,
        tools=[validate_artifacts_tool],
        system_message=(
            "你是 AL9999 验收代理。"
            "执行后必须调用 tool_validate_artifacts。"
            "当 phase 包含 6 时，必须依据 AFML 标准判定通过与否："
            "DSR>95%、PSR>95%、PBO 低（阈值见返回的 thresholds）。"
        ),
    )

    reporter = AssistantAgent(
        name="reporter",
        model_client=model_client,
        system_message=(
            "你是交付总结代理。"
            "请输出: 1) 执行范围 2) 执行结果 3) 风险与后续动作。"
            "当任务闭环时在结尾写上 WORKFLOW_DONE。"
        ),
    )

    team = SelectorGroupChat(
        [planner, executor, validator, reporter],
        model_client=model_client,
        termination_condition=MaxMessageTermination(options.max_messages),
    )

    task_prompt = (
        f"任务: {options.task}\n"
        f"允许 phase: {options.phases}\n"
        f"no_optimize: {options.no_optimize}\n"
        f"dry_run_execution: {options.dry_run_execution}\n"
        "请按 规划 -> 执行 -> 验收 -> 总结 的顺序完成。"
    )

    result = await team.run(task=task_prompt)

    # 只输出 Agent 对话内容，过滤掉框架内部事件
    print("\n" + "=" * 60)
    print("AutoGen 工作流执行结果")
    print("=" * 60)

    for message in result.messages:
        source = getattr(message, "source", "unknown")
        content = getattr(message, "content", "")
        msg_type = getattr(message, "type", "")

        # 只输出文本消息和工具调用结果，跳过内部事件
        if msg_type in ("TextMessage", "ToolCallSummaryMessage"):
            print(f"\n[{source}]")
            # 如果是 JSON 字符串，尝试美化输出
            if isinstance(content, str) and content.startswith("{"):
                try:
                    parsed = json.loads(content)
                    print(json.dumps(parsed, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print(content)
            else:
                print(content)
        elif msg_type == "ToolCallExecutionEvent":
            # 工具执行结果，简化输出
            if isinstance(content, list):
                for item in content:
                    if hasattr(item, "name") and hasattr(item, "content"):
                        print(f"\n[工具调用: {item.name}]")
                        try:
                            parsed = json.loads(item.content)
                            print(json.dumps(parsed, ensure_ascii=False, indent=2))
                        except (json.JSONDecodeError, AttributeError):
                            print(item.content)

    print("\n" + "=" * 60)
    print("工作流完成")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    :returns: 参数解析器
    """
    parser = argparse.ArgumentParser(
        description="AL9999 Microsoft AutoGen 多 Agent 工作流入口"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径，默认 strategies/AL9999/autogen_config.yaml",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="工作流业务任务描述",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help="限制可执行 phase，例如 1,2,3 或 all",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="传递给 phase 1，跳过参数优化",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI 模型名",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="直接传入 OpenAI API Key（优先级高于配置与环境变量）",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="OpenAI API Key 的环境变量名（可覆盖配置）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI 兼容服务 Base URL（可覆盖配置）",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="OpenAI organization（可覆盖配置）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="模型客户端超时秒数（可覆盖配置）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="模型客户端重试次数（可覆盖配置）",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="群聊最大消息数",
    )
    parser.add_argument(
        "--dry-run-execution",
        action="store_true",
        help="仅模拟 phase 执行，不实际运行脚本",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（CLI > YAML > 默认 INFO）",
    )
    return parser


def main() -> int:
    """
    CLI 主入口。

    :returns: shell 返回码
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        config_for_log = load_yaml_config(Path(args.config).expanduser())
    except Exception as exc:  # pragma: no cover - 启动阶段保护
        print(f"读取配置失败: {exc}")
        return 2

    workflow_cfg = config_for_log.get("workflow", {}) if isinstance(config_for_log, dict) else {}
    log_level = args.log_level if args.log_level is not None else workflow_cfg.get("log_level", DEFAULT_LOG_LEVEL)
    log_level = str(log_level).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        print(f"log_level 不合法: {log_level}")
        return 2

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # 抑制 AutoGen 框架内部日志（除非是 DEBUG 模式）
    if log_level != "DEBUG":
        logging.getLogger("autogen_core").setLevel(logging.WARNING)
        logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        options = resolve_options(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2

    try:
        asyncio.run(run_workflow(options))
    except Exception:
        LOGGER.exception("AutoGen 工作流执行失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
