"""
run_workflow.py - AL9999 CTA Trend Development (基于 cta-trend-development skill)

使用方法:
    # 运行完整工作流（Step 0 → Step 7 顺序）
    python run_workflow.py --all

    # 运行指定 Step（不跳步，必须按顺序）
    python run_workflow.py --step 0   # Dollar Bars 构建（数据边界）
    python run_workflow.py --step 1   # 验证基础趋势 edge
    python run_workflow.py --step 2   # Primary Rule 确认（DMA）
    python run_workflow.py --step 3   # Exit 策略比较
    python run_workflow.py --step 4   # Filter 机制测试
    python run_workflow.py --step 5   # Position & Risk 集成
    python run_workflow.py --step 6   # Walk-Forward & 成本压力 & 参数扰动
    python run_workflow.py --step 7   # 最终报告 & Go/No-Go 决策

    # 运行范围（如 Step 1-4）
    python run_workflow.py --range 1-4

注意：
- 必须按 Step 顺序执行，不允许跳步
- 每个 Step 都有输入输出检查，未完成则不能进入下一步
- Step 6 是综合稳健性测试，必须通过才进入 Step 7
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

# CTA Skill 8步定义
STEPS = {
    0: {
        'name': '数据边界定义',
        'script': '01_dollar_bar_builder.py',
        'description': '构建 Dollar Bars，确定数据边界（品种、周期、时间范围）',
        'outputs': ['output/bars/dollar_bars.parquet'],
    },
    1: {
        'name': '基础趋势 Edge 验证',
        'script': '01b_verify_edge.py',
        'description': '使用最简单趋势定义（DMA）验证是否存在基础 edge（OOS 为正）',
        'inputs': ['output/bars/dollar_bars.parquet'],
        'outputs': ['output/results/edge_verification.parquet'],
    },
    2: {
        'name': 'Primary Rule 确认',
        'script': '02_primary_rule.py',
        'description': '固化 Primary Rule（DMA），确定参数簇而非单点冠军',
        'inputs': ['output/bars/dollar_bars.parquet'],
        'outputs': ['output/results/primary_rule.parquet'],
    },
    3: {
        'name': 'Exit 策略比较',
        'script': '03_exit_comparison.py',
        'description': '固定 entry，比较不同 exit 策略（reverse_signal, trailing_stop, fixed_hold）',
        'inputs': ['output/results/primary_rule.parquet'],
        'outputs': ['output/results/exit_comparison.parquet'],
    },
    4: {
        'name': 'Filter 机制测试',
        'script': '04_filter_test.py',
        'description': '测试机制明确的过滤器（volatility_regime, trend_strength）',
        'inputs': ['output/results/exit_comparison.parquet'],
        'outputs': ['output/results/filter_test.parquet'],
    },
    5: {
        'name': 'Position & Risk 集成',
        'script': '05_risk_position.py',
        'description': '加入仓位控制与风险限制（最大回撤、日亏损）',
        'inputs': ['output/results/filter_test.parquet'],
        'outputs': ['output/results/risk_position.parquet'],
    },
    6: {
        'name': '稳健性测试',
        'script': '06_stress_test.py',
        'description': 'Walk-Forward + 成本压力 + 参数扰动（三合一）',
        'inputs': ['output/results/risk_position.parquet'],
        'outputs': ['output/results/stress_test.parquet'],
    },
    7: {
        'name': '最终报告与决策',
        'script': '07_final_report.py',
        'description': '汇总所有结果，按 Pass Criteria 做出 Go/No-Go 决策',
        'inputs': ['output/results/stress_test.parquet'],
        'outputs': ['output/results/final_report.md', 'output/results/go_no_go.txt'],
    },
}

def check_step_prerequisites(step_num: int) -> bool:
    """检查当前 Step 的前置条件是否满足（输入文件存在）"""
    if step_num not in STEPS:
        return False
    step = STEPS[step_num]
    if 'inputs' not in step:
        return True  # Step 0 无输入要求

    for input_file in step['inputs']:
        path = os.path.join(PROJECT_ROOT, input_file)
        if not os.path.exists(path):
            print(f"  ⚠️  前置文件缺失: {input_file}")
            return False
    return True

def run_script(script_name: str, extra_args: list = None) -> bool:
    """运行指定脚本"""
    script_path = os.path.join(PROJECT_ROOT, script_name)
    if not os.path.exists(script_path):
        print(f"❌ 脚本不存在: {script_path}")
        return False

    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"  运行: {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0

def run_step(step_num: int, extra_args: list = None) -> bool:
    """运行单个 Step，包括前置检查"""
    if step_num not in STEPS:
        print(f"❌ 无效 Step: {step_num}")
        return False

    # 检查前置条件
    if not check_step_prerequisites(step_num):
        print(f"❌ Step {step_num} 前置条件不满足，请先完成上一级 Step")
        return False

    step = STEPS[step_num]
    print(f"\n{'#'*60}")
    print(f"# Step {step_num}: {step['name']}")
    print(f"# {step['description']}")
    print(f"{'#'*60}")

    if 'script' in step:
        return run_script(step['script'], extra_args)
    elif 'scripts' in step:
        success = True
        for script in step['scripts']:
            if not run_script(script, extra_args):
                success = False
                print(f"⚠️  脚本执行失败: {script}")
        return success

def main():
    parser = argparse.ArgumentParser(description='AL9999 CTA Trend Development Workflow')
    parser.add_argument('--all', action='store_true', help='运行完整工作流（Step 0-7）')
    parser.add_argument('--step', type=str, help='运行指定 Step（如 0,1,2）')
    parser.add_argument('--range', type=str, help='运行 Step 范围（如 0-3）')
    parser.add_argument('--no-optimize', action='store_true', help='跳过参数优化')
    parser.add_argument('--list', action='store_true', help='列出所有 Steps')

    args = parser.parse_args()

    if args.list:
        print("\nCTA Workflow Steps (必须按顺序执行):")
        for num, step in STEPS.items():
            req = f"输入: {step.get('inputs', ['None'])[0]}" if 'inputs' in step else "输入: 无（起始）"
            out = f"输出: {step.get('outputs', ['None'])[0]}"
            print(f"  Step {num}: {step['name']}")
            print(f"    {step['description']}")
            print(f"    {req} | {out}")
        return

    extra_args = ['--no-optimize'] if args.no_optimize else []

    if args.all:
        print("\n" + "="*60)
        print("  AL9999 CTA 完整工作流 (Step 0 → 7)")
        print("="*60)
        for step_num in sorted(STEPS.keys()):
            if not run_step(step_num, extra_args):
                print(f"\n❌ Step {step_num} 失败，工作流终止")
                return
        print("\n" + "="*60)
        print("  ✅ 完整工作流执行成功")
        print("="*60)

    elif args.step:
        steps = [int(s.strip()) for s in args.step.split(',')]
        # 检查是否按顺序
        if len(steps) > 1 and steps != sorted(steps):
            print("❌ Steps 必须按顺序执行（0→7）")
            return
        for step_num in steps:
            if not run_step(step_num, extra_args):
                print(f"\n❌ Step {step_num} 失败")
                return

    elif args.range:
        try:
            start, end = map(int, args.range.split('-'))
            if start > end or start < 0 or end > 7:
                raise ValueError
        except:
            print("❌ range 格式错误，应为 '0-7'")
            return
        for step_num in range(start, end + 1):
            if not run_step(step_num, extra_args):
                print(f"\n❌ Step {step_num} 失败")
                return

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
