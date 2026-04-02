"""
run_workflow.py - IF9999 策略统一工作流入口

一键执行完整的 AFML 策略开发流程：
1. Dollar Bars 构建
2. 特征工程
3. Trend Scanning
4. Primary Model 回测
5. 特征平稳性检验
6. Meta Labels
7. Meta Model 训练
8. DSR 验证
9. PBO 验证（新增）

使用方法:
    uv run python strategies/IF9999/run_workflow.py --steps 1-9
    uv run python strategies/IF9999/run_workflow.py --steps 8,9  # 仅运行验证
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 工作流步骤定义
WORKFLOW_STEPS = {
    1: {
        'name': 'Dollar Bars 构建',
        'script': '01_dollar_bar_builder.py',
        'description': '构建 Dollar Bars 并验证 IID 特性',
    },
    2: {
        'name': '特征工程',
        'script': '02_feature_engineering.py',
        'description': '计算技术因子和 FracDiff 特征',
    },
    3: {
        'name': 'Trend Scanning',
        'script': '03_trend_scanning.py',
        'description': '使用 Trend Scanning 生成 Primary Model 信号',
    },
    4: {
        'name': 'Primary Model 回测',
        'script': '04_ma_primary_model.py',
        'description': 'MA Primary Model + TBM 标签回测',
    },
    5: {
        'name': '特征平稳性检验',
        'script': '05_feature_stationarity.py',
        'description': 'ADF 检验 + FracDiff 处理非平稳特征',
    },
    6: {
        'name': 'Meta Labels',
        'script': '06_meta_labels.py',
        'description': '生成 Meta Labels 和样本权重',
    },
    7: {
        'name': 'Meta Model 训练',
        'script': '07_meta_model.py',
        'description': '训练 Meta Model (Purged CV + F1 优化)',
    },
    8: {
        'name': 'DSR 验证',
        'script': '08_dsr_validation.py',
        'description': 'PSR/DSR 统计显著性验证',
    },
    9: {
        'name': 'PBO 验证',
        'script': '09_pbo_validation.py',
        'description': '回测过拟合概率验证',
    },
    10: {
        'name': '组合策略综合回测',
        'script': '10_combined_backtest.py',
        'description': '扣除成本后的实盘仿真回测 (PSR/DSR)',
    },
}


def run_step(step_num: int, strategy_dir: Path) -> bool:
    """
    执行单个工作流步骤.

    :param step_num: 步骤编号
    :param strategy_dir: 策略目录路径
    :returns: 是否成功
    """
    if step_num not in WORKFLOW_STEPS:
        print(f"❌ 未知步骤: {step_num}")
        return False

    step = WORKFLOW_STEPS[step_num]
    script_path = strategy_dir / step['script']

    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False

    print("\n" + "=" * 70)
    print(f"  Step {step_num}: {step['name']}")
    print(f"  {step['description']}")
    print("=" * 70)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            ['uv', 'run', 'python', str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            print(f"\n✅ Step {step_num} 完成 ({elapsed:.1f}s)")
            return True
        else:
            print(f"\n❌ Step {step_num} 失败 (exit code: {result.returncode})")
            return False

    except Exception as e:
        print(f"\n❌ Step {step_num} 执行出错: {e}")
        return False


def parse_steps(steps_str: str) -> list:
    """
    解析步骤字符串.

    支持格式:
    - "1-9" → [1, 2, 3, 4, 5, 6, 7, 8, 9]
    - "1,3,5" → [1, 3, 5]
    - "1-3,5,7-9" → [1, 2, 3, 5, 7, 8, 9]
    """
    steps = []
    parts = steps_str.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            steps.extend(range(int(start), int(end) + 1))
        else:
            steps.append(int(part))

    return sorted(set(steps))


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description='IF9999 策略 AFML 工作流',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行验证与回测步骤
  uv run python strategies/IF9999/run_workflow.py --steps 8-10

  # 运行特征和模型训练
  uv run python strategies/IF9999/run_workflow.py --steps 2-7
        """
    )
    parser.add_argument(
        '--steps',
        type=str,
        default='1-10',
        help='要执行的步骤 (默认: 1-10)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='列出所有可用步骤'
    )

    args = parser.parse_args()

    # 列出步骤
    if args.list:
        print("\n可用步骤:")
        print("-" * 70)
        for num, step in WORKFLOW_STEPS.items():
            print(f"  [{num}] {step['name']}: {step['description']}")
        print("-" * 70)
        return

    # 解析步骤
    try:
        steps_to_run = parse_steps(args.steps)
    except ValueError as e:
        print(f"❌ 步骤格式错误: {e}")
        return

    print("=" * 70)
    print("  IF9999 Strategy AFML Workflow")
    print("=" * 70)
    print(f"\n执行步骤: {steps_to_run}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    strategy_dir = Path(__file__).parent

    # 执行各步骤
    results = {}
    for step_num in steps_to_run:
        success = run_step(step_num, strategy_dir)
        results[step_num] = success

        if not success:
            print(f"\n⚠️ 步骤 {step_num} 失败，是否继续？(y/n)")
            # 在自动化模式下继续
            # 在交互模式下可以暂停

    # 输出总结
    print("\n" + "=" * 70)
    print("  工作流执行总结")
    print("=" * 70)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for step_num, success in results.items():
        status = "✅" if success else "❌"
        print(f"  [{step_num}] {WORKFLOW_STEPS[step_num]['name']}: {status}")

    print("-" * 70)
    print(f"  完成: {success_count}/{total_count}")
    print("=" * 70)

    # 实盘验证清单
    if 8 in steps_to_run or 9 in steps_to_run:
        print("\n📋 实盘验证清单:")
        print("-" * 70)
        print("  [ ] DSR Probability > 95%")
        print("  [ ] PBO < 0.5")
        print("  [ ] Meta Model F1 > 0.6")
        print("  [ ] 特征 PSI < 0.25")
        print("-" * 70)


if __name__ == "__main__":
    main()