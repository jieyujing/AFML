"""
run_workflow.py - AL9999 AFML 工作流统一入口

使用方法:
    # 运行完整工作流
    python run_workflow.py --all

    # 运行指定阶段
    python run_workflow.py --phase 1    # Dollar Bars 构建
    python run_workflow.py --phase 2    # 特征工程
    python run_workflow.py --phase 3    # Trend Scanning
    python run_workflow.py --phase 4    # Primary Model + TBM
    python run_workflow.py --phase 5    # Meta Model
    python run_workflow.py --phase 6    # 验证与回测

    # 运行多个阶段
    python run_workflow.py --phase 1,2,3

    # 跳过参数优化（使用当前配置）
    python run_workflow.py --phase 1 --no-optimize
"""

import os
import sys
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

# 阶段定义
PHASES = {
    1: {
        'name': 'Dollar Bars 构建',
        'script': '01_dollar_bar_builder.py',
        'description': '加载原始数据，构建 Dollar Bars，三刀验证',
    },
    2: {
        'name': '特征工程',
        'script': '02_feature_engineering.py',
        'description': 'FracDiff 平稳化，CUSUM 事件采样，特征计算',
    },
    3: {
        'name': 'Trend Scanning',
        'script': '03_trend_scanning.py',
        'description': 'Trend Scanning 标签生成',
    },
    4: {
        'name': 'Primary Model + TBM',
        'script': '04_ma_primary_model.py',
        'description': 'MA Primary Model 回测，TBM 标签',
    },
    5: {
        'name': 'Meta Model',
        'scripts': [
            '05_feature_stationarity.py',
            '06_meta_labels.py',
            '07_meta_model.py',
        ],
        'description': '特征平稳性分析，Meta Labels，Meta Model 训练',
    },
    6: {
        'name': '验证与回测',
        'scripts': [
            '08_dsr_validation.py',
            '10_combined_backtest.py',
        ],
        'description': 'DSR 验证 + Filter-First 组合回测（唯一权威回测入口：10_combined_backtest.py）',
    },
}


def run_script(script_name: str, extra_args: list = None):
    """
    运行指定脚本。

    :param script_name: 脚本文件名
    :param extra_args: 额外命令行参数
    """
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


def run_phase(phase_num: int, extra_args: list = None):
    """
    运行指定阶段。

    :param phase_num: 阶段编号
    :param extra_args: 额外命令行参数
    """
    if phase_num not in PHASES:
        print(f"❌ 无效阶段: {phase_num}")
        return False

    phase = PHASES[phase_num]
    print(f"\n{'#'*60}")
    print(f"# Phase {phase_num}: {phase['name']}")
    print(f"# {phase['description']}")
    print(f"{'#'*60}")

    if 'script' in phase:
        return run_script(phase['script'], extra_args)
    elif 'scripts' in phase:
        success = True
        for script in phase['scripts']:
            if not run_script(script, extra_args):
                success = False
                print(f"⚠️ 脚本执行失败: {script}")
        return success


def main():
    parser = argparse.ArgumentParser(description='AL9999 AFML 工作流')
    parser.add_argument('--all', action='store_true', help='运行完整工作流')
    parser.add_argument('--phase', type=str, help='运行指定阶段（如 1,2,3）')
    parser.add_argument('--no-optimize', action='store_true', help='跳过参数优化')
    parser.add_argument('--list', action='store_true', help='列出所有阶段')

    args = parser.parse_args()

    if args.list:
        print("\n可用阶段:")
        for num, phase in PHASES.items():
            print(f"  {num}: {phase['name']} - {phase['description']}")
        return

    extra_args = ['--no-optimize'] if args.no_optimize else []

    if args.all:
        print("\n" + "="*60)
        print("  AL9999 AFML 完整工作流")
        print("="*60)
        for phase_num in sorted(PHASES.keys()):
            if not run_phase(phase_num, extra_args):
                print(f"\n❌ Phase {phase_num} 失败，工作流终止")
                return
        print("\n" + "="*60)
        print("  ✅ 完整工作流执行成功")
        print("="*60)

    elif args.phase:
        phases = [int(p.strip()) for p in args.phase.split(',')]
        for phase_num in phases:
            if not run_phase(phase_num, extra_args):
                print(f"\n❌ Phase {phase_num} 失败")
                return

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
