"""
07_final_report.py - Step 7: 最终报告与 Go/No-Go 决策

目标：汇总所有步骤结果，按 PASS_CRITERIA 做出决策
任务：
1. 加载所有前置结果文件（primary_rule, exit_comparison, filter_test, risk_position, stress_test）
2. 汇总关键指标：
   - Primary Rule: fast, slow, OOS Sharpe, trade count
   - Exit: 类型
   - Filter: 是否启用
   - Risk: 最大回撤
   - Stress Test: WF 通过、成本通过、扰动通过
3. 按 PASS_CRITERIA 检查：
   - trade_count >= 30
   - OOS windows >= 3 (WF 分片数)
   - OOS Sharpe >= 0
   - OOS Sharpe >= Full Sharpe (可选)
   - 参数稳定性、成本压力通过
4. 生成 Go/No-Go 决策
输出：
- final_report.md (人类可读总结)
- go_no_go.txt (简单的结果: GO 或 NO-GO)
"""

import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import RESULTS_DIR, PASS_CRITERIA, WF_CONFIG

def load_result(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def main():
    print(f"[Step 7] 最终报告与决策")

    # 加载所有结果
    files = [
        'primary_rule.parquet',
        'exit_comparison.parquet',
        'filter_test.parquet',
        'risk_position.parquet',
        'stress_test.parquet'
    ]
    results = {}
    for f in files:
        df = load_result(f)
        if df is not None:
            results[f] = df

    if len(results) < len(files):
        missing = [f for f in files if f not in results]
        print(f"  ❌ 缺少结果文件: {missing}")
        print(f"  请确保 Step 1-6 全部成功完成")
        return 1

    # 提取关键指标
    primary = results['primary_rule.parquet'].iloc[0]
    exit_df = results['exit_comparison.parquet']
    best_exit = exit_df.loc[exit_df['sharpe'].idxmax()]
    filter_df = results['filter_test.parquet']
    best_filter = filter_df.loc[filter_df['sharpe'].idxmax()]
    risk = results['risk_position.parquet'].iloc[0]
    stress = results['stress_test.parquet'].iloc[0]

    # 构造报告
    report_lines = [
        "# AL9999 CTA 策略最终报告",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Primary Rule",
        f"- 策略: DMA({int(primary['selected_fast'])}, {int(primary['selected_slow'])})",
        f"- OOS Sharpe: {primary['oos_sharpe']:.3f}",
        f"- OOS 交易数: {int(primary['oos_trade_count'])}",
        f"- 参数簇稳定性: {'✅' if primary.get('neighbor_stable', True) else '⚠️'}",
        "",
        "## 2. Exit 策略",
        f"- 最优类型: {best_exit['exit_type']}",
        f"- OOS Sharpe: {best_exit['sharpe']:.3f}",
        f"- 均交易额: {best_exit['mean_trade']:.2f}",
        "",
        "## 3. Filter",
        f"- 启用: {'是' if best_filter['label'] != 'baseline' else '否'} ({best_filter['label']})",
        f"- OOS Sharpe: {best_filter['sharpe']:.3f}",
        "",
        "## 4. Risk",
        f"- 最大回撤: {risk['max_dd']:.2f}",
        f"- 仓位大小: {risk['position_size']}",
        "",
        "## 5. 稳健性测试",
        f"- Walk-Forward OOS Sharpe 中位数: {stress['wf_median_sharpe']:.3f}",
        f"- WF 所有分片 Sharpe > 0: {'✅' if stress['wf_all_positive'] else '❌'}",
        f"- 成本压力 Sharpe >= 0: {'✅' if stress['worst_cost_pass'] else '❌'} (worst={stress['worst_cost_sharpe']:.3f})",
        f"- 参数邻域稳定: {'✅' if stress['perturb_all_positive'] else '❌'}",
        "",
        "## 6. Pass Criteria 检查",
    ]

    checks = []
    passed = True

    # C1: trade_count >= 30
    c1 = primary['oos_trade_count'] >= PASS_CRITERIA['trade_count_min']
    checks.append(f"- 交易数: {int(primary['oos_trade_count'])} >= {PASS_CRITERIA['trade_count_min']} → {'✅' if c1 else '❌'}")
    passed = passed and c1

    # C2: OOS windows >= 3
    c2 = WF_CONFIG['n_splits'] >= PASS_CRITERIA['oos_windows_min']
    checks.append(f"- OOS 窗口数: {WF_CONFIG['n_splits']} >= {PASS_CRITERIA['oos_windows_min']} → {'✅' if c2 else '❌'}")
    passed = passed and c2

    # C3: OOS Sharpe >= 0
    c3 = primary['oos_sharpe'] >= PASS_CRITERIA['oos_sharpe_min']
    checks.append(f"- OOS Sharpe: {primary['oos_sharpe']:.3f} >= {PASS_CRITERIA['oos_sharpe_min']} → {'✅' if c3 else '❌'}")
    passed = passed and c3

    # C4: WF 和压力测试通过
    c4 = stress['wf_all_positive'] and stress['worst_cost_pass'] and stress['perturb_all_positive']
    checks.append(f"- 稳健性测试 (WF + 成本 + 扰动): {'✅' if c4 else '❌'}")
    passed = passed and c4

    report_lines.extend(checks)
    report_lines.append("")
    report_lines.append("## 7. 最终决策")
    if passed:
        report_lines.append("**GO** ✅ 策略满足最低验收标准，可作为实盘候选。")
    else:
        report_lines.append("**NO-GO** ❌ 策略未通过关键验证，建议回退到对应步骤优化或更换品种/周期。")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*本报告由 AL9999 CTA Workflow 自动生成*")

    # 输出文件
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_md = os.path.join(RESULTS_DIR, 'final_report.md')
    report_txt = os.path.join(RESULTS_DIR, 'go_no_go.txt')

    with open(report_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    with open(report_txt, 'w', encoding='utf-8') as f:
        f.write("GO\n" if passed else "NO-GO\n")

    print('\n'.join(report_lines))
    print(f"\n📁 报告保存: {report_md}")
    print(f"📁 决策文件: {report_txt}")
    print(f"\n{'🏁 Step 7 完成' if passed else '⛔️ Step 7 未通过'}")

    return 0 if passed else 1

if __name__ == '__main__':
    import pandas as pd  # 仅在 main 需要时导入
    sys.exit(main())
