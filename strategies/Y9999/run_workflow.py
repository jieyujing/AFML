"""
run_workflow.py - Y9999 豆油期货策略完整工作流

流程:
1. Dollar Bars 构建
2. 特征工程
3. Trend Scanning
4. MA Primary Model (TBM)
5. Meta Labels 生成
6. Meta Model 训练
7. DSR 验证
8. PBO 验证（新增）

使用最优参数:
- Dollar Bars: TARGET_DAILY_BARS = 6
- TBM: 止损 1.5, 止盈 2.5
"""

import os
import sys
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import PROJECT_ROOT, OUTPUT_DIR, BARS_DIR, FIGURES_DIR, FEATURES_DIR, MODELS_DIR


def run_step(step_name: str, script_path: str):
    """运行单个步骤。"""
    print("\n" + "=" * 70)
    print(f"  {step_name}")
    print("=" * 70)

    result = subprocess.run(
        ["uv", "run", "python", script_path],
        cwd=PROJECT_ROOT,
        capture_output=False
    )

    if result.returncode != 0:
        print(f"❌ {step_name} 失败")
        return False

    print(f"✅ {step_name} 完成")
    return True


def main():
    """运行 Y9999 完整工作流。"""
    print("=" * 70)
    print("  Y9999 豆油期货策略 - 完整工作流")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BARS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Step 1: Dollar Bars 构建
    if not run_step("Step 1: Dollar Bars 构建", "01_dollar_bar_builder.py"):
        return

    # Step 2: 特征工程
    if not run_step("Step 2: 特征工程", "02_feature_engineering.py"):
        return

    # Step 3: Trend Scanning
    if not run_step("Step 3: Trend Scanning", "03_trend_scanning.py"):
        return

    # Step 4: MA Primary Model (TBM)
    if not run_step("Step 4: MA Primary Model (TBM)", "04_ma_primary_model.py"):
        return

    # Step 5: Meta Labels 生成
    if not run_step("Step 5: Meta Labels 生成", "06_meta_labels.py"):
        return

    # Step 6: Meta Model 训练
    if not run_step("Step 6: Meta Model 训练", "07_meta_model.py"):
        return

    # Step 7: DSR 验证
    if not run_step("Step 7: DSR 验证", "08_dsr_validation.py"):
        return

    # Step 8: PBO 验证
    if not run_step("Step 8: PBO 验证", "09_pbo_validation.py"):
        return

    print("\n" + "=" * 70)
    print("  工作流完成")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")

    # 实盘验证清单
    print("\n📋 实盘验证清单:")
    print("-" * 70)
    print("  [ ] DSR Probability > 95%")
    print("  [ ] PBO < 0.5")
    print("  [ ] Meta Model F1 > 0.6")
    print("  [ ] 特征 PSI < 0.25")
    print("-" * 70)


if __name__ == "__main__":
    main()