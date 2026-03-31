"""
05_feature_stationarity.py - 特征平稳性分析与 FracDiff 处理

流程:
1. 加载 bars_features.parquet
2. 智能分类特征（排除收益率类、时间类、二值信号类）
3. 逐列 ADF 检验
4. 对非平稳特征应用 FracDiff
5. 输出 ADF 报告和处理后特征文件
"""

import os
import sys
from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    FEATURES_DIR,
    FRACDIFF_THRES,
    FRACDIFF_D_STEP,
    FRACDIFF_MAX_D,
)

from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd


# ============================================================
# 特征分类
# ============================================================

# 排除模式定义
EXCLUDE_PATTERNS = {
    'return': ['_roc_', '_return'],       # 收益率类（已差分）
    'time': ['_sin_', '_cos_', '_sess'],  # 时间特征（周期编码）
    'binary': ['_sig_'],                   # 二值信号类
}


def classify_features(columns: list[str]) -> tuple[list[str], list[str]]:
    """
    分类特征列：候选处理 vs 排除

    :param columns: 特征列名列表
    :returns: (candidate_features, excluded_features)
    """
    candidate_features = []
    excluded_features = []
    exclude_reasons = {}

    for col in columns:
        is_excluded = False
        reason = None

        # 检查排除模式
        for category, patterns in EXCLUDE_PATTERNS.items():
            for pattern in patterns:
                if pattern in col.lower():
                    is_excluded = True
                    reason = category
                    break
            if is_excluded:
                break

        if is_excluded:
            excluded_features.append(col)
            exclude_reasons[col] = reason
        else:
            candidate_features.append(col)

    print(f"\n[分类结果]")
    print(f"  候选特征: {len(candidate_features)} 列")
    print(f"  排除特征: {len(excluded_features)} 列")
    if excluded_features:
        for cat in EXCLUDE_PATTERNS.keys():
            cols = [c for c in excluded_features if exclude_reasons[c] == cat]
            if cols:
                print(f"    - {cat}类: {len(cols)} 列 ({cols[:3]}{'...' if len(cols) > 3 else ''})")

    return candidate_features, excluded_features


# ============================================================
# 单列处理
# ============================================================

def process_feature_stationarity(
    series: pd.Series,
    thres: float,
    d_step: float,
    max_d: float
) -> tuple[pd.Series, dict[str, Any]]:
    """
    处理单个特征的平稳性

    :param series: 特征序列
    :param thres: FracDiff 截断阈值
    :param d_step: d 搜索步长
    :param max_d: d 最大值
    :returns: (processed_series, result_dict)
    """
    # 去除 NaN
    valid_series = series.dropna()

    # 检查数据量
    if len(valid_series) < 10:
        return series, {
            'p_value': None,
            'optimal_d': None,
            'is_stationary': None,
            'is_valid': False,
            'fracdiff_success': False,
        }

    # ADF 检验
    try:
        adf_result = adfuller(valid_series.values, regression='c')
        p_value = adf_result[1]
    except Exception as e:
        return series, {
            'p_value': None,
            'optimal_d': None,
            'is_stationary': None,
            'is_valid': False,
            'fracdiff_success': False,
            'error': str(e),
        }

    # 判断平稳性
    if p_value < 0.05:
        # 已平稳，无需处理
        return series, {
            'p_value': p_value,
            'optimal_d': 0.0,
            'is_stationary': True,
            'is_valid': True,
            'fracdiff_success': True,
        }

    # 非平稳，应用 FracDiff
    try:
        optimal_d = optimize_d(
            series,
            thres=thres,
            d_step=d_step,
            max_d=max_d,
            min_corr=0.0
        )

        if optimal_d == 0.0:
            # optimize_d 判断已平稳（可能与直接 ADF 结果不一致）
            processed_series = series.copy()
        else:
            processed_series = frac_diff_ffd(series, d=optimal_d, thres=thres)

        return processed_series, {
            'p_value': p_value,
            'optimal_d': optimal_d,
            'is_stationary': False,
            'is_valid': True,
            'fracdiff_success': True,
        }

    except Exception as e:
        # FracDiff 失败，返回原序列
        return series, {
            'p_value': p_value,
            'optimal_d': None,
            'is_stationary': False,
            'is_valid': True,
            'fracdiff_success': False,
            'error': str(e),
        }


# ============================================================
# 批量处理
# ============================================================

def run_stationarity_analysis(
    features_path: str,
    output_dir: str,
    thres: float = FRACDIFF_THRES,
    d_step: float = FRACDIFF_D_STEP,
    max_d: float = FRACDIFF_MAX_D
) -> None:
    """
    运行完整的平稳性分析流程

    :param features_path: 输入特征文件路径
    :param output_dir: 输出目录
    :param thres: FracDiff 截断阈值
    :param d_step: d 搜索步长
    :param max_d: d 最大值
    """
    print("=" * 70)
    print("  Feature Stationarity Analysis & FracDiff Processing")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 加载特征
    print("\n[Step 1] 加载特征文件...")
    features_df = pd.read_parquet(features_path)
    print(f"  特征数量: {len(features_df.columns)} 列")
    print(f"  样本数量: {len(features_df)}")

    # Step 2: 分类特征
    print("\n[Step 2] 分类特征...")
    candidate_features, excluded_features = classify_features(list(features_df.columns))

    # Step 3: 处理候选特征
    print("\n[Step 3] 处理候选特征...")
    results = []
    processed_df = features_df.copy()

    for i, col in enumerate(candidate_features):
        print(f"  [{i+1}/{len(candidate_features)}] 处理: {col}")
        series = features_df[col]

        processed_series, result = process_feature_stationarity(
            series, thres, d_step, max_d  # type: ignore[arg-type]
        )

        # 更新 DataFrame（原地替换）
        processed_df[col] = processed_series

        # 记录结果
        result['feature'] = col
        result['is_excluded'] = False
        results.append(result)

        # 打印简要信息
        if result['is_valid'] and result['p_value'] is not None:
            status = "平稳" if result['is_stationary'] else f"FracDiff d={result['optimal_d']:.2f}"
            print(f"    p={result['p_value']:.4f} -> {status}")
        else:
            print(f"    无效或失败")

    # Step 4: 记录排除特征
    print("\n[Step 4] 记录排除特征...")
    for col in excluded_features:
        results.append({
            'feature': col,
            'p_value': None,
            'optimal_d': None,
            'is_stationary': None,
            'is_valid': None,
            'is_excluded': True,
            'fracdiff_success': None,
        })

    # Step 5: 输出 ADF 报告
    print("\n[Step 5] 输出 ADF 报告...")
    report_df = pd.DataFrame(results)
    report_path = os.path.join(output_dir, 'adf_report.csv')
    report_df.to_csv(report_path, index=False)
    print(f"  ADF 报告已保存: {report_path}")

    # 统计
    stationary_count = sum(1 for r in results if r.get('is_stationary') == True)
    fracdiff_count = sum(1 for r in results if r.get('optimal_d') is not None and r.get('optimal_d') > 0)
    print(f"  平稳特征: {stationary_count}")
    print(f"  FracDiff 处理: {fracdiff_count}")
    print(f"  排除特征: {len(excluded_features)}")

    # Step 6: 输出处理后特征
    print("\n[Step 6] 输出处理后特征...")
    output_path = os.path.join(output_dir, 'bars_features_fd.parquet')
    processed_df.to_parquet(output_path)
    print(f"  处理后特征已保存: {output_path}")

    # 完成
    print("\n" + "=" * 70)
    print("  Stationarity Analysis 完成")
    print("=" * 70)


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主流程入口
    """
    # 输入路径
    features_path = os.path.join(FEATURES_DIR, 'bars_features.parquet')

    # 检查输入文件是否存在
    if not os.path.exists(features_path):
        print(f"输入文件不存在: {features_path}")
        print(f"   请先运行 02_feature_engineering.py 生成特征文件")
        return

    # 运行分析
    run_stationarity_analysis(
        features_path=features_path,
        output_dir=FEATURES_DIR,
    )


if __name__ == "__main__":
    main()