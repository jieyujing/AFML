#!/usr/bin/env python3
"""
AL9999 Bar 质量三刀验证
检查 Dollar Bars 是否满足 I.I.D. 假设
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# 加载 Dollar Bars
data_dir = Path(__file__).parent / 'output'
bars = pd.read_parquet(data_dir / 'bars' / 'dollar_bars_target4.parquet')
returns = bars['close'].pct_change().dropna()

print("=" * 60)
print("AL9999 Dollar Bars 三刀验证")
print("=" * 60)

# 基础统计
print(f"\n样本数: {len(returns)}")
print(f"日均 bars: {len(returns) / ((bars.index[-1] - bars.index[0]).days + 1):.1f}")

# ==============================================
# 第一刀：独立性 (Independence)
# ==============================================
print("\n" + "=" * 60)
print("第一刀：独立性 (Independence)")
print("=" * 60)

# 一阶自相关
ac1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
ac1_pass = abs(ac1) < 0.05
print(f"AC1 (一阶自相关): {ac1:.4f}")
print(f"目标: |AC1| < 0.05 | 结果: {'PASS' if ac1_pass else 'FAIL'}")

# Ljung-Box 检验
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(returns, lags=[10], return_df=True)
    lb_pvalue = lb['lb_pvalue'].values[0]
    print(f"Ljung-Box (lag10) p-value: {lb_pvalue:.4f}")
    lb_pass = lb_pvalue > 0.05
    print(f"目标: p > 0.05 | 结果: {'PASS' if lb_pass else 'FAIL'}")
except ImportError:
    print("statsmodels 未安装，跳过 Ljung-Box")
    lb_pass = None

independence_score = 0.5 * (1 if ac1_pass else 0)
if lb_pass is not None:
    independence_score += 0.5 * (1 if lb_pass else 0)
print(f"\n独立性评分: {independence_score:.2f} / 1.0")

# ==============================================
# 第二刀：同分布 (Identically Distributed)
# ==============================================
print("\n" + "=" * 60)
print("第二刀：同分布 (Identically Distributed)")
print("=" * 60)

# 按月分组计算方差
df_returns = pd.DataFrame({'return': returns}, index=bars.index[1:])
monthly_vars = df_returns['return'].groupby(df_returns.index.to_period('M')).var()
vov = monthly_vars.var()  # 方差的方差
vov_ratio = vov / max(monthly_vars.mean(), 1e-10)

print(f"月度方差均值: {monthly_vars.mean():.8f}")
print(f"方差的方差 (VoV): {vov:.2e}")
print(f"VoV / Mean 比率: {vov_ratio:.4f}")
print(f"目标: VoV/Mean < 0.1 | 结果: {'PASS' if vov_ratio < 0.1 else 'FAIL'}")

print(f"\n月度方差分布:")
print(f"  均值: {monthly_vars.mean():.2e}")
print(f"  标准差: {monthly_vars.std():.2e}")
print(f"  min: {monthly_vars.min():.2e}")
print(f"  max: {monthly_vars.max():.2e}")

identically_score = 1 if vov_ratio < 0.1 else max(0, 1 - vov_ratio)
print(f"\n同分布评分: {identically_score:.2f} / 1.0")

# ==============================================
# 第三刀：正态性 (Normality)
# ==============================================
print("\n" + "=" * 60)
print("第三刀：正态性 (Normality)")
print("=" * 60)

# Jarque-Bera 检验
jb_stat, jb_pvalue = stats.jarque_bera(returns)
skew = stats.skew(returns)
kurt = stats.kurtosis(returns, fisher=False)  # Fisher=False 得到 3 是正态

print(f"JB 统计量: {jb_stat:,.0f}")
print(f"JB p-value: {jb_pvalue:.4f}")
print(f"偏度 (Skewness): {skew:.4f} | 目标: ≈ 0")
print(f"峰度 (Kurtosis): {kurt:.2f} | 目标: ≈ 3")

# 正态性评分（对比其他 Bar 类型）
jb_score = min(np.log10(max(jb_stat, 1)) / 9, 1.0)  # 对数缩放，参考标准
skew_score = 1 - min(abs(skew) / 2, 1.0)
kurt_score = 1 - min(abs(kurt - 3) / 47, 1.0)
normality_score = 0.5 * jb_score + 0.25 * skew_score + 0.25 * kurt_score

print(f"\n正态性评分: {normality_score:.2f} / 1.0")
print(f"  (JB: {jb_score:.2f}, Skew: {skew_score:.2f}, Kurt: {kurt_score:.2f})")

# ==============================================
# 综合评分
# ==============================================
print("\n" + "=" * 60)
print("综合评分 (AFML 加权标准)")
print("=" * 60)

# AFML 加权：独立 50% + 同分布 30% + 正态性 20%
weighted_score = 0.5 * independence_score + 0.3 * identically_score + 0.2 * normality_score
print(f"独立性 (50%): {independence_score:.2f} * 0.5 = {0.5 * independence_score:.3f}")
print(f"同分布 (30%): {identically_score:.2f} * 0.3 = {0.3 * identically_score:.3f}")
print(f"正态性 (20%): {normality_score:.2f} * 0.2 = {0.2 * normality_score:.3f}")
print(f"\n加权总分: {weighted_score:.3f}")
print(f"评分越低越好 (理想 < 0.3)")

# ==============================================
# 结论与建议
# ==============================================
print("\n" + "=" * 60)
print("结论与建议")
print("=" * 60)

issues = []
if abs(ac1) > 0.05:
    issues.append(f"AC1={ac1:.4f} 过高，存在序列相关")
if vov_ratio > 0.1:
    issues.append(f"VoV ratio={vov_ratio:.2f} 过高，分布不稳定")

if issues:
    print("\n⚠️ 发现问题:")
    for issue in issues:
        print(f"  - {issue}")
    print("\n建议:")
    if abs(ac1) > 0.05:
        print("  - 考虑降低 TARGET_DAILY_BARS (如从 4 降到 6)")
        print("  - 或使用 Tick Bars 替代 Dollar Bars")
    if vov_ratio > 0.1:
        print("  - 考虑使用 Bundles Bars 替代 Dollar Bars")
else:
    print("\n✅ Bar 质量良好，满足 I.I.D. 假设")

print("\n" + "=" * 60)
