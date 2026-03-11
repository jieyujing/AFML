"""IID 评估器 - Jarque-Bera 检验和自相关分析"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


class IIDEvaluator:
    """
    IID (独立同分布) 评估器

    用于评估 Dollar Bars 的收益率序列是否接近 IID，
    这是金融机器学习中数据质量的重要指标。
    """

    def __init__(self):
        """初始化 IID 评估器"""
        pass

    def evaluate(self, ohlcv: pd.DataFrame) -> Dict:
        """
        执行完整的 IID 评估

        :param ohlcv: OHLCV DataFrame
        :return: 评估结果字典
        """
        if len(ohlcv) < 10:
            return {
                "n_bars": len(ohlcv),
                "error": "数据量不足，至少需要 10 个 bars"
            }

        # 计算对数收益率
        log_ret = self._compute_log_returns(ohlcv)

        if len(log_ret) < 3:
            return {
                "n_bars": len(ohlcv),
                "n_returns": len(log_ret),
                "error": "有效收益率数据不足"
            }

        # JB 检验
        jb_stat, jb_pvalue = self.jarque_bera_test(log_ret)

        # 一阶自相关
        autocorr_1 = self.autocorrelation(log_ret, lag=1)

        # 高阶自相关
        autocorr_5 = self.autocorrelation(log_ret, lag=5)
        autocorr_10 = self.autocorrelation(log_ret, lag=10)

        # Ljung-Box 检验
        lb_stat, lb_pvalue = self.ljung_box_test(log_ret, lags=10)

        # 基本统计量
        mean_ret = log_ret.mean()
        std_ret = log_ret.std()
        skew = log_ret.skew()
        kurtosis = log_ret.kurtosis()

        return {
            "n_bars": len(ohlcv),
            "n_returns": len(log_ret),
            "jb_stat": jb_stat,
            "jb_pvalue": jb_pvalue,
            "autocorr_1": autocorr_1,
            "autocorr_5": autocorr_5,
            "autocorr_10": autocorr_10,
            "lb_stat": lb_stat,
            "lb_pvalue": lb_pvalue,
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "skew": skew,
            "kurtosis": kurtosis,
            "log_ret": log_ret,  # 保留用于可视化
        }

    def _compute_log_returns(self, ohlcv: pd.DataFrame) -> pd.Series:
        """
        计算对数收益率

        :param ohlcv: OHLCV DataFrame
        :return: 对数收益率 Series
        """
        if 'close' not in ohlcv.columns:
            raise ValueError("DataFrame 必须包含 'close' 列")

        # 计算对数收益率
        log_ret = np.log(ohlcv['close'] / ohlcv['close'].shift(1))

        # 移除无穷值和 NaN
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

        return log_ret

    def jarque_bera_test(self, returns: pd.Series) -> Tuple[float, float]:
        """
        Jarque-Bera 正态性检验

        :param returns: 收益率序列
        :return: (JB 统计量，p 值)
        """
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        return jb_stat, jb_pvalue

    def autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """
        计算自相关系数

        :param returns: 收益率序列
        :param lag: 滞后阶数
        :return: 自相关系数
        """
        return returns.autocorr(lag=lag)

    def ljung_box_test(self, returns: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """
        Ljung-Box 检验（检验序列是否存在自相关）

        :param returns: 收益率序列
        :param lags: 检验的滞后阶数
        :return: (LB 统计量，p 值)
        """
        # 移除 NaN
        returns_clean = returns.dropna()

        if len(returns_clean) <= lags:
            return np.nan, np.nan

        # 使用 statsmodels 的 acorr_ljungbox
        result = acorr_ljungbox(returns_clean, lags=[lags], return_df=True)
        return float(result['lb_stat'].iloc[0]), float(result['lb_pvalue'].iloc[0])

    def score(self, results: Dict[int, Dict]) -> Tuple[int, pd.DataFrame]:
        """
        综合评分，返回最优频率

        :param results: 各频率的评估结果字典 {freq: result_dict}
        :return: (最优频率，评分详情 DataFrame)
        """
        if not results:
            return 0, pd.DataFrame()

        freqs = sorted(results.keys())

        # 提取指标
        jb_stats = []
        ac_vals = []
        n_bars_list = []

        for freq in freqs:
            r = results[freq]
            if "error" in r:
                jb_stats.append(np.nan)
                ac_vals.append(np.nan)
                n_bars_list.append(0)
            else:
                jb_stats.append(r.get('jb_stat', np.nan))
                ac_vals.append(abs(r.get('autocorr_1', np.nan)))
                n_bars_list.append(r.get('n_bars', 0))

        # 转换为 numpy 数组
        jb_vals = np.array(jb_stats)
        ac_vals_arr = np.array(ac_vals)
        n_bars = np.array(n_bars_list)

        # 归一化（越小越好）
        jb_norm = self._normalize(jb_vals, lower_is_better=True)
        ac_norm = self._normalize(ac_vals_arr, lower_is_better=True)

        # n_bars 归一化（越大越好，但考虑目标频率的合理性）
        n_bars_norm = self._normalize(n_bars, lower_is_better=False)

        # 综合评分：JB 40% + 自相关 40% + 样本量 20%
        scores = 0.4 * jb_norm + 0.4 * ac_norm + 0.2 * n_bars_norm

        # 找到最优频率
        best_idx = int(np.argmin(scores))
        best_freq = freqs[best_idx]

        # 创建评分详情 DataFrame
        score_df = pd.DataFrame({
            'frequency': freqs,
            'jb_stat': jb_stats,
            'autocorr_1': ac_vals,
            'n_bars': n_bars_list,
            'jb_norm': jb_norm,
            'ac_norm': ac_norm,
            'n_bars_norm': n_bars_norm,
            'score': scores
        })

        return best_freq, score_df

    def _normalize(
        self,
        values: np.ndarray,
        lower_is_better: bool = True
    ) -> np.ndarray:
        """
        Min-Max 归一化

        :param values: 值数组
        :param lower_is_better: 是否越小越好
        :return: 归一化后的数组
        """
        # 处理 NaN
        valid_mask = ~np.isnan(values)
        normalized = np.full_like(values, np.nan, dtype=float)

        if not valid_mask.any():
            return normalized

        valid_vals = values[valid_mask]
        min_val = valid_vals.min()
        max_val = valid_vals.max()
        range_val = max_val - min_val

        if range_val < 1e-12:
            # 所有值相同
            normalized[valid_mask] = 0.5
        else:
            if lower_is_better:
                normalized[valid_mask] = (values[valid_mask] - min_val) / range_val
            else:
                normalized[valid_mask] = 1 - (values[valid_mask] - min_val) / range_val

        return normalized

    def generate_report(self, results: Dict[int, Dict]) -> str:
        """
        生成文本报告

        :param results: 各频率的评估结果
        :return: 格式化的文本报告
        """
        lines = []
        lines.append("=" * 70)
        lines.append(" IID 评估报告")
        lines.append("=" * 70)

        # 表格头部
        header = (
            f"{'频率':>8s} | {'Bar 数':>10s} | {'JB 统计量':>12s} | "
            f"{'JB p-value':>10s} | {'自相关 (1)':>10s} | {'偏度':>8s} | {'峰度':>8s}"
        )
        lines.append(header)
        lines.append("-" * 70)

        # 数据行
        for freq in sorted(results.keys()):
            r = results[freq]

            if "error" in r:
                lines.append(f"{freq:>8d} | {r.get('n_bars', 0):>10,d} | {'ERROR':>12s} | - | {r['error'][:30]}")
            else:
                line = (
                    f"{freq:>8d} | {r['n_bars']:>10,d} | {r['jb_stat']:>12.2f} | "
                    f"{r['jb_pvalue']:>10.6f} | {r['autocorr_1']:>10.4f} | "
                    f"{r['skew']:>8.4f} | {r['kurtosis']:>8.4f}"
                )
                lines.append(line)

        # 最优频率
        best_freq, _ = self.score(results)
        lines.append("-" * 70)
        lines.append(f"最优频率：{best_freq} bars/day")
        lines.append("=" * 70)

        return "\n".join(lines)


def evaluate_iid(ohlcv: pd.DataFrame) -> Dict:
    """
    便捷函数：执行 IID 评估

    :param ohlcv: OHLCV DataFrame
    :return: 评估结果字典
    """
    evaluator = IIDEvaluator()
    return evaluator.evaluate(ohlcv)


def compare_frequencies(bars_dict: Dict[int, pd.DataFrame]) -> Tuple[int, pd.DataFrame, str]:
    """
    便捷函数：比较多个频率的 IID 特性

    :param bars_dict: 各频率的 bars 字典
    :return: (最优频率，评分详情，文本报告)
    """
    evaluator = IIDEvaluator()

    # 评估所有频率
    results = {}
    for freq, bars_df in bars_dict.items():
        if len(bars_df) > 0:
            results[freq] = evaluator.evaluate(bars_df)

    # 计算评分
    best_freq, score_df = evaluator.score(results)

    # 生成报告
    report = evaluator.generate_report(results)

    return best_freq, score_df, report
