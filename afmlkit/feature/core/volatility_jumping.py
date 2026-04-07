"""
Volatility jumping factor transforms derived from ``strategies/AL9999/volatility_jumping.py``.

These transforms operate on OHLCV-like DataFrames with a ``DatetimeIndex`` and
return a Series aligned to the original bar index by broadcasting each
``(code, frequency bucket)`` factor value back to all rows inside that bucket.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import gamma, pi
from typing import Callable

import numpy as np
import pandas as pd

from afmlkit.feature.base import MISOTransform


def _ensure_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")


def _bucket_index(index: pd.Index, frequency: str) -> pd.DatetimeIndex:
    normalized = "h" if frequency == "H" else frequency
    return pd.DatetimeIndex(index).floor(normalized)


def _mu_abs_normal(q: float) -> float:
    """
    Compute ``E[|Z|^q]`` for ``Z ~ N(0, 1)``.
    """
    return (2.0 ** (q / 2.0)) * gamma((q + 1.0) / 2.0) / (pi ** 0.5)


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _powered_variation(ret: pd.Series, powers: list[float]) -> float:
    value = pd.Series(1.0, index=ret.index, dtype=np.float64)
    for i, power in enumerate(powers):
        value = value * (ret.shift(i) ** power)
    return float(value.sum())


def _safe_ratio(numerator: float, denominator: float) -> float:
    if np.isnan(denominator) or np.isclose(denominator, 0.0):
        return np.nan
    return numerator / denominator


def _safe_covariance(x: pd.Series, y: pd.Series) -> float:
    """
    Compute covariance only when at least two aligned observations exist.
    """
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2:
        return np.nan
    return float(paired.iloc[:, 0].cov(paired.iloc[:, 1]))


def _tripower_integrated_variance(ret: pd.Series) -> float:
    abs_ret = ret.abs()
    mu = _mu_abs_normal(2.0 / 3.0)
    return (mu ** -3.0) * _powered_variation(abs_ret, [2.0 / 3.0] * 3)


def _jump_threshold(ret: pd.Series, integrated_variance: float, alpha: float, power: float) -> float:
    size = int(ret.dropna().size)
    if size == 0 or np.isnan(integrated_variance) or integrated_variance < 0.0:
        return np.nan
    delta = 1.0 / size
    return alpha * (delta ** power) * (integrated_variance ** 0.5)


def _iter_groups(
    df: pd.DataFrame,
    code_col: str,
    frequency: str,
) -> tuple[pd.DatetimeIndex, list[tuple[str, pd.Timestamp, pd.DataFrame]]]:
    _ensure_datetime_index(df)
    buckets = _bucket_index(df.index, frequency)
    work = df.copy()
    work["_bucket"] = buckets
    groups: list[tuple[str, pd.Timestamp, pd.DataFrame]] = []
    for (code, bucket), group in work.groupby([code_col, "_bucket"], sort=False):
        groups.append((code, bucket, group.drop(columns="_bucket")))
    return buckets, groups


def _broadcast_group_values(
    df: pd.DataFrame,
    code_col: str,
    buckets: pd.DatetimeIndex,
    grouped_values: pd.Series,
    output_name: str,
) -> pd.Series:
    keys = pd.MultiIndex.from_arrays(
        [df[code_col].to_numpy(), buckets.to_numpy()],
        names=[code_col, "bucket"],
    )
    values = grouped_values.reindex(keys).to_numpy(dtype=np.float64)
    return pd.Series(values, index=df.index, name=output_name)


def _broadcast_scalar_per_group(
    df: pd.DataFrame,
    code_col: str,
    frequency: str,
    compute: Callable[[pd.DataFrame], float],
    output_name: str,
) -> pd.Series:
    buckets, groups = _iter_groups(df, code_col, frequency)
    grouped_values = pd.Series(
        {
            (code, bucket): compute(group)
            for code, bucket, group in groups
        },
        dtype=np.float64,
    )
    grouped_values.index = pd.MultiIndex.from_tuples(grouped_values.index, names=[code_col, "bucket"])
    return _broadcast_group_values(df, code_col, buckets, grouped_values, output_name)


class _GroupedFactorTransform(MISOTransform, ABC):
    """
    Base class for bucket-level factor transforms that broadcast results back to rows.
    """

    def __init__(
        self,
        input_cols: list[str],
        output_col: str,
        frequency: str,
        code_col: str,
    ):
        super().__init__(input_cols, output_col)
        self.frequency = frequency
        self.code_col = code_col

    def _nb(self, x: pd.DataFrame) -> pd.Series:
        return self._pd(x)

    def _validate_common(self, x: pd.DataFrame) -> None:
        _ensure_datetime_index(x)
        if self.code_col not in x.columns:
            raise ValueError(f"Input DataFrame must contain '{self.code_col}' column.")


class _ScalarGroupedFactorTransform(_GroupedFactorTransform, ABC):
    """
    Base class for scalar-per-bucket factors.
    """

    def _pd(self, x: pd.DataFrame) -> pd.Series:
        self._validate_common(x)
        return _broadcast_scalar_per_group(
            x,
            code_col=self.code_col,
            frequency=self.frequency,
            compute=self._compute_group,
            output_name=self.output_name,
        )

    @abstractmethod
    def _compute_group(self, group: pd.DataFrame) -> float:
        pass


class DownsideVolShareTransform(_ScalarGroupedFactorTransform):
    """QIML0105: Downside volatility share."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "downside_vol_share"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        total = float((ret ** 2).sum())
        down = float(((ret[ret < 0.0]) ** 2).sum())
        return _safe_ratio(down, total)


class UpsideVolShareTransform(_ScalarGroupedFactorTransform):
    """QIML0113: Upside volatility share."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "upside_vol_share"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        total = float((ret ** 2).sum())
        up = float(((ret[ret > 0.0]) ** 2).sum())
        return _safe_ratio(up, total)


class RealizedVarianceTransform(_ScalarGroupedFactorTransform):
    """QIML0206: Realized variance."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "realized_variance"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        return float((ret ** 2).sum())


class DownsideRealizedVarianceTransform(_ScalarGroupedFactorTransform):
    """QIML0219: Downside realized variance."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "downside_realized_variance"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        return float(((ret[ret < 0.0]) ** 2).sum())


class UpsideRealizedVarianceTransform(_ScalarGroupedFactorTransform):
    """QIML0226: Upside realized variance."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "upside_realized_variance"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        return float(((ret[ret > 0.0]) ** 2).sum())


class UpDownVolAsymmetryTransform(_ScalarGroupedFactorTransform):
    """QIML0304: Up/down volatility asymmetry."""

    def __init__(self, frequency: str = "H", code_col: str = "code", close_col: str = "close", output_col: str = "up_down_vol_asymmetry"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        total = float((ret ** 2).sum())
        up = float(((ret[ret > 0.0]) ** 2).sum())
        down = float(((ret[ret < 0.0]) ** 2).sum())
        return _safe_ratio(up - down, total)


class RealizedBipowerVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0411: Realized bipower variation."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "realized_bipower_variation"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col]).abs()
        mu = _mu_abs_normal(1.0)
        return (mu ** -2.0) * _powered_variation(ret, [1.0, 1.0])


class RealizedTripowerVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0424: Realized tripower variation."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "realized_tripower_variation"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col]).abs()
        mu = _mu_abs_normal(2.0 / 3.0)
        return (mu ** -3.0) * _powered_variation(ret, [2.0 / 3.0] * 3)


class RealizedJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0516: Realized jump variation."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "realized_jump_variation"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        rv = float((ret ** 2).sum())
        return max(rv - iv, 0.0)


def _upside_jump_value(ret: pd.Series, integrated_variance: float) -> float:
    rs_p = float(((ret[ret > 0.0]) ** 2).sum())
    return max(rs_p - 0.5 * integrated_variance, 0.0)


def _downside_jump_value(ret: pd.Series, integrated_variance: float) -> float:
    rs_n = float(((ret[ret < 0.0]) ** 2).sum())
    return max(rs_n - 0.5 * integrated_variance, 0.0)


class UpsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0530: Upside jump variation."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "upside_jump_variation"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        return _upside_jump_value(ret, iv)


class DownsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0611: Downside jump variation."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "downside_jump_variation"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        return _downside_jump_value(ret, iv)


class JumpAsymmetryTransform(_ScalarGroupedFactorTransform):
    """QIML0615: Jump asymmetry."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "jump_asymmetry"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        return _upside_jump_value(ret, iv) - _downside_jump_value(ret, iv)


class LargeUpsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0702: Large upside jump variation."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "large_upside_jump_variation",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        rvjp = _upside_jump_value(ret, iv)
        rslp = float(((ret[ret > gamma_value]) ** 2).sum())
        return min(rvjp, rslp)


class LargeDownsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0720: Large downside jump variation."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "large_downside_jump_variation",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        rvjn = _downside_jump_value(ret, iv)
        rsln = float(((ret[ret < -gamma_value]) ** 2).sum())
        return min(rvjn, rsln)


class IntradayJumpnessTransform(_ScalarGroupedFactorTransform):
    """QIML0726: Intraday jumpness."""

    def __init__(self, frequency: str = "D", code_col: str = "code", close_col: str = "close", output_col: str = "intraday_jumpness"):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        s_ret = group[self.close_col].pct_change()
        l_ret = _log_returns(group[self.close_col])
        v_jump = (s_ret - l_ret) * 2.0 - (l_ret ** 2)
        return float(v_jump.mean())


class SmallUpsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0804: Small upside jump variation."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "small_upside_jump_variation",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        rvjp = _upside_jump_value(ret, iv)
        rslp = float(((ret[ret > gamma_value]) ** 2).sum())
        return rvjp - min(rvjp, rslp)


class SmallDownsideJumpVariationTransform(_ScalarGroupedFactorTransform):
    """QIML0817: Small downside jump variation."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "small_downside_jump_variation",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        rvjn = _downside_jump_value(ret, iv)
        rsln = float(((ret[ret < -gamma_value]) ** 2).sum())
        return rvjn - min(rvjn, rsln)


class SmallJumpAsymmetryTransform(_ScalarGroupedFactorTransform):
    """QIML0918: Small jump asymmetry."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "small_jump_asymmetry",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        small_up = _upside_jump_value(ret, iv) - min(_upside_jump_value(ret, iv), float(((ret[ret > gamma_value]) ** 2).sum()))
        small_down = _downside_jump_value(ret, iv) - min(_downside_jump_value(ret, iv), float(((ret[ret < -gamma_value]) ** 2).sum()))
        return small_up - small_down


class LargeJumpAsymmetryTransform(_ScalarGroupedFactorTransform):
    """QIML0921: Large jump asymmetry."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        output_col: str = "large_jump_asymmetry",
        alpha: float = 4.0,
        power: float = 0.49,
    ):
        super().__init__([code_col, close_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.alpha = alpha
        self.power = power

    def _compute_group(self, group: pd.DataFrame) -> float:
        ret = _log_returns(group[self.close_col])
        iv = _tripower_integrated_variance(ret)
        gamma_value = _jump_threshold(ret, iv, self.alpha, self.power)
        if np.isnan(gamma_value):
            return np.nan
        large_up = min(_upside_jump_value(ret, iv), float(((ret[ret > gamma_value]) ** 2).sum()))
        large_down = min(_downside_jump_value(ret, iv), float(((ret[ret < -gamma_value]) ** 2).sum()))
        return large_up - large_down


class AmountVolatilityTransform(_ScalarGroupedFactorTransform):
    """QIML1023: Amount volatility."""

    def __init__(self, frequency: str = "D", code_col: str = "code", amount_col: str = "amount", output_col: str = "amount_volatility"):
        super().__init__([code_col, amount_col], output_col, frequency, code_col)
        self.amount_col = amount_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        return float(group[self.amount_col].std())


class IntradayReturnVolRatioTransform(_ScalarGroupedFactorTransform):
    """QIML1101: Intraday return-volatility ratio."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        output_col: str = "intraday_return_vol_ratio",
    ):
        super().__init__([code_col, open_col, high_col, low_col, close_col], output_col, frequency, code_col)
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        x_ret = group[self.close_col].pct_change()
        work = group[[self.open_col, self.close_col, self.high_col, self.low_col]].copy()
        for lag in range(1, 5):
            shifted = group[[self.open_col, self.close_col, self.high_col, self.low_col]].shift(lag)
            shifted.columns = [f"{col}{lag}" for col in [self.open_col, self.close_col, self.high_col, self.low_col]]
            work = pd.concat([work, shifted], axis=1)
        better_vol = (work.std(axis=1) / work.mean(axis=1)) ** 2
        ret_vol_ratio = x_ret / better_vol
        return _safe_covariance(ret_vol_ratio, better_vol)


class HighVolReturnVolRatioTransform(_ScalarGroupedFactorTransform):
    """QIML1117: Return-volatility ratio during abnormally high volatility."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        output_col: str = "high_vol_return_vol_ratio",
    ):
        super().__init__([code_col, open_col, high_col, low_col, close_col], output_col, frequency, code_col)
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        x_ret = group[self.close_col].pct_change()
        work = group[[self.open_col, self.close_col, self.high_col, self.low_col]].copy()
        for lag in range(1, 5):
            shifted = group[[self.open_col, self.close_col, self.high_col, self.low_col]].shift(lag)
            shifted.columns = [f"{col}{lag}" for col in [self.open_col, self.close_col, self.high_col, self.low_col]]
            work = pd.concat([work, shifted], axis=1)
        work_dropna = work.dropna()
        better_vol = (work_dropna.std(axis=1) / work_dropna.mean(axis=1)) ** 2
        ret_vol_ratio = x_ret / better_vol
        subset = pd.DataFrame({"better_vol": better_vol, "ret_vol_ratio": ret_vol_ratio})
        threshold = subset["better_vol"].mean() + subset["better_vol"].std()
        high_vol_time = subset[subset["better_vol"] >= threshold]
        return _safe_covariance(high_vol_time["better_vol"], high_vol_time["ret_vol_ratio"])


class DreamAmplitudeTransform(_ScalarGroupedFactorTransform):
    """QIML1230: Dream amplitude."""

    def __init__(
        self,
        frequency: str = "D",
        code_col: str = "code",
        close_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        output_col: str = "dream_amplitude",
    ):
        super().__init__([code_col, close_col, high_col, low_col], output_col, frequency, code_col)
        self.close_col = close_col
        self.high_col = high_col
        self.low_col = low_col

    def _compute_group(self, group: pd.DataFrame) -> float:
        work = pd.DataFrame({self.close_col: group[self.close_col]})
        work["amplitude"] = (group[self.high_col] / group[self.low_col]) - 1.0
        threshold_up = work[self.close_col].quantile(q=0.75)
        threshold_down = work[self.close_col].quantile(q=0.25)
        high_amp = work.loc[work[self.close_col] >= threshold_up, "amplitude"].mean()
        low_amp = work.loc[work[self.close_col] <= threshold_down, "amplitude"].mean()
        return float(high_amp - low_amp)


__all__ = [
    "DownsideVolShareTransform",
    "UpsideVolShareTransform",
    "RealizedVarianceTransform",
    "DownsideRealizedVarianceTransform",
    "UpsideRealizedVarianceTransform",
    "UpDownVolAsymmetryTransform",
    "RealizedBipowerVariationTransform",
    "RealizedTripowerVariationTransform",
    "RealizedJumpVariationTransform",
    "UpsideJumpVariationTransform",
    "DownsideJumpVariationTransform",
    "JumpAsymmetryTransform",
    "LargeUpsideJumpVariationTransform",
    "LargeDownsideJumpVariationTransform",
    "IntradayJumpnessTransform",
    "SmallUpsideJumpVariationTransform",
    "SmallDownsideJumpVariationTransform",
    "SmallJumpAsymmetryTransform",
    "LargeJumpAsymmetryTransform",
    "AmountVolatilityTransform",
    "IntradayReturnVolRatioTransform",
    "HighVolReturnVolRatioTransform",
    "DreamAmplitudeTransform",
]
