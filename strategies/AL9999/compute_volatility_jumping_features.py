"""
compute_volatility_jumping_features.py - AL9999 Volatility Jumping 因子计算

使用 FeatureKit 调用 afmlkit.feature.core.volatility_jumping 中的 Transform，
基于 AL9999 Dollar Bars 生成波动跳跃因子并保存到 output/features。

用法:
    uv run python strategies/AL9999/compute_volatility_jumping_features.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from afmlkit.feature.kit import Feature, FeatureKit
from afmlkit.feature.core.volatility_jumping import (
    AmountVolatilityTransform,
    DownsideJumpVariationTransform,
    DownsideRealizedVarianceTransform,
    DownsideVolShareTransform,
    DreamAmplitudeTransform,
    HighVolReturnVolRatioTransform,
    IntradayJumpnessTransform,
    IntradayReturnVolRatioTransform,
    JumpAsymmetryTransform,
    LargeDownsideJumpVariationTransform,
    LargeJumpAsymmetryTransform,
    LargeUpsideJumpVariationTransform,
    RealizedBipowerVariationTransform,
    RealizedJumpVariationTransform,
    RealizedTripowerVariationTransform,
    RealizedVarianceTransform,
    SmallDownsideJumpVariationTransform,
    SmallJumpAsymmetryTransform,
    SmallUpsideJumpVariationTransform,
    UpDownVolAsymmetryTransform,
    UpsideJumpVariationTransform,
    UpsideRealizedVarianceTransform,
    UpsideVolShareTransform,
)


def prepare_bars_for_featurekit(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Adapt AL9999 single-instrument bars to the transform input schema.

    :param bars: Dollar bars with ``dollar_volume`` and ``n_ticks``.
    :returns: DataFrame with added ``code``, ``amount`` and ``count`` columns.
    """
    df = bars.copy()
    df["code"] = "AL9999"
    df["amount"] = df["dollar_volume"].astype(float)
    df["count"] = df["n_ticks"].astype(int)
    return df


def build_volatility_jumping_kit() -> tuple[FeatureKit, list[str]]:
    """
    Build a FeatureKit for all implemented volatility jumping factors.

    :returns: FeatureKit instance and ordered feature names.
    """
    transforms = [
        DownsideVolShareTransform(frequency="H"),
        UpsideVolShareTransform(frequency="H"),
        RealizedVarianceTransform(frequency="H"),
        DownsideRealizedVarianceTransform(frequency="H"),
        UpsideRealizedVarianceTransform(frequency="H"),
        UpDownVolAsymmetryTransform(frequency="H"),
        RealizedBipowerVariationTransform(frequency="D"),
        RealizedTripowerVariationTransform(frequency="D"),
        RealizedJumpVariationTransform(frequency="D"),
        UpsideJumpVariationTransform(frequency="D"),
        DownsideJumpVariationTransform(frequency="D"),
        JumpAsymmetryTransform(frequency="D"),
        LargeUpsideJumpVariationTransform(frequency="D"),
        LargeDownsideJumpVariationTransform(frequency="D"),
        IntradayJumpnessTransform(frequency="D"),
        SmallUpsideJumpVariationTransform(frequency="D"),
        SmallDownsideJumpVariationTransform(frequency="D"),
        SmallJumpAsymmetryTransform(frequency="D"),
        LargeJumpAsymmetryTransform(frequency="D"),
        AmountVolatilityTransform(frequency="D"),
        IntradayReturnVolRatioTransform(frequency="D"),
        HighVolReturnVolRatioTransform(frequency="D"),
        DreamAmplitudeTransform(frequency="D"),
    ]
    features = [Feature(transform) for transform in transforms]
    feature_names = [feature.name for feature in features]
    retain = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "open_interest",
        "n_ticks",
    ]
    return FeatureKit(features, retain=retain), feature_names


def summarize_features(features_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    Build a compact summary for generated features.

    :param features_df: Generated feature DataFrame.
    :param feature_names: Factor column names.
    :returns: Summary DataFrame.
    """
    summary = pd.DataFrame(index=feature_names)
    summary["non_null"] = features_df[feature_names].notna().sum()
    summary["nan_ratio"] = features_df[feature_names].isna().mean().round(4)
    summary["mean"] = features_df[feature_names].mean().round(6)
    summary["std"] = features_df[feature_names].std().round(6)
    return summary


def main() -> None:
    from strategies.AL9999.config import BARS_DIR, FEATURES_DIR, TARGET_DAILY_BARS

    os.makedirs(FEATURES_DIR, exist_ok=True)

    bars_path = os.path.join(BARS_DIR, f"dollar_bars_target{TARGET_DAILY_BARS}.parquet")
    output_features_path = os.path.join(FEATURES_DIR, "bars_features_volatility_jumping.parquet")
    output_merged_path = os.path.join(FEATURES_DIR, "bars_features_with_volatility_jumping.parquet")
    output_summary_path = os.path.join(FEATURES_DIR, "bars_features_volatility_jumping_summary.csv")

    print("📂 加载 AL9999 Dollar Bars...")
    bars = pd.read_parquet(bars_path)
    print(f"   bars: {len(bars)} rows, {bars.shape[1]} columns")
    print(f"   时间范围: {bars.index.min()} -> {bars.index.max()}")

    kit_input = prepare_bars_for_featurekit(bars)
    kit, feature_names = build_volatility_jumping_kit()

    print("\n🧮 使用 FeatureKit 计算 Volatility Jumping 因子...")
    features_df = kit.build(kit_input, backend="pd", timeit=False)
    generated = features_df[feature_names].copy()

    generated.to_parquet(output_features_path)
    print(f"✅ 因子结果已保存: {output_features_path}")

    merged = bars.copy()
    for name in feature_names:
        merged[name] = generated[name]
    merged.to_parquet(output_merged_path)
    print(f"✅ 合并结果已保存: {output_merged_path}")

    summary = summarize_features(generated, feature_names)
    summary.to_csv(output_summary_path)
    print(f"✅ 因子摘要已保存: {output_summary_path}")

    print("\n📊 因子生成摘要:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
