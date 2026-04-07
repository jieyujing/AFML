"""Tests for volatility jumping transforms."""
import numpy as np
import pandas as pd
import pytest

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


def make_ohlcv(
    n_per_code: int = 60 * 24 * 4,
    codes: tuple[str, ...] = ("BTC", "ETH"),
    freq: str = "1min",
) -> pd.DataFrame:
    """Create a synthetic multi-code OHLCV DataFrame."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=n_per_code, freq=freq)
    frames: list[pd.DataFrame] = []

    for idx, code in enumerate(codes):
        base = 100 + idx * 10
        noise = rng.randn(n_per_code)
        close = base + np.cumsum(noise * 0.2 + 0.01)
        open_ = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.randn(n_per_code)) * 0.3 + 0.05
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        amount = rng.randint(1000, 100000, n_per_code).astype(float) * (1.0 + idx * 0.1)
        count = rng.randint(1, 100, n_per_code)
        volume = rng.randint(10, 1000, n_per_code).astype(float)

        frames.append(
            pd.DataFrame(
                {
                    "code": code,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "amount": amount,
                    "count": count,
                    "volume": volume,
                },
                index=dates,
            )
        )

    return pd.concat(frames).sort_index()


TRANSFORM_CASES = [
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


@pytest.mark.parametrize("transform", TRANSFORM_CASES, ids=lambda x: x.__class__.__name__)
def test_transform_contract(transform):
    """Each transform should return a named Series aligned with input rows."""
    df = make_ohlcv()

    result = transform(df, backend="pd")

    assert isinstance(result, pd.Series)
    assert len(result) == len(df)
    assert result.index.equals(df.index)
    assert result.name == transform.output_name
    assert not np.all(np.isnan(result.to_numpy(dtype=float)))
