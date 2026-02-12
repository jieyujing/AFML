"""
End-to-End Integration Test for Polars AFML Pipeline.

This script tests the complete Polars-based ML pipeline from
raw data to predictions.
"""

import numpy as np
import polars as pl
from afml import (
    DollarBarsProcessor,
    TripleBarrierLabeler,
    FeatureEngineer,
    SampleWeightCalculator,
    PurgedKFoldCV,
    MetaLabelingPipeline,
    BetSizer,
    to_polars,
)


def generate_sample_data(n_rows: int = 10000) -> pl.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    dates = pl.datetime_range(
        start=pl.datetime(2020, 1, 1),
        end=pl.datetime(2024, 1, 1),
        interval="1h",
        eager=True,
    )[:n_rows]

    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    open_prices = close + np.random.randn(n_rows) * 0.1
    high = np.maximum(open_prices, close) + np.abs(np.random.randn(n_rows) * 0.1)
    low = np.minimum(open_prices, close) - np.abs(np.random.randn(n_rows) * 0.1)
    volume = np.random.randint(1000, 10000, n_rows).astype(float)

    return pl.DataFrame(
        {
            "datetime": dates,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_dollar_bars(df: pl.DataFrame) -> pl.DataFrame:
    """Test Dollar Bars generation."""
    print("Testing DollarBarsProcessor...")
    processor = DollarBarsProcessor(daily_target=4, lazy=False)
    dollar_bars = processor.fit_transform(df)
    print(f"  Generated {len(dollar_bars)} dollar bars")
    print(f"  Threshold: {processor.threshold_:.2f}")
    return dollar_bars


def test_triple_barrier(df: pl.DataFrame) -> pl.DataFrame:
    """Test Triple Barrier labeling."""
    print("\nTesting TripleBarrierLabeler...")
    labeler = TripleBarrierLabeler(
        pt_sl=[1.0, 1.0],
        vertical_barrier_bars=12,
    )
    labeler.fit(df["close"])

    cusum_events = labeler.get_cusum_events(df["close"])
    print(f"  Generated {len(cusum_events)} CUSUM events")

    return cusum_events


def test_features(df: pl.DataFrame) -> pl.DataFrame:
    """Test Feature engineering."""
    print("\nTesting FeatureEngineer...")
    engineer = FeatureEngineer(
        windows=[5, 10, 20, 30, 50],
        ffd_d=0.5,
    )
    features = engineer.fit_transform(df)
    print(f"  Generated {features.width} columns")
    print(f"  Feature columns: {[c for c in features.columns if c not in df.columns]}")
    return features


def test_sample_weights(events: pl.DataFrame) -> pl.DataFrame:
    """Test Sample Weight calculation."""
    print("\nTesting SampleWeightCalculator...")
    calculator = SampleWeightCalculator(decay=0.9)
    weights = calculator.fit_transform(events)
    print(f"  Generated {len(weights)} weights")
    print(f"  Mean weight: {weights['weight'].mean():.4f}")
    return weights


def test_cv(df: pl.DataFrame) -> None:
    """Test Purged K-Fold CV."""
    print("\nTesting PurgedKFoldCV...")
    cv = PurgedKFoldCV(n_splits=3, embargo=0.1, purge=1)

    X = df.select(pl.all().exclude("datetime"))
    y = (df["close"].pct_change().shift(-1) > 0).cast(pl.Int8)

    splits = list(cv.split(X, y))
    print(f"  Generated {len(splits)} splits")

    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"    Split {i + 1}: train={len(train_idx)}, test={len(test_idx)}")


def test_pipeline(df: pl.DataFrame) -> None:
    """Test complete pipeline."""
    print("\n" + "=" * 60)
    print("Testing Complete Polars AFML Pipeline")
    print("=" * 60)

    # 1. Generate Dollar Bars
    dollar_bars = test_dollar_bars(df)
    assert len(dollar_bars) > 0, "Dollar bars generation failed"

    # 2. Generate Features
    features = test_features(dollar_bars)
    feature_cols = [c for c in features.columns if c not in dollar_bars.columns]
    assert len(feature_cols) > 0, "Feature engineering failed"

    # 3. Generate Labels
    labeler = TripleBarrierLabeler(pt_sl=[1.0, 1.0], vertical_barrier_bars=12)
    labeler.fit(features["close"])
    cusum_events = labeler.get_cusum_events(features["close"])
    events = labeler.label(features["close"], cusum_events)
    print(f"\n  Generated {len(events)} labeled events")

    # 4. Calculate Sample Weights
    if len(events) > 0:
        weights = test_sample_weights(events)

    # 5. Test CV
    test_cv(features)

    print("\n" + "=" * 60)
    print("All integration tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(n_rows=5000)
    print(f"Generated {len(df)} rows of sample data")

    test_pipeline(df)
