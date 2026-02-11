"""
Polars Pipeline - Complete ML Pipeline using Polars and AFML Methodology.

This script implements a complete financial ML pipeline using Polars-based
processors, enforcing strict AFML standards including stationarity and DSR.

Usage:
    uv run python src/afml_polars_pipeline.py [input_file] [options]

Pipeline Steps:
    1. Load Data
    2. Check Stationarity (Auto-FFD)
    3. Generate Dollar Bars
    4. Apply Triple Barrier Labeling
    5. Generate Features (Alpha158 + FFD)
    6. Calculate Sample Weights
    7. Cross-Validation with Purged K-Fold
    8. Meta-Labeling
    9. Bet Sizing
    10. Strategy Verification (DSR)

Output:
    - Data Artifacts in data/
    - Comprehensive Report
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import polars as pl
from scipy.stats import skew, kurtosis

from afml.polars import (
    PolarsDollarBarsProcessor,
    PolarsTripleBarrierLabeler,
    PolarsFeatureEngineer,
    PolarsSampleWeightCalculator,
    PolarsPurgedKFoldCV,
    PolarsMetaLabelingPipeline,
    PolarsBetSizer,
)
from afml.stationarity import get_min_d
from afml.metrics import get_strategy_metrics


# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("visual_analysis")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_raw_data(filepath: str) -> pl.DataFrame:
    """
    Load raw tick data from CSV file.

    Args:
        filepath: Path to CSV file with columns:
            datetime, open, high, low, close, volume

    Returns:
        Polars DataFrame
    """
    print(f"\n{'=' * 60}")
    print("Step 1: Loading Raw Data")
    print(f"{'=' * 60}")

    print(f"Loading data from {filepath}...")

    df = pl.read_csv(filepath, try_parse_dates=True)

    if "datetime" not in df.columns:
        raise ValueError("Data must have 'datetime' column")

    df = df.sort("datetime")

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    return df


def generate_dollar_bars(
    df: pl.DataFrame,
    daily_target: int = 4,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Generate dollar bars from raw tick data.

    Args:
        df: Raw tick data with OHLCV columns
        daily_target: Target number of bars per day
        output_path: Optional path to save results

    Returns:
        Dollar bars DataFrame
    """
    print(f"\n{'=' * 60}")
    print("Step 2: Generating Dollar Bars")
    print(f"{'=' * 60}")

    processor = PolarsDollarBarsProcessor(
        daily_target=daily_target,
        lazy=False,
    )

    print(f"  Daily target: {daily_target} bars/day")
    print("  Processing...")

    dollar_bars = processor.fit_transform(df)

    threshold_info = processor.get_threshold_info()
    print(f"  Threshold: {threshold_info['threshold']:,.0f}")
    print(f"  Generated: {len(dollar_bars):,} bars")

    if output_path or DATA_DIR / "dollar_bars_polars.csv":
        output_path = output_path or str(DATA_DIR / "dollar_bars_polars.csv")
        dollar_bars.write_csv(output_path)
        print(f"  Saved to: {output_path}")

    return dollar_bars


def apply_labels(
    df: pl.DataFrame,
    pt_sl: list = None,
    vertical_barrier_bars: int = 12,
    output_path: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply triple barrier labeling to events.

    Args:
        df: Dollar bars DataFrame
        pt_sl: Profit taking / Stop loss multipliers
        vertical_barrier_bars: Maximum holding period
        output_path: Optional path to save results

    Returns:
        Tuple of (events DataFrame, labeled DataFrame)
    """
    print(f"\n{'=' * 60}")
    print("Step 3: Applying Triple Barrier Labels")
    print(f"{'=' * 60}")

    pt_sl = pt_sl or [1.0, 1.0]

    labeler = PolarsTripleBarrierLabeler(
        pt_sl=pt_sl,
        vertical_barrier_bars=vertical_barrier_bars,
    )

    print(f"  PT/SL: {pt_sl}")
    print(f"  Vertical barrier: {vertical_barrier_bars} bars")

    # Fit labeler
    labeler.fit(df["close"])
    print("  Volatility computed")

    # Generate CUSUM events
    events = labeler.get_cusum_events(df["close"])
    print(f"  CUSUM events: {len(events):,}")

    if len(events) > 0:
        # Apply labels
        labeled = labeler.label(df["close"], events)
        print(f"  Labeled events: {len(labeled):,}")

        # Show label distribution
        if "label" in labeled.columns:
            label_dist = labeled.group_by("label").agg(pl.count().alias("count"))
            print("  Label distribution:")
            for row in label_dist.rows():
                label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}.get(
                    row[0], "Unknown"
                )
                print(f"    {label_name} ({row[0]}): {row[1]:,}")
    else:
        labeled = pl.DataFrame({"t1": [], "tr": [], "label": []})
        print("  No events generated")

    if output_path or DATA_DIR / "labeled_polars.csv":
        output_path = output_path or str(DATA_DIR / "labeled_polars.csv")
        labeled.write_csv(output_path)
        print(f"  Saved to: {output_path}")

    return events, labeled


def generate_features(
    df: pl.DataFrame,
    windows: list = None,
    ffd_d: float = 0.5,
    check_stationarity: bool = True,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Generate features using Alpha158 and FFD methods.

    Args:
        df: Dollar bars DataFrame
        windows: Rolling window sizes
        ffd_d: Fractional differentiation parameter (initial guess or fixed)
        check_stationarity: Whether to automatically find optimal d
        output_path: Optional path to save results

    Returns:
        DataFrame with features
    """
    print(f"\n{'=' * 60}")
    print("Step 4: Generating Features (Alpha158 + FFD)")
    print(f"{'=' * 60}")

    windows = windows or [5, 10, 20, 30, 50]
    optimal_d = ffd_d

    if check_stationarity:
        print("  Checking stationarity for optimal d...")
        # Use close price for determining d
        # Polars Series to numpy
        close_prices = df["close"].to_numpy()
        min_d, p_val = get_min_d(close_prices)
        
        print(f"  Optimal d found: {min_d} (p-value: {p_val:.4f})")
        if p_val > 0.05:
            print("  WARNING: Could not achieve stationarity with d < 1.0")
            print("  Using d=1.0 for integer differentiation")
            
        optimal_d = min_d
    else:
        print(f"  Using fixed d: {ffd_d}")

    engineer = PolarsFeatureEngineer(
        windows=windows,
        ffd_d=optimal_d,
    )

    print(f"  Windows: {windows}")
    print(f"  FFD d: {optimal_d}")
    print("  Processing...")

    features = engineer.fit_transform(df)

    # Get feature info
    info = engineer.get_feature_info()
    feature_cols = [c for c in features.columns if c not in df.columns]

    print(f"  Original columns: {len(df.columns)}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Total columns: {len(features.columns)}")

    if output_path or DATA_DIR / "features_polars.csv":
        output_path = output_path or str(DATA_DIR / "features_polars.csv")
        features.write_csv(output_path)
        print(f"  Saved to: {output_path}")
        
    # Store optimal d in metadata or return if needed, currently embedded in features
    return features


def calculate_weights(
    events: pl.DataFrame,
    decay: float = 0.9,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Calculate sample weights using uniqueness and time decay.

    Args:
        events: Labeled events DataFrame
        decay: Time decay factor
        output_path: Optional path to save results

    Returns:
        DataFrame with weights
    """
    print(f"\n{'=' * 60}")
    print("Step 5: Calculating Sample Weights")
    print(f"{'=' * 60}")

    if len(events) == 0:
        print("  No events, returning empty weights")
        return pl.DataFrame({"weight": [], "uniqueness": []})

    calculator = PolarsSampleWeightCalculator(decay=decay)

    print(f"  Decay factor: {decay}")

    weights = calculator.fit_transform(events)

    print(f"  Events: {len(events):,}")
    print(f"  Mean weight: {weights['weight'].mean():.4f}")
    print(f"  Std weight: {weights['weight'].std():.4f}")

    if output_path or DATA_DIR / "sample_weights_polars.csv":
        output_path = output_path or str(DATA_DIR / "sample_weights_polars.csv")
        weights.write_csv(output_path)
        print(f"  Saved to: {output_path}")

    return weights


def run_cross_validation(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    n_splits: int = 5,
    embargo: float = 0.1,
) -> None:
    """
    Run Purged K-Fold cross-validation.

    Args:
        features: Feature DataFrame
        labels: Labeled events DataFrame
        n_splits: Number of CV folds
        embargo: Embargo proportion
    """
    print(f"\n{'=' * 60}")
    print("Step 6: Cross-Validation (Purged K-Fold)")
    print(f"{'=' * 60}")

    cv = PolarsPurgedKFoldCV(
        n_splits=n_splits,
        embargo=embargo,
        purge=1,
    )

    print(f"  Splits: {n_splits}")
    print(f"  Embargo: {embargo}")

    # Prepare data
    X = features.select(pl.all().exclude("datetime"))
    y = labels["label"] if "label" in labels.columns else None

    print(f"  Features shape: {X.width}")

    # Generate splits
    splits = list(cv.split(X, y))

    print(f"  Generated {len(splits)} splits:")
    for i, (train_idx, test_idx) in enumerate(splits):
        print(f"    Split {i + 1}: train={len(train_idx):,}, test={len(test_idx):,}")


def run_meta_labeling(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    weights: pl.DataFrame,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Run meta-labeling pipeline.

    Args:
        features: Feature DataFrame
        labels: Labeled events DataFrame
        weights: Sample weights
        output_path: Optional path to save results

    Returns:
        DataFrame with predictions
    """
    print(f"\n{'=' * 60}")
    print("Step 7: Meta-Labeling Pipeline")
    print(f"{'=' * 60}")

    feature_cols = [
        c
        for c in features.columns
        if c not in ["datetime", "open", "high", "low", "close", "volume", "amount"]
    ]

    if len(feature_cols) == 0 or len(labels) == 0:
        print("  Skipping meta-labeling (no features or labels)")
        return pl.DataFrame()

    # Use simple approach - align by position
    n_samples = min(len(features), len(labels))

    X_np = features.head(n_samples).select(feature_cols).to_numpy()
    y_np = (
        labels.head(n_samples)["label"].to_numpy()
        if "label" in labels.columns
        else np.zeros(n_samples)
    )
    
    # Use weights if available (current pipeline doesn't use them in fit directly 
    # but could be used for sample_weight in scikit-learn models)
    if not weights.is_empty():
        w_np = weights.head(n_samples)["weight"].to_numpy()
    else:
        w_np = None

    print(f"  Samples: {n_samples}")
    print(f"  Features: {len(feature_cols)}")

    # Create pipeline
    pipeline = PolarsMetaLabelingPipeline(
        primary_model="random_forest",
        meta_model="logistic",
        primary_params={"n_estimators": 50, "max_depth": 3},
    )

    print("  Primary model: Random Forest")
    print("  Meta model: Logistic Regression")

    try:
        pipeline.fit(X_np, y_np)
        print("  Models fitted")

        predictions = pipeline.predict(X_np)
        print(f"  Predictions: {len(predictions)}")

        metrics = pipeline.score(X_np, y_np)
        print("\n  Performance Metrics:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
    except Exception as e:
        print(f"  Meta-labeling error: {e}")
        predictions = np.zeros(n_samples)
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    predictions_df = pl.DataFrame(
        {
            "prediction": predictions.astype(int),
            "actual": y_np.astype(int),
        }
    )

    if output_path or DATA_DIR / "predictions_polars.csv":
        output_path = output_path or str(DATA_DIR / "predictions_polars.csv")
        predictions_df.write_csv(output_path)
        print(f"\n  Saved predictions to: {output_path}")

    return predictions_df


def calculate_bet_sizes(
    features: pl.DataFrame,
    predictions: pl.DataFrame,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Calculate bet sizes from predictions.

    Args:
        features: Feature DataFrame
        predictions: Predictions DataFrame
        threshold: Probability threshold
        output_path: Optional path to save results

    Returns:
        DataFrame with bet sizes
    """
    print(f"\n{'=' * 60}")
    print("Step 8: Bet Sizing")
    print(f"{'=' * 60}")

    if len(predictions) == 0:
        print("  No predictions, skipping bet sizing")
        return pl.DataFrame()

    sizer = PolarsBetSizer(
        threshold=threshold,
        quantity=100,
    )

    print(f"  Threshold: {threshold}")
    print(f"  Quantity: 100")

    # Get bet sizes
    bet_sizes = sizer.fit_transform(
        pl.Series(predictions["prediction"].to_numpy().astype(float)),
        pl.Series(predictions["actual"].to_numpy().astype(float)),
    )

    print(f"  Mean bet size: {bet_sizes.mean():.4f}")
    print(f"  Std bet size: {bet_sizes.std():.4f}")

    # Get metrics
    metrics = sizer.get_metrics(
        pl.Series(predictions["actual"].to_numpy()),
        bet_sizes,
    )

    print("\n  Bet Sizing Metrics:")
    print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"    Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"    Max Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"    Total Return: {metrics['total_return']:.4f}")

    bet_sizes_df = pl.DataFrame(
        {
            "bet_size": bet_sizes,
            "prediction": predictions["prediction"],
            "actual": predictions["actual"],
        }
    )

    if output_path or DATA_DIR / "bet_sizes_polars.csv":
        output_path = output_path or str(DATA_DIR / "bet_sizes_polars.csv")
        bet_sizes_df.write_csv(output_path)
        print(f"\n  Saved bet sizes to: {output_path}")

    return bet_sizes_df


def verify_strategy(
    labeled: pl.DataFrame,
    bet_sizes: pl.DataFrame,
    n_trials: int = 1,
    acceptance_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Verify strategy using Deflated Sharpe Ratio.
    
    Args:
        labeled: DataFrame with actual returns ('tr').
        bet_sizes: DataFrame with bet sizes.
        n_trials: Number of trials (for DSR adjustment).
        acceptance_threshold: Threshold for DSR probability.
        
    Returns:
        Metrics dictionary.
    """
    print(f"\n{'=' * 60}")
    print("Step 9: Strategy Verification (DSR)")
    print(f"{'=' * 60}")
    
    if bet_sizes.is_empty() or labeled.is_empty():
        print("  Skipping verification (no data)")
        return {"status": "FAILED", "reason": "No data"}
        
    # Align data
    n_samples = min(len(bet_sizes), len(labeled))
    bet_s = bet_sizes["bet_size"].head(n_samples).to_numpy()
    
    if "tr" in labeled.columns:
        # Target return from triple barrier
        returns_s = labeled["tr"].head(n_samples).to_numpy()
    else:
        # Fallback if 'tr' not present (should be in standard labeler)
        print("  WARNING: 'tr' column not found in labels, using simplified verification")
        returns_s = labeled["label"].head(n_samples).to_numpy() # Proxy
        
    # Calculate strategy returns
    strategy_returns = bet_s * returns_s
    
    # Calculate metrics
    metrics = get_strategy_metrics(strategy_returns, n_trials=n_trials)
    
    print("\n  Statistical Validation:")
    print(f"    Mean Return: {metrics.get('mean_return', 0):.6f}")
    print(f"    Annualized SR: {metrics.get('annualized_sharpe_ratio', 0):.4f}")
    print(f"    Skewness: {metrics.get('skewness', 0):.4f}")
    print(f"    Kurtosis: {metrics.get('kurtosis', 0):.4f}")
    print(f"    PSR (Prob. SR > 0): {metrics.get('psr', 0):.4f}")
    print(f"    DSR (Deflated SR): {metrics.get('dsr', 0):.4f}")
    
    dsr_val = metrics.get('dsr', 0)
    passed = dsr_val > acceptance_threshold
    status = "PASSED" if passed else "FAILED"
    
    print(f"\n  ACCEPTANCE STATUS: {status}")
    print(f"  (Threshold: {acceptance_threshold})")
    
    metrics["status"] = status
    return metrics


def run_pipeline(
    input_path: str,
    daily_target: int = 4,
    pt_sl: list = None,
    vertical_barrier_bars: int = 12,
    windows: list = None,
    ffd_d: float = 0.5,
    check_stationarity: bool = True,
    decay: float = 0.9,
    n_splits: int = 5,
    embargo: float = 0.1,
    acceptance_threshold: float = 0.95,
) -> dict:
    """
    Run complete Polars ML pipeline.

    Args:
        input_path: Path to input CSV file
        daily_target: Dollar bars per day
        pt_sl: Profit taking / Stop loss multipliers
        vertical_barrier_bars: Maximum holding period
        windows: Rolling window sizes for features
        ffd_d: Fractional differentiation parameter
        check_stationarity: Auto-find min d
        decay: Time decay factor for weights
        n_splits: Number of CV folds
        embargo: Embargo proportion for CV
        acceptance_threshold: DSR threshold for acceptance

    Returns:
        Dict with pipeline results
    """
    print("\n" + "=" * 60)
    print("POLARS ML PIPELINE (AFML COMPLIANT)")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input file: {input_path}")

    results = {
        "input_path": input_path,
        "parameters": {
            "daily_target": daily_target,
            "pt_sl": pt_sl,
            "vertical_barrier_bars": vertical_barrier_bars,
            "windows": windows,
            "ffd_d": ffd_d,
            "check_stationarity": check_stationarity,
            "decay": decay,
            "n_splits": n_splits,
            "embargo": embargo,
        },
    }

    try:
        # Step 1: Load data
        df = load_raw_data(input_path)
        results["raw_rows"] = len(df)

        # Step 2: Dollar bars
        dollar_bars = generate_dollar_bars(df, daily_target=daily_target)
        results["dollar_bars_count"] = len(dollar_bars)

        # Step 3: Labels
        events, labeled = apply_labels(
            dollar_bars,
            pt_sl=pt_sl,
            vertical_barrier_bars=vertical_barrier_bars,
        )
        results["labeled_count"] = len(labeled)

        # Step 4: Features (with Stationarity)
        features = generate_features(
            dollar_bars,
            windows=windows,
            ffd_d=ffd_d,
            check_stationarity=check_stationarity,
        )
        results["feature_count"] = len(features.columns)

        # Step 5: Weights
        weights = calculate_weights(events, decay=decay)
        results["weights_count"] = len(weights)

        # Step 6: Cross-validation
        run_cross_validation(features, labeled, n_splits=n_splits, embargo=embargo)

        # Step 7: Meta-labeling
        predictions = run_meta_labeling(features, labeled, weights)

        # Step 8: Bet sizing
        bet_sizes = calculate_bet_sizes(features, predictions)
        results["predictions_count"] = len(predictions)
        
        # Step 9: Verification
        # Assume 1 trial for this run, but in real research this should increase
        metrics = verify_strategy(
            labeled, 
            bet_sizes, 
            n_trials=1, 
            acceptance_threshold=acceptance_threshold
        )
        results["metrics"] = metrics
        results["status"] = metrics.get("status", "UNKNOWN")

    except Exception as e:
        print(f"\nError: {e}")
        # import traceback
        # traceback.print_exc()
        results["status"] = "ERROR"
        results["error"] = str(e)
        raise

    # Summary
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Raw data rows: {results.get('raw_rows', 0):,}")
    print(f"  Dollar bars: {results.get('dollar_bars_count', 0):,}")
    print(f"  Labeled events: {results.get('labeled_count', 0):,}")
    print(f"  Feature columns: {results.get('feature_count', 0)}")
    print(f"  Status: {results.get('status', 'unknown')}")
    
    if "metrics" in results:
        m = results["metrics"]
        print(f"  DSR: {m.get('dsr', 0):.4f} ({m.get('status', 'FAIL')})")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run complete Polars ML pipeline",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/daily_bars.csv",
        help="Path to input CSV file (default: data/daily_bars.csv)",
    )
    parser.add_argument(
        "--daily-target",
        type=int,
        default=4,
        help="Target dollar bars per day (default: 4)",
    )
    parser.add_argument(
        "--pt-sl",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Profit taking / Stop loss multipliers (default: 1.0 1.0)",
    )
    parser.add_argument(
        "--vertical-barrier",
        type=int,
        default=12,
        help="Vertical barrier in bars (default: 12)",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30, 50],
        help="Rolling window sizes (default: 5 10 20 30 50)",
    )
    parser.add_argument(
        "--ffd-d",
        type=float,
        default=0.5,
        help="FFD differentiation parameter (initial guess) (default: 0.5)",
    )
    parser.add_argument(
        "--no-auto-stationarity",
        action="store_true",
        help="Disable automatic stationarity check (use fixed d)",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=0.9,
        help="Time decay factor for weights (default: 0.9)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of CV splits (default: 5)",
    )
    parser.add_argument(
        "--embargo",
        type=float,
        default=0.1,
        help="Embargo proportion for CV (default: 0.1)",
    )
    parser.add_argument(
        "--dsr-threshold",
        type=float,
        default=0.95,
        help="Threshold for Deflated Sharpe Ratio acceptance (default: 0.95)",
    )

    args = parser.parse_args()

    results = run_pipeline(
        input_path=args.input,
        daily_target=args.daily_target,
        pt_sl=args.pt_sl,
        vertical_barrier_bars=args.vertical_barrier,
        windows=args.windows,
        ffd_d=args.ffd_d,
        check_stationarity=not args.no_auto_stationarity,
        decay=args.decay,
        n_splits=args.n_splits,
        embargo=args.embargo,
        acceptance_threshold=args.dsr_threshold,
    )

    return results


if __name__ == "__main__":
    main()
