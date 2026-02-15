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

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
from enum import Enum

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
)
from afml.stationarity import get_min_d, get_stationarity_search_history
from afml.metrics import get_strategy_metrics
from afml.visualization import AFMLVisualizer, _compute_jb_statistics


# Step execution enum
class PipelineStep(str, Enum):
    """Pipeline step enumeration for modular execution."""

    LOAD = "load"
    BARS = "bars"
    LABELS = "labels"
    FEATURES = "features"
    WEIGHTS = "weights"
    CV = "cv"
    META = "meta"
    BET = "bet"
    VERIFY = "verify"
    ALL = "all"

    @classmethod
    def choices(cls) -> list[str]:
        """Return list of valid step choices."""
        return [step.value for step in cls]


# Default input paths for each step (auto-detection)
STEP_DEFAULT_INPUTS: Dict[str, str] = {
    "load": "data/BTCUSDT/parquet_db",
    "bars": "data/raw_data_polars.parquet",
    "labels": "data/dollar_bars_polars.parquet",
    "features": "data/dollar_bars_polars.parquet",
    "weights": "data/labeled_polars.parquet",
    "cv": "data/features_polars.parquet",
    "meta": "data/features_polars.parquet",
    "bet": "data/predictions_polars.parquet",
    "verify": "data/bet_sizes_polars.parquet",
}


# Configuration
DATA_DIR = Path("data")
OUTPUT_DIR = Path("visual_analysis")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def load_raw_data(filepath: str) -> pl.LazyFrame:
    """
    Load raw tick data lazily from CSV file or Parquet directory.

    Args:
        filepath: Path to CSV/Parquet file or directory containing Parquet files.
            Expected columns: datetime, open, high, low, close, volume
            Or (Tick data): timestamp, price, qty (will be converted)

    Returns:
        Polars LazyFrame
    """
    print(f"\n{'=' * 60}")
    print("Step 1: Loading Raw Data (Lazy)")
    print(f"{'=' * 60}")

    print(f"Scanning data from {filepath}...")
    path_obj = Path(filepath)

    if path_obj.is_dir():
        print(f"  Detected directory, scanning parquet files...")
        # Use scan_parquet with glob pattern
        try:
            df = pl.scan_parquet(str(path_obj / "*.parquet"))
        except Exception:
            # Fallback to recursive scan if needed, or specific pattern
            df = pl.scan_parquet(str(path_obj / "**/*.parquet"))
    elif str(filepath).endswith(".parquet"):
        df = pl.scan_parquet(filepath)
    else:
        df = pl.scan_csv(filepath, try_parse_dates=True)

    # Standardize column names for Tick Data
    # timestamp -> datetime, price -> close, qty -> volume
    # Cache schema to avoid repeated collect_schema() calls
    schema_names = df.collect_schema().names()
    rename_map = {
        "timestamp": "datetime",
        "price": "close",
        "qty": "volume",
        "quantity": "volume",
    }

    cols_map = {}
    for old, new in rename_map.items():
        if old in schema_names and new not in schema_names:
            cols_map[old] = new

    if cols_map:
        print(f"  Renaming columns (Lazy): {cols_map}")
        df = df.rename(cols_map)
        # Update cached schema after rename
        schema_names = df.collect_schema().names()

    # Backfill OHLC from Close if missing
    if "close" in schema_names:
        exprs = []
        if "open" not in schema_names:
            exprs.append(pl.col("close").alias("open"))
        if "high" not in schema_names:
            exprs.append(pl.col("close").alias("high"))
        if "low" not in schema_names:
            exprs.append(pl.col("close").alias("low"))

        if exprs:
            df = df.with_columns(exprs)

    # We skip global sort for massive parquet directories to avoid OOM
    # Crypto tick data is usually saved chronologically
    if not path_obj.is_dir():
        print("  Sorting data by datetime...")
        df = df.sort("datetime")
    else:
        print("  Skipping global sort for directory (assuming chronological order)")

    # We delay row count to avoid full scan
    print(f"  Data definition loaded successfully (LazyFrame)")

    return df


def generate_dollar_bars(
    df: Union[pl.DataFrame, pl.LazyFrame],
    daily_target: int = 4,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Generate dollar bars from raw tick data.

    Args:
        df: Raw tick data (LazyFrame or DataFrame)
        daily_target: Target number of bars per day
        output_path: Optional path to save results

    Returns:
        Dollar bars DataFrame (aggregated and collected)
    """
    print(f"\n{'=' * 60}")
    print("Step 2: Generating Dollar Bars")
    print(f"{'=' * 60}")

    processor = DollarBarsProcessor(
        daily_target=daily_target,
        lazy=True,  # Enable lazy processing
    )

    print(f"  Daily target: {daily_target} bars/day")
    print("  Processing (Lazy -> Collect)...")

    # This will return a LazyFrame if input is LazyFrame because we set lazy=True
    # and we modified Processor to respect laziness
    dollar_bars_lazy = processor.fit_transform_dynamic(df)

    # NOW we collect. The input was huge (7GB), output bars allow much smaller size.
    dollar_bars = dollar_bars_lazy.collect()

    threshold_info = processor.get_threshold_info()
    print(f"  Threshold: {threshold_info['threshold']:,.0f}")
    print(f"  Generated: {len(dollar_bars):,} bars")

    if output_path or DATA_DIR / "dollar_bars_polars.parquet":
        output_path = output_path or str(DATA_DIR / "dollar_bars_polars.parquet")
        dollar_bars.write_parquet(output_path)
        print(f"  Saved to: {output_path}")

    return dollar_bars


def apply_labels(
    df: pl.DataFrame,
    pt_sl: list = None,
    vertical_barrier_bars: int = 12,
    output_path: Optional[str] = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Apply Triple Barrier Labeling (AFML Ch. 3) to detected CUSUM events.

    Args:
        df: Dollar bars DataFrame (OHLCV).
        pt_sl: [PT, SL] multipliers. Barriers are set at Volatility * pt_sl.
        vertical_barrier_bars: Max bars to wait before label is set to return at t1.
        output_path: Optional path to save labeled results.

    Returns:
        Tuple of (events DataFrame, labeled events DataFrame).
    """
    print(f"\n{'=' * 60}")
    print("Step 3: Applying Triple Barrier Labels")
    print(f"{'=' * 60}")

    pt_sl = pt_sl or [1.0, 1.0]

    labeler = TripleBarrierLabeler(
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
            label_dist = labeled.group_by("label").agg(pl.len().alias("count"))
            print("  Label distribution:")
            for row in label_dist.rows():
                label_name = {-1: "Loss", 0: "Neutral", 1: "Profit"}.get(
                    row[0], "Unknown"
                )
                print(f"    {label_name} ({row[0]}): {row[1]:,}")
    else:
        labeled = pl.DataFrame({"t1": [], "tr": [], "label": []})
        print("  No events generated")

    if output_path or DATA_DIR / "labeled_polars.parquet":
        output_path = output_path or str(DATA_DIR / "labeled_polars.parquet")
        labeled.write_parquet(output_path)
        print(f"  Saved to: {output_path}")

    return events, labeled


def generate_features(
    df: pl.DataFrame,
    windows: list = None,
    ffd_d: float = 0.5,
    check_stationarity: bool = True,
    output_path: Optional[str] = None,
) -> tuple[pl.DataFrame, Optional[tuple]]:
    """
    Generate ML features including Alpha158 and FFD (AFML Ch. 5).

    Args:
        df: Dollar bars DataFrame.
        windows: List of window sizes for technical indicators (MA, STD, RSI, etc.).
        ffd_d: Coefficient for Fractional Differentiation (initial guess).
        check_stationarity: Automatically find the optimal d coefficient.
        output_path: Optional path to save feature DataFrame.

    Returns:
        Tuple of (DataFrame with features, stationarity history plot data).
    """
    print(f"\n{'=' * 60}")
    print("Step 4: Generating Features (Alpha158 + FFD)")
    print(f"{'=' * 60}")

    windows = windows or [5, 10, 20, 30, 50]
    optimal_d = ffd_d
    stationarity_history = None

    if check_stationarity:
        print("  Checking stationarity for optimal d...")
        # Use get_stationarity_search_history which computes ALL d values
        # This avoids the double-computation (get_min_d + get_stationarity_search_history)
        close_prices = df["close"].to_numpy()
        d_vals, p_vals = get_stationarity_search_history(
            close_prices, max_d=1.0, step_size=0.05
        )
        stationarity_history = (d_vals, p_vals)

        # Find min d from the history
        stationary_mask = p_vals < 0.05
        if stationary_mask.any():
            first_stat_idx = stationary_mask.argmax()
            optimal_d = round(float(d_vals[first_stat_idx]), 2)
            p_val = float(p_vals[first_stat_idx])
        else:
            optimal_d = 1.0
            p_val = float(p_vals[-1])

        print(f"  Optimal d found: {optimal_d} (p-value: {p_val:.4f})")
        if p_val > 0.05:
            print("  WARNING: Could not achieve stationarity with d < 1.0")
            print("  Using d=1.0 for integer differentiation")
    else:
        print(f"  Using fixed d: {ffd_d}")

    engineer = FeatureEngineer(
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

    if output_path or DATA_DIR / "features_polars.parquet":
        output_path = output_path or str(DATA_DIR / "features_polars.parquet")
        features.write_parquet(output_path)
        print(f"  Saved to: {output_path}")

    return features, stationarity_history


def calculate_weights(
    events: pl.DataFrame,
    decay: float = 0.9,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Calculate sample weights based on uniqueness and time decay (AFML Ch. 4).

    Args:
        events: Labeled events (output of apply_labels).
        decay: Exponential decay for older samples (1.0 = equal weights).
        output_path: Optional path to save weights.

    Returns:
        DataFrame containing uniqueness scores and final weights.
    """
    print(f"\n{'=' * 60}")
    print("Step 5: Calculating Sample Weights")
    print(f"{'=' * 60}")

    if len(events) == 0:
        print("  No events, returning empty weights")
        return pl.DataFrame({"weight": [], "uniqueness": []})

    calculator = SampleWeightCalculator(decay=decay)

    print(f"  Decay factor: {decay}")

    weights = calculator.fit_transform(events)

    print(f"  Events: {len(events):,}")
    print(f"  Mean weight: {weights['weight'].mean():.4f}")
    print(f"  Std weight: {weights['weight'].std():.4f}")

    if output_path or DATA_DIR / "sample_weights_polars.parquet":
        output_path = output_path or str(DATA_DIR / "sample_weights_polars.parquet")
        weights.write_parquet(output_path)
        print(f"  Saved to: {output_path}")

    return weights


def run_cross_validation(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    n_splits: int = 5,
    embargo: float = 0.1,
) -> list:
    """
    Run Purged K-Fold Cross-Validation (AFML Ch. 7).

    Args:
        features: Feature DataFrame.
        labels: Labeled events DataFrame.
        n_splits: Number of CV folds.
        embargo: Data embargo fraction to prevent information leakage.

    Returns:
        List of splits (train_idx, test_idx).
    """
    print(f"\n{'=' * 60}")
    print("Step 6: Cross-Validation (Purged K-Fold)")
    print(f"{'=' * 60}")

    cv = PurgedKFoldCV(
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

    return splits


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
    pipeline = MetaLabelingPipeline(
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
        # Get combined probabilities [primary_proba, meta_proba]
        probabilities = pipeline.predict_proba(X_np)

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
        probabilities = np.zeros((n_samples, 2))
        metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}

    predictions_df = pl.DataFrame(
        {
            "prediction": predictions.astype(int),
            "actual": y_np.astype(int),
            "primary_proba": probabilities[:, 0],
            "meta_proba": probabilities[:, 1],
        }
    )

    if output_path or DATA_DIR / "predictions_polars.parquet":
        output_path = output_path or str(DATA_DIR / "predictions_polars.parquet")
        predictions_df.write_parquet(output_path)
        print(f"\n  Saved predictions to: {output_path}")

    return predictions_df


def calculate_bet_sizes(
    features: pl.DataFrame,
    predictions: pl.DataFrame,
    labeled: pl.DataFrame,
    threshold: float = 0.5,
    output_path: Optional[str] = None,
) -> pl.DataFrame:
    """
    Calculate bet sizes from predictions.

    Args:
        features: Feature DataFrame
        predictions: Predictions DataFrame (with primary_proba and meta_proba)
        labeled: Labeled data (with actual returns 'tr')
        threshold: Probability threshold
        output_path: Optional path to save results

    Returns:
        DataFrame with bet sizes
    """
    print(f"\n{'=' * 60}")
    print("Step 8: Bet Sizing")
    print(f"{'=' * 60}")

    if len(predictions) == 0 or len(labeled) == 0:
        print("  No predictions or labels, skipping bet sizing")
        return pl.DataFrame()

    sizer = BetSizer(
        threshold=threshold,
        quantity=100,
    )

    print(f"  Threshold: {threshold}")
    print("  Quantity: 100")

    # Align data - use head(n_samples) to ensure matching lengths
    n_samples = min(len(predictions), len(labeled))

    # We use meta_proba for 'confidence' and 'tr' for the empirical training
    # meta_proba is probability that primary model is correct (1)
    # primary_proba determines the direction if it's > 0.5 (optional, pipeline uses predictions)

    bet_sizes = sizer.fit_transform(
        pl.Series(predictions["meta_proba"].head(n_samples)),
        pl.Series(labeled["tr"].head(n_samples)),
    )

    # Apply direction to bet size
    # predictions["prediction"] already contains (direction * meta_filter)
    # where direction is 1 (Long) or -1 (Short) and meta_filter is 0 or 1.
    direction = predictions["prediction"].head(n_samples).to_numpy()
    final_bet_sizes = bet_sizes.to_numpy() * direction

    print(f"  Mean bet size: {np.mean(final_bet_sizes):.4f}")
    print(f"  Std bet size: {np.std(final_bet_sizes):.4f}")

    # Get metrics
    # Note: sizer.get_metrics uses strategy_returns = returns * bet_sizes
    # Here final_bet_sizes already includes direction.
    metrics = sizer.get_metrics(
        pl.Series(labeled["tr"].head(n_samples)),
        pl.Series(final_bet_sizes),
    )

    print("\n  Bet Sizing Metrics:")
    print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"    Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"    Max Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"    Total Return: {metrics['total_return']:.4f}")

    bet_sizes_df = pl.DataFrame(
        {
            "bet_size": final_bet_sizes,
            "prediction": predictions["prediction"].head(n_samples),
            "actual_label": predictions["actual"].head(n_samples),
            "actual_return": labeled["tr"].head(n_samples),
        }
    )

    if output_path or DATA_DIR / "bet_sizes_polars.parquet":
        output_path = output_path or str(DATA_DIR / "bet_sizes_polars.parquet")
        bet_sizes_df.write_parquet(output_path)
        print(f"\n  Saved bet sizes to: {output_path}")

    return bet_sizes_df


def verify_strategy(
    labeled: pl.DataFrame,
    bet_sizes: pl.DataFrame,
    n_trials: int = 1,
    acceptance_threshold: float = 0.95,
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
        print(
            "  WARNING: 'tr' column not found in labels, using simplified verification"
        )
        returns_s = labeled["label"].head(n_samples).to_numpy()  # Proxy

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

    dsr_val = metrics.get("dsr", 0)
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
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> dict:
    """
    Run complete Polars ML pipeline following AFML standards.

    Args:
        input_path: Path to raw data (CSV or Parquet).
        daily_target: Target number of Dollar Bars per day (AFML Ch. 2).
        pt_sl: [Profit-Taking, Stop-Loss] multipliers (AFML Ch. 3).
        vertical_barrier_bars: Max holding period in bars (AFML Ch. 3).
        windows: Window sizes for feature rolling calculations.
        ffd_d: Initial guess for Fractional Differentiation (AFML Ch. 5).
        check_stationarity: If True, search for minimum d to make series stationary.
        decay: Time decay factor for sample weights (AFML Ch. 4).
        n_splits: Folds for Purged K-Fold Cross-Validation (AFML Ch. 7).
        embargo: Data embargo fraction to prevent leakage (AFML Ch. 7).
        acceptance_threshold: Minimum DSR to consider strategy 'passed'.
        visualize: If True, generate diagnostic plots for each step.
        visual_analysis_dir: Directory to save generated plots.

    Returns:
        Dict: Final metrics including DSR, Sharpes, and acceptance status.
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
            "visualize": visualize,
        },
    }

    viz = None
    if visualize:
        print("\n" + "-" * 60)
        print(f"Initializing Visualization (Dir: {visual_analysis_dir})")
        print("-" * 60)
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)

    try:
        # Step 1: Load data
        df = load_raw_data(input_path)

        # Optimized row count: only if needed or use metadata if possible
        # For now, we still count but the logic around it is more careful
        if not visualize and not check_stationarity:
            print("  Skipping full row count for speed...")
            row_count = 1_000_000_000  # Placeholder
        else:
            print("  Counting rows (lazy)...")
            row_count = df.select(pl.len()).collect().item()
            print(f"  Rows: {row_count:,}")

        results["raw_rows"] = row_count

        # Step 2: Dollar bars
        dollar_bars = generate_dollar_bars(df, daily_target=daily_target)
        results["dollar_bars_count"] = len(dollar_bars)

        # Compute JB statistics for dollar bars
        dollar_close = dollar_bars["close"].to_numpy()
        dollar_returns = np.log(dollar_close[1:] / dollar_close[:-1])
        dollar_jb = _compute_jb_statistics(dollar_returns)

        # Compute JB statistics for time bars (raw data) if available
        # Collecting raw data into memory might be too heavy.
        # We can take a sample (first 1M rows) for JB stat comparison if dataset is huge
        time_jb = None
        if row_count > 100:  # Only compute if we have enough data
            print("  Computing Time Bar statistics (on sample)...")
            # Take a sample for statistic
            time_close = (
                df.select("close").limit(1_000_000).collect()["close"].to_numpy()
            )
            time_returns = np.log(time_close[1:] / time_close[:-1])
            time_jb = _compute_jb_statistics(time_returns)

        # Print JB statistics to console
        print("\n  Jarque-Bera Statistics:")
        print(
            f"    Dollar Bars - JB: {dollar_jb['jb_stat']:.2f}, p-value: {dollar_jb['p_value']:.4f}, "
            f"Skew: {dollar_jb['skewness']:.4f}, Kurtosis: {dollar_jb['kurtosis']:.4f}"
        )
        if time_jb:
            print(
                f"    Time Bars   - JB: {time_jb['jb_stat']:.2f}, p-value: {time_jb['p_value']:.4f}, "
                f"Skew: {time_jb['skewness']:.4f}, Kurtosis: {time_jb['kurtosis']:.4f}"
            )
            if dollar_jb["jb_stat"] < time_jb["jb_stat"]:
                print(
                    f"    ✓ Dollar Bars reduce JB statistic: {dollar_jb['jb_stat']:.2f} < {time_jb['jb_stat']:.2f}"
                )
            else:
                print("    [FAIL] Dollar Bars do not improve normality")

        if viz:
            print("  [Viz] Plotting Dollar Bar statistics...")
            viz.plot_bar_stats(dollar_bars, filename="step2_dollar_bars.png")

        # Step 3: Labels
        events, labeled = apply_labels(
            dollar_bars,
            pt_sl=pt_sl,
            vertical_barrier_bars=vertical_barrier_bars,
        )
        results["labeled_count"] = len(labeled)

        if viz:
            print("  [Viz] Plotting Triple Barrier labels...")
            # We pass dollar_bars as price_df
            viz.plot_triple_barrier_sample(
                dollar_bars, events, labeled, filename="step3_triple_barrier_sample.png"
            )
            viz.plot_label_distribution(
                labeled, filename="step3_label_distribution.png"
            )

        # Step 4: Features (with Stationarity)
        features, stationarity_history = generate_features(
            dollar_bars,
            windows=windows,
            ffd_d=ffd_d,
            check_stationarity=check_stationarity,
        )
        results["feature_count"] = len(features.columns)

        if viz:
            print("  [Viz] Plotting Features...")
            # Reuse stationarity_history from generate_features (no double computation)
            if check_stationarity and stationarity_history is not None:
                d_vals, p_vals = stationarity_history
                viz.plot_stationarity_search(
                    d_vals, p_vals, filename="step4_stationarity_search.png"
                )

            viz.plot_feature_heatmap(
                features, filename="step4_feature_correlations.png"
            )

        # Step 5: Weights
        weights = calculate_weights(events, decay=decay)
        results["weights_count"] = len(weights)

        # Step 6: Cross-validation
        splits = run_cross_validation(
            features, labeled, n_splits=n_splits, embargo=embargo
        )

        if viz:
            print("  [Viz] Plotting CV Timeline...")
            viz.plot_cv_timeline(
                splits, n_samples=len(features), filename="step6_cv_timeline.png"
            )

        # Step 7: Meta-labeling
        predictions = run_meta_labeling(features, labeled, weights)

        # Step 8: Bet sizing
        bet_sizes = calculate_bet_sizes(features, predictions, labeled)
        results["predictions_count"] = len(predictions)

        if viz and not predictions.is_empty():
            print("  [Viz] Plotting Meta-Model Performance...")
            # Use actual/predicted from predictions df
            # prediction col is predicted label (0/1), actual is actual label (0/1)
            # But for ROC/PR we ideally want probabilities. The pipeline currently outputs hard text labels
            # or maybe 0/1 integers.
            # check run_meta_labeling: "prediction": predictions.astype(int)
            # It uses `pipeline.predict(X_np)`. To get probabilities we need `predict_proba`.
            # For now we plot with binary predictions.
            # Convert actual to binary for meta-label evaluation (1 if profit, 0 otherwise)
            y_true_binary = (predictions["actual"].to_numpy() > 0).astype(int)

            viz.plot_meta_performance(
                y_true_binary,
                predictions["prediction"].to_numpy(),
                filename="step7_meta_performance.png",
            )

        # Step 9: Verification
        # Assume 1 trial for this run, but in real research this should increase
        metrics = verify_strategy(
            labeled, bet_sizes, n_trials=1, acceptance_threshold=acceptance_threshold
        )
        results["metrics"] = metrics
        results["status"] = metrics.get("status", "UNKNOWN")

        if viz:
            # Calculate equity curve for visualization
            # Similar logic to verify_strategy
            n_samples = min(len(bet_sizes), len(labeled))
            if n_samples > 0:
                bet_s = bet_sizes["bet_size"].head(n_samples).to_numpy()
                returns_s = (
                    labeled["tr"].head(n_samples).to_numpy()
                    if "tr" in labeled.columns
                    else labeled["label"].head(n_samples).to_numpy()
                )
                strategy_returns = bet_s * returns_s

                print("  [Viz] Plotting Equity Curve...")
                viz.plot_equity_curve(
                    strategy_returns, filename="step9_equity_curve.png"
                )

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


def run_jb_parameter_sweep(
    data_dir: str = "data/BTCUSDT/parquet_db",
    daily_targets: list = None,
    output_dir: str = "visual_analysis",
    chunk_size: int = 50_000_000,
) -> dict:
    """
    Search for the optimal sampling frequency (daily_target) using Jarque-Bera
    normality test with rich comparative visualization (AFML Ch. 2).

    For large datasets, uses chunked processing to avoid OOM. The sweep produces
    a 6-panel diagnostic plot comparing return distributions, QQ plots, daily
    bar count stability, and JB statistics across different daily_target values.

    Args:
        data_dir: Source directory/file for raw ticks. Also checks
            data/raw_data_polars.parquet as a fallback.
        daily_targets: List of sampling frequencies to test.
        output_dir: Directory to save the sweep diagnostic plots.
        chunk_size: Chunk size for transform_chunked (default: 50M rows).

    Returns:
        Dict with 'results' list and 'best' parameter entry.
    """
    if daily_targets is None:
        daily_targets = [4, 10, 20, 50, 80, 100]

    print("\n" + "=" * 60)
    print("DOLLAR BARS PARAMETER SWEEP (JB Normality)")
    print("=" * 60)

    # --- Step 1: Resolve data source ---
    path_obj = Path(data_dir)
    raw_parquet = Path("data/raw_data_polars.parquet")
    use_chunked = False

    if raw_parquet.exists():
        # Prefer the pre-processed parquet (already sorted, standardized)
        data_source = str(raw_parquet)
        print(f"  Data source: {data_source} (pre-processed)")
        total_rows = pl.scan_parquet(data_source).select(pl.len()).collect().item()
        use_chunked = total_rows > 100_000_000
    elif path_obj.is_dir():
        data_source = data_dir
        print(f"  Data source: {data_source} (raw directory)")
        scan_path = str(path_obj / "*.parquet")
        total_rows = pl.scan_parquet(scan_path).select(pl.len()).collect().item()
        use_chunked = total_rows > 100_000_000
    elif path_obj.exists():
        data_source = data_dir
        print(f"  Data source: {data_source}")
        total_rows = pl.scan_parquet(data_source).select(pl.len()).collect().item()
        use_chunked = total_rows > 100_000_000
    else:
        raise FileNotFoundError(
            f"Data source not found: {data_dir}. "
            "Run --step load first, or specify a valid --jb-data-dir."
        )

    print(f"  Total rows: {total_rows:,}")
    print(f"  Processing mode: {'chunked' if use_chunked else 'in-memory'}")
    print(f"  Testing daily_targets: {daily_targets}")

    # --- Step 2: For each daily_target, generate dollar bars and compute JB ---
    results = []
    all_returns = {}  # Store returns for visualization

    for dt in daily_targets:
        print(f"\n  {'─' * 50}")
        print(f"  Testing daily_target = {dt}")
        print(f"  {'─' * 50}")

        processor = DollarBarsProcessor(daily_target=dt, lazy=True)

        if use_chunked:
            # Fit on the data source
            if Path(data_source).is_dir():
                fit_df = pl.scan_parquet(str(Path(data_source) / "*.parquet"))
            else:
                fit_df = pl.scan_parquet(data_source)

            # Standardize columns for fitting
            schema = fit_df.collect_schema().names()
            rename_cols = {}
            rmap = {"timestamp": "datetime", "price": "close", "qty": "volume"}
            for old, new in rmap.items():
                if old in schema and new not in schema:
                    rename_cols[old] = new
            if rename_cols:
                fit_df = fit_df.rename(rename_cols)
            schema = fit_df.collect_schema().names()
            backfill = []
            if "close" in schema:
                if "open" not in schema:
                    backfill.append(pl.col("close").alias("open"))
                if "high" not in schema:
                    backfill.append(pl.col("close").alias("high"))
                if "low" not in schema:
                    backfill.append(pl.col("close").alias("low"))
            if backfill:
                fit_df = fit_df.with_columns(backfill)

            print(f"    Fitting threshold (streaming, dynamic)...")
            processor.fit_dynamic(fit_df)
            print(f"    Threshold (initial): {processor.threshold_:,.0f}")

            print(f"    Generating dollar bars (chunked)...")
            dollar_bars = processor.transform_chunked(data_source, chunk_size=chunk_size)
        else:
            # In-memory processing (small datasets)
            if Path(data_source).is_dir():
                df = load_raw_data(data_source)
                dollar_bars = generate_dollar_bars(df, daily_target=dt)
            elif data_source.endswith(".parquet"):
                df = pl.read_parquet(data_source)
                dollar_bars = generate_dollar_bars(df, daily_target=dt)
            else:
                df = pl.read_csv(data_source, try_parse_dates=True)
                dollar_bars = generate_dollar_bars(df, daily_target=dt)

        # Compute log returns and JB statistics
        close_arr = dollar_bars["close"].to_numpy()
        returns = np.log(close_arr[1:] / close_arr[:-1])
        # Remove NaN/Inf
        returns = returns[np.isfinite(returns)]
        jb = _compute_jb_statistics(returns)

        # Daily bar count distribution
        daily_counts = (
            dollar_bars.with_columns(pl.col("datetime").dt.date().alias("date"))
            .group_by("date")
            .agg(pl.len().alias("count"))["count"]
            .to_numpy()
        )

        result = {
            "daily_target": dt,
            "bars_count": len(dollar_bars),
            "jb_stat": jb["jb_stat"],
            "p_value": jb["p_value"],
            "skewness": jb["skewness"],
            "kurtosis": jb["kurtosis"],
            "is_normal": jb["is_normal"],
            "daily_bars_mean": float(np.mean(daily_counts)),
            "daily_bars_std": float(np.std(daily_counts)),
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
        }
        results.append(result)
        all_returns[dt] = returns

        print(
            f"    ✓ Bars: {result['bars_count']:,}, "
            f"JB: {result['jb_stat']:.2f}, "
            f"p: {result['p_value']:.4f}, "
            f"Skew: {result['skewness']:.4f}, "
            f"Kurt: {result['kurtosis']:.4f}, "
            f"Normal: {'✓' if result['is_normal'] else '✗'}"
        )

    # --- Step 3: Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    header = (
        f"{'target':>8} | {'bars':>8} | {'JB stat':>10} | {'p-value':>8} | "
        f"{'skew':>8} | {'kurt':>8} | {'bars/day':>10} | {'normal':>6}"
    )
    print(header)
    print("─" * len(header))
    for r in results:
        marker = " ◀" if r["p_value"] == max(x["p_value"] for x in results) else ""
        print(
            f"{r['daily_target']:>8} | {r['bars_count']:>8,} | {r['jb_stat']:>10.2f} | "
            f"{r['p_value']:>8.4f} | {r['skewness']:>8.4f} | {r['kurtosis']:>8.4f} | "
            f"{r['daily_bars_mean']:>7.1f}±{r['daily_bars_std']:<4.1f} | "
            f"{'✓' if r['is_normal'] else '✗':>4}{marker}"
        )

    best = max(results, key=lambda x: x["p_value"])
    print(f"\n✓ Best parameter: daily_target={best['daily_target']} "
          f"(p-value={best['p_value']:.4f}, {best['bars_count']:,} bars)")

    # --- Step 4: Rich Visualization ---
    Path(output_dir).mkdir(exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy import stats as sp_stats
        import seaborn as sns

        plt.style.use("seaborn-v0_8-darkgrid")
        n_params = len(daily_targets)
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_params))

        fig = plt.figure(figsize=(20, 22))
        gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

        # ── Panel 1: JB p-value vs daily_target ──
        ax1 = fig.add_subplot(gs[0, 0])
        dt_vals = [r["daily_target"] for r in results]
        p_vals = [r["p_value"] for r in results]

        ax1.plot(dt_vals, p_vals, "o-", linewidth=2.5, markersize=10,
                 color="#2196F3", markerfacecolor="white", markeredgewidth=2)
        ax1.axhline(y=0.05, color="#E91E63", linestyle="--", linewidth=1.5,
                    label="p=0.05 (normality threshold)")
        # Highlight best
        ax1.scatter([best["daily_target"]], [best["p_value"]],
                    s=200, color="#4CAF50", zorder=5, marker="*",
                    label=f"Best: dt={best['daily_target']}")
        ax1.set_xlabel("daily_target (bars/day)", fontsize=12)
        ax1.set_ylabel("JB p-value", fontsize=12)
        ax1.set_title("Jarque-Bera Normality Test", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(dt_vals)

        # ── Panel 2: Skewness & Kurtosis ──
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(n_params)
        width = 0.35
        skews = [r["skewness"] for r in results]
        kurts = [r["kurtosis"] for r in results]

        bars1 = ax2.bar(x - width / 2, skews, width, label="Skewness",
                        color="#FF9800", alpha=0.85, edgecolor="white")
        bars2 = ax2.bar(x + width / 2, kurts, width, label="Kurtosis",
                        color="#9C27B0", alpha=0.85, edgecolor="white")
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
        ax2.axhline(y=3.0, color="#E91E63", linestyle="--", linewidth=1.5,
                    label="Normal kurtosis = 3")
        ax2.set_xticks(x)
        ax2.set_xticklabels(dt_vals)
        ax2.set_xlabel("daily_target", fontsize=12)
        ax2.set_ylabel("Value", fontsize=12)
        ax2.set_title("Skewness & Kurtosis Comparison", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # ── Panel 3: Return Distribution Overlay ──
        ax3 = fig.add_subplot(gs[1, 0])
        for i, dt in enumerate(daily_targets):
            ret = all_returns[dt]
            # Normalize for comparison
            ax3.hist(ret, bins=80, alpha=0.4, color=colors[i], density=True,
                     label=f"dt={dt} (n={len(ret):,})")
            # KDE
            try:
                kde_x = np.linspace(ret.min(), ret.max(), 300)
                kde = sp_stats.gaussian_kde(ret)
                ax3.plot(kde_x, kde(kde_x), color=colors[i], linewidth=1.5)
            except Exception:
                pass

        # Normal reference
        combined = np.concatenate(list(all_returns.values()))
        x_norm = np.linspace(combined.min(), combined.max(), 300)
        ax3.plot(x_norm, sp_stats.norm.pdf(x_norm, 0, np.std(combined)),
                 "k--", linewidth=2, alpha=0.5, label="Normal (ref)")
        ax3.set_xlabel("Log Return", fontsize=12)
        ax3.set_ylabel("Density", fontsize=12)
        ax3.set_title("Return Distributions Overlay", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=9, loc="upper right")
        ax3.grid(True, alpha=0.3)

        # ── Panel 4: QQ Plots Grid ──
        ax4 = fig.add_subplot(gs[1, 1])
        # Create sub-grid for QQ plots
        n_qq = min(n_params, 6)
        qq_rows = (n_qq + 2) // 3
        qq_gs = gridspec.GridSpecFromSubplotSpec(qq_rows, 3, subplot_spec=gs[1, 1],
                                                  hspace=0.4, wspace=0.3)
        ax4.set_visible(False)  # Hide parent axes
        for i in range(n_qq):
            ax_qq = fig.add_subplot(qq_gs[i // 3, i % 3])
            dt = daily_targets[i]
            ret = all_returns[dt]
            sp_stats.probplot(ret, dist="norm", plot=ax_qq)
            ax_qq.set_title(f"dt={dt}", fontsize=10, fontweight="bold")
            ax_qq.get_lines()[0].set(markersize=2, color=colors[i], alpha=0.5)
            ax_qq.get_lines()[1].set(color="#E91E63", linewidth=1.5)
            ax_qq.set_xlabel("")
            ax_qq.set_ylabel("")

        # ── Panel 5: Daily Bar Count Distribution ──
        ax5 = fig.add_subplot(gs[2, 0])
        daily_data = []
        daily_labels = []
        for r in results:
            daily_labels.append(f"dt={r['daily_target']}")
            daily_data.append(r["daily_bars_mean"])

        bar_colors = ["#4CAF50" if r["daily_target"] == best["daily_target"]
                      else "#2196F3" for r in results]
        bars_plot = ax5.bar(range(n_params), daily_data, color=bar_colors,
                           alpha=0.85, edgecolor="white")
        # Add error bars for std
        stds = [r["daily_bars_std"] for r in results]
        ax5.errorbar(range(n_params), daily_data, yerr=stds,
                     fmt="none", ecolor="gray", capsize=5)
        # Add target reference lines
        for i, r in enumerate(results):
            ax5.axhline(y=r["daily_target"], color=colors[i], linestyle=":",
                        alpha=0.3, linewidth=1)
        ax5.set_xticks(range(n_params))
        ax5.set_xticklabels(daily_labels, fontsize=10)
        ax5.set_xlabel("Configuration", fontsize=12)
        ax5.set_ylabel("Mean Bars per Day", fontsize=12)
        ax5.set_title("Actual vs Target Bars per Day", fontsize=14, fontweight="bold")
        ax5.grid(True, alpha=0.3, axis="y")

        # ── Panel 6: Summary Table ──
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis("off")

        table_data = []
        for r in results:
            table_data.append([
                str(r["daily_target"]),
                f"{r['bars_count']:,}",
                f"{r['jb_stat']:.1f}",
                f"{r['p_value']:.4f}",
                f"{r['skewness']:.3f}",
                f"{r['kurtosis']:.3f}",
                f"{r['daily_bars_mean']:.1f}",
                "YES" if r["is_normal"] else "NO",
            ])

        col_labels = ["Target", "Bars", "JB", "p-val", "Skew", "Kurt",
                       "Bars/Day", "Normal"]
        table = ax6.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.auto_set_column_width(range(len(col_labels)))
        table.scale(1.0, 1.6)

        # Style header
        for j, label in enumerate(col_labels):
            cell = table[0, j]
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")

        # Highlight best row
        best_idx = next(i for i, r in enumerate(results)
                        if r["daily_target"] == best["daily_target"])
        for j in range(len(col_labels)):
            cell = table[best_idx + 1, j]
            cell.set_facecolor("#E8F5E9")
            cell.set_edgecolor("#4CAF50")
            cell.set_linewidth(2)

        ax6.set_title("Parameter Comparison Summary", fontsize=14,
                      fontweight="bold", pad=20)

        # ── Overall title ──
        fig.suptitle(
            f"Dollar Bars Parameter Sweep — {total_rows:,} ticks\n"
            f"Best: daily_target={best['daily_target']} "
            f"(p={best['p_value']:.4f}, {best['bars_count']:,} bars)",
            fontsize=16, fontweight="bold", y=0.98
        )

        output_path = f"{output_dir}/dollar_bars_parameter_sweep.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"\n✓ Visualization saved to {output_path}")
        plt.close()

        # Also save individual return distributions for detail
        fig2, axes2 = plt.subplots(1, n_params, figsize=(5 * n_params, 4),
                                    sharey=True, squeeze=False)
        for i, dt in enumerate(daily_targets):
            ax = axes2[0, i]
            ret = all_returns[dt]
            ax.hist(ret, bins=60, color=colors[i], alpha=0.7, density=True,
                    edgecolor="white")
            # Overlay normal
            mu, sigma = np.mean(ret), np.std(ret)
            x_norm = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            ax.plot(x_norm, sp_stats.norm.pdf(x_norm, mu, sigma),
                    "r--", linewidth=2, alpha=0.7)
            r_info = results[i]
            ax.set_title(f"dt={dt}\nJB={r_info['jb_stat']:.1f} "
                        f"p={r_info['p_value']:.4f}", fontsize=11)
            ax.set_xlabel("Log Return")
            if i == 0:
                ax.set_ylabel("Density")

        fig2.suptitle("Return Distributions with Normal Fit", fontsize=14,
                      fontweight="bold")
        plt.tight_layout()
        detail_path = f"{output_dir}/dollar_bars_sweep_distributions.png"
        plt.savefig(detail_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"✓ Detail distributions saved to {detail_path}")
        plt.close()

    except ImportError as e:
        print(f"\n[Note] Visualization skipped: {e}")

    return {"results": results, "best": best}


def get_step_input_path(step: str, user_input: Optional[str]) -> str:
    """
    Determine input path for a given step.

    Args:
        step: Pipeline step name
        user_input: User-provided input path (or None)

    Returns:
        Resolved input path for the step
    """
    if user_input and user_input != "data/BTCUSDT/parquet_db":
        return user_input

    if step in STEP_DEFAULT_INPUTS:
        default_path = STEP_DEFAULT_INPUTS[step]
        if Path(default_path).exists():
            print(f"  Auto-detected input: {default_path}")
            return default_path
        else:
            print(f"  Warning: Default input {default_path} not found")
            return user_input or "data/BTCUSDT/parquet_db"

    return user_input or "data/BTCUSDT/parquet_db"


def run_step_load(
    input_path: str,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute step: load raw data."""
    print(f"\n{'=' * 60}")
    print("STEP: Loading Raw Data")
    print(f"{'=' * 60}")

    df = load_raw_data(input_path)

    # Use sink_parquet for massive datasets to avoid OOM
    output_path = output_path or "data/raw_data_polars.parquet"
    try:
        df.sink_parquet(output_path)
        print(f"  Saved to: {output_path} (streaming sink)")
        # Get row count from the written file metadata (cheap)
        row_count = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    except Exception:
        # Fallback to collect if sink_parquet not supported for this source
        df_collected = df.collect()
        row_count = len(df_collected)
        df_collected.write_parquet(output_path)
        print(f"  Saved to: {output_path} (collected)")
        del df_collected  # Allow GC immediately

    print(f"  Loaded {row_count:,} rows")

    result = {
        "step": "load",
        "row_count": row_count,
        # Don't store massive DataFrame in result dict - allows GC
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        print("  [Viz] Initialized visualizer")

    return result


def run_step_bars(
    input_path: str,
    daily_target: int = 4,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
    streaming: bool = False,
) -> Dict[str, Any]:
    """Execute step: generate dollar bars.

    For large datasets (>100M rows), this function always uses lazy scanning
    and streaming collection to avoid OOM. The `streaming` flag is kept for
    backwards compatibility but parquet files are now always scanned lazily.
    """
    print(f"\n{'=' * 60}")
    print("STEP: Generating Dollar Bars")
    print(f"{'=' * 60}")

    path_obj = Path(input_path)
    is_lazy = False

    # ALWAYS use lazy scanning for parquet to avoid OOM on large datasets
    if path_obj.is_dir():
        print("  Scanning parquet directory (lazy)...")
        try:
            df = pl.scan_parquet(str(path_obj / "*.parquet"))
        except Exception:
            df = pl.scan_parquet(str(path_obj / "**/*.parquet"))
        is_lazy = True
    elif str(input_path).endswith(".parquet"):
        print("  Scanning parquet file (lazy)...")
        df = pl.scan_parquet(input_path)
        is_lazy = True
    else:
        df = pl.read_csv(input_path, try_parse_dates=True)

    # Column renaming (works on both LazyFrame and DataFrame)
    rename_map = {
        "timestamp": "datetime",
        "price": "close",
        "qty": "volume",
        "quantity": "volume",
        "amount": "volume",
    }
    current_cols = df.collect_schema().names() if hasattr(df, 'collect_schema') else df.columns
    rename_cols = {
        k: v
        for k, v in rename_map.items()
        if k in current_cols and v not in current_cols
    }
    if rename_cols:
        df = df.rename(rename_cols)
        current_cols = df.collect_schema().names() if hasattr(df, 'collect_schema') else df.columns

    # Backfill OHLC from close if missing
    if "close" in current_cols:
        exprs = []
        if "open" not in current_cols:
            exprs.append(pl.col("close").alias("open"))
        if "high" not in current_cols:
            exprs.append(pl.col("close").alias("high"))
        if "low" not in current_cols:
            exprs.append(pl.col("close").alias("low"))
        if exprs:
            df = df.with_columns(exprs)

    # Skip global sort for lazy data — load step already wrote sorted parquet.
    # Sorting 1.5B rows eagerly would OOM. For CSV files (small), we still sort.
    if not is_lazy:
        df = df.sort("datetime")
    else:
        print("  Skipping global sort (assuming pre-sorted from load step)")

    # --- Use DollarBarsProcessor directly with lazy=True for memory safety ---
    print(f"\n{'=' * 60}")
    print("Step 2: Generating Dollar Bars")
    print(f"{'=' * 60}")

    processor = DollarBarsProcessor(
        daily_target=daily_target,
        lazy=True,
    )
    print(f"  Daily target: {daily_target} bars/day")
    print("  Fitting threshold (streaming)...")

    # fit() internally uses streaming=True for daily aggregation
    processor.fit(df)

    threshold_info_obj = processor.get_threshold_info()
    print(f"  Threshold: {threshold_info_obj['threshold']:,.0f}")

    if is_lazy:
        # For massive datasets: use chunked processing to avoid OOM
        # cum_sum() over 1.5B rows is not streamable — must process in chunks
        print("  Using chunked transform (memory-safe for large datasets)...")
        dollar_bars = processor.transform_chunked(input_path, chunk_size=50_000_000)
    else:
        # For small datasets (CSV): standard lazy transform
        print("  Transforming to dollar bars...")
        dollar_bars_lazy = processor.transform(df)
        if isinstance(dollar_bars_lazy, pl.LazyFrame):
            dollar_bars = dollar_bars_lazy.collect()
        else:
            dollar_bars = dollar_bars_lazy

    print(f"  Generated: {len(dollar_bars):,} bars")

    # Save
    output_path = str(DATA_DIR / "dollar_bars_polars.parquet")
    dollar_bars.write_parquet(output_path)
    print(f"  Saved to: {output_path}")

    result = {
        "step": "bars",
        "dollar_bars": dollar_bars,
        "bars_count": len(dollar_bars),
        "threshold": threshold_info_obj,
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        print("  [Viz] Plotting bar statistics...")
        viz.plot_bar_stats(dollar_bars, filename="step2_dollar_bars.png")

    return result


def run_step_labels(
    input_path: str,
    pt_sl: list = None,
    vertical_barrier_bars: int = 12,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: apply triple barrier labels."""
    print(f"\n{'=' * 60}")
    print("STEP: Applying Triple Barrier Labels")
    print(f"{'=' * 60}")

    dollar_bars = pl.read_parquet(input_path)
    print(f"  Loaded {len(dollar_bars):,} bars")

    events, labeled = apply_labels(
        dollar_bars,
        pt_sl=pt_sl,
        vertical_barrier_bars=vertical_barrier_bars,
    )

    result = {
        "step": "labels",
        "events": events,
        "labeled": labeled,
        "labeled_count": len(labeled),
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        print("  [Viz] Plotting triple barrier sample...")
        viz.plot_triple_barrier_sample(
            dollar_bars, events, labeled, filename="step3_triple_barrier_sample.png"
        )
        print("  [Viz] Plotting label distribution...")
        viz.plot_label_distribution(labeled, filename="step3_label_distribution.png")

    return result


def run_step_features(
    input_path: str,
    windows: list = None,
    ffd_d: float = 0.5,
    check_stationarity: bool = True,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: generate features."""
    print(f"\n{'=' * 60}")
    print("STEP: Generating Features (Alpha158 + FFD)")
    print(f"{'=' * 60}")

    dollar_bars = pl.read_parquet(input_path)
    print(f"  Loaded {len(dollar_bars):,} bars")

    features, stationarity_history = generate_features(
        dollar_bars,
        windows=windows,
        ffd_d=ffd_d,
        check_stationarity=check_stationarity,
    )

    result = {
        "step": "features",
        "features": features,
        "feature_count": len(features.columns),
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        if check_stationarity and stationarity_history is not None:
            print("  [Viz] Plotting stationarity search...")
            d_vals, p_vals = stationarity_history
            viz.plot_stationarity_search(
                d_vals, p_vals, filename="step4_stationarity_search.png"
            )
        print("  [Viz] Plotting feature correlations...")
        viz.plot_feature_heatmap(features, filename="step4_feature_correlations.png")

    return result


def run_step_weights(
    input_path: str,
    decay: float = 0.9,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: calculate sample weights."""
    print(f"\n{'=' * 60}")
    print("STEP: Calculating Sample Weights")
    print(f"{'=' * 60}")

    labeled = pl.read_parquet(input_path)
    print(f"  Loaded {len(labeled):,} labeled events")

    events = labeled.select(["t1", "tr"]) if "t1" in labeled.columns else labeled
    weights = calculate_weights(events, decay=decay)

    result = {
        "step": "weights",
        "weights": weights,
        "weights_count": len(weights),
    }

    if visualize:
        print("  [Viz] Sample weights calculated (no viz for this step)")

    return result


def run_step_cv(
    features_path: str,
    labels_path: str,
    n_splits: int = 5,
    embargo: float = 0.1,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: cross-validation."""
    print(f"\n{'=' * 60}")
    print("STEP: Cross-Validation (Purged K-Fold)")
    print(f"{'=' * 60}")

    features = pl.read_parquet(features_path)
    labeled = pl.read_parquet(labels_path)
    print(f"  Features: {len(features.columns)} columns, {len(features):,} rows")

    splits = run_cross_validation(features, labeled, n_splits=n_splits, embargo=embargo)

    result = {
        "step": "cv",
        "splits": splits,
        "n_splits": len(splits),
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        print("  [Viz] Plotting CV timeline...")
        viz.plot_cv_timeline(
            splits, n_samples=len(features), filename="step6_cv_timeline.png"
        )

    return result


def run_step_meta(
    features_path: str,
    labels_path: str,
    weights_path: str,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: meta-labeling."""
    print(f"\n{'=' * 60}")
    print("STEP: Meta-Labeling Pipeline")
    print(f"{'=' * 60}")

    features = pl.read_parquet(features_path)
    labeled = pl.read_parquet(labels_path)
    weights = (
        pl.read_parquet(weights_path) if Path(weights_path).exists() else pl.DataFrame()
    )

    print(f"  Features: {len(features.columns)} columns")
    print(f"  Labels: {len(labeled):,} rows")

    predictions = run_meta_labeling(features, labeled, weights)

    result = {
        "step": "meta",
        "predictions": predictions,
        "predictions_count": len(predictions),
    }

    if visualize and not predictions.is_empty():
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        print("  [Viz] Plotting meta-model performance...")
        y_true_binary = (predictions["actual"].to_numpy() > 0).astype(int)
        viz.plot_meta_performance(
            y_true_binary,
            predictions["prediction"].to_numpy(),
            filename="step7_meta_performance.png",
        )

    return result


def run_step_bet(
    features_path: str,
    predictions_path: str,
    labels_path: str,
    threshold: float = 0.5,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: bet sizing."""
    print(f"\n{'=' * 60}")
    print("STEP: Bet Sizing")
    print(f"{'=' * 60}")

    features = pl.read_parquet(features_path)
    predictions = pl.read_parquet(predictions_path)
    labeled = pl.read_parquet(labels_path)

    print(f"  Predictions: {len(predictions):,}")

    bet_sizes = calculate_bet_sizes(features, predictions, labeled, threshold=threshold)

    result = {
        "step": "bet",
        "bet_sizes": bet_sizes,
        "bet_sizes_count": len(bet_sizes),
    }

    if visualize and not bet_sizes.is_empty():
        print("  [Viz] Bet sizes calculated (no viz for this step)")

    return result


def run_step_verify(
    labels_path: str,
    bet_sizes_path: str,
    n_trials: int = 1,
    acceptance_threshold: float = 0.95,
    visualize: bool = True,
    visual_analysis_dir: str = "visual_analysis",
) -> Dict[str, Any]:
    """Execute step: strategy verification."""
    print(f"\n{'=' * 60}")
    print("STEP: Strategy Verification (DSR)")
    print(f"{'=' * 60}")

    labeled = pl.read_parquet(labels_path)
    bet_sizes = pl.read_parquet(bet_sizes_path)

    print(f"  Bet sizes: {len(bet_sizes):,}")

    metrics = verify_strategy(
        labeled, bet_sizes, n_trials=n_trials, acceptance_threshold=acceptance_threshold
    )

    result = {
        "step": "verify",
        "metrics": metrics,
        "status": metrics.get("status", "UNKNOWN"),
    }

    if visualize:
        viz = AFMLVisualizer(output_dir=visual_analysis_dir)
        n_samples = min(len(bet_sizes), len(labeled))
        if n_samples > 0:
            bet_s = bet_sizes["bet_size"].head(n_samples).to_numpy()
            returns_s = (
                labeled["tr"].head(n_samples).to_numpy()
                if "tr" in labeled.columns
                else labeled["label"].head(n_samples).to_numpy()
            )
            strategy_returns = bet_s * returns_s
            print("  [Viz] Plotting equity curve...")
            viz.plot_equity_curve(strategy_returns, filename="step9_equity_curve.png")

    return result


def run_step_dispatcher(
    step: PipelineStep,
    args,
) -> Dict[str, Any]:
    """
    Dispatch execution to appropriate step runner.

    Args:
        step: Pipeline step to execute
        args: Parsed command line arguments

    Returns:
        Step execution result dictionary
    """
    viz = args.visualize
    viz_dir = args.visual_analysis_dir
    step_name = step.value if isinstance(step, PipelineStep) else step

    if step == PipelineStep.LOAD or step_name == "load":
        input_path = get_step_input_path(step_name, args.input)
        return run_step_load(
            input_path=input_path,
            visualize=viz,
            visual_analysis_dir=viz_dir,
            output_path=args.output,
        )

    elif step == PipelineStep.BARS or step_name == "bars":
        input_path = get_step_input_path(step_name, args.input)
        return run_step_bars(
            input_path=input_path,
            daily_target=args.daily_target,
            visualize=viz,
            visual_analysis_dir=viz_dir,
            streaming=args.streaming,
        )

    elif step == PipelineStep.LABELS or step_name == "labels":
        input_path = get_step_input_path(step_name, args.input)
        return run_step_labels(
            input_path=input_path,
            pt_sl=args.pt_sl,
            vertical_barrier_bars=args.vertical_barrier,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.FEATURES or step_name == "features":
        input_path = get_step_input_path(step_name, args.input)
        return run_step_features(
            input_path=input_path,
            windows=args.windows,
            ffd_d=args.ffd_d,
            check_stationarity=not args.no_auto_stationarity,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.WEIGHTS or step_name == "weights":
        input_path = get_step_input_path(step_name, args.input)
        return run_step_weights(
            input_path=input_path,
            decay=args.decay,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.CV or step_name == "cv":
        features_path = get_step_input_path("features", args.input)
        labels_path = get_step_input_path("labels", args.input)
        return run_step_cv(
            features_path=features_path,
            labels_path=labels_path,
            n_splits=args.n_splits,
            embargo=args.embargo,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.META or step_name == "meta":
        features_path = get_step_input_path("features", args.input)
        labels_path = get_step_input_path("labels", args.input)
        weights_path = get_step_input_path("weights", args.input)
        return run_step_meta(
            features_path=features_path,
            labels_path=labels_path,
            weights_path=weights_path,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.BET or step_name == "bet":
        features_path = get_step_input_path("features", args.input)
        predictions_path = get_step_input_path("meta", args.input)
        labels_path = get_step_input_path("labels", args.input)
        return run_step_bet(
            features_path=features_path,
            predictions_path=predictions_path,
            labels_path=labels_path,
            threshold=args.dsr_threshold,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    elif step == PipelineStep.VERIFY or step_name == "verify":
        labels_path = get_step_input_path("labels", args.input)
        bet_sizes_path = get_step_input_path("bet", args.input)
        return run_step_verify(
            labels_path=labels_path,
            bet_sizes_path=bet_sizes_path,
            n_trials=1,
            acceptance_threshold=args.dsr_threshold,
            visualize=viz,
            visual_analysis_dir=viz_dir,
        )

    else:
        raise ValueError(f"Unknown step: {step}")


def main():
    """Main entry point with documented parameters."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AFML Quant Factory: High-performance Polars ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Data & Execution Groups ---
    data_group = parser.add_argument_group("Data & Execution")
    data_group.add_argument(
        "input",
        nargs="?",
        default="data/BTCUSDT/parquet_db",
        help="Path to input (Parquet directory or file). Must contain Ticks or OHLCV."
    )
    data_group.add_argument(
        "--step",
        type=str,
        choices=PipelineStep.choices(),
        default=None,
        help=(
            "Run specific pipeline step: "
            "load (Standardize Ticks/OHLCV), "
            "bars (Generate Dollar Bars), "
            "labels (Triple Barrier), "
            "features (Alpha158 + FFD), "
            "weights (Uniqueness/Decay), "
            "cv (Purged K-Fold), "
            "meta (Model Training/Pred), "
            "bet (Sizing), "
            "verify (DSR/Backtest)."
        )
    )
    data_group.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for huge datasets (reduces memory usage)."
    )
    data_group.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override default output path for the current step."
    )

    # --- Sampling (AFML Chapter 2) ---
    sampling_group = parser.add_argument_group("Bar Sampling (AFML Ch. 2)")
    sampling_group.add_argument(
        "--daily-target",
        type=int,
        default=4,
        help="Target number of Dollar Bars per day. Adjust for sampling frequency."
    )

    # --- Labeling (AFML Chapter 3) ---
    label_group = parser.add_argument_group("Triple Barrier Labeling (AFML Ch. 3)")
    label_group.add_argument(
        "--pt-sl",
        nargs=2,
        type=float,
        default=[1.0, 1.0],
        help="Profit Taking (PT) and Stop Loss (SL) multipliers. Multiplied by volatility."
    )
    label_group.add_argument(
        "--vertical-barrier",
        type=int,
        default=12,
        help="Maximum holding period in number of bars (Vertical Barrier)."
    )

    # --- Features & Stationarity (AFML Chapter 5) ---
    feat_group = parser.add_argument_group("Features & Stationarity (AFML Ch. 5)")
    feat_group.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[5, 10, 20, 30, 50],
        help="Window sizes for technical indicators and rolling stats."
    )
    feat_group.add_argument(
        "--ffd-d",
        type=float,
        default=0.5,
        help="Fractional Differentiation (FFD) coefficient (initial guess)."
    )
    feat_group.add_argument(
        "--no-auto-stationarity",
        action="store_true",
        help="Use fixed --ffd-d; do not search for minimum d to achieve stationarity."
    )

    # --- Weights & Cross-Validation (AFML Chapter 4 & 7) ---
    cv_group = parser.add_argument_group("Sample Weights & CV (AFML Ch. 4/7)")
    cv_group.add_argument(
        "--decay",
        type=float,
        default=0.9,
        help="Time decay factor for sample weights. 1.0 = no decay."
    )
    cv_group.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds in Purged K-Fold Cross-Validation."
    )
    cv_group.add_argument(
        "--embargo",
        type=float,
        default=0.1,
        help="Embargo period as a fraction of the dataset (prevents leakage)."
    )
    cv_group.add_argument(
        "--dsr-threshold",
        type=float,
        default=0.95,
        help="Threshold for Deflated Sharpe Ratio (DSR) to accept strategy."
    )

    # --- Visualization ---
    viz_group = parser.add_argument_group("Visualization")
    viz_group.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Disable automatic plot generation for all pipeline steps."
    )
    parser.set_defaults(visualize=True)
    viz_group.add_argument(
        "--visual-analysis-dir",
        type=str,
        default="visual_analysis",
        help="Directory to save PNG plots and analysis artifacts."
    )

    # --- Optimization / Sweeps ---
    sweep_group = parser.add_argument_group("Parameter Sweeps (Optimization)")
    sweep_group.add_argument(
        "--jb-sweep",
        action="store_true",
        help="Run Jarque-Bera normality sweep for --daily-target instead of pipeline."
    )
    sweep_group.add_argument(
        "--jb-targets",
        nargs="+",
        type=int,
        default=[4, 20, 50, 60, 80, 100],
        help="List of daily_target values to test during JB sweep."
    )
    sweep_group.add_argument(
        "--jb-data-dir",
        type=str,
        default="data/BTCUSDT/parquet_db",
        help="Data source for the JB parameter sweep."
    )

    args = parser.parse_args()

    if args.jb_sweep:
        results = run_jb_parameter_sweep(
            data_dir=args.jb_data_dir,
            daily_targets=args.jb_targets,
            output_dir=args.visual_analysis_dir,
        )
    elif args.step:
        step = PipelineStep(args.step)
        print(f"\n{'=' * 60}")
        print(f"Running step: {step.value}")
        print(f"{'=' * 60}")
        results = run_step_dispatcher(step, args)
    else:
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
            visualize=args.visualize,
            visual_analysis_dir=args.visual_analysis_dir,
        )

    return results


if __name__ == "__main__":
    main()
