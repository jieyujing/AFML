"""
Feature Importance Analysis Script — AFML Pipeline

Reads the feature matrix produced by the upstream pipeline, performs:
1. Feature clustering via correlation-distance hierarchical clustering
2. Clustered MDA (Mean Decrease Accuracy/Log-loss) with PurgedKFold CV
3. Outputs importance rankings and generates visualisation plots

Usage:
    uv run python scripts/feature_importance_analysis.py

References:
    López de Prado, M. "Advances in Financial Machine Learning." Wiley, 2018. Ch. 4, 7, 8.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from afmlkit.importance.clustering import (
    cluster_features,
    get_feature_distance_matrix,
    hierarchical_clustering,
)
from afmlkit.importance.mda import clustered_mda

# --------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------- #
INPUT_PATH = Path("outputs/dollar_bars/feature_matrix.csv")
OUTPUT_DIR = Path("outputs/feature_importance")

# OHLCV and meta columns — excluded from feature importance analysis
# log_return 是当前 bar 的对数收益率 log(close/close_prev)，
# 与 TBM 标签存在直接因果关系，不可作为因子
META_COLS = [
    "open", "high", "low", "close", "volume", "vwap",
    "trades", "median_trade_size", "log_return",
    "bin", "t1", "avg_uniqueness", "return_attribution",
]

CV_N_SPLITS = 5
CV_EMBARGO_PCT = 0.01
MDA_N_REPEATS = 1
RANDOM_STATE = 42


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load feature_matrix.csv and split into features, labels, t1, sample_weight.

    :returns: (X_features, y_label, t1_series, sample_weight_series)
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"[Data] Loaded {df.shape[0]} samples × {df.shape[1]} columns from {path}")

    # Drop rows with NaN in critical columns
    df = df.dropna(subset=["bin", "t1"])
    df["t1"] = pd.to_datetime(df["t1"])

    y = df["bin"].astype(int)
    t1 = df["t1"]
    sample_weight = df["return_attribution"] if "return_attribution" in df.columns else None

    # Feature columns = everything except META_COLS
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()

    # Drop any feature columns that are entirely NaN
    nan_cols = X.columns[X.isna().all()]
    if len(nan_cols) > 0:
        print(f"[Data] Dropping all-NaN features: {nan_cols.tolist()}")
        X = X.drop(columns=nan_cols)

    # Forward-fill then backward-fill remaining NaN (edge cases at start/end)
    X = X.ffill().bfill()

    print(f"[Data] Features: {X.columns.tolist()}")
    print(f"[Data] Label distribution: {y.value_counts().to_dict()}")
    return X, y, t1, sample_weight


def plot_dendrogram(X: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save a dendrogram of the feature clustering."""
    dist = get_feature_distance_matrix(X)
    link = hierarchical_clustering(dist, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(
        link,
        labels=X.columns.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    ax.set_title("Feature Clustering Dendrogram (Ward Linkage)", fontsize=14)
    ax.set_ylabel("Distance", fontsize=12)
    fig.tight_layout()

    fpath = output_dir / "feature_dendrogram.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved dendrogram → {fpath}")


def plot_importance(df_result: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save a horizontal bar chart of cluster importance."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_result) * 0.6)))

    # Sort ascending for bottom-to-top display
    df_plot = df_result.sort_values("mean_importance", ascending=True)

    labels = [
        f"C{int(row.cluster_id)}: {', '.join(row.features[:3])}"
        + ("…" if len(row.features) > 3 else "")
        for _, row in df_plot.iterrows()
    ]
    y_pos = np.arange(len(labels))
    colors = [
        "#2ecc71" if v > 0 else "#e74c3c"
        for v in df_plot["mean_importance"]
    ]

    ax.barh(
        y_pos,
        df_plot["mean_importance"],
        xerr=df_plot["std_importance"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean Importance (neg-log-loss drop)", fontsize=11)
    ax.set_title("Clustered MDA Feature Importance", fontsize=14)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()

    fpath = output_dir / "clustered_mda_importance.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved importance chart → {fpath}")


def plot_distance_heatmap(X: pd.DataFrame, output_dir: Path) -> None:
    """Plot and save a correlation-distance heatmap."""
    dist = get_feature_distance_matrix(X)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(dist.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(dist.columns)))
    ax.set_yticks(range(len(dist.index)))
    ax.set_xticklabels(dist.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(dist.index, fontsize=7)
    ax.set_title("Feature Distance Matrix  D = √(0.5 × (1 − ρ))", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    fpath = output_dir / "feature_distance_heatmap.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved distance heatmap → {fpath}")


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    """Run the full feature importance analysis pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load data ---
    X, y, t1, sample_weight = load_data(INPUT_PATH)

    # --- Step 2: Feature clustering ---
    print("\n" + "=" * 60)
    print("STEP 2: Feature Clustering")
    print("=" * 60)

    clusters = cluster_features(X, method="ward")

    # Visualise clustering
    plot_dendrogram(X, OUTPUT_DIR)
    plot_distance_heatmap(X, OUTPUT_DIR)

    # --- Step 3: Clustered MDA with PurgedKFold ---
    print("\n" + "=" * 60)
    print("STEP 3: Clustered MDA (PurgedKFold + Log-loss)")
    print("=" * 60)

    df_importance = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        sample_weight=sample_weight,
        n_splits=CV_N_SPLITS,
        embargo_pct=CV_EMBARGO_PCT,
        n_repeats=MDA_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # Save results
    csv_path = OUTPUT_DIR / "clustered_mda_results.csv"
    df_importance.to_csv(csv_path, index=False)
    print(f"\n[Output] Saved importance results → {csv_path}")

    # Visualise importance
    plot_importance(df_importance, OUTPUT_DIR)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  CV Folds: {CV_N_SPLITS} (embargo={CV_EMBARGO_PCT:.1%})")
    print("  Scoring:  Weighted Log-loss")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("  Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"    - {f.name}")


if __name__ == "__main__":
    main()
